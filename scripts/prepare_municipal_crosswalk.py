"""Prepare a coded, versioned crosswalk from the unmodified SberIndex archive.

Public source: https://www.sberbank.com/common/files/t_dict_municipal.rar
Metadata covers January 1, 2018 through January 1, 2024. ``year_to=9999``
is an open-ended source sentinel, not evidence of completeness after that date.
Names are retained as attributes and are never used to join territories.

Extract the original RAR with an existing archiver before running this command.
The XLSX reader uses the standard library; it does not evaluate workbook formulas.
GeoPackage inspection uses read-only SQLite and does not transform geometry.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from municipal_profiles.data import load_panel

SOURCE_URL = "https://www.sberbank.com/common/files/t_dict_municipal.rar"
METADATA_URL = (
    "https://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/"
    "metadata_municipal_dict_sberindex_2.pdf"
)
COVERAGE_START = "2018-01-01"
COVERAGE_END = "2024-01-01"
OPEN_END = 9999
XLSX_NAME = "t_dict_municipal_districts.xlsx"
GPKG_NAME = "t_dict_municipal_districts_poly.gpkg"
SHEET_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REQUIRED_COLUMNS = (
    "territory_id",
    "oktmo",
    "region_code",
    "region_name",
    "municipal_district_name",
    "municipal_district_name_short",
    "municipal_district_type",
    "municipal_district_status",
    "municipal_district_center",
    "municipal_district_center_lat",
    "municipal_district_center_lon",
    "shape",
    "shape_linked_oktmo",
    "source_rosstat",
    "change_id_from",
    "change_id_to",
    "year_from",
    "year_to",
)


def sha256_file(path: Path) -> str:
    """Hash original bytes without loading the entire file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_source_xlsx(path: Path) -> pd.DataFrame:
    """Read the archive's single flat worksheet, preserving text codes verbatim."""
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", SHEET_NS)
        if len(sheets) != 1:
            raise ValueError("Expected exactly one crosswalk worksheet")
        rel_id = sheets[0].attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = [rel.attrib["Target"] for rel in rels if rel.attrib["Id"] == rel_id]
        if len(targets) != 1 or ".." in Path(targets[0]).parts:
            raise ValueError("Ambiguous or unsafe worksheet target")
        target = targets[0].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(text.text or "" for text in item.findall(".//s:t", SHEET_NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        rows = []
        for row in ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", SHEET_NS):
            values = {}
            for cell in row:
                if cell.find("s:f", SHEET_NS) is not None:
                    raise ValueError("Formula cell in the source crosswalk")
                column = re.sub(r"\d", "", cell.attrib["r"])
                value = cell.find("s:v", SHEET_NS)
                value = value.text if value is not None else None
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", SHEET_NS))
                elif cell.attrib.get("t") in {"e", "b"}:
                    raise ValueError("Error or boolean cell in the source crosswalk")
                values[column] = value
            if any(value is not None for value in values.values()):
                rows.append(values)
    if len(rows) < 2 or len(set(rows[0].values())) != len(rows[0]):
        raise ValueError("Empty source or duplicate crosswalk headers")
    if any(set(row) - set(rows[0]) for row in rows[1:]):
        raise ValueError("Data cell has no column header")
    return pd.DataFrame(rows[1:]).reindex(columns=list(rows[0])).rename(columns=rows[0])


def validate_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous keys/intervals; retain nullable statuses and source text."""
    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Unexpected crosswalk schema; inspect the source before adapting it")
    data = frame.loc[:, REQUIRED_COLUMNS].copy().replace("", None)
    if data.empty:
        raise ValueError("Empty crosswalk")
    for field in ("territory_id", "region_code", "shape", "year_from", "year_to"):
        values = pd.to_numeric(data[field], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or not values.eq(values // 1).all():
            raise ValueError(f"{field} must contain finite integers")
        data[field] = values.astype("int64")
    if not data["territory_id"].gt(0).all() or not data["region_code"].between(1, 99).all():
        raise ValueError("Invalid territory_id or region_code")
    if not data["shape"].isin([1, 2, 3]).all():
        raise ValueError("Unexpected shape category")
    if not data["year_from"].between(2018, 2024).all():
        raise ValueError("Version starts outside documented source coverage")
    if not (data["year_to"].between(2019, 2024) | data["year_to"].eq(OPEN_END)).all():
        raise ValueError("Version ends outside documented source coverage")
    if not data["year_from"].lt(data["year_to"]).all():
        raise ValueError("Invalid or zero-length validity interval")
    for field in (
        "oktmo",
        "region_name",
        "municipal_district_name",
        "municipal_district_name_short",
        "municipal_district_type",
        "source_rosstat",
    ):
        if data[field].isna().any():
            raise ValueError(f"Null required attribute: {field}")
    if (
        not data["oktmo"]
        .map(
            lambda value: (
                isinstance(value, str)
                and re.fullmatch(r"\d{2}-\d{3}-\d{3}-\d{3}", value) is not None
            )
        )
        .all()
    ):
        raise ValueError("OKTMO must be the original 2-3-3-3 hyphenated text code")
    data["oktmo_digits"] = data["oktmo"].str.replace("-", "", regex=False)
    if data.duplicated(["territory_id", "year_from"]).any():
        raise ValueError("Duplicate territory_id/year_from version key")
    for key in ("territory_id", "oktmo_digits"):
        ordered = data.sort_values([key, "year_from"])
        previous_end = ordered.groupby(key)["year_to"].shift()
        if ordered["year_from"].lt(previous_end).any():
            raise ValueError(f"Overlapping validity intervals for {key}")
    for field, bound in (
        ("municipal_district_center_lat", 90),
        ("municipal_district_center_lon", 180),
    ):
        data[field] = pd.to_numeric(data[field], errors="raise")
        present = data[field].notna()
        if not data.loc[present, field].between(-bound, bound).all():
            raise ValueError(f"Out-of-range center coordinate: {field}")
    if (
        not data["municipal_district_center_lat"]
        .isna()
        .eq(data["municipal_district_center_lon"].isna())
        .all()
    ):
        raise ValueError("Unpaired center coordinates")
    return data.sort_values(["territory_id", "year_from"]).reset_index(drop=True)


def snapshot_crosswalk(data: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return a January 1 snapshot, refusing extrapolation of the 9999 sentinel."""
    if isinstance(year, bool) or year not in range(2018, 2025):
        raise ValueError("Supported snapshots are January 1 of 2018 through 2024 only")
    snapshot = data.loc[data["year_from"].le(year) & data["year_to"].gt(year)].copy()
    if snapshot["territory_id"].duplicated().any() or snapshot["oktmo_digits"].duplicated().any():
        raise ValueError("Snapshot is not one-to-one between territory_id and OKTMO")
    snapshot.insert(0, "snapshot_date", f"{year}-01-01")
    snapshot["metadata_coverage_through"] = COVERAGE_END
    return snapshot.sort_values("territory_id").reset_index(drop=True)


def inspect_geopackage(path: Path, crosswalk: pd.DataFrame) -> dict:
    """Check coded geometry coverage with read-only SQL, without spatial refitting."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT territory_id, year_from, year_to, osm_ref, osm_vers, "
            "geom IS NULL AS geometry_null, length(geom) AS geometry_bytes "
            "FROM t_dict_municipal_districts_poly",
            connection,
        )
        content = pd.read_sql_query("SELECT * FROM gpkg_contents", connection).to_dict("records")
        geometry = pd.read_sql_query("SELECT * FROM gpkg_geometry_columns", connection).to_dict(
            "records"
        )
    if frame.isna().any().any() or frame["geometry_null"].any():
        raise ValueError("Null GeoPackage key, provenance or geometry")
    for field in ("territory_id", "year_from", "year_to"):
        values = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(values).all() or not values.eq(values // 1).all():
            raise ValueError("Invalid integer geometry key or validity bound")
        frame[field] = values.astype("int64")
    if frame["territory_id"].duplicated().any():
        raise ValueError("GeoPackage has multiple geometries per territory_id; review cardinality")
    if set(frame["territory_id"]) != set(crosswalk["territory_id"]):
        raise ValueError("GeoPackage and crosswalk territory sets differ")
    extents = crosswalk.groupby("territory_id").agg(
        year_from=("year_from", "min"), year_to=("year_to", "max")
    )
    actual = frame.set_index("territory_id").sort_index()
    if not actual[["year_from", "year_to"]].equals(extents):
        raise ValueError("Geometry validity differs from coded territory lifetime")
    return {
        "rows": len(frame),
        "unique_territory_ids": int(frame["territory_id"].nunique()),
        "geometry_null_rows": int(frame["geometry_null"].sum()),
        "minimum_geometry_bytes": int(frame["geometry_bytes"].min()),
        "territory_ids_equal_crosswalk": True,
        "validity_equal_territory_lifetime": True,
        "contents": content,
        "geometry_columns": geometry,
        "limitation": "Non-null geometry and key coverage checked; topology not validated.",
    }


def coverage_record(snapshot: pd.DataFrame, ids: set[int]) -> dict:
    """Measure an ID-only join without imputation or name matching."""
    present = set(snapshot["territory_id"])
    return {
        "requested_ids": len(ids),
        "matched_ids": len(ids & present),
        "unmatched_ids": sorted(ids - present),
    }


def prepare(source_dir: Path, consumption: Path, output_dir: Path, archive: Path | None) -> dict:
    """Prepare source versions and two dated snapshots, with byte-level lineage."""
    xlsx, gpkg = source_dir / XLSX_NAME, source_dir / GPKG_NAME
    data = validate_crosswalk(read_source_xlsx(xlsx))
    geometry_audit = inspect_geopackage(gpkg, data)
    full_panel = load_panel(consumption, complete_only=False)
    all_ids = set(full_panel["territory_id"])
    month_counts = full_panel.groupby("territory_id")["date"].nunique()
    primary_ids = set(month_counts.index[month_counts.eq(full_panel["date"].nunique())])
    snapshots = {year: snapshot_crosswalk(data, year) for year in (2023, 2024)}
    sources = [xlsx, gpkg, consumption] + ([archive] if archive is not None else [])
    audit = {
        "source_url": SOURCE_URL,
        "metadata_url": METADATA_URL,
        "license": "CC BY-SA 4.0 as stated by SberIndex; attribute SberIndex, Rosstat and OSM.",
        "metadata_coverage_start": COVERAGE_START,
        "metadata_coverage_through": COVERAGE_END,
        "validity_semantics": "year_from inclusive, year_to exclusive; 9999 open source sentinel",
        "interpretation_limit": (
            "The 2024 snapshot describes January 1 only. No completeness during or after 2024 "
            "is inferred. This coded crosswalk does not establish TERSON-MO equivalence."
        ),
        "source_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sources
        ],
        "rows": len(data),
        "unique_territory_ids": int(data["territory_id"].nunique()),
        "regions": int(data["region_code"].nunique()),
        "duplicate_version_keys": int(data.duplicated(["territory_id", "year_from"]).sum()),
        "null_counts": {column: int(count) for column, count in data.isna().sum().items()},
        "version_start_counts": data["year_from"].value_counts().sort_index().to_dict(),
        "version_end_counts": data["year_to"].value_counts().sort_index().to_dict(),
        "codes_per_territory_counts": (
            data.groupby("territory_id")["oktmo_digits"].nunique().value_counts().sort_index()
        ).to_dict(),
        "snapshots": {},
        "geometry": geometry_audit,
        "join_policy": "Exact territory_id only. No joins, repairs or disambiguation by name.",
    }
    for year, snapshot in snapshots.items():
        audit["snapshots"][str(year)] = {
            "date": f"{year}-01-01",
            "rows": len(snapshot),
            "unique_territory_ids": int(snapshot["territory_id"].nunique()),
            "unique_oktmo_codes": int(snapshot["oktmo_digits"].nunique()),
            "all_consumption": coverage_record(snapshot, all_ids),
            "balanced_consumption": coverage_record(snapshot, primary_ids),
            "type_counts": snapshot["municipal_district_type"].value_counts().to_dict(),
            "primary_type_counts": snapshot.loc[
                snapshot["territory_id"].isin(primary_ids), "municipal_district_type"
            ]
            .value_counts()
            .to_dict(),
        }
    paired = snapshots[2023].merge(
        snapshots[2024],
        on="territory_id",
        how="inner",
        suffixes=("_2023", "_2024"),
        validate="one_to_one",
    )
    primary = paired.loc[paired["territory_id"].isin(primary_ids)]
    audit["primary_2023_to_2024"] = {
        "common_ids": len(primary),
        "changed_oktmo": int(primary["oktmo_2023"].ne(primary["oktmo_2024"]).sum()),
        "changed_type": int(
            primary["municipal_district_type_2023"]
            .ne(primary["municipal_district_type_2024"])
            .sum()
        ),
        "changed_region_code": int(
            primary["region_code_2023"].ne(primary["region_code_2024"]).sum()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = output_dir / "municipal_crosswalk_versions.csv"
    data.to_csv(versions, index=False)
    outputs = [versions]
    for year, snapshot in snapshots.items():
        path = output_dir / f"municipal_crosswalk_{year}-01-01.csv"
        snapshot.to_csv(path, index=False)
        outputs.append(path)
    audit["output_hashes"] = {path.name: sha256_file(path) for path in outputs}
    (output_dir / "municipal_crosswalk_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, required=True, help="Extracted original RAR folder"
    )
    parser.add_argument(
        "--consumption", type=Path, required=True, help="Original consumption.parquet"
    )
    parser.add_argument(
        "--archive", type=Path, help="Optional original RAR for additional SHA lineage"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/enrichment"))
    args = parser.parse_args()
    audit = prepare(args.source_dir, args.consumption, args.output_dir, args.archive)
    print(
        json.dumps(
            {
                "status": "PASS",
                "versions": audit["rows"],
                "territories": audit["unique_territory_ids"],
                "balanced_coverage": {
                    year: value["balanced_consumption"]
                    for year, value in audit["snapshots"].items()
                },
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
