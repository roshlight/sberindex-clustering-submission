"""Offline, hash-checked replay of round-four geographic evidence.

This inspects the publisher's delivered version records. Identical stored geometry
is deliberately not promoted to independent legal boundary equivalence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import pandas as pd

IDS = (208, 217, 221, 1051)
YEARS = (2023, 2024)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_version(records: list[dict], territory_id: int, year: int) -> dict:
    selected = [
        row
        for row in records
        if int(row["territory_id"]) == territory_id
        and int(row["year_from"]) <= year < int(row["year_to"])
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one version: {territory_id}/{year}: {len(selected)}")
    return selected[0]


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def inspect_versions(gpkg: Path, xlsx: Path) -> tuple[list[dict], list[dict], dict]:
    with sqlite3.connect(gpkg.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        geometry = [
            dict(row)
            for row in connection.execute(
                "SELECT fid, territory_id, year_from, year_to, osm_ref, osm_vers, geom "
                "FROM t_dict_municipal_districts_poly "
                "WHERE CAST(territory_id AS INTEGER) IN (208,217,221,1051)"
            )
        ]
        contents = dict(
            connection.execute(
                "SELECT table_name, last_change, srs_id FROM gpkg_contents "
                "WHERE table_name='t_dict_municipal_districts_poly'"
            ).fetchone()
        )
    attributes = pd.read_excel(xlsx).to_dict("records")
    annual, identities, blobs = [], [], {}
    for territory_id in IDS:
        selected_geometries = []
        for year in YEARS:
            geo = select_version(geometry, territory_id, year)
            attr = select_version(attributes, territory_id, year)
            blob = geo["geom"]
            if not isinstance(blob, bytes) or not blob.startswith(b"GP"):
                raise ValueError("Missing/invalid GeoPackage geometry blob")
            digest = hashlib.sha256(blob).hexdigest()
            blobs[f"geometry/{territory_id}_{digest[:12]}.gpkgblob"] = blob
            selected_geometries.append(blob)
            annual.append(
                {
                    "territory_id": territory_id,
                    "reference_year": year,
                    "name": attr["municipal_district_name_short"],
                    "oktmo_verbatim": attr["oktmo"],
                    "municipal_type": attr["municipal_district_type"],
                    "attribute_year_from": int(attr["year_from"]),
                    "attribute_year_to_exclusive": int(attr["year_to"]),
                    "geometry_fid": geo["fid"],
                    "geometry_year_from": geo["year_from"],
                    "geometry_year_to_exclusive": geo["year_to"],
                    "osm_ref_verbatim": geo["osm_ref"],
                    "osm_vers_verbatim": geo["osm_vers"],
                    "geometry_blob_bytes": len(blob),
                    "geometry_blob_sha256": digest,
                    "geometry_blob_path": f"geometry/{territory_id}_{digest[:12]}.gpkgblob",
                }
            )
        identities.append(
            {
                "territory_id": territory_id,
                "year_a": YEARS[0],
                "year_b": YEARS[1],
                "publisher_geometry_bytes_identical": selected_geometries[0]
                == selected_geometries[1],
                "independent_legal_boundary_equivalence": "unknown",
                "polygon_overlap_computed": False,
                "existing_geography_exclusion_retained": True,
                "city_1051_2023_calendar_review_retained": territory_id == 1051,
                "automatic_mask_change": False,
            }
        )
    return annual, identities, {"blobs": blobs, "gpkg_contents": contents}


def verified_inputs(raw_root: Path) -> list[dict]:
    sources = json.loads((raw_root / "inputs/source_ledger.json").read_text("utf-8"))
    for row in sources:
        if row["download_status"] != "downloaded":
            continue
        path = (raw_root / row["path"]).resolve()
        if not path.is_relative_to(raw_root.resolve()):
            raise ValueError("Source path escapes raw root")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Source fingerprint changed: {row['source_id']}")
    return sources


def build(raw_root: Path, output: Path, verify_only: bool = False) -> dict:
    sources = verified_inputs(raw_root)
    source_map = {row["source_id"]: row for row in sources}
    annual, identities, extra = inspect_versions(
        raw_root / source_map["sber_geometry"]["path"],
        raw_root / source_map["sber_versions"]["path"],
    )
    evidence = json.loads((raw_root / "inputs/reviewed_evidence.json").read_text("utf-8"))
    for row in evidence:
        row["source_url"] = source_map[row["source_id"]]["source_url"]
        row["source_sha256"] = source_map[row["source_id"]]["sha256"]
    requests = json.loads((raw_root / "inputs/missing_requests.json").read_text("utf-8"))
    outputs = {
        "source_ledger.csv": csv_bytes(sources),
        "source_ledger.json": json.dumps(sources, ensure_ascii=False, indent=2).encode(),
        "publisher_annual_versions.csv": csv_bytes(annual),
        "publisher_geometry_identity.csv": csv_bytes(identities),
        "evidence.csv": csv_bytes(evidence),
        "missing_requests.csv": csv_bytes(requests),
        **extra["blobs"],
    }
    raw_files = [
        {
            "path": path.relative_to(raw_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(raw_root.rglob("*"))
        if path.is_file()
    ]
    outputs["raw_inventory.json"] = json.dumps(raw_files, indent=2).encode()
    qa = {
        "status": "PASS",
        "scope": "publisher_version_selection_and_collected_source_integrity_only",
        "source_records": len(sources),
        "downloaded_or_reused_sources": sum(
            row["download_status"] == "downloaded" for row in sources
        ),
        "new_content_downloads": sum(row["origin"] == "new_download" for row in sources)
        - sum(row["download_status"] != "downloaded" for row in sources),
        "fresh_metadata_matches_previous_sha": source_map["sber_metadata_fresh"]["sha256"]
        == source_map["sber_metadata_fresh"]["previous_sha256"],
        "case_count": len(identities),
        "annual_records": len(annual),
        "equal_publisher_geometry_cases": sum(
            row["publisher_geometry_bytes_identical"] for row in identities
        ),
        "accepted_independent_historical_boundary_pairs": 0,
        "polygon_overlap_computed": False,
        "mask_changes": 0,
        "gpkg_contents": extra["gpkg_contents"],
        "raw_file_count": len(raw_files),
        "raw_bytes": sum(row["bytes"] for row in raw_files),
    }
    outputs["QA.json"] = json.dumps(qa, ensure_ascii=False, indent=2).encode()
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "input_configuration_sha256": {
            path.name: sha256(path) for path in sorted((raw_root / "inputs").glob("*.json"))
        },
        "outputs": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in outputs.items()
        },
        "legal_boundary_claim": "unresolved; no exclusions changed",
    }
    outputs["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    if verify_only:
        for name, data in outputs.items():
            if (output / name).read_bytes() != data:
                raise ValueError(f"Offline replay differs: {name}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in outputs.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.raw_root, args.output, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
