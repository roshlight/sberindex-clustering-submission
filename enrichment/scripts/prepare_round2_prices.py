"""Reconcile published city ISZh observations with annual official municipality codes.

No web access, fuzzy matching, imputation, or municipality-wide price estimation.
Inputs include preserved OCR text; exact corrections are documented, not inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

REGISTRY_COLUMNS = [
    "TER",
    "KOD1",
    "KOD2",
    "KOD3",
    "KC",
    "RAZDEL",
    "NAME",
    "CENTER",
    "NOTE",
    "CHANGE",
    "ACTION",
    "DateUtv",
    "DateVved",
]
REGISTRIES = {
    2023: "data-20230208T0902-structure-20260210T1102.csv",
    2024: "data-20240125T1401-structure-20260210T1102.csv",
}
ORDERS = {
    2023: ("975", "Prikaz_975_23122022.pdf", 60, 69, 282, "2023-01-01"),
    2024: ("665", "Prikaz_665_19122023.pdf", 13, 22, 279, "2024-01-01"),
}
# Literal OCR repairs verified against the corresponding preserved page image.
# A city name is never repaired using similarity to the target municipality.
OCR_REPAIRS = {
    ("665", 13, "18 Белгород"): "1. Белгород",
    ("665", 13, "э. Навля"): "5. Навля",
    ("665", 13, "7: Новозыбков"): "7. Новозыбков",
    ("665", 14, "44. (Скопин"): "44. Скопин",
    ("665", 16, "104. Ялта,"): "104. Ялта",
    ("975", 68, "259. шт Агинское"): "259. пгт Агинское",
    ("975", 68, "263. шт Палана"): "263. пгт Палана",
    ("975", 69, "278. Корсаков ."): "278. Корсаков",
}
REGION_HEADER_ALIASES = {
    "Архангельская область (кроме Ненецкого автономного округа)": "Архангельская область",
    "Тюменская область (кроме Ханты-Мансийского автономного": "Тюменская область",
    "Ханты-Мансийский авт. округ — Югра": "Ханты-Мансийский автономный округ — Югра",
    "Ненецкий авт. округ": "Ненецкий автономный округ",
    "Ямало-Ненецкий авт. округ": "Ямало-Ненецкий автономный округ",
    "Чукотский авт. округ": "Чукотский автономный округ",
    "Еврейская авт. область": "Еврейская автономная область",
    "Кемеровская область — Кузбасс": "Кемеровская область",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def city_literal(value: str) -> str:
    """Remove only explicit settlement type tokens; preserve case and ё/е."""
    return re.sub(r"^(?:г\.\s*|пгт(?:\.\s*|\s+))", "", value.strip())


def parse_city_lists(raw: Path, crosswalk: pd.DataFrame) -> pd.DataFrame:
    regions = crosswalk[["region_name", "region_code"]].drop_duplicates()
    region_map = dict(regions.itertuples(index=False, name=None))
    rows = []
    for year, (order, filename, first, last, count, effective) in ORDERS.items():
        pdf = raw / filename
        region = district = region_header = None
        year_rows = []
        for page in range(first, last + 1):
            for line_number, raw_line in enumerate(
                (raw / f"{order}-{page:02}.txt").read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                line = raw_line.strip()
                line = OCR_REPAIRS.get((order, page, line), line)
                # These are OCR margin/bullet characters, not text normalization.
                header = line.lstrip("(, ")
                if "федеральный округ" in header:
                    district = header
                    continue
                region_lookup = REGION_HEADER_ALIASES.get(header, header)
                if region_lookup in region_map:
                    region = region_lookup
                    region_header = header
                    continue
                match = re.fullmatch(r"\.?\s*(\d{1,3})\.\s*(?:—\s*)?(.+)", line)
                if not match:
                    continue
                ordinal, label = int(match[1]), match[2].strip()
                if region is None or district is None:
                    continue  # Order header/date occurs before the first region.
                city = city_literal(label)
                city_region = (
                    city if city in {"Москва", "Санкт-Петербург", "Севастополь"} else region
                )
                year_rows.append(
                    {
                        "year": year,
                        "order_number": order,
                        "order_effective_from": effective,
                        "ordinal": ordinal,
                        "official_city_label": label,
                        "city_name_key": city,
                        "official_region_header": region_header,
                        "region_name": city_region,
                        "region_code": region_map[city_region],
                        "federal_district": district,
                        "pdf_page": page,
                        "ocr_line_number": line_number,
                        "ocr_literal_line": raw_line,
                        "ocr_exact_repair_applied": (order, page, raw_line.strip()) in OCR_REPAIRS,
                        "city_list_source_url": "https://rosstat.gov.ru/storage/mediabank/"
                        + filename,
                        "city_list_source_sha256": sha(pdf),
                    }
                )
        if [r["ordinal"] for r in year_rows] != list(range(1, count + 1)):
            raise ValueError(f"Incomplete or unordered city list {order}")
        rows.extend(year_rows)
    result = pd.DataFrame(rows)
    if result.duplicated(["year", "city_name_key", "federal_district"]).any():
        raise ValueError("Ambiguous official city identity within year/federal district")
    return result


def load_registry(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, sep=";", names=REGISTRY_COLUMNS, dtype=str, keep_default_na=False)
    result["registry_row_number"] = range(1, len(result) + 1)
    result["official_oktmo_11"] = result["TER"] + result["KOD1"] + result["KOD2"] + result["KOD3"]
    result["city_name_key"] = result["NAME"].str.extract(
        r"^(?:г|пгт|рп|кп|дп|гп|п) (.+)$", expand=False
    )
    # Official OKTMO introduction: cities 001..049, urban settlements 051..099;
    # rural settlements 101..999. In particular literal "п" is not sufficient.
    result["urban_settlement_code"] = result.KOD3.astype(int).between(1, 49) | result.KOD3.astype(
        int
    ).between(51, 99)
    if not result["official_oktmo_11"].str.fullmatch(r"\d{11}").all():
        raise ValueError("Invalid official code structure")
    return result


def temporal_review_roles(role_dates: dict[str, str], year: int) -> list[str]:
    """Require separate evidence for unknown or post-January role dates.

    DateVved is retained as registry evidence, not asserted to be a legal boundary
    date. An adjudication of a parent code cannot automatically approve a city.
    """
    review = []
    for role, literal in role_dates.items():
        try:
            parsed = datetime.strptime(literal, "%d.%m.%Y")
        except (TypeError, ValueError):
            review.append(role)
            continue
        if parsed > datetime(year, 1, 1):
            review.append(role)
    return review


def bridge_year(prices, official, registry, crosswalk, cohort, year):
    """Use full city+region identity, then an existing two-level official parent row."""
    settlement_counts = (
        registry.loc[(registry.RAZDEL == "2") & registry.KOD3.ne("000")]
        .groupby(["TER", "KOD1"])
        .size()
    )
    parents = registry.loc[
        (registry.RAZDEL == "1") & (registry.KOD2 == "000") & (registry.KOD3 == "000"),
        ["official_oktmo_11", "NAME", "registry_row_number", "DateUtv", "DateVved", "CHANGE"],
    ].rename(
        columns={
            "NAME": "registry_parent_name",
            "registry_row_number": "registry_parent_row",
            "DateUtv": "parent_DateUtv",
            "DateVved": "parent_DateVved",
            "CHANGE": "parent_CHANGE",
        }
    )
    if parents.official_oktmo_11.duplicated().any():
        raise ValueError("Ambiguous parent registry code")
    parents = parents.merge(
        crosswalk, left_on="official_oktmo_11", right_on="oktmo_digits", validate="one_to_one"
    )
    parent_keys = parents.assign(
        TER=parents.oktmo_digits.str[:2], KOD1=parents.oktmo_digits.str[2:5]
    )
    city_rows = registry.loc[
        (registry.RAZDEL == "2") & registry.city_name_key.notna() & registry.urban_settlement_code
    ].merge(
        parent_keys[
            [
                "TER",
                "KOD1",
                "territory_id",
                "region_code",
                "region_name",
                "municipal_district_type",
                "municipal_district_name",
                "registry_parent_name",
                "registry_parent_row",
                "parent_DateUtv",
                "parent_DateVved",
                "parent_CHANGE",
                "oktmo_digits",
            ]
        ],
        on=["TER", "KOD1"],
        validate="many_to_one",
    )
    immediate_parents = registry.loc[
        registry.RAZDEL == "1",
        ["official_oktmo_11", "NAME", "registry_row_number", "DateUtv", "DateVved", "CHANGE"],
    ]
    city_rows["immediate_parent_key"] = city_rows.official_oktmo_11.str[:8] + "000"
    city_rows = city_rows.merge(
        immediate_parents.rename(
            columns={
                "official_oktmo_11": "immediate_parent_key",
                "NAME": "immediate_parent_name",
                "registry_row_number": "immediate_parent_row",
                "DateUtv": "immediate_parent_DateUtv",
                "DateVved": "immediate_parent_DateVved",
                "CHANGE": "immediate_parent_CHANGE",
            }
        ),
        on="immediate_parent_key",
        validate="many_to_one",
    )
    merged = prices.merge(
        official,
        on=["year", "city_name_key", "federal_district"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for record in merged.to_dict("records"):
        matches = city_rows.loc[
            (city_rows.city_name_key == record["city_name_key"])
            & (city_rows.region_code == record.get("region_code"))
        ]
        row = dict(record)
        row["registry_city_candidates"] = len(matches)
        row["candidate_official_city_codes"] = "|".join(matches.official_oktmo_11)
        row["bridge_status"] = "candidate_unresolved"
        row["temporal_review_required"] = False
        row["temporal_review_roles"] = ""
        row["context_scope"] = "city_context_only"
        row["whole_municipality_price_estimate"] = False
        row["year_geography_status"] = "conditional_annual_reference_codes_not_boundary_audit"
        if pd.isna(record.get("order_number")):
            row["bridge_reason"] = "not_in_corresponding_year_official_city_list"
        elif record["city_name_key"] in {"Москва", "Санкт-Петербург", "Севастополь"}:
            row["bridge_reason"] = "federal_city_no_broadcast_to_intracity_municipalities"
        elif len(matches) != 1:
            row["bridge_reason"] = "no_unique_exact_city_region_type_and_parent_hierarchy"
        else:
            match = matches.iloc[0]
            row.update(
                {
                    "territory_id": match.territory_id,
                    "source_city_oktmo_11": match.official_oktmo_11,
                    "source_city_registry_label": match.NAME,
                    "source_parent_oktmo_11": match.oktmo_digits,
                    "source_parent_registry_label": match.registry_parent_name,
                    "registry_city_row": match.registry_row_number,
                    "registry_parent_row": match.registry_parent_row,
                    "source_immediate_parent_oktmo_11": match.immediate_parent_key,
                    "source_immediate_parent_registry_label": match.immediate_parent_name,
                    "registry_immediate_parent_row": match.immediate_parent_row,
                    "registry_city_DateUtv": match.DateUtv,
                    "registry_city_DateVved": match.DateVved,
                    "registry_city_CHANGE": match.CHANGE,
                    "registry_parent_DateUtv": match.parent_DateUtv,
                    "registry_parent_DateVved": match.parent_DateVved,
                    "registry_parent_CHANGE": match.parent_CHANGE,
                    "registry_immediate_parent_DateUtv": match.immediate_parent_DateUtv,
                    "registry_immediate_parent_DateVved": match.immediate_parent_DateVved,
                    "registry_immediate_parent_CHANGE": match.immediate_parent_CHANGE,
                    "source_parent_listed_settlement_count": int(
                        settlement_counts[(match.TER, match.KOD1)]
                    ),
                    "city_only_listed_settlement_in_parent": int(
                        settlement_counts[(match.TER, match.KOD1)]
                    )
                    == 1,
                    "municipal_district_name": match.municipal_district_name,
                    "municipal_district_type": match.municipal_district_type,
                    "in_primary_cohort": int(match.territory_id) in cohort,
                    "bridge_status": "approved_evidence",
                    "bridge_reason": (
                        "exact_city_region_type_unique_official_section2_parent_section1_sber_code"
                    ),
                }
            )
            roles = temporal_review_roles(
                {
                    "city": match.DateVved,
                    "parent": match.parent_DateVved,
                    "immediate_parent": match.immediate_parent_DateVved,
                },
                year,
            )
            if roles:
                row.update(
                    {
                        "bridge_status": "temporal_review",
                        "bridge_reason": "role_specific_date_evidence_required_no_parent_overlay",
                        "temporal_review_required": True,
                        "temporal_review_roles": "|".join(roles),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; old results are immutable")
    repo, raw = args.repo_root, args.raw_dir
    inputs = []
    price_path = repo / "reports/enrichment_v3/prices/city_cost_of_living_2023_2024.csv"
    price = read_csv(price_path)
    inputs.extend([price_path, args.workbook])
    price["year"] = price.year.astype(int)
    workbook = load_workbook(args.workbook, read_only=True, data_only=True)
    for row in price.itertuples(index=False):
        cell = workbook[row.source_sheet][row.source_cell].value
        if row.value_status == "observed" and float(row.value) != cell:
            raise ValueError(f"CSV value differs from workbook at {row.source_cell}")
        if row.source_sha256 != sha(args.workbook):
            raise ValueError("Unexpected workbook source hash")
    # Country is a reference denominator, not an observed city.
    price = price.loc[price.territorial_level == "surveyed_city"].copy()
    price["city_name_key"] = price.source_city_name.map(city_literal)
    price["value"] = pd.to_numeric(price.value, errors="raise")
    assignments = repo / "reports/initial_study/assignments.parquet"
    import duckdb

    cohort = set(
        duckdb.sql(f"SELECT DISTINCT territory_id FROM read_parquet('{assignments.as_posix()}')")
        .df()
        .territory_id.astype(int)
    )
    inputs.append(assignments)
    crosswalks = {}
    for year in (2023, 2024):
        path = repo / f"data/derived/enrichment/municipal_crosswalk_{year}-01-01.csv"
        crosswalks[year] = read_csv(path)
        inputs.append(path)
    lists = parse_city_lists(raw, crosswalks[2024])
    bridges = []
    for year in (2023, 2024):
        registry_path = args.registry_dir / REGISTRIES[year]
        metadata_path = registry_path.with_suffix(".csv.download.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if sha(registry_path) != metadata["sha256"]:
            raise ValueError("Registry changed since downloaded")
        frame = bridge_year(
            price.loc[price.year == year],
            lists.loc[lists.year == year],
            load_registry(registry_path),
            crosswalks[year],
            cohort,
            year,
        )
        frame["registry_source_url"] = metadata["url"]
        frame["registry_source_sha256"] = metadata["sha256"]
        frame["registry_retrieved_at"] = metadata["retrieved_utc"]
        frame["registry_vintage"] = REGISTRIES[year].split("-structure")[0][5:]
        bridges.append(frame)
        inputs.extend([registry_path, metadata_path])
    result = pd.concat(bridges, ignore_index=True)
    approved = result.loc[result.bridge_status == "approved_evidence"]
    observed = approved.loc[approved.in_primary_cohort.eq(True) & approved.value.notna()].copy()
    counts = (
        observed.groupby(["year", "territory_id"])
        .size()
        .rename("observed_cities_per_municipality_year")
    )
    observed = observed.merge(counts, on=["year", "territory_id"], validate="many_to_one")
    # Emit the cohort grid only if one observed city attaches to this MO/year.
    grid = pd.MultiIndex.from_product(
        [[2023, 2024], sorted(cohort)], names=["year", "territory_id"]
    ).to_frame(index=False)
    grid["territory_id"] = grid.territory_id.astype(str)
    single = observed.loc[
        observed.observed_cities_per_municipality_year == 1,
        [
            "year",
            "territory_id",
            "source_city_name",
            "source_city_oktmo_11",
            "value",
            "context_scope",
        ],
    ]
    grid = grid.merge(single, on=["year", "territory_id"], how="left", validate="one_to_one")
    grid["value_status"] = grid.value.notna().map(
        {True: "observed_city_context", False: "no_approved_single_city_context"}
    )
    summary = {
        "scope": (
            "city_context_only; no municipality price estimate, no inflation or category deflator"
        ),
        "cohort_municipalities": len(cohort),
        "city_year_source_rows": len(price),
        "source_numeric_city_years": int(price.value.notna().sum()),
        "official_city_list_counts": lists.groupby("year").size().to_dict(),
        "approved_city_year_links_all_municipalities": len(approved),
        "approved_numeric_city_years_primary": len(observed),
        "cohort_numeric_context_by_year": grid.loc[grid.value.notna()]
        .groupby("year")
        .size()
        .to_dict(),
        "unresolved_reasons": result.loc[result.bridge_status != "approved_evidence"]
        .bridge_reason.value_counts()
        .to_dict(),
        "multiple_observed_cities_per_municipality_year": int((counts > 1).sum()),
        "methodology": (
            "ISZh yearly national weights; missing goods may use region/federal district/Russia "
            "prices (Order440 pp12-13)"
        ),
        "available_at": "Latest downloaded source vintage; not a historical point-in-time release",
    }
    args.output.mkdir(parents=True)
    tables = {
        "official_base_cities.csv": lists,
        "bridge_candidates.csv": result,
        "bridge_approved_evidence.csv": approved,
        "city_context_observations.csv": observed,
        "primary_cohort_city_context.csv": grid,
    }
    for filename, table in tables.items():
        table.to_csv(args.output / filename, index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    for order, filename, first, last, *_ in ORDERS.values():
        inputs.append(raw / filename)
        inputs.extend(raw / f"{order}-{page:02}.txt" for page in range(first, last + 1))
    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "script_sha256": sha(Path(__file__)),
        "inputs": [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in inputs],
        "outputs": [
            {"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(args.output.iterdir())
        ],
        "no_network": True,
        "no_numeric_imputation": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
