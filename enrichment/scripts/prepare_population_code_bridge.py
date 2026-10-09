"""Reproduce the official-section-1 code bridge; fetch no population or other data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

REGISTRY_COLUMNS = [
    "TER",
    "KOD1",
    "KOD2",
    "KOD3",
    "KC",
    "RAZDEL",
    "NAME1",
    "Centrum",
    "NomDescr",
    "NomAkt",
    "Status",
    "DateUtv",
    "DateVved",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(
    registry_csv, crosswalk_2023, crosswalk_2024, assignments, population, selector, output
):
    inputs = {
        name: Path(value)
        for name, value in {
            "registry_csv": registry_csv,
            "crosswalk_2023": crosswalk_2023,
            "crosswalk_2024": crosswalk_2024,
            "assignments": assignments,
            "population": population,
            "selector": selector,
        }.items()
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    registry = pd.read_csv(registry_csv, sep=";", header=None, names=REGISTRY_COLUMNS, dtype=str)
    section = registry.loc[registry.RAZDEL == "1"].copy()
    for field, width in (("TER", 2), ("KOD1", 3), ("KOD2", 3), ("KOD3", 3)):
        if not section[field].str.fullmatch(rf"[0-9]{{{width}}}").all():
            raise ValueError(f"Official field width invalid: {field}")
    if section.empty or not section.KOD3.eq("000").all():
        raise ValueError("Expected municipality section with KOD3=000")
    section["official_oktmo_8"] = section.TER + section.KOD1 + section.KOD2
    section["official_oktmo_11"] = section.official_oktmo_8 + section.KOD3
    if section.official_oktmo_8.duplicated().any():
        raise ValueError("Official section 1 is not unique by code")
    with duckdb.connect() as connection:
        primary = set(
            connection.execute(
                "SELECT DISTINCT CAST(territory_id AS VARCHAR) FROM read_parquet(?)",
                [str(assignments)],
            )
            .df()
            .iloc[:, 0]
        )
    crosses = {
        year: pd.read_csv(path, dtype=str)
        for year, path in (
            ("2023", crosswalk_2023),
            ("2024", crosswalk_2024),
        )
    }
    for frame in crosses.values():
        if frame.territory_id.duplicated().any() or frame.oktmo_digits.duplicated().any():
            raise ValueError("Snapshot codes and territory IDs must be unique")
    cohort = crosses["2024"].loc[lambda frame: frame.territory_id.isin(primary)]
    if len(cohort) != len(primary):
        raise ValueError("The crosswalk must contain every primary territory")
    joined = cohort.merge(
        section,
        left_on="oktmo_digits",
        right_on="official_oktmo_11",
        how="left",
        validate="one_to_one",
    )
    safe = joined.loc[joined.official_oktmo_8.notna()]
    missing = joined.loc[joined.official_oktmo_8.isna()]
    regional_urls = set(
        re.findall(
            r"https://rosstat.gov.ru/dbscripts/munst/munst[0-9]{2}/?",
            Path(selector).read_bytes().decode("cp1251"),
        )
    )
    regional_urls = {url.rstrip("/") + "/" for url in regional_urls}
    targets = []
    for row in safe.itertuples():
        url = f"https://rosstat.gov.ru/dbscripts/munst/munst{row.TER}/"
        if url in regional_urls:
            targets.append(
                {
                    "region_url": url,
                    "oktmo": str(int(row.official_oktmo_8)),
                    "official_oktmo_8": row.official_oktmo_8,
                    "sber_oktmo_11": row.oktmo_digits,
                    "territory_id": int(row.territory_id),
                    "status": "candidate_request_exact_source_option_validation_required",
                }
            )
    source_population = pd.read_csv(population, dtype={"source_oktmo": str})
    if source_population.duplicated(
        [
            "source_region_url",
            "source_oktmo",
            "reference_date",
            "population_type",
        ]
    ).any():
        raise ValueError("Duplicate source population observation")
    linked = []
    for row in source_population.to_dict("records"):
        year = row["reference_date"][:4]
        if year not in crosses or not re.fullmatch(r"[0-9]+", row["source_oktmo"]):
            raise ValueError("Unexpected source date or code")
        region = re.fullmatch(
            r"https://rosstat.gov.ru/dbscripts/munst/munst([0-9]{2})/", row["source_region_url"]
        )
        if region is None:
            raise ValueError("Unrecognized regional source URL")
        candidate = section.loc[
            section.TER.eq(region[1])
            & section.official_oktmo_8.map(int).eq(int(row["source_oktmo"]))
        ]
        if len(candidate) != 1:
            raise ValueError("No unique official section 1 code for source option")
        official = candidate.iloc[0]
        year_cross = crosses[year]
        match = year_cross.loc[year_cross.oktmo_digits.eq(official.official_oktmo_11)]
        if len(match) != 1:
            raise ValueError("No unique municipality in reference-year crosswalk")
        row.update(
            territory_id=int(match.iloc[0].territory_id),
            official_oktmo_8=official.official_oktmo_8,
            sber_oktmo_11=official.official_oktmo_11,
            crosswalk_snapshot_date=f"{year}-01-01",
            join_status="accepted_pilot_source_option_and_official_section1",
            registry_file_version="2024-01-01",
            in_primary_cohort=match.iloc[0].territory_id in primary,
        )
        linked.append(row)
    if not linked:
        raise ValueError("Population pilot must not be empty")
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(linked)
    frame.to_csv(output / "population_linked_pilot.csv", index=False)
    with duckdb.connect() as connection:
        connection.register("linked_rows", frame)
        connection.execute(
            "COPY linked_rows TO ? (FORMAT PARQUET)",
            [str(output / "population_linked_pilot.parquet")],
        )
    missing[["territory_id", "oktmo", "region_name", "municipal_district_name"]].to_csv(
        output / "primary_unmatched_2024.csv",
        index=False,
    )
    (output / "primary_population_candidate_targets.json").write_text(
        json.dumps(targets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "primary_count": len(primary),
        "exact_official_section1_matches": len(safe),
        "unmatched_count": len(missing),
        "prepared_candidate_targets": len(targets),
        "linked_pilot_cells": len(frame),
        "linked_pilot_municipalities": int(frame.territory_id.nunique()),
        "population_mass_fetch_run": False,
        "input_hashes": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "processor_sha256": sha256(__file__),
        "output_hashes": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "registry-csv",
        "crosswalk-2023",
        "crosswalk-2024",
        "assignments",
        "population",
        "selector",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
