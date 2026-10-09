"""Join reviewed external city evidence to municipality IDs without changing H1-H3."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(cohort, reference):
    if cohort.territory_id.duplicated().any() or reference.region_code.duplicated().any():
        raise ValueError("Nonunique cohort or regional reference")
    if reference.center_territory_id.duplicated().any():
        raise ValueError("One center ID assigned to multiple regions")
    result = cohort.merge(
        reference,
        left_on="region_code_2024",
        right_on="region_code",
        how="left",
        validate="many_to_one",
    )
    known = result.center_territory_id.notna()
    result["matches_reference_center_municipality"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result.loc[known, "matches_reference_center_municipality"] = (
        result.loc[known, "territory_id"] == result.loc[known, "center_territory_id"]
    )
    result["center_review_status"] = "unverified_region_reference"
    result.loc[known, "center_review_status"] = "other_municipality_in_covered_region"
    result.loc[
        result.matches_reference_center_municipality.fillna(False), "center_review_status"
    ] = "matches_reviewed_center_municipality"
    result.loc[result.region_code_2024.isin(["77", "78", "92"]), "center_review_status"] = (
        "federal_city_submunicipal_unit_not_comparable"
    )
    result.loc[result.region_code_2024.eq("47"), "center_review_status"] = (
        "city_verified_municipal_grain_unresolved"
    )
    result["run_id"] = "evaluation-20260914-v1"
    return result


def run(repo, config_path, output):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name, expected in config["inputs"].items():
        if sha(repo / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    cohort = pd.read_csv(repo / config["cohort"], dtype=str)
    reference = pd.read_csv(repo / config["reference"], dtype=str)
    versions = pd.read_csv(repo / config["versions"], dtype=str)
    snapshot = versions.loc[
        (versions.year_from.astype(int) <= 2024) & (versions.year_to.astype(int) > 2024)
    ]
    if snapshot.territory_id.duplicated().any():
        raise ValueError("Ambiguous source snapshot")
    linked = reference.merge(
        snapshot,
        left_on="center_territory_id",
        right_on="territory_id",
        validate="one_to_one",
        suffixes=("_ref", "_source"),
        how="left",
    )
    if not (
        linked.region_code_ref.eq(linked.region_code_source)
        & linked.center_oktmo.eq(linked.oktmo)
        & linked.municipality_name.eq(linked.municipal_district_name)
        & linked.municipal_district_type.eq("городской округ")
        & linked.center_city.eq(linked.municipal_district_center.str.removeprefix("г "))
    ).all():
        raise ValueError("Reviewed center identity changed")
    result = classify(cohort, reference)
    if len(result) != 2016 or len(reference) != 80:
        raise ValueError("Unexpected audited cohort or reference size")
    fields = [
        "territory_id",
        "region_code_2024",
        "region_name_2024",
        "oktmo_2024",
        "municipal_district_type_2024",
        "municipal_district_status_2024",
        "municipal_district_center_2024",
        "center_city",
        "center_territory_id",
        "matches_reference_center_municipality",
        "center_review_status",
        "reference_id",
        "printed_pages",
        "temporal_scope",
        "review_date",
        "run_id",
    ]
    table = result[fields]
    coverage = table.groupby("center_review_status", dropna=False).size().reset_index(name="n_ids")
    flagged = table.loc[table.municipal_district_status_2024.eq("административный_центр_субъекта")]
    output.mkdir(parents=True)
    tables = {
        "municipality_center_review.csv": table,
        "coverage.csv": coverage,
        "source_flag_review.csv": flagged,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, lineterminator="\n")
    manifest = {
        "config": config,
        "config_sha256": sha(config_path),
        "processor_sha256": sha(__file__),
        "output_hashes": {name: sha(output / name) for name in tables},
        "n_ids": len(table),
        "n_reference_regions": len(reference),
        "coverage": coverage.to_dict("records"),
        "limits": [
            "External statistical reference published in 2024, reporting 2023/end-year context",
            "Not exhaustive legal status history for every day of 2024",
            "Unknown and noncomparable municipality grains are not converted to false",
            "Original Sber status and all H1-H3 calculations remain unchanged",
            "Reference city names were manually linked to explicit region and municipality IDs",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/regional_center_review.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.config, args.output)
