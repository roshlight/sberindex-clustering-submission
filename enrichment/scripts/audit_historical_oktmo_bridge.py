"""Audit official registry vintages and prepare year-specific population fetch candidates.

This is a code-identity audit, not a legal-effectivity or boundary adjudication.
DateUtv/DateVved are retained verbatim. They never select a population year, change
a Sber code, or supply an automatic replacement for an unmatched municipality.
No names are used to join. The source form must still confirm exact p_oktmo options.
"""

import argparse
import hashlib
import json
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
VINTAGES = {
    "2023-01-12": "data-20230112T1301-structure-20260210T1102.csv",
    "2023-02-08": "data-20230208T0902-structure-20260210T1102.csv",
    "2023-12-19": "data-20231219T1512-structure-20260210T1102.csv",
    "2024-01-01": "data-20240101T1101-structure-20260210T1102.csv",
    "2024-01-25": "data-20240125T1401-structure-20260210T1102.csv",
    "2024-12-27": "data-20241227T1412-structure-20260210T1102.csv",
}
SOURCE_BASE = "https://rosstat.gov.ru/opendata/7708234640-oktmo/"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_registry(path):
    frame = pd.read_csv(path, sep=";", header=None, names=REGISTRY_COLUMNS, dtype=str)
    section = frame.loc[frame.RAZDEL.eq("1")].copy()
    if section.empty:
        raise ValueError("Missing municipality section RAZDEL=1")
    for field, width in (("TER", 2), ("KOD1", 3), ("KOD2", 3), ("KOD3", 3)):
        if not section[field].str.fullmatch(rf"[0-9]{{{width}}}").fillna(False).all():
            raise ValueError(f"Invalid official code component {field}")
    if not section.KOD3.eq("000").all():
        raise ValueError("Municipality section requires KOD3=000")
    section["official_oktmo_8"] = section.TER + section.KOD1 + section.KOD2
    section["official_oktmo_11"] = section.official_oktmo_8 + section.KOD3
    if section.official_oktmo_11.duplicated().any():
        raise ValueError("Duplicate official section-1 code")
    return frame, section.set_index("official_oktmo_11", drop=False)


def code_evidence(code, reference_date, sber_row, registries, preferred):
    """Keep official presence, annual inclusion and independent confirmation separate."""
    matched = [v for v in preferred if code in registries[v].index]
    if not matched:
        raise ValueError(f"No official code evidence for {code} / {reference_date}")
    primary = registries[matched[0]].loc[code]
    dates = pd.to_datetime(
        pd.Series([primary.DateUtv, primary.DateVved]), dayfirst=True, errors="raise"
    )
    after_reference = bool((dates > pd.Timestamp(reference_date)).any())
    return {
        "reference_date": reference_date,
        "territory_id": int(sber_row.territory_id),
        "official_oktmo_8": primary.official_oktmo_8,
        "sber_oktmo_11": code,
        "official_code_present": True,
        "sber_annual_code_present": True,
        "independent_legal_effectivity_confirmed": None,
        "independent_boundary_equivalence_confirmed": None,
        "geography_basis": "Sber annual version; not an independent legal boundary audit",
        "sber_year_from": int(sber_row.year_from),
        "sber_year_to_exclusive": int(sber_row.year_to),
        "sber_source_rosstat": sber_row.source_rosstat,
        "registry_vintages": matched,
        "registry_sources": [SOURCE_BASE + VINTAGES[v] for v in matched],
        "registry_nomakt": primary.NomAkt,
        "registry_status": primary.Status,
        "registry_DateUtv": primary.DateUtv,
        "registry_DateVved": primary.DateVved,
        "registry_date_after_reference": after_reference,
        "temporal_review_required": after_reference,
        "fetch_eligible": True,
        "evidence_status": (
            "exact_code_and_sber_annual_version_temporal_review_required"
            if after_reference
            else "exact_code_and_sber_annual_version_source_option_validation_required"
        ),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(repo_root, registry_dir, output_dir):
    repo_root, registry_dir, output_dir = map(Path, (repo_root, registry_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; previous results are immutable")
    inputs = {}
    crosswalks = {}
    for year in (2023, 2024):
        path = repo_root / f"data/derived/enrichment/municipal_crosswalk_{year}-01-01.csv"
        frame = pd.read_csv(path, dtype=str)
        if frame.territory_id.duplicated().any() or frame.oktmo_digits.duplicated().any():
            raise ValueError("Crosswalk must be one-to-one by ID and full code")
        crosswalks[year] = frame
        inputs[f"crosswalk_{year}"] = path
    assignments = repo_root / "reports/initial_study/assignments.parquet"
    with duckdb.connect() as db:
        ids = set(
            db.execute(
                "SELECT DISTINCT CAST(territory_id AS VARCHAR) FROM read_parquet(?)",
                [str(assignments)],
            )
            .df()
            .iloc[:, 0]
        )
    inputs["assignments"] = assignments
    cohort = {
        year: frame.loc[frame.territory_id.isin(ids)].copy() for year, frame in crosswalks.items()
    }
    if any(len(frame) != len(ids) for frame in cohort.values()):
        raise ValueError("Incomplete primary-cohort crosswalk")
    registries, registry_audit = {}, []
    for version, name in VINTAGES.items():
        path = registry_dir / name
        full, section = read_registry(path)
        registries[version] = section
        inputs[f"registry_{version}"] = path
        row = {
            "file_version_label": version,
            "filename": name,
            "url": SOURCE_BASE + name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "total_rows": len(full),
            "section1_rows": len(section),
            "section1_unique_codes": section.index.nunique(),
            "section1_kod1_5prefix": int(section.KOD1.str.startswith("5").sum()),
            "max_DateVved": pd.to_datetime(section.DateVved, dayfirst=True)
            .max()
            .date()
            .isoformat(),
            "primary_2023_exact_codes": int(cohort[2023].oktmo_digits.isin(section.index).sum()),
            "primary_2024_exact_codes": int(cohort[2024].oktmo_digits.isin(section.index).sum()),
            "population_reference_date_inferred_from_filename": False,
        }
        registry_audit.append(row)
    year_preferences = {
        2023: ("2023-01-12", "2023-02-08"),
        2024: ("2023-12-19", "2024-01-25"),
    }
    links, grouped_targets = [], {}
    for year, frame in cohort.items():
        for row in frame.sort_values("territory_id", key=lambda s: s.astype(int)).itertuples():
            evidence = code_evidence(
                row.oktmo_digits, f"{year}-01-01", row, registries, year_preferences[year]
            )
            links.append(evidence)
            official8 = evidence["official_oktmo_8"]
            region_url = f"https://rosstat.gov.ru/dbscripts/munst/munst{official8[:2]}/"
            source_option_candidate = str(int(official8))
            key = region_url, source_option_candidate
            target = grouped_targets.setdefault(
                key,
                {
                    "region_url": region_url,
                    "oktmo": source_option_candidate,
                    "reference_links": [],
                    "provenance": {
                        "status": "fetch_candidate_exact_p_oktmo_validation_required",
                        "code_representation": (
                            "numeric representation of a unique official RAZDEL1 code"
                        ),
                        "name_join_used": False,
                        "legal_or_boundary_confirmation_implied": False,
                    },
                },
            )
            target["reference_links"].append(evidence)
    link_frame = pd.DataFrame(links)
    old = pd.read_csv(
        repo_root / "reports/enrichment_v1/population_bridge/primary_unmatched_2024.csv", dtype=str
    )
    inputs["previous_unmatched"] = (
        repo_root / "reports/enrichment_v1/population_bridge/primary_unmatched_2024.csv"
    )
    actual_missing = set(
        cohort[2024].loc[
            ~cohort[2024].oktmo_digits.isin(registries["2024-01-01"].index), "territory_id"
        ]
    )
    if set(old.territory_id) != actual_missing:
        raise ValueError("Previous unmatched list does not reproduce")
    compared = cohort[2023].merge(
        cohort[2024], on="territory_id", suffixes=("_2023", "_2024"), validate="one_to_one"
    )
    compared["code_changed"] = compared.oktmo_digits_2023.ne(compared.oktmo_digits_2024)
    compared["type_changed"] = compared.municipal_district_type_2023.ne(
        compared.municipal_district_type_2024
    )
    changed = compared.loc[compared.code_changed | compared.type_changed].copy()
    versions_path = repo_root / "data/derived/enrichment/municipal_crosswalk_versions.csv"
    inputs["sber_versions"] = versions_path
    versions = pd.read_csv(versions_path, dtype=str)
    historical_candidates = (
        versions.loc[
            versions.territory_id.isin(actual_missing)
            & versions.oktmo_digits.isin(registries["2024-01-01"].index)
        ]
        .groupby("territory_id")
        .oktmo_digits.agg(lambda values: "|".join(sorted(set(values))))
    )
    recovery = compared.loc[compared.territory_id.isin(actual_missing)].copy()
    recovery["class_389"] = recovery.code_changed.map(
        {
            True: "code_changed_between_sber_2023_2024",
            False: "2024_code_already_present_in_sber_2023",
        }
    )
    recovery["exact_code_in_2023_12_19_registry"] = recovery.oktmo_digits_2024.isin(
        registries["2023-12-19"].index
    )
    recovery["exact_code_in_2024_01_25_registry"] = recovery.oktmo_digits_2024.isin(
        registries["2024-01-25"].index
    )
    recovery["older_sber_version_codes_in_problematic_file"] = recovery.territory_id.map(
        historical_candidates
    ).fillna("")
    recovery["old_code_used_as_replacement"] = False
    recovery["recovery_basis"] = "unchanged exact 2024 code in two neighboring official vintages"
    for frame in (changed, recovery):
        frame["same_sber_territory_id"] = True
        frame["independent_legal_effectivity_confirmed"] = None
        frame["independent_boundary_equivalence_confirmed"] = None
    all_compared = crosswalks[2023].merge(
        crosswalks[2024], on="territory_id", suffixes=("_2023", "_2024"), validate="one_to_one"
    )
    all_changes = int(
        (
            all_compared.oktmo_digits_2023.ne(all_compared.oktmo_digits_2024)
            | all_compared.municipal_district_type_2023.ne(
                all_compared.municipal_district_type_2024
            )
        ).sum()
    )
    summary = {
        "status": "code_bridge_ready_with_explicit_temporal_and_source_option_conditions",
        "primary_territories": len(ids),
        "reference_links": len(links),
        "unique_fetch_targets": len(grouped_targets),
        "official_code_present_links": int(link_frame.official_code_present.sum()),
        "sber_annual_code_present_links": int(link_frame.sber_annual_code_present.sum()),
        "independently_confirmed_legal_effectivity_links": 0,
        "independently_confirmed_boundary_equivalence_links": 0,
        "temporal_review_links": int(link_frame.temporal_review_required.sum()),
        "temporal_review_ids": sorted(
            link_frame.loc[link_frame.temporal_review_required, "territory_id"].unique().tolist()
        ),
        "previous_unmatched": len(old),
        "exactly_recovered_in_both_neighboring_vintages": int(
            (
                recovery.exact_code_in_2023_12_19_registry
                & recovery.exact_code_in_2024_01_25_registry
            ).sum()
        ),
        "remaining_missing_official_2024_codes": int(
            (~cohort[2024].oktmo_digits.isin(registries["2023-12-19"].index)).sum()
        ),
        "recovered_with_older_sber_code_in_problematic_file": len(historical_candidates),
        "old_code_replacements_applied": 0,
        "class_389_counts": recovery.class_389.value_counts().to_dict(),
        "type_389_counts": recovery.municipal_district_type_2024.value_counts().to_dict(),
        "primary_code_or_type_changes": len(changed),
        "all_territories_code_or_type_changes": all_changes,
        "primary_change_type_counts": (
            changed.municipal_district_type_2023 + " -> " + changed.municipal_district_type_2024
        )
        .value_counts()
        .to_dict(),
        "population_fetched_by_this_script": False,
        "population_joined_by_this_script": False,
        "name_joins_used": False,
        "DateUtv_DateVved_used_to_assign_population_year": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(registry_audit).to_csv(output_dir / "registry_vintage_audit.csv", index=False)
    link_frame.to_csv(output_dir / "reference_code_evidence.csv", index=False)
    link_frame.loc[link_frame.temporal_review_required].to_csv(
        output_dir / "temporal_review_2023.csv", index=False
    )
    review_links = link_frame.loc[
        link_frame.temporal_review_required, ["territory_id", "reference_date"]
    ].copy()
    review_links["reason"] = (
        "Sber annual code present, but registry DateUtv or DateVved is later than reference_date; "
        "independent early applicability and boundary/date equivalence remain unconfirmed"
    )
    review_links.to_csv(output_dir / "temporal_review_links.csv", index=False)
    recovery.to_csv(output_dir / "recovered_389.csv", index=False)
    changed.to_csv(output_dir / "primary_78_code_type_changes.csv", index=False)
    targets = [grouped_targets[key] for key in sorted(grouped_targets)]
    write_json(output_dir / "population_fetch_candidate_targets.json", targets)
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "status": "complete",
        "scope": __doc__,
        "inputs": {
            k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for k, p in inputs.items()
        },
        "processor_sha256": sha256(__file__),
        "output_hashes": {p.name: sha256(p) for p in sorted(output_dir.iterdir())},
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
