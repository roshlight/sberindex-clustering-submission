"""Offline context analysis: investment paths, SME record eligibility, geometry limits.

No model fitting, municipal SME join, workforce sum or source replacement is performed.
SME paths require the unmodified entire Parquet files from the round 4 release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

INPUTS = {
    "cases": "reports/data_round4/cases/evidence_ledger.csv",
    "rounding": "reports/data_round4/cases/rounding_comparison.csv",
    "prior_cases": "reports/round3_analysis/context/six_cases_economy_pairs.csv",
    "geography": "reports/data_round4/geography/publisher_geometry_identity.csv",
    "versions": "reports/data_round4/geography/publisher_annual_versions.csv",
    "sme_inventory": "reports/data_round4/sme/annual_inventory.csv",
    "sme_manifest": "reports/data_round4/sme/manifest.json",
    "sme_dictionary": "reports/data_round4/sme/source_dictionary_pages.json",
}
SCOPE = "excludes_small_business_and_investment_not_directly_observed"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8f")


def rounded_ratio(old: str, new: str) -> dict:
    """Conservative arithmetic bounds from displayed 0.1-million rounding, not a CI."""
    a, b, half = Decimal(old), Decimal(new), Decimal("0.05")
    if not a.is_finite() or not b.is_finite() or a <= half or b < half:
        raise ValueError("Positive values exceeding the rounding half-step are required")
    ratio = b / a
    low, high = (b - half) / (a + half), (b + half) / (a - half)
    return {
        "nominal_ratio_from_rounded_values": float(ratio),
        "nominal_ratio_rounding_lower": float(low),
        "nominal_ratio_rounding_upper": float(high),
        "nominal_growth_pct_from_rounded_values": float((ratio - 1) * 100),
        "nominal_growth_rounding_lower_pct": float((low - 1) * 100),
        "nominal_growth_rounding_upper_pct": float((high - 1) * 100),
        "uncertainty_kind": "arithmetic_rounding_bounds_not_confidence_interval",
    }


def collapse_case_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "unit": "million_rubles",
        "price_basis": "actual_current_prices",
        "scope": SCOPE,
        "period_type": "calendar_year",
        "model_join_accepted": "False",
        "independent_measurement": "False",
    }
    if any(not evidence[key].eq(value).all() for key, value in expected.items()):
        raise ValueError("Changed units, scope, period or evidence role")
    rows = []
    for (territory, year), cells in evidence.groupby(
        ["territory_id_context_candidate", "period"], sort=True
    ):
        if cells.value.nunique() != 1 or cells.source_municipality_label.nunique() != 1:
            raise ValueError("Conflicting published values/names cannot be silently collapsed")
        rows.append(
            {
                "territory_id_context_candidate": territory,
                "municipality": cells.source_municipality_label.iloc[0],
                "year": int(year),
                "value_million_rubles": cells.value.iloc[0],
                "rounding_half_step_million_rubles": "0.05",
                "price_basis": "actual_current_prices",
                "scope": SCOPE,
                "publication_cells": len(cells),
                "distinct_editions": cells.edition_year.nunique(),
                "evidence_ids": ";".join(sorted(cells.evidence_id)),
                "source_urls": json.dumps(sorted(set(cells.source_url)), ensure_ascii=False),
                "source_sha256s": ";".join(sorted(set(cells.source_sha256))),
                "model_join_accepted": False,
                "respondent_stability_verified": False,
            }
        )
    return pd.DataFrame(rows)


def investment_analysis(repo: Path, out: Path) -> dict:
    evidence = read_csv(repo / INPUTS["cases"])
    series = collapse_case_evidence(evidence)
    if len(series) != 10 or set(series.year) != set(range(2020, 2025)):
        raise ValueError("Expected exactly two five-year documentary paths")
    indexed = series.set_index(["territory_id_context_candidate", "year"])
    lower = [Decimal(indexed.loc[("331", year)].value_million_rubles) for year in range(2020, 2025)]
    upper = [Decimal(indexed.loc[("318", year)].value_million_rubles) for year in range(2020, 2025)]
    if lower[3] / max(lower[:3]) <= 40 or not lower[4] < min(lower[1:3]):
        raise ValueError("The documented Lower Kolyma interpretation needs reassessment")
    if any(b <= a for a, b in zip(upper, upper[1:], strict=False)):
        raise ValueError("The documented Upper Kolyma monotonic path needs reassessment")
    write_csv(series, out / "investment_paths.csv")
    comparisons = []
    for territory, cells in series.groupby("territory_id_context_candidate", sort=True):
        values = cells.set_index("year").value_million_rubles.to_dict()
        for a, b in [(2020, 2021), (2021, 2022), (2022, 2023), (2021, 2024), (2022, 2024)]:
            comparisons.append(
                {
                    "territory_id_context_candidate": territory,
                    "year_a": a,
                    "year_b": b,
                    "value_a_million_rubles": values[a],
                    "value_b_million_rubles": values[b],
                    **rounded_ratio(values[a], values[b]),
                    "use": "exploratory_retrospective_description_not_change_point_test",
                }
            )
    write_csv(pd.DataFrame(comparisons), out / "investment_longer_comparisons.csv")
    prior = read_csv(repo / INPUTS["prior_cases"])
    exact = prior.loc[
        prior.territory_id.isin(["318", "331"]) & prior.indicator_id.eq("8109001")
    ].copy()
    if len(exact) != 2:
        raise ValueError("Two frozen investment comparisons required")
    rounding = read_csv(repo / INPUTS["rounding"])
    if len(rounding) != 10 or not rounding.within_rounding_interval.eq("True").all():
        raise ValueError("Published comparisons did not preserve the established rounding check")
    write_csv(
        exact[
            [
                "territory_id",
                "value_2023",
                "value_2024",
                "unit",
                "nominal_growth_pct",
                "comparison_status",
                "respondent_circle_stable_independently_verified",
                "price_basis",
                "availability",
                "automatic_mask_change",
            ]
        ],
        out / "investment_2023_2024_exact_prior.csv",
    )
    return {"publication_cells": len(evidence), "documentary_case_years": len(series)}


def entity_audit(
    connection: duckdb.DuckDBPyConnection, table: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ephemeral row per INN, with ambiguity explicit; never sum workforce."""
    if table != "sme_source":
        raise ValueError("Only the local read-only source view is accepted")
    connection.execute("DROP TABLE IF EXISTS entity_audit")
    connection.execute(
        """
        CREATE TEMP TABLE entity_audit AS
        SELECT tax_number,
          count(*) AS source_rows,
          min(is_sole_trader) AS is_sole_trader,
          count(DISTINCT is_sole_trader) AS entity_types,
          count(DISTINCT nullif(trim(municipality_code), '')) AS code_variants,
          bool_or(municipality_code IS NULL OR trim(municipality_code) = '') AS any_missing_code,
          count(DISTINCT sme_category) AS category_variants,
          count(DISTINCT nullif(trim(main_nace_code), '')) AS nace_full_variants,
          count(DISTINCT CASE WHEN regexp_full_match(main_nace_code, '[0-9]{2}(\\.[0-9]+)*')
            THEN substring(main_nace_code, 1, 2) END) AS nace2_variants,
          min(CASE WHEN regexp_full_match(main_nace_code, '[0-9]{2}(\\.[0-9]+)*')
            THEN substring(main_nace_code, 1, 2) END) AS nace2,
          bool_or(main_nace_code IS NULL OR NOT coalesce(
            regexp_full_match(main_nace_code, '[0-9]{2}(\\.[0-9]+)*'), false)) AS any_bad_nace,
          count(DISTINCT employees_count) FILTER (WHERE employees_count >= 0)
            AS worker_value_variants,
          bool_or(employees_count < 0) AS any_negative_workers,
          bool_or(employees_count = 0) AS any_zero_workers,
          bool_or(employees_count > 0) AS any_positive_workers
        FROM sme_source GROUP BY tax_number
        """
    )
    audit = connection.execute(
        """
        SELECT is_sole_trader, count(*) AS tax_ids,
          sum(source_rows)::BIGINT AS source_rows,
          count(*) FILTER (WHERE source_rows > 1) AS tax_ids_multiple_rows,
          count(*) FILTER (WHERE entity_types != 1) AS tax_ids_ambiguous_entity_type,
          count(*) FILTER (WHERE worker_value_variants = 1 AND NOT
            coalesce(any_negative_workers, false)) AS tax_ids_valid_known_workers,
          count(*) FILTER (WHERE worker_value_variants = 0) AS tax_ids_no_valid_workers,
          count(*) FILTER (WHERE worker_value_variants > 1) AS tax_ids_conflicting_workers,
          count(*) FILTER (WHERE coalesce(any_negative_workers, false)) AS tax_ids_negative_workers,
          count(*) FILTER (WHERE any_zero_workers) AS tax_ids_zero_workers,
          count(*) FILTER (WHERE any_positive_workers) AS tax_ids_positive_workers,
          count(*) FILTER (WHERE code_variants > 1) AS tax_ids_multiple_codes,
          count(*) FILTER (WHERE any_missing_code) AS tax_ids_any_missing_code,
          count(*) FILTER (WHERE category_variants > 1) AS tax_ids_multiple_categories,
          count(*) FILTER (WHERE nace_full_variants > 1) AS tax_ids_multiple_full_nace,
          count(*) FILTER (WHERE nace2_variants > 1) AS tax_ids_multiple_nace2,
          count(*) FILTER (WHERE any_bad_nace) AS tax_ids_any_missing_or_invalid_nace,
          count(*) FILTER (WHERE nace2_variants = 1 AND NOT any_bad_nace)
            AS tax_ids_unambiguous_nace2,
          count(*) FILTER (WHERE entity_types = 1 AND code_variants = 1 AND NOT any_missing_code
            AND nace2_variants = 1 AND NOT any_bad_nace AND worker_value_variants = 1
            AND NOT coalesce(any_negative_workers, false)) AS record_candidate_known_workers
        FROM entity_audit GROUP BY is_sole_trader ORDER BY is_sole_trader
        """
    ).df()
    for column in ["tax_ids_valid_known_workers", "record_candidate_known_workers"]:
        audit[f"{column}_pct_of_tax_ids"] = 100 * audit[column] / audit.tax_ids
    audit["row_count_inflation_pct_vs_distinct_tax_ids"] = 100 * (
        audit.source_rows / audit.tax_ids - 1
    )
    audit["historical_municipal_join_ready"] = False
    sector = connection.execute(
        """
        SELECT CASE WHEN nace2_variants = 1 AND NOT any_bad_nace THEN nace2
          ELSE 'unresolved' END AS primary_nace2_code, count(*) AS legal_entity_tax_ids,
          count(*) FILTER (WHERE worker_value_variants = 1 AND NOT
            coalesce(any_negative_workers, false)) AS tax_ids_valid_known_workers
        FROM entity_audit WHERE is_sole_trader = 0 AND entity_types = 1
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    sector["known_workers_pct_of_sector_tax_ids"] = (
        100 * sector.tax_ids_valid_known_workers / sector.legal_entity_tax_ids
    )
    sector["measure"] = "availability_of_annual_workers_not_sector_employment_share"
    return audit, sector


def sme_analysis(repo: Path, raw: Path, out: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((repo / INPUTS["sme_manifest"]).read_text(encoding="utf-8"))
    inventory = read_csv(repo / INPUTS["sme_inventory"]).set_index("year")
    connection = duckdb.connect()
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='2GB'")
    all_audit, all_sector, lineage = [], [], []
    for year in [2023, 2024]:
        relative = f"extracted/{year}/entire/data_sme_153_entire_year_{year}_v20260605.parquet"
        source = raw / relative
        sha = digest(source)
        if sha != manifest["inputs"][relative]:
            raise ValueError(f"Unfrozen SME Parquet: {relative}")
        lineage.append({"root": "sme_raw", "path": relative, "sha256": sha})
        connection.read_parquet(str(source)).create_view("sme_source", replace=True)
        if connection.execute("SELECT count(*) FROM sme_source WHERE year != ?", [year]).fetchone()[
            0
        ]:
            raise ValueError("Unexpected year in annual file")
        if connection.execute(
            "SELECT count(*) FROM sme_source WHERE tax_number IS NULL"
        ).fetchone()[0]:
            raise ValueError("Null entity identifiers cannot be grouped")
        audit, sector = entity_audit(connection, "sme_source")
        if int(audit.tax_ids.sum()) != int(inventory.loc[str(year), "distinct_tax_ids"]):
            raise ValueError("Distinct INN denominator differs from the frozen inventory")
        if int(audit.source_rows.sum()) != int(inventory.loc[str(year), "source_rows"]):
            raise ValueError("Raw row denominator differs from the frozen inventory")
        audit.insert(0, "year", year)
        sector.insert(0, "year", year)
        all_audit.append(audit)
        all_sector.append(sector)
    connection.close()
    audit_frame, sector_frame = pd.concat(all_audit), pd.concat(all_sector)
    write_csv(audit_frame, out / "sme_entity_eligibility.csv")
    write_csv(sector_frame, out / "sme_worker_observation_by_primary_nace2.csv")
    if (
        audit_frame.tax_ids_conflicting_workers.sum()
        or audit_frame.tax_ids_ambiguous_entity_type.sum()
    ):
        raise ValueError("Unexpected annual worker/type conflict needs separate review")
    return {"sme_entity_years_audited": int(audit_frame.tax_ids.sum())}, lineage


def geography_analysis(repo: Path, out: Path) -> dict:
    identities = read_csv(repo / INPUTS["geography"])
    versions = read_csv(repo / INPUTS["versions"])
    if len(identities) != 4 or not identities.publisher_geometry_bytes_identical.eq("True").all():
        raise ValueError("Changed publisher geometry evidence")
    if not identities.independent_legal_boundary_equivalence.eq("unknown").all():
        raise ValueError("New legal evidence requires explicit reassessment")
    if not identities.automatic_mask_change.eq("False").all():
        raise ValueError("Masks must not be changed by the context pipeline")
    rows = []
    for row in identities.to_dict("records"):
        code_pair = versions.loc[versions.territory_id.eq(row["territory_id"])].sort_values(
            "reference_year"
        )
        if len(code_pair) != 2 or code_pair.geometry_blob_sha256.nunique() != 1:
            raise ValueError("Cannot reproduce equal annual polygon selection")
        row["publisher_code_changed"] = code_pair.oktmo_verbatim.nunique() != 1
        row["analytical_effect"] = (
            "same_polygon_selection_removes_publisher_polygon_change_as_mechanical_difference"
        )
        row["does_not_establish"] = "legal_boundaries_or_Rosstat_respondent_geography"
        row["mask_decision"] = "retain_existing_exclusions"
        rows.append(row)
    write_csv(pd.DataFrame(rows), out / "geography_evidence_decisions.csv")
    return {"publisher_geometry_pairs": len(rows), "mask_changes": 0}


def write_decisions(out: Path) -> None:
    claims = [
        (
            "investment_lower_kolyma_history",
            "supported",
            "2023 published investment is more than 40 times every 2020-2022 published amount; "
            "2024 is below 2021 and 2022 in current prices, including rounding bounds.",
            "investment_longer_comparisons.csv",
            "Descriptive retrospective selected case only.",
        ),
        (
            "investment_lower_kolyma_economic_collapse",
            "unsupported",
            "The 2023-2024 fall establishes a collapse of the whole local economy.",
            "investment_paths.csv",
            "Narrow investment scope; cause and respondent stability unknown.",
        ),
        (
            "investment_upper_kolyma_history",
            "supported",
            "Published nominal investment increases in every observed adjacent year 2020-2024.",
            "investment_paths.csv;investment_2023_2024_exact_prior.csv",
            "Not a real-growth estimate, persistent-trend forecast, or causal result.",
        ),
        (
            "new_independent_investment_measurement",
            "unsupported",
            "Repeated yearbook and BDPMO agreement is an independent economic measurement.",
            "../../data_round4/cases/evidence_ledger.csv",
            "Shared statistical base is possible.",
        ),
        (
            "sme_complements_sector_structure",
            "partially_supported",
            "Registered SME primary-activity composition could add a distinct "
            "structural perspective.",
            "sme_entity_eligibility.csv",
            "Requires historical geocoding and explicit annual entity rule.",
        ),
        (
            "sme_worker_share_all_business",
            "unsupported",
            "Available workers identify total municipal SME employment or its economic share.",
            "sme_worker_observation_by_primary_nace2.csv",
            "Missing IP workers, incomplete legal-entity coverage, "
            "registered address not workplace.",
        ),
        (
            "sme_stable_codes_prove_historical_geography",
            "unsupported",
            "A single code within an annual file proves a correct historical municipal join.",
            "sme_entity_eligibility.csv",
            "Geocoding vintage is June 2026; code stability is only internal.",
        ),
        (
            "identical_geometry_implies_constant_legal_boundary",
            "unsupported",
            "Four byte-identical annual publisher geometries justify removing boundary exclusions.",
            "geography_evidence_decisions.csv",
            "No independent paired legal coordinates; retain masks.",
        ),
    ]
    write_csv(
        pd.DataFrame(claims, columns=["claim_id", "status", "claim", "evidence", "limit"]),
        out / "claim_evidence_matrix.csv",
    )
    routes = [
        (
            "primary_activity_entity_mix",
            "preferred_next_candidate",
            "One INN per defined annual stock/event population; "
            "distribution of main NACE among SMEs.",
            "Historical code hierarchy, annual date rule, multi-code/NACE policy, denominator.",
            "Do not call company counts production or employment shares.",
        ),
        (
            "corporate_observed_worker_mix",
            "conditional_secondary_candidate",
            "One annual employee value per legal-entity INN, only observed nonnegative values.",
            "Historical geography, missingness analysis, workplace limitation, category overlap.",
            "No IP imputation to zero; no sum with non-small-business BDPMO totals.",
        ),
        (
            "enterprise_entries_and_exits",
            "not_ready",
            "Separately defined changes in register presence, "
            "not births and deaths of all businesses.",
            "Monthly observation calendar, interval endpoints, "
            "register eligibility changes, revisions.",
            "Start/end fields alone do not establish genuine economic entry/exit.",
        ),
        (
            "nominal_revenue_structure",
            "defer",
            "Observed company income with annual deduplication and source financial definition.",
            "Resolve documentation labels, missingness, float precision, geography and overlap.",
            "Income is not local value added, consumer spending or municipal GDP.",
        ),
    ]
    write_csv(
        pd.DataFrame(routes, columns=["route", "priority", "estimand", "prerequisites", "limit"]),
        out / "sme_construction_routes.csv",
    )


def build(repo: Path, raw: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory to preserve existing evidence")
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = [
        {"root": "repo", "path": p, "sha256": digest(repo / p)} for p in INPUTS.values()
    ]
    summary = {"status": "PASS_WITH_CONDITIONS", **investment_analysis(repo, out)}
    sme_summary, raw_hashes = sme_analysis(repo, raw, out)
    summary.update(sme_summary)
    summary.update(geography_analysis(repo, out))
    write_decisions(out)
    summary.update(
        {
            "models_fit": 0,
            "municipal_sme_joins": 0,
            "workers_sums_published": 0,
            "causes_established": 0,
            "new_sources_searched": 0,
            "inputs_unchanged": all(
                digest(repo / row["path"]) == row["sha256"] for row in input_hashes
            ),
        }
    )
    (out / "QA.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "script_sha256": digest(Path(__file__)),
        "inputs": input_hashes + raw_hashes,
        "outputs": [
            {"path": p.name, "sha256": digest(p)}
            for p in sorted(out.iterdir())
            if p.suffix in {".csv", ".json"}
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sme-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.sme_raw, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
