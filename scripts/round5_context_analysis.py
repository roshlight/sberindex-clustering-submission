"""Describe SME geography resolution and documentary investment context, without model joins."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

FEDERAL_GROUP_CODES = {
    "45000000": "Москва",
    "40000000": "Санкт-Петербург",
    "67000000": "Севастополь",
}
KNOWN_KINDS = {
    "region_group",
    "section2_settlement",
    "section1_top_level_or_group",
    "section1_submunicipal_or_group",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def code_dimension(frame):
    """Preserve exact keys, including distinct NULL and empty-string representations."""
    keys = ["year", "source_code", "source_code_is_null"]
    if frame.duplicated(keys).any():
        raise ValueError("Geography dimension is not many-to-one")
    if frame.v20260601_ambiguous.any():
        raise ValueError("Ambiguous registry codes require separate adjudication")
    matched = frame.v20260601_present
    if frame.loc[~matched, "v20260601_kind"].notna().any():
        raise ValueError("An absent exact code cannot carry an accepted structural kind")
    if not frame.loc[matched, "v20260601_kind"].isin(KNOWN_KINDS).all():
        raise ValueError("Unknown code kind")
    if frame.loc[matched, "source_code_is_null"].any():
        raise ValueError("NULL cannot be an exact registry match")
    if not frame.loc[matched, "source_code"].str.fullmatch(r"(?:[0-9]{8}|[0-9]{11})").all():
        raise ValueError("Matched codes have invalid source syntax")
    result = frame[keys + ["source_rows", "v20260601_present", "v20260601_kind"]].copy()
    result["geo_stratum"] = result.v20260601_kind.fillna("no_exact_code")
    result.loc[result.geo_stratum.eq("region_group"), "geo_stratum"] = "other_region_group"
    federal = result.source_code.isin(FEDERAL_GROUP_CODES)
    if not result.loc[federal, "v20260601_kind"].eq("region_group").all():
        raise ValueError("Federal-city grouping requires its reviewed exact registry entry")
    result.loc[federal, "geo_stratum"] = "federal_city_region_group"
    return result


def create_base(con, source, dimension, year):
    con.read_parquet(str(source)).create_view("source", replace=True)
    invalid = con.execute(
        "SELECT count(*) FROM source WHERE year IS NULL OR year != ? "
        "OR tax_number IS NULL OR tax_number = '' "
        "OR is_sole_trader IS NULL OR is_sole_trader NOT IN (0,1)",
        [year],
    ).fetchone()[0]
    if invalid:
        raise ValueError("Source year, identity or legal-kind contract failed")
    dim = dimension.loc[dimension.year.eq(year)]
    con.register("code_dim", dim)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT s.tax_number, s.year, s.region_code, s.region_iso_code, s.region,
          s.municipality_code, d.geo_stratum, d.v20260601_present AS code_present,
          d.source_rows AS dimension_source_rows,
          CASE WHEN s.is_sole_trader=1 THEN 'IP' ELSE 'legal_entity' END AS legal_kind,
          CASE WHEN s.employees_count IS NULL THEN 'missing'
               WHEN s.employees_count < 0 THEN 'invalid_negative'
               WHEN s.employees_count = 0 THEN 'zero' ELSE 'positive' END AS employee_state,
          CASE WHEN s.region_code IN ('77','78','92') THEN 'federal_city_source_region'
               WHEN s.region_code IS NULL OR s.region_code='' THEN 'missing_source_region'
               ELSE 'other_source_region' END AS source_region_scope,
          CASE WHEN s.municipality_code IS NULL THEN 'source_null'
               WHEN s.municipality_code='' THEN 'source_empty'
               WHEN NOT regexp_full_match(s.municipality_code, '(?:[0-9]{8}|[0-9]{11})')
                 THEN 'source_malformed'
               ELSE 'valid_syntax_not_in_june2026' END AS unmatched_reason
        FROM source s LEFT JOIN code_dim d
        ON s.year=d.year AND coalesce(s.municipality_code,'')=d.source_code
          AND (s.municipality_code IS NULL)=d.source_code_is_null
    """)
    n = con.execute("SELECT count(*) FROM source").fetchone()[0]
    joined, missing = con.execute(
        "SELECT count(*),count(*) FILTER(WHERE dimension_source_rows IS NULL) FROM base"
    ).fetchone()
    if n != joined or missing or int(dim.source_rows.sum()) != n:
        raise ValueError("Exact code annotation lost or multiplied source rows")
    # Detect drift within groups too; the national row count alone is insufficient.
    changed = con.execute("""
        SELECT count(*) FROM (
          SELECT municipality_code, count(*) n, min(dimension_source_rows) expected
          FROM base GROUP BY municipality_code HAVING n != expected
        )
    """).fetchone()[0]
    if changed:
        raise ValueError("Raw code-group counts differ from the audited dimension")
    return n


def annual_tables(con, year, n):
    result = {}
    result["annual_population"] = con.execute("""
        SELECT year, CASE WHEN grouping(legal_kind)=1 THEN 'ALL' ELSE legal_kind END legal_kind,
          count(*) source_rows, count(DISTINCT tax_number) distinct_tax_ids,
          count(*) FILTER(WHERE employee_state='missing') missing_employee_rows,
          count(*) FILTER(WHERE employee_state='invalid_negative') negative_employee_rows,
          count(*) FILTER(WHERE employee_state='zero') zero_employee_rows,
          count(*) FILTER(WHERE employee_state='positive') positive_employee_rows
        FROM base GROUP BY GROUPING SETS((year),(year,legal_kind)) ORDER BY year,legal_kind
    """).df()
    annual = result["annual_population"]
    annual["source_rows_per_distinct_tax_id"] = annual.source_rows / annual.distinct_tax_ids
    annual["nonnegative_employee_rows"] = annual.zero_employee_rows + annual.positive_employee_rows
    annual["nonnegative_employee_row_share_pct"] = (
        100 * annual.nonnegative_employee_rows / annual.source_rows
    )
    result["geography_strata"] = con.execute("""
        SELECT year,geo_stratum,count(*) source_rows,count(DISTINCT tax_number) distinct_tax_ids,
          count(*) FILTER(WHERE legal_kind='IP') ip_rows,
          count(*) FILTER(WHERE legal_kind='legal_entity') legal_entity_rows,
          count(*) FILTER(WHERE employee_state IN ('zero','positive')) nonnegative_employee_rows,
          count(*) FILTER(WHERE employee_state='missing') missing_employee_rows,
          count(*) FILTER(WHERE employee_state='invalid_negative') negative_employee_rows
        FROM base GROUP BY year,geo_stratum ORDER BY year,geo_stratum
    """).df()
    strata = result["geography_strata"]
    strata["denominator_year_rows"] = n
    strata["share_year_rows_pct"] = 100 * strata.source_rows / n
    strata["nonnegative_employee_row_share_pct"] = (
        100 * strata.nonnegative_employee_rows / strata.source_rows
    )
    detail = con.execute("""
        SELECT year,geo_stratum,legal_kind,employee_state,count(*) source_rows,
          count(DISTINCT tax_number) distinct_tax_ids
        FROM base GROUP BY ALL ORDER BY year,geo_stratum,legal_kind,employee_state
    """).df()
    detail["denominator_year_rows"] = n
    detail["denominator_stratum_rows"] = detail.groupby("geo_stratum").source_rows.transform("sum")
    detail["share_year_rows_pct"] = 100 * detail.source_rows / n
    detail["share_stratum_rows_pct"] = 100 * detail.source_rows / detail.denominator_stratum_rows
    result["geography_legal_workers"] = detail
    profile = con.execute("""
        SELECT year,region_code,region_iso_code,region,geo_stratum,count(*) source_rows,
          count(DISTINCT tax_number) distinct_tax_ids,
          count(*) FILTER(WHERE employee_state IN ('zero','positive')) nonnegative_employee_rows
        FROM base GROUP BY ALL ORDER BY year,region_code NULLS FIRST,region,geo_stratum
    """).df()
    region_keys = ["year", "region_code", "region_iso_code", "region"]
    profile["denominator_source_region_rows"] = profile.groupby(
        region_keys, dropna=False
    ).source_rows.transform("sum")
    profile["denominator_year_stratum_rows"] = profile.groupby("geo_stratum").source_rows.transform(
        "sum"
    )
    profile["share_source_region_rows_pct"] = (
        100 * profile.source_rows / profile.denominator_source_region_rows
    )
    profile["share_year_stratum_rows_pct"] = (
        100 * profile.source_rows / profile.denominator_year_stratum_rows
    )
    result["regional_code_profile"] = profile
    result["unmatched_audit"] = con.execute("""
        SELECT year,region_code,region,unmatched_reason,legal_kind,count(*) source_rows
        FROM base WHERE NOT code_present GROUP BY ALL
        ORDER BY year,region_code NULLS FIRST,unmatched_reason,legal_kind
    """).df()
    coverage = con.execute("""
        SELECT year,CASE WHEN grouping(source_region_scope)=1 THEN 'ALL'
                    ELSE source_region_scope END source_region_scope,
          count(*) source_rows,
          count(*) FILTER(WHERE code_present) exact_code_rows,
          count(*) FILTER(WHERE geo_stratum='federal_city_region_group') federal_city_group_rows,
          count(*) FILTER(WHERE geo_stratum='other_region_group') other_region_group_rows,
          count(*) FILTER(WHERE geo_stratum IN ('section2_settlement',
             'section1_top_level_or_group','section1_submunicipal_or_group')) subregional_code_rows,
          count(*) FILTER(WHERE geo_stratum='no_exact_code') unmatched_rows
        FROM base GROUP BY GROUPING SETS((year),(year,source_region_scope))
        ORDER BY year,source_region_scope
    """).df()
    for field in [
        "exact_code",
        "federal_city_group",
        "other_region_group",
        "subregional_code",
        "unmatched",
    ]:
        coverage[field + "_share_pct"] = 100 * coverage[field + "_rows"] / coverage.source_rows
    coverage["historical_municipal_join_proven_rows"] = 0
    coverage["zero_proven_is_a_readiness_status_not_true_coverage"] = True
    result["coverage_interpretation"] = coverage
    result["entity_multiplicity"] = con.execute("""
        WITH entities AS (
          SELECT tax_number,count(*) n,count(DISTINCT geo_stratum) strata,
            count(DISTINCT coalesce(region_code,'<NULL>')) regions,
            count(DISTINCT legal_kind) kinds
          FROM base GROUP BY tax_number
        )
        SELECT count(*) distinct_tax_ids, count(*) FILTER(WHERE n>1) multirow_tax_ids,
          count(*) FILTER(WHERE strata>1) multiple_geo_strata_tax_ids,
          count(*) FILTER(WHERE regions>1) multiple_source_regions_tax_ids,
          count(*) FILTER(WHERE kinds>1) multiple_legal_kinds_tax_ids
        FROM entities
    """).df()
    result["entity_multiplicity"].insert(0, "year", year)
    for name in ["geography_strata", "geography_legal_workers", "regional_code_profile"]:
        if int(result[name].source_rows.sum()) != n:
            raise ValueError(f"Nonadditive row partition: {name}")
    return result


def fns_contrast(stock, annual):
    rows = []
    for year in [2023, 2024]:
        selected = stock.loc[
            stock.snapshot_date.eq(f"{year}-12-10") & stock.geography_level.eq("country")
        ]
        if len(selected) != 1:
            raise ValueError("Require one national FNS stock row per reviewed snapshot")
        s = selected.iloc[0]
        if int(s.total) != int(s.legal_total) + int(s.ip_total):
            raise ValueError("FNS legal/IP stock partition does not reconcile")
        for kind, column in [("ALL", "total"), ("legal_entity", "legal_total"), ("IP", "ip_total")]:
            a = annual.loc[annual.year.eq(year) & annual.legal_kind.eq(kind)].iloc[0]
            rows.append(
                {
                    "year": year,
                    "legal_kind": kind,
                    "fns_snapshot_date": s.snapshot_date,
                    "fns_stock_subjects": int(s[column]),
                    "tochno_period_rows": int(a.source_rows),
                    "tochno_annual_distinct_tax_ids": int(a.distinct_tax_ids),
                    "tochno_version": "v20260605",
                    "fns_source_file": s.source_file,
                    "fns_excel_row": int(s.excel_row),
                    "same_population_and_moment": False,
                    "coverage_error_or_growth_estimate": False,
                    "territorial_scope": "each_source_as_published_no_population_harmonization",
                    "interpretation": (
                        "snapshot_stock_and_annual_period_population_are_not_interchangeable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def budget_revision(ledger):
    rows = []
    flags = [
        "model_join_accepted",
        "causal_attribution_confirmed",
        "annual_bdpm_investment_reconciled",
    ]
    if not ledger[flags].eq("False").all().all():
        raise ValueError("Upstream documentary evidence was promoted beyond reviewed scope")
    for year in ["2022", "2023", "2024", "2022–2024"]:
        versions = []
        for sid in ["gov_contract_uplift_b_http", "gov_contract_uplift_a_http"]:
            selected = ledger.loc[ledger.source_id.eq(sid) & ledger.reported_year.eq(year)]
            if len(selected) != 1 or not selected.unit.eq("thousand_RUB").all():
                raise ValueError("Budget revision requires unique source cells with matching units")
            versions.append(selected.iloc[0])
        old, new = versions
        old_s, new_s = old.value_as_published, new.value_as_published
        delta = (
            ""
            if "-" in (old_s, new_s)
            else str(Decimal(new_s.replace(",", ".")) - Decimal(old_s.replace(",", ".")))
        )
        rows.append(
            {
                "project": "Аэропорт Черский",
                "year_column": year,
                "old_document_date": old.source_publication_date,
                "new_document_date": new.source_publication_date,
                "old_value_as_published": old_s,
                "new_value_as_published": new_s,
                "revision_delta_thousand_rub": delta,
                "revision_delta_million_rub": str(Decimal(delta) / 1000) if delta else "",
                "meaning": "revision_of_budget_contract_uplift_not_annual_capex",
                "dash_means_zero": False,
                "share_of_bdpm_peak_computed": False,
                "old_source_sha256": old.source_sha256,
                "new_source_sha256": new.source_sha256,
                "old_source_url": old.source_url,
                "new_source_url": new.source_url,
                "old_pdf_physical_page": old.physical_page,
                "new_pdf_physical_page": new.physical_page,
            }
        )
    return pd.DataFrame(rows)


def case_timeline(ledger, paths):
    ids = [
        "airport_timeline",
        "airport_permit_announcement",
        "airport_first_flight",
        "airport_annual_report",
        "tlc_chersky",
        "andryushkino_clinic",
        "fiber_route_2024",
    ]
    timeline = ledger.loc[ledger.evidence_id.isin(ids)].copy()
    timeline["historical_model_eligibility"] = False
    timeline["availability_note"] = "publication_time_separate_from_reported_event_year"
    lower = paths.loc[paths.municipality.eq("Нижнеколымский")].copy()
    if set(lower.year) != {2020, 2021, 2022, 2023, 2024}:
        raise ValueError("Expected reviewed five-year Lower Kolyma path")
    counts = {
        2020: "",
        2021: "airport_timeline",
        2022: "airport_timeline",
        2023: ";".join(ids[:6]),
        2024: "fiber_route_2024",
    }
    lower["event_period_alignment_evidence_ids"] = lower.year.map(counts)
    lower["project_contribution_estimated"] = False
    lower["interpretation"] = (
        "coinciding_project_period_or_event_does_not_allocate_annual_investment"
    )
    return timeline, lower


def run(repo, sme_raw, out):
    if out.exists():
        raise FileExistsError("Choose a new output directory; old results are immutable")
    inputs = []

    def checked(root_name, relative, expected=None):
        root = repo if root_name == "repo" else sme_raw
        p = root / relative
        actual = sha(p)
        if expected and actual != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
        inputs.append({"root": root_name, "path": relative, "sha256": actual})
        return p

    gm = load(checked("repo", "reports/data_round5/geography/manifest.json"))
    sm = load(checked("repo", "reports/data_round4/sme/manifest.json"))
    fm = load(checked("repo", "reports/data_round5/sme/artifact_manifest.json"))
    cm = load(checked("repo", "reports/data_round5/cases/manifest.json"))
    pm = load(checked("repo", "reports/round4_analysis/context/manifest.json"))
    checked("repo", "uv.lock")
    checked("repo", "pyproject.toml")
    geo_path = checked(
        "repo",
        "reports/data_round5/geography/sme_code_presence.parquet",
        gm["outputs"]["sme_code_presence.parquet"],
    )
    with duckdb.connect(config={"threads": 2, "memory_limit": "2GB"}) as con:
        dimension = code_dimension(con.read_parquet(str(geo_path)).df())
        all_tables = {}
        years_qa = []
        for year in [2023, 2024]:
            rel = f"extracted/{year}/entire/data_sme_153_entire_year_{year}_v20260605.parquet"
            source = checked("sme_raw", rel, sm["inputs"][rel])
            n = create_base(con, source, dimension, year)
            tables = annual_tables(con, year, n)
            for key, value in tables.items():
                all_tables.setdefault(key, []).append(value)
            years_qa.append(
                {
                    "year": year,
                    "source_rows": n,
                    "joined_rows": n,
                    "missing_code_dimension_rows": 0,
                    "model_join": False,
                }
            )
    tables = {key: pd.concat(frames, ignore_index=True) for key, frames in all_tables.items()}
    fhash = next(item["sha256"] for item in fm if item["path"] == "official_stock_counts.csv")
    stock = pd.read_csv(checked("repo", "reports/data_round5/sme/official_stock_counts.csv", fhash))
    ledger = pd.read_csv(
        checked(
            "repo",
            "reports/data_round5/cases/evidence_ledger.csv",
            cm["outputs"]["evidence_ledger.csv"],
        ),
        dtype=str,
        keep_default_na=False,
    )
    phash = next(item["sha256"] for item in pm["outputs"] if item["path"] == "investment_paths.csv")
    paths = pd.read_csv(
        checked("repo", "reports/round4_analysis/context/investment_paths.csv", phash)
    )
    tables["fns_population_contrast"] = fns_contrast(stock, tables["annual_population"])
    tables["budget_revision"] = budget_revision(ledger)
    tables["chersky_timeline"], tables["chersky_investment_alignment"] = case_timeline(
        ledger, paths
    )
    qa = {
        "status": "ready_with_conditions",
        "purpose": "context_and_source_readiness_only",
        "year_row_reconciliation": years_qa,
        "workers_summed": False,
        "historical_municipal_joins": 0,
        "model_features_added": 0,
        "new_network_requests": 0,
        "causal_attributions": 0,
        "source_table_reextraction": "not repeated; verified frozen upstream derived tables",
        "distinct_ids_additive_across_strata_or_regions": False,
        "fns_stock_is_annual_panel_denominator": False,
        "dashes_converted_to_zero": False,
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
    }
    out.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")
    save_json(out / "QA.json", qa)
    save_json(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "inputs": inputs,
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    print(
        json.dumps(
            {
                "status": qa["status"],
                "tables": len(tables),
                "source_rows": sum(q["source_rows"] for q in years_qa),
            }
        )
    )
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sme-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.sme_raw, args.out)


if __name__ == "__main__":
    main()
