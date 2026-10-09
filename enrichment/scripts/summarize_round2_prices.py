"""Rebuild price QA and an explicit delivery inventory from canonical extracted tables.

Only this new report's generated summaries/manifest are replaced. No fitting, source
changes, bridge changes, or network access. Human-authored prose is never edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from prepare_round2_prices import sha


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_dir
    primary = report / "monthly_v3/primary_city_price_context.parquet"
    source = report / "monthly_v3/city_monthly_price_cells.parquet"
    con = duckdb.connect()
    con.from_parquet(str(primary)).create_view("primary_prices")
    con.from_parquet(str(source)).create_view("source_prices")
    qa = con.sql("""SELECT year, count(*) AS n_cells, count(value) AS n_numeric,
        count(*) FILTER (WHERE value>0) AS n_positive,
        count(*) FILTER (WHERE value=0) AS n_zeros
        FROM primary_prices GROUP BY year ORDER BY year""").df()
    qa.to_csv(report / "primary_numeric_qa.csv", index=False)
    con.sql("""SELECT year,product_code,count(*) AS zero_cells
        FROM source_prices WHERE value=0 GROUP BY 1,2 ORDER BY 1,2""").df().to_csv(
        report / "source_zero_review.csv", index=False
    )
    numeric, positive, zeros = con.sql("""SELECT count(value),
        count(*) FILTER (WHERE value>0),count(*) FILTER (WHERE value=0)
        FROM source_prices""").fetchone()
    constraints = {
        "zero_use_status": "unclear_zero_use",
        "source_numeric_includes_zero": True,
        "primary_numeric_includes_zero": True,
        "primary_positive_prices": int(qa.n_positive.sum()),
        "primary_numeric_prices": int(qa.n_numeric.sum()),
        "primary_source_zero_prices": int(qa.n_zeros.sum()),
        "source_positive_prices": positive,
        "source_numeric_prices": numeric,
        "source_zero_prices": zeros,
        "exclude_zero_from_log_ratio_deflation_until_source_review": True,
        "whole_municipality_price_estimate": False,
        "scope": "city_context_only",
        "allow_nearest_city_fill": False,
        "allow_automatic_sber_category_deflation": False,
        "single_common_index_cancels_when_profiles_renormalized": True,
        "allow_parent_only_adjudication_for_city": False,
    }
    candidates = report / "bridge_v2/bridge_candidates.csv"
    con.read_csv(str(candidates)).create_view("candidates")
    constraints["temporal_review_city_years"] = con.sql(
        "SELECT count(*) FROM candidates WHERE bridge_status='temporal_review'"
    ).fetchone()[0]
    constraints["temporal_review_primary_city_years"] = con.sql("""SELECT count(*) FROM candidates
        WHERE bridge_status='temporal_review' AND in_primary_cohort""").fetchone()[0]
    monthly_summary = json.loads((report / "monthly_v3/summary.json").read_text(encoding="utf8"))
    constraints["temporal_review_source_cells"] = monthly_summary["temporal_review_source_cells"]
    write_json(report / "analysis_constraints.json", constraints)
    raw = json.loads(args.raw_inventory.read_text(encoding="utf8"))
    if args.raw_inventory.resolve() != (report / "raw_inventory.json").resolve():
        write_json(report / "raw_inventory.json", raw)
    inventory = {
        "scope": "city_context_only; source and candidate rows are not automatic model inputs",
        "bridge": "bridge_v2",
        "monthly": "monthly_v3",
        "source_verification": "source_cell_verification_v3.json",
        "source_cells": monthly_summary["source_cell_count"],
        "primary_cells": int(qa.n_cells.sum()),
        "raw_inventory": "raw_inventory.json",
        "raw_file_count": raw["total_files"],
        "raw_bytes": raw["total_bytes"],
        "table_roles": {
            "bridge_v2/bridge_candidates.csv": "city-year; all candidates and review reasons",
            "bridge_v2/bridge_approved_evidence.csv": "city-year; documentary approved links",
            "bridge_v2/primary_cohort_city_context.csv": "municipality-year grid; annual city ISZh",
            "monthly_v3/city_monthly_price_cells.parquet": "all published city-month-product cells",
            "monthly_v3/primary_city_price_context.parquet": "approved primary city context cells",
            "monthly_v3/city_monthly_dimensions.csv": "source city-month; exact context/status",
            "monthly_v3/product_dictionary.csv": "product-month; full name/unit/source column",
            "monthly_v3/yearly_source_coverage.csv": (
                "year; nominal/published/numeric/missing counts"
            ),
            "monthly_v3/product_primary_coverage.csv": "product-year; accepted cohort coverage",
            "primary_numeric_qa.csv": "year; numeric, positive and zero denominators",
            "source_zero_review.csv": "year-product; literal source zero counts, no imputation",
        },
    }
    write_json(report / "inventory.json", inventory)
    scripts = [
        args.repo_root / "enrichment/scripts" / name
        for name in [
            "prepare_round2_prices.py",
            "extract_round2_prices.py",
            "verify_round2_prices.py",
            "summarize_round2_prices.py",
        ]
    ]
    code = scripts + [
        args.repo_root / "enrichment/tests/test_round2_prices.py",
        args.repo_root / "enrichment/uv.lock",
    ]
    write_json(
        report / "manifest.json",
        {
            "scope": (
                "replayable_with_constraints; no fitting or historical publication reconstruction"
            ),
            "code_and_environment": [
                {"path": p.relative_to(args.repo_root).as_posix(), "sha256": sha(p)} for p in code
            ],
            "outputs": [
                {
                    "path": p.relative_to(report).as_posix(),
                    "sha256": sha(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(report.rglob("*"))
                if p.is_file() and p != report / "manifest.json"
            ],
        },
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
