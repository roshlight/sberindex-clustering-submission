"""Profile archived Tochno parquet files without accepting territorial joins."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, coverage, years = [], [], []
    with duckdb.connect() as db:
        db.execute("SET memory_limit='768MB'")
        db.execute("SET threads=2")
        paths = sorted(args.raw.glob("data_section*.parquet"))
        if len(paths) != 9:
            raise ValueError("Expected the nine explicitly selected source sections")
        for path in paths:
            db.read_parquet(str(path)).create_view("source", replace=True)
            schema = db.execute("DESCRIBE source").fetchall()
            year_counts = db.execute(
                "SELECT year, count(*) AS n FROM source GROUP BY year ORDER BY year"
            ).fetchall()
            for year, count in year_counts:
                years.append({"source_file": path.name, "year": year, "source_rows": count})
            db.execute(
                "CREATE OR REPLACE VIEW period AS SELECT * FROM source WHERE year IN (2023,2024)"
            )
            table = db.execute(
                """SELECT indicator_section_code, indicator_section, indicator_code,
                indicator_name, indicator_unit, year, indicator_period, mun_level,
                count(*) AS source_rows, count(indicator_value) AS numeric_rows,
                count(DISTINCT oktmo) AS source_codes,
                count(DISTINCT region_id) AS source_regions
                FROM period GROUP BY ALL ORDER BY 1,3,6,7,8,5"""
            ).fetchdf()
            table.insert(0, "source_file", path.name)
            coverage.append(table)
            total, numeric = db.execute(
                "SELECT count(*), count(indicator_value) FROM period"
            ).fetchone()
            naive_keys = db.execute(
                "SELECT count(*) FROM (SELECT DISTINCT indicator_code,oktmo,year FROM period)"
            ).fetchone()[0]
            out_of_range = db.execute(
                """SELECT count(*) FROM period WHERE year < try_cast(oktmo_year_from AS INTEGER)
                OR year > try_cast(oktmo_year_to AS INTEGER)"""
            ).fetchone()[0]
            records.append(
                {
                    "source_file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_bytes": path.stat().st_size,
                    "full_source_rows": sum(n for _, n in year_counts),
                    "selected_years": [2023, 2024],
                    "selected_rows": total,
                    "numeric_rows": numeric,
                    "null_rows": total - numeric,
                    "indicators": int(table.indicator_code.nunique()),
                    "naive_indicator_code_year_keys": naive_keys,
                    "excess_rows_over_naive_key": total - naive_keys,
                    "outside_publisher_inclusive_code_year_range": out_of_range,
                    "schema": schema,
                    "join_status": "unlinked_secondary_source_not_an_accepted_municipal_panel",
                }
            )
    combined = pd.concat(coverage, ignore_index=True)
    combined.to_csv(args.out / "indicator_year_scope_coverage.csv", index=False)
    pd.DataFrame(years).to_csv(args.out / "section_year_counts.csv", index=False)
    compact = (
        combined.groupby(
            [
                "indicator_section_code",
                "indicator_section",
                "indicator_code",
                "indicator_name",
                "year",
            ],
            dropna=False,
        )
        .agg(
            source_rows=("source_rows", "sum"),
            numeric_rows=("numeric_rows", "sum"),
            units=("indicator_unit", lambda x: " | ".join(sorted(set(x)))),
            periods=("indicator_period", lambda x: " | ".join(sorted(set(x)))),
            levels=("mun_level", lambda x: " | ".join(sorted(set(x)))),
        )
        .reset_index()
    )
    compact.to_csv(args.out / "indicator_shortlist.csv", index=False)
    summary = {
        "status": "source_profiled_not_joined",
        "sections": len(records),
        "sections_with_selected_years": sum(r["selected_rows"] > 0 for r in records),
        "selected_rows": sum(r["selected_rows"] for r in records),
        "numeric_rows": sum(r["numeric_rows"] for r in records),
        "indicators": int(combined.indicator_code.nunique()),
        "records": records,
        "limitations": [
            "Rows are dimensional source records, not independent municipalities or features.",
            "Publisher harmonized codes and history require a separate audited bridge.",
            "Secondary processing of Rosstat is not an independent measurement source.",
            "Nulls, period variants, geography levels and category dimensions must be preserved.",
            "No model imputation, panel join, aggregation or baseline change performed.",
        ],
    }
    (args.out / "profile_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "limitations"}}))


if __name__ == "__main__":
    main()
