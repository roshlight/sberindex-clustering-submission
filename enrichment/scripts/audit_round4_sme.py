"""Audit record grain and historical eligibility of downloaded SME data, without model joins."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
from bs4 import BeautifulSoup


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    summaries, groups, code_lengths, conflicts, inputs = [], [], [], [], {}
    for year in [2023, 2024]:
        path = next((raw / "extracted" / str(year) / "entire").glob("*.parquet"))
        filtered = next((raw / "extracted" / str(year) / "filtered").glob("*.parquet"))
        inputs[path.relative_to(raw).as_posix()] = sha(path)
        inputs[filtered.relative_to(raw).as_posix()] = sha(filtered)
        con.read_parquet(str(path)).create_view("s", replace=True)
        con.read_parquet(str(filtered)).create_view("f", replace=True)
        con.sql("DESCRIBE s").df().to_csv(out / f"schema_{year}.csv", index=False)
        result = con.sql("""
            SELECT year, count(*) AS source_rows, count(DISTINCT tax_number) AS distinct_tax_ids,
                count(DISTINCT (tax_number,start_date,end_date)) AS distinct_tax_periods,
                count(*) FILTER (WHERE tax_number IS NULL) AS missing_tax_id,
                count(*) FILTER (WHERE municipality_code IS NULL) AS missing_code_rows,
                count(employees_count) AS nonnull_workers_rows,
                count(*) FILTER (WHERE employees_count >= 0) AS valid_nonnegative_workers_rows,
                count(*) FILTER (WHERE employees_count = 0) AS zero_workers_rows,
                count(*) FILTER (WHERE employees_count < 0) AS negative_workers_rows,
                min(start_date)::VARCHAR AS first_start,
                max(start_date)::VARCHAR AS last_start,
                min(end_date)::VARCHAR AS first_end,
                max(end_date)::VARCHAR AS last_end,
                count(*) FILTER (WHERE start_date>end_date) AS reversed_periods,
                count(*) FILTER (WHERE NOT
                    regexp_full_match(municipality_code,'[0-9]{8}|[0-9]{11}'))
                    AS malformed_nonnull_code_rows,
                count(*) FILTER (WHERE lat IS NULL OR lon IS NULL) AS sql_null_coordinate_rows,
                count(*) FILTER (WHERE NOT isfinite(lat) OR NOT isfinite(lon))
                    AS nonfinite_coordinate_rows,
                count(*) FILTER (WHERE isfinite(lat) AND isfinite(lon)
                    AND (lat < -90 OR lat > 90 OR lon < -180 OR lon > 180))
                    AS finite_out_of_range_coordinate_rows
            FROM s GROUP BY year
        """).df()
        assert len(result) == 1 and int(result.year.iloc[0]) == year
        result["filtered_rows"] = con.sql("SELECT count(*) FROM f").fetchone()[0]
        result["source_rows_minus_distinct_tax_ids"] = result.source_rows - result.distinct_tax_ids
        summaries.append(result)
        groups.append(
            con.sql("""
            SELECT year,is_sole_trader,sme_category,count(*) AS source_rows,
                count(DISTINCT tax_number) AS distinct_tax_ids,
                count(employees_count) AS nonnull_workers_rows,
                count(*) FILTER (WHERE employees_count>=0) AS valid_nonnegative_workers_rows,
                count(*) FILTER (WHERE isfinite(revenue)) AS finite_revenue_rows,
                count(*) FILTER (WHERE isfinite(expenditure)) AS finite_expenditure_rows
            FROM s GROUP BY ALL ORDER BY 1,2,3
        """).df()
        )
        code_lengths.append(
            con.sql("""
            SELECT year,length(municipality_code) AS code_digits,count(*) AS source_rows
            FROM s GROUP BY ALL ORDER BY 1,2
        """).df()
        )
        conflicts.append(
            con.sql("""
            WITH entities AS (
                SELECT tax_number,count(*) AS n,count(DISTINCT municipality_code) AS codes,
                    count(DISTINCT sme_category) AS categories,
                    count(DISTINCT employees_count) AS workers,
                    count(DISTINCT (start_date,end_date)) AS periods
                FROM s GROUP BY tax_number
            )
            SELECT count(*) AS tax_ids, count(*) FILTER (WHERE n>1) AS tax_ids_multiple_rows,
                count(*) FILTER (WHERE codes>1) AS tax_ids_multiple_codes,
                count(*) FILTER (WHERE categories>1) AS tax_ids_multiple_categories,
                count(*) FILTER (WHERE workers>1) AS tax_ids_conflicting_workers,
                count(*) FILTER (WHERE n>periods) AS tax_ids_repeated_period_key
            FROM entities
        """)
            .df()
            .assign(year=year)
        )
        print(f"Audited SME {year}: {int(result.source_rows.iloc[0])} records", flush=True)
    pd.concat(summaries).to_csv(out / "annual_inventory.csv", index=False)
    pd.concat(groups).to_csv(out / "coverage_by_category.csv", index=False)
    pd.concat(code_lengths).to_csv(out / "code_lengths.csv", index=False)
    pd.concat(conflicts).to_csv(out / "entity_multiplicity.csv", index=False)

    # The page declares an encoding that does not match its actual Windows-1251 bytes.
    sakhalin = raw / "raw/sakhalin_sme_share.htm"
    text = BeautifulSoup(sakhalin.read_bytes().decode("cp1251"), "html.parser").get_text(
        " ", strip=True
    )
    text = re.sub(r"\s+", " ", text)
    assert "2015" in text and "2020" in text
    assert "2023" not in text and "2024" not in text
    (out / "sakhalin_source_text.txt").write_text(text, encoding="utf-8")
    pages = json.loads((raw / "description_pages.json").read_text("utf-8"))
    normalized = [{"pdf_page": i + 1, "text": re.sub(r"\s+", " ", t)} for i, t in enumerate(pages)]
    (out / "source_dictionary_pages.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), "utf-8"
    )
    qa = {
        "status": "ready_with_conditions_for_record_audit_only",
        "historical_municipal_attributes": "NOT_READY",
        "main_panel_joins": 0,
        "models_fit": 0,
        "source_corrections": 0,
        "issues": [
            "Geographic coding is June2026, not a documented historical2023/2024 snapshot.",
            "One INN/year can have multiple change-period records; rows are not company counts.",
            "Workers for sole traders are missing; missing cannot be interpreted as zero.",
            "Registered address / settlement coordinate is not a verified workplace location.",
            "Current geography and mixed8/11digit codes need audited hierarchy/time mapping.",
            "Medium enterprises may overlap with the existing non-small-business workforce layer.",
            "PDF p3 reverses revenue/expenditure labels; p7 dictionary defines revenue as income.",
            "Sakhalin municipal workers-share page updated2025 contains2015/2020, not2023/2024.",
            "Each annual file contains one workers_count=-1; not accepted as a worker count.",
            "Coordinates contain both SQL NULL and floating NaN; neither is a valid location.",
        ],
        "sakhalin_encoding": "Windows-1251 bytes decoded explicitly; original bytes retained",
        "raw_archive_crc": "verified by reading every archive member during collection",
        "no_annual_workers_sum_published": True,
    }
    (out / "QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), "utf-8")
    for name in [
        "raw/tochno_sme.html",
        "raw/description_sme.pdf",
        "raw/sakhalin_sme_share.htm",
        "retrievals.json",
        "description_retrieval.json",
        "sakhalin_retrieval.json",
    ]:
        inputs[name] = sha(raw / name)
    generated = [
        "schema_2023.csv",
        "schema_2024.csv",
        "annual_inventory.csv",
        "coverage_by_category.csv",
        "code_lengths.csv",
        "entity_multiplicity.csv",
        "sakhalin_source_text.txt",
        "source_dictionary_pages.json",
        "QA.json",
    ]
    manifest = {
        "inputs": inputs,
        "script_sha256": sha(Path(__file__)),
        "outputs": {name: sha(out / name) for name in generated},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw, args.out)
