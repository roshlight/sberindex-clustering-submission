"""Independent offline verification of SME inventory, grain and source semantics.

Does not import the production audit, identify individuals, sum workers, or join
municipal attributes. All output consists of aggregate checks and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def query_dict(connection, sql):
    cursor = connection.execute(sql)
    return dict(zip([x[0] for x in cursor.description], cursor.fetchone(), strict=True))


def verify(raw, report, output):
    manifest = json.loads((report / "manifest.json").read_text("utf-8"))
    checks = []

    def check(name, observed, expected):
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "pass": observed == expected,
            }
        )

    for name, expected in manifest["inputs"].items():
        check("input_sha:" + name, sha(raw / name), expected)
    for name, expected in manifest["outputs"].items():
        check("output_sha:" + name, sha(report / name), expected)
    # Reading every member recomputes ZIP CRC and ties it to the extracted file.
    archive_checks = []
    for record in json.loads((raw / "retrievals.json").read_text("utf-8")):
        archive = raw / record["path"]
        check("zip_sha:" + str(record["year_in_filename"]), sha(archive), record["sha256"])
        with zipfile.ZipFile(archive) as bundle:
            members = {x.filename: x for x in bundle.infolist() if not x.is_dir()}
            check("zip_member_count:" + str(record["year_in_filename"]), len(members), 2)
            for entry in record["entries"]:
                member = "/".join(Path(entry["path"]).parts[2:])
                with bundle.open(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                check("zip_member_sha:" + entry["path"], digest, sha(raw / entry["path"]))
                check("zip_member_record:" + entry["path"], digest, entry["sha256"])
                archive_checks.append(
                    {
                        "archive": record["path"],
                        "member": member,
                        "bytes": members[member].file_size,
                        "sha256": digest,
                        "crc_verified_by_complete_read": True,
                    }
                )
    expected_annual = pd.read_csv(report / "annual_inventory.csv").set_index("year")
    expected_entities = pd.read_csv(report / "entity_multiplicity.csv").set_index("year")
    coverage = pd.read_csv(report / "coverage_by_category.csv")
    lengths = pd.read_csv(report / "code_lengths.csv")
    results = {}
    with tempfile.TemporaryDirectory(prefix="round4-sme-qa-") as temporary:
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='2GB'")
        con.execute("SET temp_directory = ?", [temporary])
        for year in [2023, 2024]:
            entire = list((raw / "extracted" / str(year) / "entire").glob("*.parquet"))
            filtered = list((raw / "extracted" / str(year) / "filtered").glob("*.parquet"))
            assert len(entire) == len(filtered) == 1
            con.read_parquet(str(entire[0])).create_view("source", replace=True)
            con.read_parquet(str(filtered[0])).create_view("filtered", replace=True)
            stats = query_dict(
                con,
                """
                SELECT count(*) source_rows, count(DISTINCT tax_number) distinct_tax_ids,
                    sum(CASE WHEN tax_number IS NULL THEN 1 ELSE 0 END) missing_tax_id,
                    sum(CASE WHEN municipality_code IS NULL THEN 1 ELSE 0 END) missing_code_rows,
                    sum(CASE WHEN employees_count IS NOT NULL THEN 1 ELSE 0 END)
                        nonnull_workers_rows,
                    sum(CASE WHEN employees_count>=0 THEN 1 ELSE 0 END)
                        valid_nonnegative_workers_rows,
                    sum(CASE WHEN employees_count=0 THEN 1 ELSE 0 END) zero_workers_rows,
                    sum(CASE WHEN employees_count<0 THEN 1 ELSE 0 END) negative_workers_rows,
                    min(employees_count) minimum_workers,
                    min(start_date)::VARCHAR first_start, max(start_date)::VARCHAR last_start,
                    min(end_date)::VARCHAR first_end, max(end_date)::VARCHAR last_end,
                    sum(CASE WHEN start_date>end_date THEN 1 ELSE 0 END) reversed_periods,
                    sum(CASE WHEN municipality_code IS NOT NULL AND
                        (length(municipality_code) NOT IN (8,11) OR
                         regexp_matches(municipality_code,'[^0-9]')) THEN 1 ELSE 0 END)
                         malformed_nonnull_code_rows,
                    sum(CASE WHEN lat IS NULL OR lon IS NULL THEN 1 ELSE 0 END)
                        sql_null_coordinate_rows,
                    sum(CASE WHEN isnan(lat) OR isnan(lon) OR isinf(lat) OR isinf(lon)
                        THEN 1 ELSE 0 END) nonfinite_coordinate_rows,
                    sum(CASE WHEN isfinite(lat) AND isfinite(lon) AND
                        NOT (lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180)
                        THEN 1 ELSE 0 END) finite_out_of_range_coordinate_rows,
                    min(year) minimum_year, max(year) maximum_year,
                    sum(CASE WHEN is_sole_trader=1 AND employees_count IS NOT NULL
                        THEN 1 ELSE 0 END) sole_trader_nonnull_workers,
                    sum(CASE WHEN year < year(start_date) OR year > year(end_date)
                        THEN 1 ELSE 0 END) year_outside_source_interval
                FROM source
            """,
            )
            stats["filtered_rows"] = con.sql("SELECT count(*) FROM filtered").fetchone()[0]
            stats["distinct_tax_periods"] = con.sql("""
                SELECT count(*) FROM (SELECT tax_number,start_date,end_date
                    FROM source GROUP BY tax_number,start_date,end_date)
            """).fetchone()[0]
            stats["source_rows_minus_distinct_tax_ids"] = (
                stats["source_rows"] - stats["distinct_tax_ids"]
            )
            for column, value in expected_annual.loc[year].items():
                value = value.item() if hasattr(value, "item") else value
                check(f"annual:{year}:{column}", stats[column], value)
            for column, expected in [
                ("minimum_year", year),
                ("maximum_year", year),
                ("minimum_workers", -1),
                ("sole_trader_nonnull_workers", 0),
                ("year_outside_source_interval", 0),
            ]:
                check(f"extra:{year}:{column}", stats[column], expected)
            # Materialized INN groups independently expose record, period and code multiplicity.
            con.execute("""
                CREATE OR REPLACE TEMP TABLE per_inn AS
                SELECT tax_number, count(*) records,
                    count(DISTINCT municipality_code) codes,
                    count(DISTINCT sme_category) categories,
                    count(DISTINCT employees_count) workers,
                    count(DISTINCT (start_date,end_date)) periods
                FROM source GROUP BY tax_number
            """)
            entity = query_dict(
                con,
                """
                SELECT count(*) tax_ids,
                    sum(CASE WHEN records>1 THEN 1 ELSE 0 END) tax_ids_multiple_rows,
                    sum(CASE WHEN codes>1 THEN 1 ELSE 0 END) tax_ids_multiple_codes,
                    sum(CASE WHEN categories>1 THEN 1 ELSE 0 END) tax_ids_multiple_categories,
                    sum(CASE WHEN workers>1 THEN 1 ELSE 0 END) tax_ids_conflicting_workers,
                    sum(CASE WHEN records>periods THEN 1 ELSE 0 END) tax_ids_repeated_period_key,
                    sum(records-periods) rows_above_distinct_inn_period_key
                FROM per_inn
            """,
            )
            for column, value in expected_entities.loc[year].items():
                check(f"entity:{year}:{column}", entity[column], int(value))
            cats = con.sql("""
                SELECT year,is_sole_trader,sme_category,count(*) source_rows,
                    count(DISTINCT tax_number) distinct_tax_ids,
                    count(employees_count) nonnull_workers_rows,
                    sum(CASE WHEN employees_count>=0 THEN 1 ELSE 0 END)
                        valid_nonnegative_workers_rows,
                    sum(CASE WHEN revenue IS NOT NULL AND NOT isnan(revenue) AND NOT isinf(revenue)
                        THEN 1 ELSE 0 END) finite_revenue_rows,
                    sum(CASE WHEN expenditure IS NOT NULL AND NOT isnan(expenditure)
                        AND NOT isinf(expenditure)
                        THEN 1 ELSE 0 END) finite_expenditure_rows
                FROM source GROUP BY year,is_sole_trader,sme_category ORDER BY 1,2,3
            """).df()
            expected_cats = coverage[coverage.year == year].reset_index(drop=True)
            check(
                f"coverage:{year}:all_cells",
                bool((cats.to_numpy() == expected_cats.to_numpy()).all()),
                True,
            )
            actual_lengths = con.sql("""
                SELECT coalesce(length(municipality_code),-1) code_digits,count(*) source_rows
                FROM source GROUP BY 1 ORDER BY 1
            """).fetchall()
            expected_lengths = sorted(
                (int(-1 if pd.isna(r.code_digits) else r.code_digits), int(r.source_rows))
                for r in lengths[lengths.year == year].itertuples()
            )
            check(f"code_lengths:{year}", actual_lengths, expected_lengths)
            coordinates = con.sql("""
                SELECT CASE WHEN lat IS NULL OR lon IS NULL THEN 'sql_null'
                    WHEN isnan(lat) OR isnan(lon) THEN 'float_nan'
                    WHEN isinf(lat) OR isinf(lon) THEN 'float_infinity'
                    WHEN NOT (lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180)
                        THEN 'finite_outside_range' ELSE 'finite_in_range' END coordinate_class,
                    count(*) record_count FROM source GROUP BY 1 ORDER BY 1
            """).fetchall()
            overlap = con.sql("""
                SELECT count(*) FROM source WHERE (lat IS NULL OR lon IS NULL)
                    AND (isnan(lat) OR isnan(lon) OR isinf(lat) OR isinf(lon))
            """).fetchone()[0]
            check(
                f"coordinate_exclusive_partition:{year}",
                sum(x[1] for x in coordinates),
                stats["source_rows"],
            )
            check(f"coordinate_null_nonfinite_overlap:{year}", overlap, 0)
            results[str(year)] = {
                "annual": stats,
                "entity_multiplicity": entity,
                "exclusive_coordinate_classes": dict(coordinates),
            }
            print(f"Independent SME verification {year} complete", flush=True)
        con.close()
    # Extract directly from the PDF, not from production cached text.
    pages = [
        re.sub(r"\s+", " ", p.extract_text() or "")
        for p in PdfReader(raw / "raw/description_sme.pdf").pages
    ]
    check("pdf_pages", len(pages), 9)
    check(
        "pdf_page3_reversed_labels",
        bool(re.search(r"расходы\s*\(\s*revenue\s*\).*доходы\s*\(\s*expenditure", pages[2])),
        True,
    )
    check(
        "pdf_page7_revenue_is_income",
        bool(re.search(r"revenue\s+Доходы\s+организации\s+за\s+год", pages[6])),
        True,
    )
    check(
        "pdf_page7_expenditure_is_expense",
        bool(re.search(r"expenditure\s+Расходы\s+организации\s+за\s+год", pages[6])),
        True,
    )
    check(
        "pdf_page7_workers_annual_average",
        bool(
            re.search(
                r"employees_count\s+Среднесписочная\s+численность\s+работников\s+за\s+год", pages[6]
            )
        ),
        True,
    )
    check(
        "pdf_page6_coordinates_registered_settlement",
        "населённого пункта, в котором зарегистрирована организация" in pages[5],
        True,
    )
    check("pdf_page5_year_can_repeat", "Может повторяться для каждого ИНН" in pages[4], True)
    landing = re.sub(
        r"\s+",
        " ",
        BeautifulSoup((raw / "raw/tochno_sme.html").read_bytes(), "html.parser").get_text(
            " ", strip=True
        ),
    )
    check("landing_geography_june2026", "по состоянию на июнь 2026 года" in landing, True)
    sakhalin = re.sub(
        r"\s+",
        " ",
        BeautifulSoup(
            (raw / "raw/sakhalin_sme_share.htm").read_bytes().decode("cp1251"), "html.parser"
        ).get_text(" ", strip=True),
    )
    check("sakhalin_header2015_2020", "2015 г. 2020 г." in sakhalin, True)
    check("sakhalin_no2023_2024_cells", "2023" not in sakhalin and "2024" not in sakhalin, True)
    qa = json.loads((report / "QA.json").read_text("utf-8"))
    check("claimed_main_panel_joins", qa["main_panel_joins"], 0)
    check("claimed_no_annual_workers_sum", qa["no_annual_workers_sum_published"], True)
    check("historical_attributes_not_ready", qa["historical_municipal_attributes"], "NOT_READY")
    result = {
        "status": "PASS_WITH_RESTRICTIONS" if all(x["pass"] for x in checks) else "FAIL",
        "checks_total": len(checks),
        "checks_pass": sum(x["pass"] for x in checks),
        "checks": checks,
        "counts": results,
        "archive_members_verified": archive_checks,
        "total_entire_period_records": sum(x["annual"]["source_rows"] for x in results.values()),
        "models_fit": 0,
        "municipal_joins_performed": 0,
        "workers_summed": False,
        "review_scope": (
            "Independent offline record/source audit; historical municipal aggregation NOT_READY."
        ),
        "manual_visual_review": "See independent_review.md for PDF pages3/5/6/7 visual review.",
        "production_manifest_sha256": sha(report / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "duckdb_version": duckdb.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw, args.report, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_total": result["checks_total"],
                "checks_pass": result["checks_pass"],
            }
        )
    )
    raise SystemExit(0 if result["status"] == "PASS_WITH_RESTRICTIONS" else 1)
