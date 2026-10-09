"""Extract source-keyed population layers; no OKTMO/SberIndex matching is performed."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from sber_enrichment import population
from sber_enrichment.population import POPULATION_FIELDS, parse_population_workbook


def prepare(inputs, source_urls, output, retrieved_at):
    if len(inputs) != len(source_urls):
        raise ValueError("Provide one source URL per workbook")
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; existing outputs are never overwritten"
        )
    frames, audits = [], []
    for path, url in zip(inputs, source_urls, strict=True):
        frame, audit = parse_population_workbook(path, source_url=url, retrieved_at=retrieved_at)
        frames.append(frame)
        audits.append(audit)
    combined = pd.concat(frames, ignore_index=True)
    if combined.source_record_id.duplicated().any():
        raise ValueError("The same source rows were supplied more than once")
    issue_mask = combined.balance_status.ne("pass") | combined.source_code_duplicate
    issue_mask |= combined.source_code_status.isin(["numeric_precision_risk", "unparsed_text"])
    issue_mask |= combined.region_name_source.isna()
    for field in POPULATION_FIELDS:
        issue_mask |= ~combined[f"{field}_status"].isin(["observed", "source_absence_marker"])
    issues = combined.loc[issue_mask].copy()
    regions = combined.loc[combined.row_kind.eq("region_header")].copy()
    counts = (
        combined.groupby(["reference_date", "region_name_source"], dropna=False)
        .agg(
            n_source_rows=("source_record_id", "size"),
            n_numeric_source_codes=("source_code", "count"),
            n_distinct_source_codes=("source_code", "nunique"),
        )
        .reset_index()
    )
    output.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as connection:
        connection.register("population_frame", combined)
        connection.execute(
            "COPY population_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(output / "population_unlinked.parquet")],
        )
    issues.to_csv(output / "population_qa_rows.csv", index=False, encoding="utf-8")
    regions.to_csv(output / "population_region_headers.csv", index=False, encoding="utf-8")
    counts.to_csv(output / "population_region_coverage.csv", index=False, encoding="utf-8")
    (output / "population_source_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "status": "complete_unlinked",
        "operation": "read_only_source_extraction_no_crosswalk",
        "n_rows": len(combined),
        "n_quality_flag_rows": len(issues),
        "reference_dates": sorted(combined.reference_date.unique()),
        "processor_sha256": {
            "prepare_population.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "population.py": hashlib.sha256(Path(population.__file__).read_bytes()).hexdigest(),
        },
        "sources": [
            {
                field: audit[field]
                for field in (
                    "source_url",
                    "source_filename",
                    "source_sha256",
                    "source_bytes",
                    "retrieved_at",
                    "reference_date",
                    "published_at",
                )
            }
            for audit in audits
        ],
        "output_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
        },
    }
    (output / "population_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--source-urls", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retrieved-at", required=True, help="Actual retrieval date, YYYY-MM-DD")
    args = parser.parse_args(argv)
    manifest = prepare(args.inputs, args.source_urls, args.output, args.retrieved_at)
    print(
        json.dumps(
            {key: manifest[key] for key in ("status", "n_rows", "n_quality_flag_rows")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
