"""Build an explicitly conditional annual-code candidate grid; do not fit models."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_panel(source, links, statuses):
    source_key = ["source_region_url", "source_oktmo", "indicator_id", "year"]
    annual_key = ["territory_id", "reference_date"]
    if source.duplicated(source_key).any() or links.duplicated(annual_key).any():
        raise ValueError("Duplicate source or annual municipality keys")
    if not links.reference_date.str.fullmatch(r"202[34]-01-01").all():
        raise ValueError("Annual code reference date is not a supported January snapshot")
    links = links.rename(columns={"reference_date": "code_reference_date"}).copy()
    links["year"] = links.code_reference_date.str[:4]
    grid = links.merge(pd.DataFrame({"indicator_id": ["8423005", "8423007"]}), how="cross")
    source = source.drop(columns=["territory_id", "join_status"], errors="ignore")
    joined = grid.merge(source, on=source_key, how="left", validate="many_to_one", indicator=True)
    status_key = source_key[:-1]
    if statuses.duplicated(status_key).any():
        raise ValueError("Duplicate extraction status")
    joined = joined.merge(
        statuses[status_key + ["status", "error"]].rename(columns={"status": "extraction_status"}),
        on=status_key,
        how="left",
        validate="many_to_one",
    )
    joined["cell_status"] = "not_returned"
    present = joined._merge.eq("both")
    joined.loc[present, "cell_status"] = "returned_source_marker"
    joined.loc[present & joined.value_status.eq("observed"), "cell_status"] = "observed"
    joined["geography_interpretation"] = (
        "conditional: exact source code and Sber January version; "
        "no independent equivalence of full-year economic geography"
    )
    return joined.drop(columns="_merge").sort_values(["territory_id", "year", "indicator_id"])


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError("Use a new annual candidate output directory")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_with_gaps":
        raise ValueError("The source collection has not completed")
    for name, expected in manifest["output_hashes"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Source output hash mismatch: {name}")
    names = ["economy_source.csv", "annual_code_reference_links.csv", "target_status.csv"]
    frames = [pd.read_csv(directory / name, dtype=str, keep_default_na=False) for name in names]
    result = candidate_panel(*frames)
    output.mkdir(parents=True)
    result.to_csv(output / "annual_candidate_panel.csv", index=False)
    with duckdb.connect() as db:
        db.register("candidate", result)
        db.execute(
            "COPY candidate TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(output / "annual_candidate_panel.parquet")],
        )
    coverage = result.groupby(["year", "indicator_id", "cell_status"]).size().unstack(fill_value=0)
    coverage["expected_cells"] = coverage.sum(axis=1)
    coverage.to_csv(output / "annual_coverage.csv")
    result.loc[result.cell_status.ne("observed")].to_csv(output / "annual_gaps.csv", index=False)
    audit = {
        "status": "PASS",
        "scope": "conditional annual-code candidate layer; no legal boundary audit or model result",
        "created_at": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": sha(directory / "manifest.json"),
        "script_sha256": sha(__file__),
        "expected_cells": len(result),
        "unique_territories": int(result.territory_id.nunique()),
        "cell_status_counts": result.cell_status.value_counts().to_dict(),
        "coverage": coverage.reset_index().to_dict("records"),
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.output), ensure_ascii=False))
