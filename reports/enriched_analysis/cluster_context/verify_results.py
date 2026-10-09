"""Recompute saved context sums of squares and permutation summaries without fitting."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["outputs"].items():
        assert sha(REPORT / name) == digest
    for value in manifest["input_hashes"].values():
        assert sha(ROOT / value["path"]) == value["sha256"]
    assert sha(ROOT / "scripts/analyze_cluster_context.py") == manifest["script_sha256"]
    with duckdb.connect() as db:
        cohort = db.execute(
            "SELECT * FROM read_parquet(?)", [str(REPORT / "analytical_cohort.parquet")]
        ).df()
    results = pd.read_csv(REPORT / "feature_dispersion.csv")
    draws = pd.read_csv(REPORT / "permutation_draws.csv")
    for row in results.itertuples(index=False):
        frame = cohort.loc[cohort.month.eq(row.month) & cohort.method.eq(row.method)]
        frame = frame.loc[np.isfinite(frame[row.variable]) & frame.region_code.notna()]
        overall = frame[row.variable].mean()
        grouped = frame.groupby("label")[row.variable].agg(["mean", "count"])
        between = float(np.sum(grouped["count"] * (grouped["mean"] - overall) ** 2))
        total = float(((frame[row.variable] - overall) ** 2).sum())
        assert len(frame) == row.n_available
        assert np.isclose(between, row.ss_between) and np.isclose(total, row.ss_total)
        assert np.isclose(total - between, row.ss_within)
        null = draws.loc[
            draws.month.eq(row.month)
            & draws.method.eq(row.method)
            & draws.variable.eq(row.variable),
            "ss_between",
        ]
        assert len(null) == 499
        p = (1 + (null >= between - 1e-12).sum()) / 500
        assert np.isclose(p, row.p_within_region_one_sided)
        assert np.isclose(null.median() / total, row.null_between_share_median)
    report = {
        "status": "PASS",
        "source_and_output_hashes": True,
        "recomputed_dispersion_rows": len(results),
        "verified_permutation_rows": len(draws),
        "method": "separate pandas group-mean SS and saved null summary reconciliation; same agent",
        "verifier_sha256": sha(Path(__file__)),
    }
    (REPORT / "independent_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
