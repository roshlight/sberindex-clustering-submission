"""Exercise the real CLI against a small complete synthetic source panel."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from municipal_profiles.data import EXPECTED_PERIODS, PUBLISHED_CATEGORIES, TOTAL_CATEGORY


def test_development_cli_preserves_evidence_and_missing_value_semantics(tmp_path):
    project = Path(__file__).resolve().parents[1]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # The CLI hashes files below its workspace; populate that tree as in a real run.
    shutil.copytree(project / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    rng = np.random.default_rng(817)
    n_entities = 36  # Every k in the fixed 10/20/30 comparison grid is feasible.
    base = rng.lognormal(mean=7, sigma=0.65, size=(n_entities, 5))
    rows = []
    for month_index, month in enumerate(EXPECTED_PERIODS):
        values = np.rint(
            base * np.exp(rng.normal(0, 0.035, base.shape) + month_index * 0.004)
        ).astype(int)
        for territory_index, category_values in enumerate(values):
            for category, value in zip(PUBLISHED_CATEGORIES, category_values, strict=True):
                rows.append((month, territory_index + 1, category, int(value)))
            rows.append(
                (month, territory_index + 1, TOTAL_CATEGORY, int(category_values.sum()) + 2000)
            )
    frame = pd.DataFrame(rows, columns=["date", "territory_id", "category", "value"])
    source_path = raw / "consumption.parquet"
    with duckdb.connect() as connection:
        connection.from_df(frame).write_parquet(str(source_path))

    config = json.loads((project / "configs" / "robustness_v1.json").read_text(encoding="utf-8"))
    config.update(anchor_months=[], permutation_repeats=0)
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "run_study.py"),
            "--phase",
            "development",
            "--run-id",
            "synthetic-smoke",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    output = tmp_path / "results" / "synthetic-smoke"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["phase"] == "development"
    assert manifest["year"] == "2023"
    assert manifest["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest["data_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    code_digest = hashlib.sha256()
    for path in sorted((tmp_path / "src" / "municipal_profiles").glob("*.py")):
        code_digest.update(path.name.encode())
        code_digest.update(path.read_bytes())
    assert manifest["code_sha256"] == code_digest.hexdigest()
    assert manifest["code_sha256"] != hashlib.sha256(b"").hexdigest()
    for filename, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    assert {"variant_evidence.npz", "variant_keys.json", "metrics.csv"}.issubset(
        manifest["output_hashes"]
    )
    assert len(list(output.glob("checkpoint_*.csv"))) == 12

    tables = {}
    for name, row_count in manifest["table_rows"].items():
        path = output / f"{name}.csv"
        # The optional null experiment is disabled, so its empty CSV has no schema.
        tables[name] = pd.read_csv(path) if path.read_bytes().strip() else pd.DataFrame()
        assert len(tables[name]) == row_count
    methods = {"leiden_hellinger", "leiden_cosine", "kmeans", "kmeans_matched_k"}
    months = set(EXPECTED_PERIODS[:12])
    assignments = tables["assignments"]
    assert len(assignments) == 12 * n_entities * 4
    assert set(assignments["month"]) == months
    assert set(assignments["method"]) == methods
    assert not assignments.duplicated(["month", "territory_id", "method"]).any()
    assert assignments.groupby(["month", "method"])["territory_id"].nunique().eq(n_entities).all()
    metrics = tables["metrics"]
    assert len(metrics) == 12 * 4
    assert not metrics.duplicated(["month", "method"]).any()
    assert metrics["n_entities"].eq(n_entities).all()
    assert metrics["n_clusters"].between(1, n_entities).all()
    cluster_counts = assignments.groupby(["month", "method"])["label"].nunique()
    for row in metrics.itertuples():
        assert row.n_clusters == cluster_counts.loc[row.month, row.method]
        statuses = json.loads(row.metric_status)
        for metric, status in statuses.items():
            value = getattr(row, metric)
            if status.startswith("undefined"):
                assert pd.isna(value), (metric, status, value)
            else:
                assert np.isfinite(value), (metric, status, value)
    k_counts = metrics.pivot(index="month", columns="method", values="n_clusters")
    assert k_counts["kmeans"].eq(8).all()
    assert k_counts["kmeans_matched_k"].eq(k_counts["leiden_hellinger"]).all()

    robustness = tables["robustness"]
    variant_key = ["month", "geometry", "k", "resolution", "seed", "feature_mode"]
    assert len(robustness) == 12 * 21
    assert robustness["family"].eq("core_grid").all()
    assert not robustness.duplicated(variant_key).any()
    assert robustness.groupby("month").size().eq(21).all()
    membership = tables["membership_agreement"]
    assert len(membership) == 12 * n_entities
    assert not membership.duplicated(["month", "territory_id"]).any()
    assert membership["n_comparisons"].eq(21).all()
    assert membership["agreement"].between(0, 1).all()
    variants = json.loads((output / "variant_keys.json").read_text(encoding="utf-8"))
    assert len(variants) == len({tuple(key) for key in variants}) == 21
    with np.load(output / "variant_evidence.npz", allow_pickle=False) as evidence:
        assert len(evidence.files) == 12 * 3
        for month in sorted(months):
            np.testing.assert_array_equal(evidence[f"{month}_ids"], np.arange(1, n_entities + 1))
            assert evidence[f"{month}_labels"].shape == (21, n_entities)
            support = evidence[f"{month}_support"]
            assert support.shape == (21, n_entities)
            assert support.dtype == np.dtype(bool)
            monthly = membership.loc[membership["month"].eq(month)].sort_values("territory_id")
            np.testing.assert_allclose(support.mean(axis=0), monthly["agreement"])

    transitions = tables["transitions"]
    transition_key = ["from_month", "to_month", "lag", "method", "territory_id"]
    assert len(transitions) == 11 * n_entities * 4
    assert not transitions.duplicated(transition_key).any()
    assert transitions["lag"].eq(1).all()  # A single-year run cannot contain lag=12.
    assessed = transitions["method"].eq("leiden_hellinger")
    assert transitions.loc[assessed, "transition_reproduction_fraction"].between(0, 1).all()
    assert transitions.loc[~assessed, "robust_switch"].isna().all()
    assert transitions.loc[~assessed, "transition_reproduction_fraction"].isna().all()
    summary = tables["temporal_summary"]
    assert len(summary) == 11 * 4
    assert summary["ncommon"].eq(n_entities).all()
    assert (
        summary.loc[~summary["method"].eq("leiden_hellinger"), "robust_switch_share_of_all"]
        .isna()
        .all()
    )
