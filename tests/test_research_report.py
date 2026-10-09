"""Check report combination and provenance using saved synthetic partitions."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from municipal_profiles.dynamics import PROFILE_COLUMNS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_research_report.py"
build_report = runpy.run_path(str(SCRIPT))["build_report"]
METHODS = ["leiden_hellinger", "leiden_cosine", "kmeans", "kmeans_matched_k"]


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved_runs(tmp_path):
    runs = []
    for year in (2023, 2024):
        directory = tmp_path / str(year)
        directory.mkdir()
        labels = np.array([0, 0, 1, 1] if year == 2023 else [9, 2, 2, 2])
        assignments, members, metrics, clusters, robustness, arrays = [], [], [], [], [], {}
        for month_index in range(1, 13):
            month = f"{year}-{month_index:02d}"
            for method in METHODS:
                for territory, label in enumerate(labels, start=1):
                    assignments.append(
                        {
                            "month": month,
                            "method": method,
                            "territory_id": territory,
                            "label": label,
                            **dict.fromkeys(PROFILE_COLUMNS, 0.2),
                        }
                    )
                metrics.append(
                    {
                        "month": month,
                        "method": method,
                        "n_entities": 4,
                        "n_clusters": 2,
                        "silhouette": month_index / 100,
                        "calinski_harabasz": 10,
                        "s_dbw": np.nan,
                        "avi": 0.9,
                        "avu": 1,
                        "modularity": 0.4,
                        "smallest_cluster_size": int(
                            np.unique(labels, return_counts=True)[1].min()
                        ),
                        "largest_cluster_share": float(
                            np.unique(labels, return_counts=True)[1].max() / 4
                        ),
                    }
                )
                for label in np.unique(labels):
                    clusters.append(
                        {
                            "month": month,
                            "method": method,
                            "label": label,
                            "size": int((labels == label).sum()),
                        }
                    )
            for territory in range(1, 5):
                members.append(
                    {
                        "month": month,
                        "territory_id": territory,
                        "agreement": 0.0 if territory == 1 else 1.0,
                        "mean_co_membership_jaccard": 0.5 if territory == 1 else 0.9,
                        "n_comparisons": 21,
                    }
                )
            for variant in range(21):
                robustness.append(
                    {
                        "month": month,
                        "family": "core_grid",
                        "seed": variant,
                        "ari_to_primary": 0.75,
                        "aligned_agreement": 0.9,
                        "median_co_membership_jaccard": 0.8,
                        "candidate_n_clusters": 2,
                    }
                )
            arrays[f"{month}_ids"] = np.arange(1, 5)
            arrays[f"{month}_labels"] = np.tile(labels, (21, 1))
            arrays[f"{month}_support"] = np.tile([False, True, True, True], (21, 1))
        tables = {
            "assignments": pd.DataFrame(assignments),
            "membership_agreement": pd.DataFrame(members),
            "metrics": pd.DataFrame(metrics),
            "cluster_profiles": pd.DataFrame(clusters),
            "robustness": pd.DataFrame(robustness),
            "null_metrics": pd.DataFrame(columns=["month", "repeat", "silhouette"]),
        }
        for name, frame in tables.items():
            frame.to_csv(directory / f"{name}.csv", index=False)
        np.savez_compressed(directory / "variant_evidence.npz", **arrays)
        (directory / "variant_keys.json").write_text(
            json.dumps([["hellinger", i, 1, 13] for i in range(21)])
        )
        manifest = {
            "status": "complete",
            "run_id": f"synthetic-{year}",
            "year": str(year),
            "phase": "development" if year == 2023 else "evaluation",
            "config_sha256": "a" * 64,
            "data_sha256": "b" * 64,
            "code_sha256": "c" * 64,
            "config": {
                "primary": {"geometry": "hellinger"},
                "agreement_threshold": 0.8,
                "co_membership_threshold": 0.8,
            },
            "table_rows": {name: len(frame) for name, frame in tables.items()},
            "output_hashes": {path.name: _hash(path) for path in directory.iterdir()},
        }
        (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        runs.append(directory)
    return runs


def test_cli_combines_december_boundary_year_pairs_and_hashed_outputs(saved_runs, tmp_path):
    output = tmp_path / "combined"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs", *map(str, saved_runs), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["no_model_fitting"] is True
    assert summary["n_primary_entity_months"] == 96
    assert summary["n_strong_entity_months"] == 72
    assert summary["strong_share_of_all"] == 0.75
    temporal = pd.read_csv(output / "temporal_summary.csv")
    assert len(temporal) == (23 + 12) * 4
    assert len(temporal.loc[temporal.lag.eq(12)]) == 12 * 4
    december = temporal.loc[temporal.from_month.eq("2023-12") & temporal.to_month.eq("2024-01")]
    assert len(december) == 4
    assert december.lag.eq(1).all()
    assert december.switch_share.eq(0.25).all()
    with duckdb.connect() as connection:
        transitions = connection.read_parquet(str(output / "transitions.parquet")).df()
        assignments = connection.read_parquet(str(output / "assignments.parquet")).df()
        membership = connection.read_parquet(str(output / "membership_agreement.parquet")).df()
    assert len(assignments) == 24 * 4 * 4
    assert len(membership) == 24 * 4
    assert membership.n_comparisons.eq(21).all()
    assert len(transitions) == (23 + 12) * 4 * 4
    assert transitions.loc[~transitions.method.eq("leiden_hellinger"), "robust_switch"].isna().all()
    counts = {row["lag"]: row for row in summary["primary_transition_counts"]}
    assert counts[1]["n_robust_switches"] == 1
    assert counts[12]["n_robust_switches"] == 12
    support = pd.read_csv(output / "monthly_support.csv")
    assert support.n_entities.eq(4).all()
    assert support.n_strong.eq(3).all()
    metrics = pd.read_csv(output / "method_metrics.csv")
    assert len(metrics) == 2 * 4
    assert metrics.s_dbw_count.eq(0).all()
    assert metrics.s_dbw_median.isna().all()
    assert metrics.silhouette_count.eq(12).all()
    np.testing.assert_allclose(metrics.silhouette_median, 0.065)
    assert len(pd.read_csv(output / "robustness.csv")) == 24 * 21
    assert pd.read_csv(output / "null_metrics.csv").empty
    manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["sources"]) == 2
    for filename, digest in manifest["output_hashes"].items():
        assert _hash(output / filename) == digest


@pytest.mark.parametrize("problem", ["incomplete", "code_hash", "variant_order", "tampered_csv"])
def test_rejects_incompatible_or_tampered_sources(saved_runs, tmp_path, problem):
    directory = saved_runs[1]
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if problem == "incomplete":
        manifest["status"] = "running"
    elif problem == "code_hash":
        manifest["code_sha256"] = "d" * 64
    elif problem == "variant_order":
        keys_path = directory / "variant_keys.json"
        keys = json.loads(keys_path.read_text(encoding="utf-8"))
        keys.reverse()
        keys_path.write_text(json.dumps(keys), encoding="utf-8")
        manifest["output_hashes"][keys_path.name] = _hash(keys_path)
    else:
        with (directory / "assignments.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        build_report(saved_runs, output)
    assert not output.exists()


def test_support_must_match_saved_variant_evidence(saved_runs, tmp_path):
    directory = saved_runs[0]
    membership_path = directory / "membership_agreement.csv"
    membership = pd.read_csv(membership_path)
    membership.loc[0, "agreement"] = 1
    membership.to_csv(membership_path, index=False)
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_hashes"][membership_path.name] = _hash(membership_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="saved variant support"):
        build_report(saved_runs, tmp_path / "bad-support")


def test_one_year_is_supported_and_existing_output_is_never_overwritten(saved_runs, tmp_path):
    output = tmp_path / "one-year"
    summary = build_report(saved_runs[:1], output)
    assert summary["years"] == ["2023"]
    assert summary["primary_transition_counts"][0]["n_month_pairs"] == 11
    digest = _hash(output / "report_manifest.json")
    with pytest.raises(FileExistsError):
        build_report(saved_runs[:1], output)
    assert _hash(output / "report_manifest.json") == digest
    with pytest.raises(ValueError, match="distinct"):
        build_report([saved_runs[0], saved_runs[0]], tmp_path / "duplicate")
