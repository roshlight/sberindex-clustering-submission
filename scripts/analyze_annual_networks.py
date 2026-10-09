"""Exploratory annual-mean smoothing sensitivity; no economic attributes enter geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from municipal_profiles.dynamics import PROFILE_COLUMNS, temporal_tables
from municipal_profiles.networks import (
    co_membership_jaccard,
    compare_partitions,
    graph_statistics,
    knn_graph,
    leiden_partition,
    similarity_matrix,
)

PRIMARY = ("hellinger", 20, 1.0, 13)
INPUTS = {
    "assignments": "reports/initial_study/assignments.parquet",
    "variant_keys": "reports/initial_study/variant_keys.json",
    "december_comparator": "reports/initial_study/temporal_summary.csv",
    "monthly_support": "reports/initial_study/membership_agreement.parquet",
    "networks_code": "src/municipal_profiles/networks.py",
    "dynamics_code": "src/municipal_profiles/dynamics.py",
    "uv_lock": "uv.lock",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_parquet(path):
    with duckdb.connect() as db:
        return db.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def variant_keys():
    keys = []
    for geometry in ("hellinger", "cosine"):
        for k in (10, 20, 30):
            for resolution in (0.75, 1.0, 1.25):
                for seed in (13, 29, 47) if (k, resolution) == (20, 1.0) else (13,):
                    key = (geometry, k, resolution, seed)
                    if key != PRIMARY:
                        keys.append(key)
    if len(keys) != 21 or len(set(keys)) != 21:
        raise AssertionError("Expected 21 distinct alternatives")
    return keys


def protocol(repo):
    keys = variant_keys()
    original = json.loads((repo / INPUTS["variant_keys"]).read_text(encoding="utf-8"))
    if [list(k) for k in keys] != original:
        raise ValueError("Alternative set/order differs from original study")
    return {
        "experiment": "annual_mean_smoothing_exploratory_v1",
        "status": "prospective_for_this_new_sensitivity_only; 2023/2024 already inspected",
        "years": [2023, 2024],
        "n_cohort": 2016,
        "profile_aggregation": "mean of twelve normalized five-category monthly profiles",
        "month_field_convention": "YYYY-12 denotes the annual window, not a December profile",
        "primary": list(PRIMARY),
        "alternative_keys": [list(k) for k in keys],
        "leiden_fits": 44,
        "kmeans_fits": 2,
        "kmeans": {"n_clusters": 8, "n_init": 20, "seed": 13, "features": "sqrt annual mean p"},
        "peer_jaccard_threshold": 0.8,
        "support_fraction_threshold": 0.8,
        "paired_switch_rule": "same alternative must support both endpoints and reproduce "
        "primary Hungarian-aligned switch status; fraction>=0.8",
        "geometry_inputs": "consumption only; no economic, population or spatial attributes",
        "comparators": "original December2023 to December2024 partitions on identical 2016 IDs",
        "limitations": "no new holdout; smoothing changes the question; do not promote a failed "
        "monthly typology to confirmed annual economic classes",
        "input_hashes": {
            name: {"path": path, "sha256": sha(repo / path)} for name, path in INPUTS.items()
        },
        "script_sha256": sha(__file__),
    }


def annual_profiles(frame, expected_n=2016):
    frame = frame.loc[frame.method.eq("leiden_hellinger")].copy()
    if frame.duplicated(["month", "territory_id"]).any():
        raise ValueError("Duplicate primary municipality/month")
    expected = {f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)}
    if set(frame.month.unique()) != expected:
        raise ValueError("Exactly 24 complete calendar months are required")
    count = frame.groupby("territory_id").month.nunique()
    if len(count) != expected_n or not count.eq(24).all():
        raise ValueError("Every cohort municipality needs all 24 months")
    values = frame[list(PROFILE_COLUMNS)].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.allclose(values.sum(axis=1), 1)
    ):
        raise ValueError("Monthly profiles must be finite normalized compositions")
    frame["year"] = frame.month.str[:4].astype(int)
    means = (
        frame.groupby(["year", "territory_id"], sort=True)[list(PROFILE_COLUMNS)]
        .mean()
        .reset_index()
    )
    if not np.allclose(means[list(PROFILE_COLUMNS)].sum(axis=1), 1):
        raise AssertionError("Arithmetic mean profiles do not sum to one")
    return means


def self_checks():
    rows = []
    for year in (2023, 2024):
        for month in range(1, 13):
            for territory in (1, 2):
                p = [0.6, 0.1, 0.1, 0.1, 0.1] if month <= 6 else [0.2] * 5
                rows.append(
                    {
                        "method": "leiden_hellinger",
                        "month": f"{year}-{month:02d}",
                        "territory_id": territory,
                        **dict(zip(PROFILE_COLUMNS, p, strict=True)),
                    }
                )
    frame = pd.DataFrame(rows)
    mean = annual_profiles(frame, expected_n=2)
    assert len(mean) == 4 and np.allclose(
        mean[list(PROFILE_COLUMNS)], [0.4, 0.15, 0.15, 0.15, 0.15]
    )
    try:
        annual_profiles(frame.iloc[:-1], expected_n=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing source month was accepted")
    assert len(variant_keys()) == 21 and PRIMARY not in variant_keys()
    return {
        "status": "PASS",
        "checks": 3,
        "scope": "hand-calculated annual mean, missing month, exact variants",
    }


def write_parquet(frame, destination):
    with duckdb.connect() as db:
        db.register("result", frame)
        db.execute("COPY result TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(destination)])


def run(repo, output):
    frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    intended = protocol(repo)
    if {k: v for k, v in frozen.items() if k != "prepared_at"} != intended:
        raise ValueError("Protocol/input/code identity changed since preparation")
    if set(p.name for p in output.iterdir()) != {"protocol.json"}:
        raise FileExistsError("Use a fresh prepared directory; prior outcomes remain immutable")
    manifest = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha(output / "protocol.json"),
        "self_checks": self_checks(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    source = read_parquet(repo / INPUTS["assignments"])
    annual = annual_profiles(source)
    keys = variant_keys()
    assignments, memberships, robustness, yearly, cross_profile = [], [], [], [], []
    evidence = {}
    try:
        with threadpool_limits(limits=1):
            for year, part in annual.groupby("year", sort=True):
                part = part.sort_values("territory_id")
                ids, p = (
                    part.territory_id.to_numpy(dtype=int),
                    part[list(PROFILE_COLUMNS)].to_numpy(),
                )
                graphs = {}
                for geometry in ("hellinger", "cosine"):
                    similarity = similarity_matrix(p, geometry)
                    for k in (10, 20, 30):
                        graphs[geometry, k] = knn_graph(similarity, k)
                partitions = {
                    key: leiden_partition(graphs[key[0], key[1]], key[2], key[3])
                    for key in [PRIMARY, *keys]
                }
                reference = partitions[PRIMARY]
                jaccard = np.stack([co_membership_jaccard(reference, partitions[k]) for k in keys])
                support = jaccard >= 0.8
                fractions = support.mean(axis=0)
                month = f"{year}-12"
                evidence[month] = {
                    "ids": ids,
                    "labels": np.stack([partitions[k] for k in keys]),
                    "support": support,
                }
                membership = pd.DataFrame(
                    {
                        "month": month,
                        "territory_id": ids,
                        "agreement": fractions,
                        "mean_co_membership_jaccard": jaccard.mean(axis=0),
                        "n_comparisons": 21,
                    }
                )
                memberships.append(membership)
                methods = {
                    "leiden_hellinger": reference,
                    "kmeans": KMeans(n_clusters=8, random_state=13, n_init=20).fit_predict(
                        np.sqrt(p)
                    ),
                }
                for method, labels in methods.items():
                    frame = pd.DataFrame(p, columns=PROFILE_COLUMNS)
                    frame = frame.assign(month=month, territory_id=ids, method=method, label=labels)
                    assignments.append(frame)
                    baseline = source.loc[
                        source.method.eq(method) & source.month.eq(month)
                    ].sort_values("territory_id")
                    if not np.array_equal(ids, baseline.territory_id.to_numpy()):
                        raise ValueError("December comparator cohort differs")
                    cross_profile.append(
                        {
                            "year": int(year),
                            "method": method,
                            "n": len(ids),
                            "ari_annual_vs_december": adjusted_rand_score(labels, baseline.label),
                            **compare_partitions(labels, baseline.label.to_numpy()),
                        }
                    )
                for index, key in enumerate(keys):
                    robustness.append(
                        {
                            "year": int(year),
                            "geometry": key[0],
                            "k": key[1],
                            "resolution": key[2],
                            "seed": key[3],
                            "median_peer_jaccard": np.median(jaccard[index]),
                            "peer_support_share": support[index].mean(),
                            **compare_partitions(reference, partitions[key]),
                            **graph_statistics(graphs[key[0], key[1]]),
                        }
                    )
                yearly.append(
                    {
                        "year": int(year),
                        "n": len(ids),
                        "primary_n_clusters": len(np.unique(reference)),
                        "primary_smallest_cluster": int(np.bincount(reference).min()),
                        "primary_largest_cluster_share": np.bincount(reference).max() / len(ids),
                        "n_strong_memberships": int(np.sum(fractions >= 0.8)),
                        "strong_membership_share": np.mean(fractions >= 0.8),
                        "median_support_fraction": np.median(fractions),
                        "median_mean_peer_jaccard": np.median(jaccard.mean(axis=0)),
                    }
                )
                print(json.dumps(yearly[-1]), flush=True)
        assignments_frame, membership_frame = pd.concat(assignments), pd.concat(memberships)
        transitions, flows, summary = temporal_tables(
            assignments_frame, membership_frame, 0.8, evidence
        )
        baseline_summary = pd.read_csv(repo / INPUTS["december_comparator"])
        baseline_summary = baseline_summary.loc[
            baseline_summary.from_month.eq("2023-12")
            & baseline_summary.to_month.eq("2024-12")
            & baseline_summary.method.isin(("leiden_hellinger", "kmeans"))
        ].copy()
        if len(baseline_summary) != 2 or not baseline_summary.ncommon.eq(2016).all():
            raise ValueError("Original December comparison is not two full-cohort methods")
        comparison = pd.concat(
            [
                summary.assign(profile_basis="annual_mean_12months"),
                baseline_summary.assign(profile_basis="original_december"),
            ]
        )
        original_membership = read_parquet(repo / INPUTS["monthly_support"])
        december_support = original_membership.loc[
            original_membership.month.isin(("2023-12", "2024-12"))
        ]
        december_support = december_support.assign(strong=december_support.agreement.ge(0.8))
        december_support = (
            december_support.groupby("month")
            .agg(
                n=("territory_id", "size"),
                n_strong_memberships=("strong", "sum"),
                median_support_fraction=("agreement", "median"),
            )
            .reset_index()
        )
        tables = {
            "yearly_support.csv": pd.DataFrame(yearly),
            "variant_robustness.csv": pd.DataFrame(robustness),
            "annual_vs_december_partitions.csv": pd.DataFrame(cross_profile),
            "temporal_summary.csv": summary,
            "temporal_comparison.csv": comparison,
            "cluster_flows.csv": flows,
            "original_december_support.csv": december_support,
        }
        for name, frame in tables.items():
            frame.to_csv(output / name, index=False)
        write_parquet(annual, output / "annual_profiles.parquet")
        write_parquet(assignments_frame, output / "assignments.parquet")
        write_parquet(membership_frame, output / "membership_agreement.parquet")
        write_parquet(transitions, output / "transitions.parquet")
        np.savez_compressed(
            output / "variant_evidence.npz",
            **{
                f"{month}__{kind}": array
                for month, values in evidence.items()
                for kind, array in values.items()
            },
        )
        (output / "variant_keys.json").write_text(json.dumps(keys, indent=2), encoding="utf-8")
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            yearly_support=yearly,
            temporal_results=json.loads(comparison.to_json(orient="records", double_precision=15)),
            python=platform.python_version(),
            versions={
                name: version(name)
                for name in ["numpy", "pandas", "scikit-learn", "igraph", "leidenalg", "duckdb"]
            },
            outputs={p.name: sha(p) for p in output.iterdir() if p.name != "manifest.json"},
        )
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    destination = args.output or args.repo_root / "reports/enriched_analysis/annual_networks"
    if args.self_test:
        print(json.dumps(self_checks()))
    elif args.prepare:
        if destination.exists():
            raise FileExistsError("Prepare a new sensitivity directory")
        definition = protocol(args.repo_root)
        definition["prepared_at"] = datetime.now(UTC).isoformat()
        destination.mkdir(parents=True)
        (destination / "protocol.json").write_text(
            json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"status": "prepared_before_fit", "path": str(destination)}))
    else:
        result = run(args.repo_root, destination)
        print(json.dumps({"status": result["status"], "output": str(destination)}))
