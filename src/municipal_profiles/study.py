"""Run the preregistered descriptive robustness study on one calendar year."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from municipal_profiles.data import PUBLISHED_CATEGORIES, load_panel, make_profiles
from municipal_profiles.dynamics import temporal_tables
from municipal_profiles.metrics import evaluate_partition
from municipal_profiles.networks import (
    co_membership_jaccard,
    compare_partitions,
    equal_edge_budget_graph,
    graph_statistics,
    knn_graph,
    leiden_partition,
    neighbor_jaccard,
    similarity_matrix,
)

FEATURE_COLUMNS = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "municipal_profiles").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_lock(lock: dict, config_hash: str, source_hash: str, implementation_hash: str):
    expected = {
        "config_sha256": config_hash,
        "data_sha256": source_hash,
        "code_sha256": implementation_hash,
    }
    for field, value in expected.items():
        if lock.get(field) != value:
            raise ValueError(f"Evaluation lock mismatch: {field}")
    if lock.get("development_status") != "complete":
        raise ValueError("Evaluation requires a completed development run")


def _metric_record(x, labels, graph, distances, **identifiers):
    result = evaluate_partition(x, labels, graph, distances=distances)
    statuses = result.pop("metric_status", {})
    return {
        **identifiers,
        **result,
        "metric_status": json.dumps(statuses, ensure_ascii=False),
        **graph_statistics(graph),
    }


def run_study(root: Path, config_path: Path, run_id: str, phase: str, lock_path: Path | None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = root / "data" / "raw" / "consumption.parquet"
    hashes = {
        "config_sha256": file_hash(config_path),
        "data_sha256": file_hash(source_path),
        "code_sha256": code_hash(root),
    }
    if phase == "evaluation":
        if lock_path is None:
            raise ValueError("--lock is required for evaluation")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        validate_lock(lock, *hashes.values())
    year = config[f"{phase}_year"]
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a directory name without parent traversal")
    output = root / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "year": year,
        "status": "running",
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **hashes,
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "threadpool_limit": 1,
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scipy", "scikit-learn", "igraph", "leidenalg"]
        },
        "claim": "Retrospective procedural stability; not forecasts or true economic classes",
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    start = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            tables = _compute(source_path, config, year, output)
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8")
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - start,
            finished_at=pd.Timestamp.now(tz="UTC").isoformat(),
            table_rows={name: len(frame) for name, frame in tables.items()},
            output_hashes={
                path.name: file_hash(path)
                for path in output.iterdir()
                if path.suffix in {".csv", ".npz"} or path.name == "variant_keys.json"
            },
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"run_id": run_id, "status": "complete", "seconds": round(time.monotonic() - start, 1)}
        ),
        flush=True,
    )
    return manifest


def _compute(source_path, config, year, output):
    balanced = load_panel(source_path)
    available = load_panel(source_path, complete_only=False)
    balanced = balanced.loc[balanced.date.str.startswith(year)]
    available = available.loc[available.date.str.startswith(year)]
    profiles = make_profiles(balanced)
    full_profiles = make_profiles(available)
    ablations = {
        f"drop_{index}": make_profiles(balanced, drop_category=category)
        for index, category in enumerate(PUBLISHED_CATEGORIES)
    }
    ablations["arithmetic_residual"] = make_profiles(balanced, include_residual=True)
    assignments, metric_rows, robustness, membership, null_rows = [], [], [], [], []
    category_rows, edge_overlap = [], []
    variant_evidence = {}
    core_variant_keys = None
    main = config["primary"]
    primary_key = (main["geometry"], main["k"], main["resolution"], main["seed"])
    for month, (ids, p, categories) in profiles.items():
        month_start = time.monotonic()
        assert categories == list(PUBLISHED_CATEGORIES)
        x = np.sqrt(p)
        distances = cdist(x, x, metric="euclidean")
        np.fill_diagonal(distances, 0)
        anchor = month[-2:] in config["anchor_months"]
        similarities = {g: similarity_matrix(p, g) for g in config["geometries"]}
        graphs = {
            (g, k): knn_graph(similarities[g], k)
            for g in config["geometries"]
            for k in config["neighbor_counts"]
        }
        partitions = {}
        for g in config["geometries"]:
            for k in config["neighbor_counts"]:
                for resolution in config["resolutions"]:
                    seeds = (
                        config["seeds"]
                        if anchor or (k == main["k"] and resolution == main["resolution"])
                        else [main["seed"]]
                    )
                    for seed in seeds:
                        key = (g, k, resolution, seed)
                        partitions[key] = leiden_partition(graphs[g, k], resolution, seed)
        reference = partitions[primary_key]
        agreements = []
        core_labels, current_keys, co_membership_scores = [], [], []
        for (g, k, resolution, seed), labels in partitions.items():
            key = (g, k, resolution, seed)
            is_core = seed == main["seed"] or (k == main["k"] and resolution == main["resolution"])
            if key != primary_key:
                jaccard = co_membership_jaccard(reference, labels)
                agreement = jaccard >= config["co_membership_threshold"]
                if is_core:
                    agreements.append(agreement)
                    core_labels.append(labels)
                    current_keys.append(key)
                    co_membership_scores.append(jaccard)
                robustness.append(
                    {
                        "month": month,
                        "family": "core_grid" if is_core else "anchor_extra_seed",
                        "geometry": g,
                        "k": k,
                        "resolution": resolution,
                        "seed": seed,
                        "feature_mode": "five",
                        "reference_geometry": main["geometry"],
                        "median_co_membership_jaccard": float(np.median(jaccard)),
                        **compare_partitions(reference, labels),
                        **graph_statistics(graphs[g, k]),
                    }
                )
        if len(agreements) != 21:
            raise ValueError(f"Expected 21 alternatives, got {len(agreements)}")
        if core_variant_keys is None:
            core_variant_keys = current_keys
        elif current_keys != core_variant_keys:
            raise ValueError("Core variant order changed across months")
        variant_evidence[month] = {
            "ids": ids,
            "labels": np.asarray(core_labels),
            "support": np.asarray(agreements),
        }
        scores = np.mean(agreements, axis=0)
        membership.append(
            pd.DataFrame(
                {
                    "month": month,
                    "territory_id": ids,
                    "agreement": scores,
                    "mean_co_membership_jaccard": np.mean(co_membership_scores, axis=0),
                    "n_comparisons": len(agreements),
                }
            )
        )
        primary_labels = {
            f"leiden_{g}": partitions[g, main["k"], main["resolution"], main["seed"]]
            for g in config["geometries"]
        }
        primary_labels["kmeans"] = KMeans(
            n_clusters=config["baseline_clusters"], random_state=main["seed"], n_init=20
        ).fit_predict(x)
        primary_labels["kmeans_matched_k"] = KMeans(
            n_clusters=len(np.unique(reference)), random_state=main["seed"], n_init=20
        ).fit_predict(x)
        for method, labels in primary_labels.items():
            g = method.removeprefix("leiden_") if method.startswith("leiden_") else main["geometry"]
            graph = graphs[g, main["k"]]
            metric_rows.append(
                _metric_record(
                    x,
                    labels,
                    graph,
                    distances,
                    month=month,
                    method=method,
                    n_entities=len(ids),
                    geometry=g,
                    k=main["k"],
                    resolution=main["resolution"],
                    seed=main["seed"],
                )
            )
            frame = pd.DataFrame(p, columns=FEATURE_COLUMNS)
            frame = frame.assign(month=month, territory_id=ids, method=method, label=labels)
            assignments.append(frame)
            for label, group in frame.groupby("label"):
                row = {"month": month, "method": method, "label": label, "size": len(group)}
                for field in FEATURE_COLUMNS:
                    row[f"{field}_median"] = group[field].median()
                    row[f"{field}_q25"] = group[field].quantile(0.25)
                    row[f"{field}_q75"] = group[field].quantile(0.75)
                category_rows.append(row)
        overlap = neighbor_jaccard(graphs["hellinger", main["k"]], graphs["cosine", main["k"]])
        edge_overlap.append(
            {
                "month": month,
                "median_neighbor_jaccard": float(np.median(overlap)),
                "q10_neighbor_jaccard": float(np.quantile(overlap, 0.1)),
                "q90_neighbor_jaccard": float(np.quantile(overlap, 0.9)),
            }
        )
        if anchor:
            for mode, alternative_profiles in ablations.items():
                alternate_ids, alternate_p, _ = alternative_profiles[month]
                np.testing.assert_array_equal(alternate_ids, ids)
                for g in config["geometries"]:
                    graph = knn_graph(similarity_matrix(alternate_p, g), main["k"])
                    labels = leiden_partition(graph, main["resolution"], main["seed"])
                    reference_g = primary_labels[f"leiden_{g}"]
                    robustness.append(
                        {
                            "month": month,
                            "family": "feature_sensitivity",
                            "geometry": g,
                            "k": main["k"],
                            "resolution": main["resolution"],
                            "seed": main["seed"],
                            "feature_mode": mode,
                            "reference_geometry": g,
                            **compare_partitions(reference_g, labels),
                            **graph_statistics(graph),
                        }
                    )
            full_ids, full_p, _ = full_profiles[month]
            common = np.searchsorted(full_ids, ids)
            np.testing.assert_array_equal(full_ids[common], ids)
            for g in config["geometries"]:
                graph = knn_graph(similarity_matrix(full_p, g), main["k"])
                labels = leiden_partition(graph, main["resolution"], main["seed"])
                robustness.append(
                    {
                        "month": month,
                        "family": "cohort_sensitivity",
                        "geometry": g,
                        "feature_mode": "five",
                        "reference_geometry": g,
                        "available_n": len(full_ids),
                        **compare_partitions(primary_labels[f"leiden_{g}"], labels[common]),
                        **graph_statistics(graph),
                    }
                )
            budget = min(graphs[g, main["k"]].nnz // 2 for g in config["geometries"])
            controlled = {}
            for g in config["geometries"]:
                graph = equal_edge_budget_graph(graphs[g, main["k"]], budget)
                controlled[g] = leiden_partition(graph, main["resolution"], main["seed"])
                robustness.append(
                    {
                        "month": month,
                        "family": "equal_edge_budget_to_own_primary",
                        "geometry": g,
                        "edge_budget": budget,
                        "feature_mode": "five",
                        "reference_geometry": g,
                        **compare_partitions(primary_labels[f"leiden_{g}"], controlled[g]),
                        **graph_statistics(graph),
                    }
                )
            robustness.append(
                {
                    "month": month,
                    "family": "equal_edge_budget_cross_geometry",
                    "geometry": "cosine",
                    "edge_budget": budget,
                    "feature_mode": "five",
                    "reference_geometry": "hellinger",
                    **compare_partitions(controlled["hellinger"], controlled["cosine"]),
                }
            )
            generator = np.random.default_rng(main["seed"] + int(month.replace("-", "")))
            for repeat in range(config["permutation_repeats"]):
                labels = generator.permutation(reference)
                null_rows.append(
                    _metric_record(
                        x,
                        labels,
                        graphs[main["geometry"], main["k"]],
                        distances,
                        month=month,
                        repeat=repeat,
                        null_kind="fixed_size_random_labels",
                    )
                )
        # Checkpoints preserve completed work if a later month fails.
        month_metrics = [row for row in metric_rows if row["month"] == month]
        pd.DataFrame(month_metrics).to_csv(output / f"checkpoint_{month}.csv", index=False)
        print(
            json.dumps(
                {
                    "month": month,
                    "seconds": round(time.monotonic() - month_start, 1),
                    "partitions": len(partitions),
                    "primary_clusters": int(len(np.unique(reference))),
                    "median_agreement": float(np.median(scores)),
                }
            ),
            flush=True,
        )
    assignment_table = pd.concat(assignments, ignore_index=True)
    membership_table = pd.concat(membership, ignore_index=True)
    transitions, flows, temporal_summary = temporal_tables(
        assignment_table,
        membership_table,
        threshold=config["agreement_threshold"],
        variant_evidence=variant_evidence,
    )
    evidence_arrays = {
        f"{month}_{field}": values
        for month, evidence in variant_evidence.items()
        for field, values in evidence.items()
    }
    np.savez_compressed(output / "variant_evidence.npz", **evidence_arrays)
    (output / "variant_keys.json").write_text(json.dumps(core_variant_keys), encoding="utf-8")
    return {
        "assignments": assignment_table,
        "cluster_profiles": pd.DataFrame(category_rows),
        "metrics": pd.DataFrame(metric_rows),
        "robustness": pd.DataFrame(robustness),
        "membership_agreement": membership_table,
        "null_metrics": pd.DataFrame(null_rows),
        "neighbor_overlap": pd.DataFrame(edge_overlap),
        "transitions": transitions,
        "flows": flows,
        "temporal_summary": temporal_summary,
    }
