"""Spatial brief H1/H2: fixed 2024 partitions, road pairs and market access.

No cluster fitting, geocoding, imputation or causal identification is performed.
Road permutations operate on municipality labels, never independently on pairs.
"""

import argparse
import json
import platform
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from review_spatial_transitions import holm, sha, unique

ROOT = Path(__file__).resolve().parents[1]


def canonical_roads(frame):
    """Keep highway only; reject duplicate directions/conflicting reverse rows."""
    required = ["territory_id_x", "territory_id_y", "type", "distance"]
    if frame[required].isna().any().any():
        raise ValueError("Missing connection fields")
    road = frame.loc[frame.type.eq("highway")].copy()
    unique(road, ["territory_id_x", "territory_id_y", "type"])
    for field in ["territory_id_x", "territory_id_y"]:
        values = road[field].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values > 0) & (values == np.floor(values))).all():
            raise ValueError("Invalid territory ID")
    if road.territory_id_x.eq(road.territory_id_y).any():
        raise ValueError("Unexpected self-pair")
    distance = road.distance.to_numpy(dtype=float)
    if not (np.isfinite(distance) & (distance >= 0)).all():
        raise ValueError("Invalid road distance")
    road["a"] = np.minimum(road.territory_id_x, road.territory_id_y)
    road["b"] = np.maximum(road.territory_id_x, road.territory_id_y)
    grouped = road.groupby(["a", "b"], sort=True).distance.agg(["min", "max", "size"])
    if not np.allclose(grouped["min"], grouped["max"], atol=1e-9, rtol=0):
        raise ValueError("Conflicting reverse road distances")
    result = grouped.reset_index().rename(columns={"min": "distance"})[["a", "b", "distance"]]
    return result, {
        "highway_source_rows": len(road),
        "unordered_pairs": len(result),
        "consistent_reverse_rows_collapsed": len(road) - len(result),
    }


def between_share(values, labels):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite complete market observations required")
    _, codes = np.unique(labels, return_inverse=True)
    counts = np.bincount(codes)
    centered = values - values.mean()
    total = float(centered @ centered)
    if total == 0:
        raise ValueError("Constant market index: no dispersion to compare")
    sums = np.bincount(codes, weights=centered)
    return float(np.sum(sums * sums / counts) / total)


def shuffle_labels(labels, blocks, rng):
    shuffled = labels.copy()
    for indices in blocks:
        shuffled[indices] = rng.permutation(labels[indices])
    return shuffled


def road_statistic(labels, a, b, distance, bins, n_bins):
    same = labels[a] == labels[b]
    count = int(same.sum())
    if count == 0:
        raise ValueError("No observed within-group pairs")
    mean = float(distance[same].mean())
    cdf = np.cumsum(np.bincount(bins[same], minlength=n_bins + 1))[:n_bins] / count
    return mean, cdf, count


def null_summary(observed, draws, lower_tail):
    if not np.isfinite(draws).all():
        raise ValueError("Nonfinite permutation result")
    extreme = draws <= observed + 1e-12 if lower_tail else draws >= observed - 1e-12
    return {
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_q025": float(np.quantile(draws, 0.025)),
        "null_q975": float(np.quantile(draws, 0.975)),
        "difference_from_null": float(observed - draws.mean()),
        "p_raw": float((1 + extreme.sum()) / (len(draws) + 1)),
        "null_status": "tested" if np.ptp(draws) > 1e-12 else "degenerate",
    }


def check_coverage(n_observed, n_total, threshold, kind):
    if n_total == 0 or n_observed / n_total < threshold:
        raise ValueError(f"Insufficient {kind} coverage: {n_observed}/{n_total}")


def run(repo, config_path, output):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["road_type"] != "highway" or config["permutations"] < 1:
        raise ValueError("Expected highway and positive permutation count")
    for relative, expected in config["inputs"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()) or sha(path) != expected:
            raise ValueError(f"Input checksum mismatch: {relative}")
    assignments = duckdb.read_parquet(str(repo / "reports/initial_study/assignments.parquet")).df()
    assignments = assignments.loc[assignments.method.eq(config["method"])]
    unique(assignments, ["territory_id", "month"])
    ids = sorted(assignments.territory_id.unique())
    if len(ids) != config["expected_cohort_size"]:
        raise ValueError("Unexpected cohort")
    crosswalk = pd.read_csv(
        repo / "data/derived/reviews/spatial_transitions_v1/territory_crosswalk.csv",
        dtype={"region_code_2024": str, "oktmo_2024": str},
    )
    unique(crosswalk, ["territory_id"])
    if set(crosswalk.territory_id) != set(ids) or crosswalk.region_code_2024.isna().any():
        raise ValueError("Crosswalk cohort/region mismatch")
    if not pd.api.types.is_bool_dtype(crosswalk.attribute_changed):
        raise ValueError("Attribute flag must be Boolean")
    context = duckdb.read_parquet(
        str(repo / "reports/enrichment_v1/core_context/municipal_context_2024.parquet")
    ).df()
    unique(context, ["territory_id"])
    if set(context.territory_id) != set(ids) or not context.measurement_year.eq(2024).all():
        raise ValueError("Market context must describe the exact cohort and year 2024")
    observed = context.market_access.dropna().to_numpy(float)
    if not (np.isfinite(observed) & (observed >= 0) & (observed <= 1000)).all():
        raise ValueError("Market index outside documented 0..1000 scale")
    with duckdb.connect() as db:
        source_counts = db.execute(
            "SELECT type, count(*) AS n FROM read_parquet(?) GROUP BY type",
            [str(repo / "data/raw/connection.parquet")],
        ).df()
        connections = db.execute(
            "SELECT * FROM read_parquet(?) WHERE type='highway'",
            [str(repo / "data/raw/connection.parquet")],
        ).df()
    roads, road_audit = canonical_roads(connections)
    del connections
    base = crosswalk.merge(
        context[["territory_id", "market_access"]],
        on="territory_id",
        how="left",
        validate="one_to_one",
    )
    thresholds = np.asarray(config["distance_thresholds_km"], dtype=float)
    if not (np.isfinite(thresholds).all() and (np.diff(thresholds) > 0).all()):
        raise ValueError("Distance thresholds must be finite and increasing")
    checks, coverage, missing, draws, curves, summaries, memberships = [], [], [], [], [], [], []
    seed_index = 0
    for sample in config["samples"]:
        if sample not in {"all", "no_attribute_change"}:
            raise ValueError("Unknown sample")
        selected = base if sample == "all" else base.loc[~base.attribute_changed]
        sample_ids = set(selected.territory_id)
        pairs = roads.loc[roads.a.isin(sample_ids) & roads.b.isin(sample_ids)].reset_index(
            drop=True
        )
        endpoints = set(pairs.a) | set(pairs.b)
        n_possible = len(selected) * (len(selected) - 1) // 2
        check_coverage(len(endpoints), len(selected), config["minimum_id_coverage"], "road ID")
        check_coverage(len(pairs), n_possible, config["minimum_pair_coverage"], "road pair")
        check_coverage(
            selected.market_access.notna().sum(),
            len(selected),
            config["minimum_id_coverage"],
            "market ID",
        )
        for month in config["months"]:
            labels = assignments.loc[assignments.month.eq(month), ["territory_id", "label"]]
            if set(labels.territory_id) != set(ids) or labels.label.isna().any():
                raise ValueError(f"Incomplete partition: {month}")
            panel = selected.merge(labels, on="territory_id", how="left", validate="one_to_one")
            for hypothesis in ("H1", "H2"):
                keep = (
                    panel.territory_id.isin(endpoints)
                    if hypothesis == "H1"
                    else panel.market_access.notna()
                )
                part = panel.loc[keep].sort_values("territory_id").reset_index(drop=True)
                key = {"sample": sample, "month": month, "hypothesis": hypothesis}
                for row in panel.loc[~keep].itertuples():
                    missing.append(
                        {
                            **key,
                            "territory_id": row.territory_id,
                            "reason": "no_observed_road_pair"
                            if hypothesis == "H1"
                            else "missing_market_access",
                        }
                    )
                coverage.append(
                    {
                        **key,
                        "cohort_ids": len(selected),
                        "observed_ids": len(part),
                        "missing_ids": int((~keep).sum()),
                        "ambiguous_ids": 0,
                        "id_coverage": len(part) / len(selected),
                        "available_pairs": len(pairs) if hypothesis == "H1" else None,
                        "possible_pairs": n_possible if hypothesis == "H1" else None,
                        "pair_coverage": len(pairs) / n_possible if hypothesis == "H1" else None,
                    }
                )
                memberships.append(
                    part[
                        [
                            "territory_id",
                            "region_code_2024",
                            "label",
                            "market_access",
                            "attribute_changed",
                        ]
                    ].assign(**key)
                )
                labels_array = part.label.to_numpy()
                regions = part.region_code_2024.to_numpy()
                blocks = [np.flatnonzero(regions == r) for r in np.unique(regions)]
                blocks = [p for p in blocks if len(np.unique(labels_array[p])) > 1]
                seed = config["seed"] + seed_index
                seed_index += 1
                rng = np.random.default_rng(seed)
                null = np.empty(config["permutations"])
                if hypothesis == "H1":
                    index = pd.Index(part.territory_id)
                    a, b = index.get_indexer(pairs.a), index.get_indexer(pairs.b)
                    if (a < 0).any() or (b < 0).any():
                        raise ValueError("Pair endpoint outside eligible panel")
                    distance = pairs.distance.to_numpy(float)
                    bins = np.searchsorted(thresholds, distance, side="left")
                    value, cdf, within_count = road_statistic(
                        labels_array, a, b, distance, bins, len(thresholds)
                    )
                    null_cdf = np.empty((len(null), len(thresholds)))
                    for i in range(len(null)):
                        shuffled = shuffle_labels(labels_array, blocks, rng)
                        null[i], null_cdf[i], _ = road_statistic(
                            shuffled, a, b, distance, bins, len(thresholds)
                        )
                    for k, threshold in enumerate(thresholds):
                        curves.append(
                            {
                                **key,
                                "threshold_km": threshold,
                                "observed_cdf": cdf[k],
                                "null_mean_cdf": float(null_cdf[:, k].mean()),
                                "null_q025_cdf": float(np.quantile(null_cdf[:, k], 0.025)),
                                "null_q975_cdf": float(np.quantile(null_cdf[:, k], 0.975)),
                            }
                        )
                    same = labels_array[a] == labels_array[b]
                    for label in sorted(part.label.unique()):
                        group = panel.loc[panel.label.eq(label)]
                        group_distances = distance[same & (labels_array[a] == label)]
                        quantiles = (
                            np.quantile(group_distances, [0.25, 0.5, 0.75])
                            if len(group_distances)
                            else [None] * 3
                        )
                        summaries.append(
                            {
                                **key,
                                "label": label,
                                "n_total": len(group),
                                "n_observed": int(group.territory_id.isin(endpoints).sum()),
                                "n_pairs": len(group_distances),
                                "q25": quantiles[0],
                                "median": quantiles[1],
                                "q75": quantiles[2],
                            }
                        )
                else:
                    values = part.market_access.to_numpy(float)
                    value = between_share(values, labels_array)
                    within_count = None
                    for i in range(len(null)):
                        null[i] = between_share(values, shuffle_labels(labels_array, blocks, rng))
                    for label, group in panel.groupby("label", sort=True):
                        available = group.market_access.dropna()
                        q = (
                            available.quantile([0.25, 0.5, 0.75]).to_numpy()
                            if len(available)
                            else [None] * 3
                        )
                        summaries.append(
                            {
                                **key,
                                "label": label,
                                "n_total": len(group),
                                "n_observed": len(available),
                                "n_pairs": None,
                                "q25": q[0],
                                "median": q[1],
                                "q75": q[2],
                            }
                        )
                checks.append(
                    {
                        **key,
                        "method": config["method"],
                        "n_ids": len(part),
                        "n_within_pairs": within_count,
                        "seed": seed,
                        "permutations": len(null),
                        "permutable_regions": len(blocks),
                        "permutable_ids": sum(map(len, blocks)),
                        "unit": "km" if hypothesis == "H1" else "fraction_SS",
                        **null_summary(value, null, hypothesis == "H1"),
                    }
                )
                draws.append(
                    pd.DataFrame({**key, "draw": np.arange(1, len(null) + 1), "statistic": null})
                )
                print(f"Completed {sample} {month} {hypothesis}", flush=True)
    checks = pd.DataFrame(checks)
    for _, indices in checks.groupby("sample").groups.items():
        checks.loc[indices, "p_holm"] = holm(checks.loc[indices, "p_raw"])
    direction = np.where(
        checks.hypothesis.eq("H1"), checks.difference_from_null < 0, checks.difference_from_null > 0
    )
    checks["status"] = np.where(
        (checks.p_holm <= config["alpha"]) & direction & checks.null_status.eq("tested"),
        "supported_association",
        "not_supported",
    )
    checks.loc[checks.null_status.ne("tested"), "status"] = "not_testable"
    checks["run_id"] = "evaluation-20260914-v1"
    tables = {
        "spatial_checks.csv": checks,
        "spatial_coverage.csv": pd.DataFrame(coverage),
        "missing_ids.csv": pd.DataFrame(missing),
        "group_summary.csv": pd.DataFrame(summaries),
        "road_cdf.csv": pd.DataFrame(curves),
        "permutation_draws.csv": pd.concat(draws),
        "eligible_memberships.csv": pd.concat(memberships),
    }
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, lineterminator="\n")
    manifest = {
        "status": "complete_H1_H2",
        "config": config,
        "config_sha256": sha(config_path),
        "processor_sha256": sha(__file__),
        "shared_helper_sha256": sha(Path(__file__).with_name("review_spatial_transitions.py")),
        "road_audit": road_audit,
        "source_type_counts": source_counts.to_dict("records"),
        "market_input_scope": (
            "Previously audited exact-cohort subset, not a fresh full source download"
        ),
        "network_requests": 0,
        "run_id": "evaluation-20260914-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
        },
        "output_hashes": {name: sha(output / name) for name in tables},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/spatial_external_review.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.config, args.output)
