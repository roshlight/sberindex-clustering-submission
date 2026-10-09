"""Combine compatible completed runs into audited tables; perform no model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from municipal_profiles import dynamics, networks
from municipal_profiles.dynamics import temporal_tables

METRICS = ["silhouette", "calinski_harabasz", "s_dbw", "avi", "avu", "modularity"]
INPUT_TABLES = [
    "assignments",
    "membership_agreement",
    "cluster_profiles",
    "metrics",
    "robustness",
    "null_metrics",
]
HASH_FIELDS = ["config_sha256", "data_sha256", "code_sha256"]
PRIMARY = "leiden_hellinger"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _unique(frame: pd.DataFrame, key: list[str], name: str) -> None:
    if frame.duplicated(key).any():
        raise ValueError(f"Duplicate {name} keys: {key}")


def _load_runs(run_dirs: list[Path]):
    tables = {name: [] for name in INPUT_TABLES}
    evidence, sources, months_seen = {}, [], set()
    reference_manifest = reference_keys = None
    if not run_dirs or len(set(run_dirs)) != len(run_dirs):
        raise ValueError("Supply one or more distinct run directories")
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Run is not complete: {run_dir}")
        if not all(isinstance(manifest.get(key), str) and manifest[key] for key in HASH_FIELDS):
            raise ValueError(f"Missing source hashes: {run_dir}")
        if reference_manifest is not None:
            for key in [*HASH_FIELDS, "config"]:
                if manifest.get(key) != reference_manifest.get(key):
                    raise ValueError(f"Incompatible runs: {key}")
        else:
            reference_manifest = manifest
        consumed = [f"{name}.csv" for name in INPUT_TABLES]
        consumed += ["variant_evidence.npz", "variant_keys.json"]
        hashes = manifest.get("output_hashes", {})
        for filename in consumed:
            path = run_dir / filename
            if filename not in hashes or _hash(path) != hashes[filename]:
                raise ValueError(f"Source artifact hash mismatch: {path}")
        keys = json.loads((run_dir / "variant_keys.json").read_text(encoding="utf-8"))
        if len(keys) != 21 or len({tuple(key) for key in keys}) != 21:
            raise ValueError("Expected 21 distinct core variant keys")
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Variant keys or their order differ between runs")
        reference_keys = keys
        loaded = {name: _read_table(run_dir / f"{name}.csv") for name in INPUT_TABLES}
        for name, frame in loaded.items():
            if len(frame) != manifest.get("table_rows", {}).get(name):
                raise ValueError(f"Source row count mismatch: {name}")
            if name != "null_metrics" and frame.empty:
                raise ValueError(f"Required table is empty: {name}")
        run_months = set(loaded["assignments"]["month"])
        if months_seen & run_months:
            raise ValueError("Runs contain overlapping months")
        if any(not month.startswith(str(manifest.get("year")) + "-") for month in run_months):
            raise ValueError("Assignment months do not match the declared run year")
        months_seen.update(run_months)
        with np.load(run_dir / "variant_evidence.npz", allow_pickle=False) as arrays:
            expected = {
                f"{month}_{field}" for month in run_months for field in ("ids", "labels", "support")
            }
            if set(arrays.files) != expected:
                raise ValueError("Variant evidence months or fields differ from assignments")
            for month in run_months:
                evidence[month] = {
                    field: arrays[f"{month}_{field}"].copy()
                    for field in ("ids", "labels", "support")
                }
        for name, frame in loaded.items():
            tables[name].append(frame)
        sources.append(
            {
                "run_id": manifest.get("run_id"),
                "phase": manifest.get("phase"),
                "year": manifest.get("year"),
                "manifest_sha256": _hash(manifest_path),
                "consumed_hashes": {name: hashes[name] for name in consumed},
            }
        )
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    return combined, evidence, reference_keys, reference_manifest, sources


def _support_table(assignments, membership, evidence, threshold):
    _unique(membership, ["month", "territory_id"], "membership")
    primary = assignments.loc[assignments.method.eq(PRIMARY), ["month", "territory_id"]]
    _unique(primary, ["month", "territory_id"], "primary assignment")
    checked = primary.merge(membership, how="outer", validate="one_to_one", indicator=True)
    if not checked["_merge"].eq("both").all():
        raise ValueError("Membership must cover exactly the primary assignment keys")
    if not checked["n_comparisons"].eq(21).all():
        raise ValueError("Membership requires exactly 21 core comparisons")
    for column in ("agreement", "mean_co_membership_jaccard"):
        if not np.isfinite(checked[column]).all() or not checked[column].between(0, 1).all():
            raise ValueError(f"Invalid membership values: {column}")
    records = []
    for month, frame in checked.groupby("month", sort=True):
        frame = frame.sort_values("territory_id")
        values = evidence[month]
        if not np.array_equal(values["ids"], frame["territory_id"].to_numpy()):
            raise ValueError("Variant IDs differ from primary assignment IDs")
        if values["support"].shape != (21, len(frame)) or values["support"].dtype != bool:
            raise ValueError("Variant support must be a Boolean (21, n_entities) array")
        if not np.allclose(
            values["support"].mean(axis=0), frame["agreement"], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Membership agreement differs from saved variant support")
        strong = int(frame["agreement"].ge(threshold).sum())
        records.append(
            {
                "month": month,
                "year": month[:4],
                "method": PRIMARY,
                "n_entities": len(frame),
                "n_assessed": len(frame),
                "n_strong": strong,
                "strong_share_of_all": strong / len(frame),
                "agreement_threshold": threshold,
                "median_agreement": float(frame["agreement"].median()),
                "median_co_membership_jaccard": float(frame["mean_co_membership_jaccard"].median()),
                "n_comparisons": 21,
            }
        )
    return pd.DataFrame(records)


def _aggregate(frame: pd.DataFrame, groups: list[str], columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = frame["month"].str[:4]
    grouped = frame.groupby(["year", *groups], sort=True, dropna=False)
    # Per-metric counts retain the denominator when undefined values are excluded.
    result = grouped[[name for name in columns if name in frame]].agg(
        ["median", "min", "max", "count"]
    )
    result.columns = [f"{name}_{statistic}" for name, statistic in result.columns]
    result["n_rows"] = grouped.size()
    result["n_months"] = grouped["month"].nunique()
    return result.reset_index()


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    # DuckDB COPY supports bound output paths; no SQL interpolation is needed.
    with duckdb.connect() as connection:
        connection.register("report_frame", frame)
        connection.execute("COPY report_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])


def build_report(run_dirs: list[Path], output: Path) -> dict:
    run_dirs = [Path(path).resolve() for path in run_dirs]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if any(output.is_relative_to(path) for path in run_dirs):
        raise ValueError("Output must be outside the source run directories")
    tables, evidence, variant_keys, reference, sources = _load_runs(run_dirs)
    config = reference["config"]
    if config.get("primary", {}).get("geometry") != "hellinger":
        raise ValueError("Temporal robustness currently supports primary Hellinger only")
    threshold = float(config["agreement_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid agreement threshold")
    assignments = tables["assignments"].sort_values(["month", "method", "territory_id"])
    membership = tables["membership_agreement"]
    support = _support_table(assignments, membership, evidence, threshold)
    transitions, flows, temporal = temporal_tables(
        assignments, membership, threshold=threshold, variant_evidence=evidence
    )
    _unique(tables["metrics"], ["month", "method"], "metric")
    _unique(tables["cluster_profiles"], ["month", "method", "label"], "cluster profile")
    metric_summary = _aggregate(
        tables["metrics"],
        ["method"],
        [*METRICS, "n_clusters", "n_entities", "smallest_cluster_size", "largest_cluster_share"],
    )
    robustness_summary = _aggregate(
        tables["robustness"],
        ["family"],
        [
            "ari_to_primary",
            "aligned_agreement",
            "median_co_membership_jaccard",
            "candidate_n_clusters",
        ],
    )
    csv_tables = {
        "temporal_summary": temporal,
        "flows": flows,
        "cluster_profiles": tables["cluster_profiles"],
        "monthly_support": support,
        "method_metrics": metric_summary,
        "robustness_summary": robustness_summary,
        "robustness": tables["robustness"],
        "null_metrics": tables["null_metrics"],
    }
    primary_transitions = transitions.loc[transitions.method.eq(PRIMARY)]
    lag_counts = []
    for lag, group in primary_transitions.groupby("lag", sort=True):
        assessed = group["robust_switch"].notna()
        lag_counts.append(
            {
                "lag": int(lag),
                "n_entity_pairs": len(group),
                "n_month_pairs": len(group[["from_month", "to_month"]].drop_duplicates()),
                "n_switched": int(group["switched"].sum()),
                "n_assessed": int(assessed.sum()),
                "n_robust_switches": int(group.loc[assessed, "robust_switch"].sum())
                if assessed.any()
                else None,
            }
        )
    summary = {
        "status": "complete",
        "no_model_fitting": True,
        "months": sorted(assignments["month"].unique()),
        "years": sorted(assignments["month"].str[:4].unique()),
        "n_territories": int(assignments["territory_id"].nunique()),
        "n_assignments": len(assignments),
        "n_primary_entity_months": int(support.n_entities.sum()),
        "n_strong_entity_months": int(support.n_strong.sum()),
        "strong_share_of_all": float(support.n_strong.sum() / support.n_entities.sum()),
        "agreement_threshold": threshold,
        "co_membership_threshold": config["co_membership_threshold"],
        "core_variants": len(variant_keys),
        "primary_transition_counts": lag_counts,
        "scope": "Retrospective procedural stability; no forecast, causal or economic-class claim",
        "aggregation": (
            "Month/variant medians and extrema; counts exclude undefined values. "
            "No composite ranking."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "operation": "combine_saved_partitions_and_recompute_temporal_tables_without_fitting",
        "sources": sources,
        **{key: reference[key] for key in HASH_FIELDS},
        "processor_hashes": {
            "build_research_report.py": _hash(Path(__file__)),
            "dynamics.py": _hash(Path(dynamics.__file__)),
            "networks.py": _hash(Path(networks.__file__)),
        },
    }
    manifest_path = output / "report_manifest.json"
    _json(manifest_path, manifest)
    try:
        for name, frame in csv_tables.items():
            frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8")
        _parquet(output / "assignments.parquet", assignments)
        _parquet(output / "membership_agreement.parquet", membership)
        _parquet(output / "transitions.parquet", transitions)
        np.savez_compressed(
            output / "variant_evidence.npz",
            **{
                f"{month}_{field}": values
                for month, part in sorted(evidence.items())
                for field, values in part.items()
            },
        )
        _json(output / "variant_keys.json", variant_keys)
        _json(output / "summary.json", summary)
        manifest.update(
            status="complete",
            table_rows={
                **{name: len(frame) for name, frame in csv_tables.items()},
                "assignments": len(assignments),
                "membership_agreement": len(membership),
                "transitions": len(transitions),
            },
            output_hashes={
                path.name: _hash(path) for path in sorted(output.iterdir()) if path != manifest_path
            },
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _json(manifest_path, manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_report(args.runs, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
