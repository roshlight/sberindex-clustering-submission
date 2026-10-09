"""Audit versioned joins and compare transition rates without refitting clusters.

An exploratory contribution to spatial brief H3. A change of OKTMO/type is a
source attribute flag, NOT a verified change of borders or a dated legal event.
The permutation unit is a municipality, not its repeated monthly observations.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.sberbank.com/common/files/t_dict_municipal.rar"
FIELDS = ["oktmo", "municipal_district_type"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Null or duplicate keys: {keys}")


def snapshot(versions, year):
    """Select the half-open source interval before any join."""
    if versions[["year_from", "year_to"]].isna().any().any():
        raise ValueError("Unknown interval endpoint; do not infer an open interval")
    if not (versions.year_from < versions.year_to).all():
        raise ValueError("Invalid version interval")
    result = versions.loc[(versions.year_from <= year) & (year < versions.year_to)].copy()
    unique(result, ["territory_id"])
    return result


def build_crosswalk(versions, ids):
    """One row per cohort ID, with both versions and a narrowly defined flag."""
    cohort = pd.DataFrame({"territory_id": sorted(ids)})
    unique(cohort, ["territory_id"])
    fields = [
        "territory_id",
        "region_code",
        "region_name",
        "oktmo",
        "oktmo_digits",
        "municipal_district_name",
        "municipal_district_type",
        "municipal_district_status",
        "municipal_district_center",
        "year_from",
        "year_to",
    ]
    coverage = []
    for year in (2023, 2024):
        selected = snapshot(versions, year)
        count = int(selected.territory_id.isin(ids).sum())
        coverage.append(
            {
                "source": "crosswalk",
                "snapshot": f"{year}-01-01",
                "source_rows": len(selected),
                "cohort_ids": len(ids),
                "matched_ids": count,
                "missing_ids": len(ids) - count,
                "ambiguous_ids": 0,
            }
        )
        selected = selected[fields].rename(
            columns={f: f"{f}_{year}" for f in fields if f != "territory_id"}
        )
        cohort = cohort.merge(selected, on="territory_id", how="left", validate="one_to_one")
    required = [f"{f}_{y}" for f in [*FIELDS, "region_code"] for y in (2023, 2024)]
    if cohort[required].isna().any().any():
        raise ValueError("Incomplete crosswalk; do not classify missing versions as unchanged")
    if not cohort.region_code_2023.eq(cohort.region_code_2024).all():
        raise ValueError("Region changes need a separate stratification policy")
    for field in FIELDS:
        cohort[f"{field}_changed"] = cohort[f"{field}_2023"].ne(cohort[f"{field}_2024"])
    cohort["attribute_changed"] = cohort[[f"{f}_changed" for f in FIELDS]].any(axis=1)
    cohort["join_status"] = "matched_unique_both_snapshots"
    cohort["source_url"] = SOURCE
    cohort["metadata_coverage_through"] = "2024-01-01"
    cohort["center_reference_year"] = 2018
    return cohort, pd.DataFrame(coverage)


def municipality_rates(transitions, ids, window, method):
    """Keep the saved pairwise alignment; never compare raw label numbers."""
    t = transitions.loc[
        transitions.method.eq(method)
        & transitions.from_month.str.startswith(window["from_year"] + "-")
        & transitions.to_month.str.startswith(window["to_year"] + "-")
        & transitions.lag.eq(window["lag"])
    ].copy()
    unique(t, ["territory_id", "from_month", "to_month", "lag", "method"])
    if set(t.territory_id) != set(ids):
        raise ValueError("Transition cohort differs from assignment cohort")
    pairs = t[["from_month", "to_month", "lag"]].drop_duplicates()
    if len(pairs) != window["pairs_per_territory"]:
        raise ValueError("Unexpected calendar pairs")
    if not t.groupby("territory_id").size().eq(window["pairs_per_territory"]).all():
        raise ValueError("Incomplete municipality history in comparison window")
    for field in ("switched", "robust_switch"):
        if not pd.api.types.is_bool_dtype(t[field]) or t[field].isna().any():
            raise ValueError(f"{field}: expected observed Booleans, not missing support")
    if (t.robust_switch & ~t.switched).any():
        raise ValueError("Robust switch without a switch")
    aligned = t.from_label.ne(t.aligned_to_label)
    if not t.switched.eq(aligned).all():
        raise ValueError("Saved switched disagrees with pairwise aligned labels")
    result = t.groupby("territory_id", as_index=False).agg(
        pairs=("switched", "size"),
        switches=("switched", "sum"),
        robust_switches=("robust_switch", "sum"),
        mean_profile_change=("hellinger_profile_change", "mean"),
    )
    result["switch_rate"] = result.switches / result.pairs
    result["robust_switch_rate"] = result.robust_switches / result.pairs
    return result


def permuted_flags(flags, strata, rng):
    result = flags.copy()
    for indices in strata:
        result[indices] = rng.permutation(flags[indices])
    return result


def regional_permutation(values, changed, regions, *, permutations, seed):
    """Upper-tail test of 'changed territories switch MORE', conditional on region.

    The regional null may have a nonzero mean: region composition is preserved.
    No individual municipality-month is treated as an independent observation.
    """
    values = np.asarray(values, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    regions = np.asarray(regions)
    if not (len(values) == len(changed) == len(regions)):
        raise ValueError("Input lengths differ")
    if not np.isfinite(values).all() or pd.isna(regions).any():
        raise ValueError("Nonfinite rates or missing regions")
    if not changed.any() or changed.all() or permutations < 1:
        raise ValueError("Need both groups and at least one permutation")
    strata = [np.flatnonzero(regions == r) for r in np.unique(regions)]
    mobile = [s for s in strata if changed[s].any() and not changed[s].all()]
    observed = float(values[changed].mean() - values[~changed].mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(permutations)
    for i in range(permutations):
        shuffled = permuted_flags(changed, mobile, rng)
        draws[i] = values[shuffled].mean() - values[~shuffled].mean()
    return {
        "difference": observed,
        "null_mean": float(draws.mean()),
        "excess_over_null": float(observed - draws.mean()),
        "null_q025": float(np.quantile(draws, 0.025)),
        "null_q975": float(np.quantile(draws, 0.975)),
        "p_upper": float((1 + np.sum(draws >= observed - 1e-12)) / (permutations + 1)),
        "permutable_regions": len(mobile),
        "permutable_ids": sum(map(len, mobile)),
        "changed_ids_in_permutable_regions": sum(int(changed[s].sum()) for s in mobile),
        "status": "tested" if mobile and np.ptp(draws) > 1e-12 else "degenerate_null",
    }, draws


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return result


def run(repo, config_path, output):
    if output.exists():
        raise FileExistsError("Use a new output directory; saved results are never overwritten")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["comparison_fields"] != FIELDS:
        raise ValueError("This analysis implements OKTMO/type flags only")
    for relative, expected in config["inputs"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()) or sha(path) != expected:
            raise ValueError(f"Input checksum mismatch: {relative}")
    report = json.loads((repo / "reports/initial_study/report_manifest.json").read_text())
    run_ids = ";".join(x["run_id"] for x in report["sources"])
    assignments = duckdb.read_parquet(str(repo / "reports/initial_study/assignments.parquet")).df()
    ids = sorted(assignments.territory_id.unique())
    if len(ids) != config["expected_cohort_size"]:
        raise ValueError("Unexpected cohort size")
    versions = pd.read_csv(
        repo / "reports/enrichment_v1/crosswalk/municipal_crosswalk_versions.csv",
        dtype={"oktmo": str, "oktmo_digits": str, "region_code": str},
    )
    crosswalk, coverage = build_crosswalk(versions, ids)
    prior = pd.read_csv(
        repo / "reports/enrichment_v2/market_access_followup/sber_attribute_changes_78.csv"
    )
    unique(prior, ["territory_id"])
    if set(prior.territory_id) != set(crosswalk.loc[crosswalk.attribute_changed, "territory_id"]):
        raise ValueError("Recomputed attribute changes disagree with the earlier source audit")
    transitions = duckdb.read_parquet(str(repo / "reports/initial_study/transitions.parquet")).df()
    summaries, panels, draws = [], [], []
    for index, (name, window) in enumerate(config["windows"].items()):
        panel = (
            municipality_rates(transitions, ids, window, config["method"])
            .merge(
                crosswalk[["territory_id", "region_code_2024", "attribute_changed"]],
                on="territory_id",
                validate="one_to_one",
            )
            .sort_values("territory_id")
            .reset_index(drop=True)
        )
        panel["window"] = name
        panel["run_id"] = run_ids
        panels.append(panel)
        selected = panel.attribute_changed
        comparison, null = regional_permutation(
            panel.switch_rate,
            selected,
            panel.region_code_2024,
            permutations=config["permutations"],
            seed=config["seed"] + index,
        )
        draws.append(
            pd.DataFrame({"window": name, "draw": np.arange(1, len(null) + 1), "difference": null})
        )
        row = {
            "hypothesis": "H3",
            "window": name,
            "run_id": run_ids,
            "method": config["method"],
            "analysis_status": config["analysis_status"],
            "seed": config["seed"] + index,
            "permutations": len(null),
            **comparison,
        }
        for group, mask in [
            ("all", np.ones(len(panel), dtype=bool)),
            ("changed", selected),
            ("no_flag", ~selected),
        ]:
            part = panel.loc[mask]
            row.update(
                {
                    f"n_{group}": len(part),
                    f"pairs_{group}": int(part.pairs.sum()),
                    f"switch_rate_{group}": float(part.switch_rate.mean()),
                    f"robust_rate_{group}": float(part.robust_switch_rate.mean()),
                    f"robust_count_{group}": int(part.robust_switches.sum()),
                }
            )
        summaries.append(row)
    checks = pd.DataFrame(summaries)
    checks["p_holm"] = holm(checks.p_upper)
    checks["hypothesis_status"] = np.where(
        checks.status.eq("tested")
        & checks.p_holm.le(config["alpha"])
        & checks.excess_over_null.gt(0),
        "supported_association",
        "not_supported",
    )
    checks.loc[checks.status.ne("tested"), "hypothesis_status"] = "not_testable"
    output.mkdir(parents=True)
    tables = {
        "territory_crosswalk.csv": crosswalk,
        "spatial_coverage.csv": coverage,
        "spatial_checks.csv": checks,
        "municipality_transition_rates.csv": pd.concat(panels),
        "permutation_draws.csv": pd.concat(draws),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding="utf-8", lineterminator="\n")
    manifest = {
        "status": "complete_H3_only",
        "network_requests": 0,
        "config": config,
        "config_sha256": sha(config_path),
        "script_sha256": sha(__file__),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
        },
        "run_ids": run_ids.split(";"),
        "prior_changed_ids_match": True,
        "output_hashes": {name: sha(output / name) for name in tables},
        "limits": [
            "Exploratory, not an untouched 2024 holdout",
            "No flag means no OKTMO/type difference between two source snapshots",
            "No legal event dates or verified border changes inferred",
            "H1 road-pair and H2 market-access comparisons are outside this contribution",
            "Permutation quantiles are null reference intervals, not confidence intervals",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        checks[
            [
                "window",
                "n_changed",
                "n_no_flag",
                "difference",
                "null_mean",
                "p_holm",
                "robust_count_all",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/spatial_transition_review.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.config, args.output)
