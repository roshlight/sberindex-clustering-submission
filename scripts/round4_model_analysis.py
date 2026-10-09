"""Fixed comparisons of restored activity values on unchanged municipal cohorts."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from round2_model_analysis import KEY, TARGETS, P, design, sha
from round3_model_analysis import NEW, NEW_FLAGS, PREFIXES, RICH
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from municipal_profiles.enriched_analysis import clr, paired_region_loss_ci, r2_matrix

OLD = ["old_" + c for c in NEW]
OLD_FLAGS = ["old_" + c for c in NEW_FLAGS]
MODELS = {
    "rich_baseline": RICH,
    "new_flags": [*RICH, *NEW_FLAGS],
    "recovered_full": [*RICH, *NEW_FLAGS, *NEW],
    "legacy_flags": [*RICH, *OLD_FLAGS],
    "legacy_full": [*RICH, *OLD_FLAGS, *OLD],
    "dual_flags_old_values": [*RICH, *OLD_FLAGS, *NEW_FLAGS, *OLD],
    "dual_flags_new_values": [*RICH, *OLD_FLAGS, *NEW_FLAGS, *NEW],
}
PAIR_TYPES = {
    "primary": ("new_flags", "recovered_full"),
    "state_update": ("legacy_full", "recovered_full"),
    "shared_flags_recovery": ("dual_flags_old_values", "dual_flags_new_values"),
}


def build_panel(repo):
    old = pd.read_csv(repo / "reports/round3_analysis/models/model_panel.csv")
    new = pd.read_csv(repo / "reports/round4_analysis/audit/annual_features.csv")
    assert len(old) == len(new) == 4032
    assert not old.duplicated(KEY).any() and not new.duplicated(KEY).any()
    panel = old.rename(columns={c: "old_" + c for c in [*NEW, *NEW_FLAGS]}).merge(
        new[[*KEY, *NEW, *NEW_FLAGS]], on=KEY, validate="one_to_one", indicator=True
    )
    assert panel._merge.eq("both").all()
    panel = (
        panel.drop(columns="_merge").sort_values(["year", "territory_id"]).reset_index(drop=True)
    )
    for fresh, previous, flag in zip(NEW, OLD, NEW_FLAGS, strict=True):
        assert panel.loc[panel[previous].notna(), fresh].eq(panel[previous].dropna()).all()
        assert (
            panel.loc[panel.core_eligible, fresh]
            .isna()
            .equals(panel.loc[panel.core_eligible, flag].eq(1))
        )
    panel["old_observed_count"] = panel[OLD].notna().sum(axis=1)
    panel["new_observed_count"] = panel[NEW].notna().sum(axis=1)
    panel["recovered_feature_count"] = panel.new_observed_count - panel.old_observed_count
    assert panel.recovered_feature_count.ge(0).all()
    panel["recovered_any"] = panel.recovered_feature_count.gt(0)
    panel["common_complete_new3"] = panel.old_observed_count.eq(3) & panel.new_observed_count.eq(3)
    panel["expanded_complete_new3"] = panel.new_observed_count.eq(3)
    return panel


def frozen_folds(data, prior, year, sample, regime, seed):
    old_sample = "stable_code" if sample == "stable_code" else "core"
    old = prior.loc[
        prior.year.eq(year)
        & prior["sample"].eq(old_sample)
        & prior.regime.eq(regime)
        & prior.seed.eq(seed)
        & prior.model.eq("economic_sectors_newvalues"),
        [*KEY, "fold"],
    ]
    result = data[KEY].merge(old, on=KEY, how="left", validate="one_to_one")
    assert len(result) == len(data) and result.fold.notna().all()
    folds = result.fold.to_numpy(dtype=int)
    assert set(folds) == set(range(5))
    return folds


def shuffled(frame, seed):
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for feature in NEW:
        for group in sorted(frame.region_code.unique()):
            mask = frame.region_code.eq(group) & frame[feature].notna()
            result.loc[mask, feature] = rng.permutation(frame.loc[mask, feature].to_numpy())
        assert result[feature].isna().equals(frame[feature].isna())
    return result


def fit_oof(data, y, numeric, folds, regime, label, audit, permute_seed=None):
    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        a, b = data.loc[train], data.loc[test]
        if regime == "heldout_regions":
            assert not set(a.region_code) & set(b.region_code)
        if permute_seed is not None:
            a, b = shuffled(a, permute_seed), shuffled(b, permute_seed)
        x, z, log = design(a, b, numeric, regime == "within_region")
        prediction[test] = Ridge(alpha=1.0).fit(x, y[train]).predict(z)
        audit.append(
            {
                **label,
                "fold": fold,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "numeric": numeric,
                "permutation_seed": permute_seed,
                **log,
            }
        )
    assert np.isfinite(prediction).all()
    return prediction


def check_legacy(data, prediction, prior, label):
    old = prior
    for key, value in label.items():
        old = old.loc[old[key].eq(value)]
    old = old.loc[old.model.eq("economic_sectors_newvalues")]
    matched = data[KEY].merge(old, on=KEY, how="left", validate="one_to_one")
    expected = matched[["predicted_" + t for t in TARGETS]].to_numpy()
    np.testing.assert_allclose(prediction, expected, atol=1e-9, rtol=1e-9)
    return float(np.abs(prediction - expected).max())


def run(repo, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = repo / "configs/round4_analysis_v1.json"
    config = json.loads(protocol.read_text("utf-8"))
    started = datetime.now(UTC).isoformat()
    (out / "protocol_lock.json").write_text(
        json.dumps(
            {
                "started_at": started,
                "protocol_sha256": sha(protocol),
                "model_code_sha256": sha(__file__),
                "source_base_commit": config["base_commit"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    panel = build_panel(repo)
    panel.to_csv(out / "model_panel.csv", index=False)
    prior = duckdb.read_parquet(
        str(repo / "reports/round3_analysis/models/oof_predictions.parquet")
    ).df()
    saved, metrics, comparisons, slices, checks, preprocessing = [], [], [], [], [], []
    for year in config["years"]:
        core = panel.loc[panel.year.eq(year) & panel.core_eligible]
        assert len(core) == config["core_n"][str(year)]
        samples = {
            "core": core,
            "stable_code": core.loc[core.conservative_eligible],
            "common_complete_new3": core.loc[core.common_complete_new3],
            "expanded_complete_new3": core.loc[core.expanded_complete_new3],
        }
        assert len(samples["stable_code"]) == config["stable_n"][str(year)]
        for sample, subset in samples.items():
            data = subset.reset_index(drop=True)
            assert len(data) >= 100 and data.region_code.nunique() >= 10
            y = np.column_stack([data.log_total, clr(data[P])])
            for regime in config["regimes"]:
                seeds = (
                    config["within_region_core_seeds"]
                    if (sample == "core" and regime == "within_region")
                    else [13]
                )
                for seed in seeds:
                    label = {"year": year, "sample": sample, "regime": regime, "seed": seed}
                    folds = frozen_folds(data, prior, **label)
                    selected = (
                        MODELS
                        if sample in {"core", "stable_code"}
                        else {
                            name: MODELS[name]
                            for name in (
                                "rich_baseline",
                                "new_flags",
                                "recovered_full",
                                "legacy_full",
                            )
                        }
                    )
                    results = {
                        name: fit_oof(
                            data, y, columns, folds, regime, {**label, "model": name}, preprocessing
                        )
                        for name, columns in selected.items()
                    }
                    if sample in {"core", "stable_code"}:
                        maximum = check_legacy(data, results["legacy_full"], prior, label)
                        checks.append({**label, "check": "legacy_predictions", "max_abs": maximum})
                    if sample == "common_complete_new3":
                        np.testing.assert_allclose(
                            results["legacy_full"], results["recovered_full"], rtol=1e-9, atol=1e-9
                        )
                        checks.append(
                            {
                                **label,
                                "check": "common_complete_invariance",
                                "max_abs": float(
                                    np.abs(results["legacy_full"] - results["recovered_full"]).max()
                                ),
                            }
                        )
                    if sample == "core" and seed == 13:
                        for permutation in [91013, 91029, 91047]:
                            name = f"permuted_{permutation}"
                            results[name] = fit_oof(
                                data,
                                y,
                                MODELS["recovered_full"],
                                folds,
                                regime,
                                {**label, "model": name},
                                preprocessing,
                                permutation,
                            )
                        for prefix, feature in zip(PREFIXES, NEW, strict=True):
                            name = f"without_{prefix}"
                            columns = [c for c in MODELS["recovered_full"] if c != feature]
                            results[name] = fit_oof(
                                data,
                                y,
                                columns,
                                folds,
                                regime,
                                {**label, "model": name},
                                preprocessing,
                            )
                    for name, prediction in results.items():
                        rows = data[
                            [
                                *KEY,
                                "region_code",
                                "recovered_any",
                                "recovered_feature_count",
                                "old_observed_count",
                                "new_observed_count",
                            ]
                        ].copy()
                        for key, value in {**label, "model": name}.items():
                            rows[key] = value
                        rows["fold"] = folds
                        for i, target in enumerate(TARGETS):
                            rows["actual_" + target] = y[:, i]
                            rows["predicted_" + target] = prediction[:, i]
                        saved.append(rows)
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            metrics.append(
                                {
                                    **label,
                                    "model": name,
                                    "target": target,
                                    "n": len(data),
                                    "regions": data.region_code.nunique(),
                                    "mse_oof": float(
                                        np.square(y[:, idx] - prediction[:, idx]).mean()
                                    ),
                                    "r2_oof": r2_matrix(y[:, idx], prediction[:, idx]),
                                }
                            )
                    pairs = {
                        **PAIR_TYPES,
                        "missingness": ("rich_baseline", "new_flags"),
                        "whole_block": ("rich_baseline", "recovered_full"),
                        "legacy_increment": ("legacy_flags", "legacy_full"),
                    }
                    for name in results:
                        if name.startswith("permuted_"):
                            pairs[name] = ("new_flags", name)
                        elif name.startswith("without_"):
                            pairs[name] = (name, "recovered_full")
                    for comparison, (baseline, alternative) in pairs.items():
                        if baseline not in results or alternative not in results:
                            continue
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            effect, low, high = paired_region_loss_ci(
                                y[:, idx],
                                results[baseline][:, idx],
                                results[alternative][:, idx],
                                data.region_code.to_numpy(),
                                repeats=2000,
                                seed=13,
                            )
                            comparisons.append(
                                {
                                    **label,
                                    "comparison": comparison,
                                    "baseline": baseline,
                                    "alternative": alternative,
                                    "target": target,
                                    "n": len(data),
                                    "regions": data.region_code.nunique(),
                                    "relative_mse_reduction": effect,
                                    "ci_low": low,
                                    "ci_high": high,
                                }
                            )
                            if sample != "core" or seed != 13 or comparison not in PAIR_TYPES:
                                continue
                            for scope in config["slices"]:
                                for group in sorted(data[scope].unique()):
                                    mask = data[scope].eq(group).to_numpy()
                                    a = float(
                                        np.square(
                                            y[mask][:, idx] - results[baseline][mask][:, idx]
                                        ).sum()
                                    )
                                    b = float(
                                        np.square(
                                            y[mask][:, idx] - results[alternative][mask][:, idx]
                                        ).sum()
                                    )
                                    slices.append(
                                        {
                                            **label,
                                            "comparison": comparison,
                                            "target": target,
                                            "slice": scope,
                                            "group": str(group),
                                            "n": int(mask.sum()),
                                            "regions": data.loc[mask, "region_code"].nunique(),
                                            "baseline_sse": a,
                                            "alternative_sse": b,
                                            "relative_mse_reduction": 1 - b / a if a else np.nan,
                                        }
                                    )
                    print(json.dumps({**label, "n": len(data), "models": len(results)}), flush=True)
    for name, records in [
        ("cv_metrics", metrics),
        ("cv_comparisons", comparisons),
        ("slice_losses", slices),
        ("prior_invariance_checks", checks),
    ]:
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False)
    duckdb.from_df(pd.concat(saved, ignore_index=True)).write_parquet(
        str(out / "oof_predictions.parquet")
    )
    (out / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    inputs = [
        "configs/round4_analysis_v1.json",
        "scripts/round2_model_analysis.py",
        "scripts/round3_model_analysis.py",
        "src/municipal_profiles/enriched_analysis.py",
        "uv.lock",
        "reports/round3_analysis/models/model_panel.csv",
        "reports/round3_analysis/models/oof_predictions.parquet",
        "reports/round4_analysis/audit/annual_features.csv",
    ]
    manifest = {
        "status": "complete",
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "inputs": {p: sha(repo / p) for p in inputs},
        "code_sha256": sha(__file__),
        "models": MODELS,
        "fits": len(preprocessing),
        "tuning_performed": False,
        "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "fits": len(preprocessing),
                "metrics": len(metrics),
                "comparisons": len(comparisons),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo, args.out)
