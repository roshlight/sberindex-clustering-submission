"""Frozen fifth-round comparison of 64 additions on unchanged municipal cohorts."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from round2_model_analysis import KEY, TARGETS, P, design, sha
from round3_model_analysis import NEW, NEW_FLAGS, RICH
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


def combine_panels(old, new):
    """Reset round4 annotations before constructing the 4-to-5 feature vintages."""
    assert not old.duplicated(KEY).any() and not new.duplicated(KEY).any()
    stale = [c for c in old if c.startswith("old_")]
    stale += [
        "new_observed_count",
        "recovered_feature_count",
        "recovered_any",
        "common_complete_new3",
        "expanded_complete_new3",
    ]
    old = old.drop(columns=stale, errors="ignore")
    panel = old.rename(columns={c: "old_" + c for c in [*NEW, *NEW_FLAGS]}).merge(
        new[[*KEY, *NEW, *NEW_FLAGS]],
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    assert panel._merge.eq("both").all()
    panel = (
        panel.drop(columns="_merge").sort_values(["year", "territory_id"]).reset_index(drop=True)
    )
    for fresh, previous, flag in zip(NEW, OLD, NEW_FLAGS, strict=True):
        observed = panel[previous].notna()
        np.testing.assert_allclose(
            panel.loc[observed, fresh], panel.loc[observed, previous], atol=1e-12, rtol=1e-12
        )
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
    panel["recovery_region"] = panel.region_code.isin([25, 59])
    panel["completeness_stratum"] = np.select(
        [panel.common_complete_new3, panel.expanded_complete_new3],
        ["common_complete", "newly_complete"],
        default="still_incomplete",
    )
    return panel


def build_panel(repo, audit_root=None):
    old = pd.read_csv(repo / "reports/round4_analysis/models/model_panel.csv")
    audit_root = audit_root or repo / "reports/round5_analysis/audit"
    new = pd.read_csv(audit_root / "annual_features.csv")
    assert len(old) == len(new) == 4032
    panel = combine_panels(old, new)
    recovered = panel.loc[panel.recovered_any]
    assert len(recovered) == 64 and recovered.territory_id.nunique() == 32
    assert recovered.recovered_feature_count.sum() == 64
    assert set(recovered.region_code) == {25, 59}
    assert recovered.core_eligible.all() and recovered.conservative_eligible.all()
    return panel


def frozen_folds(data, prior, year, sample, regime, seed):
    old_sample = "stable_code" if sample == "stable_code" else "core"
    old = prior.loc[
        prior.year.eq(year)
        & prior["sample"].eq(old_sample)
        & prior.regime.eq(regime)
        & prior.seed.eq(seed)
        & prior.model.eq("recovered_full"),
        [*KEY, "fold"],
    ]
    result = data[KEY].merge(old, on=KEY, how="left", validate="one_to_one")
    assert len(result) == len(data) and result.fold.notna().all()
    folds = result.fold.to_numpy(dtype=int)
    assert set(folds) == set(range(5))
    return folds


def fit_oof(data, y, numeric, folds, regime, label, audit):
    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        a, b = data.loc[train], data.loc[test]
        if regime == "heldout_regions":
            assert not set(a.region_code) & set(b.region_code)
        x, z, log = design(a, b, numeric, regime == "within_region")
        prediction[test] = Ridge(alpha=1.0).fit(x, y[train]).predict(z)
        audit.append(
            {
                **label,
                "fold": fold,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "numeric": numeric,
                **log,
            }
        )
    assert np.isfinite(prediction).all()
    return prediction


def check_legacy(data, prediction, prior, label):
    old = prior
    for key, value in label.items():
        old = old.loc[old[key].eq(value)]
    old = old.loc[old.model.eq("recovered_full")]
    matched = data[KEY].merge(old, on=KEY, how="left", validate="one_to_one")
    expected = matched[["predicted_" + t for t in TARGETS]].to_numpy()
    np.testing.assert_allclose(prediction, expected, atol=1e-9, rtol=1e-9)
    return float(np.abs(prediction - expected).max())


def run(repo, out, audit_root=None):
    out.mkdir(parents=True, exist_ok=False)
    protocol = repo / "configs/round5_analysis_v1.json"
    audit_root = audit_root or repo / "reports/round5_analysis/audit"
    audit_sha256 = sha(audit_root / "annual_features.csv")
    config = json.loads(protocol.read_text("utf-8"))
    started = datetime.now(UTC).isoformat()
    (out / "protocol_lock.json").write_text(
        json.dumps(
            {
                "started_at": started,
                "protocol_sha256": sha(protocol),
                "model_code_sha256": sha(__file__),
                "source_base_commit": config["base_commit"],
                "annual_features_sha256": audit_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    panel = build_panel(repo, audit_root)
    panel.to_csv(out / "model_panel.csv", index=False)
    prior = duckdb.read_parquet(
        str(repo / "reports/round4_analysis/models/oof_predictions.parquet")
    ).df()
    saved, metrics, comparisons, slices, checks, preprocessing = [], [], [], [], [], []
    for year in config["years"]:
        core = panel.loc[panel.year.eq(year) & panel.core_eligible]
        assert len(core) == config["core_n"][str(year)]
        samples = {
            "core": core,
            "stable_code": core.loc[core.conservative_eligible],
            "common_complete_new3": core.loc[core.common_complete_new3],
            "core_without_recovery_regions": core.loc[~core.recovery_region],
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
                        if sample != "common_complete_new3"
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
                    if sample in {"common_complete_new3", "core_without_recovery_regions"}:
                        np.testing.assert_allclose(
                            data[OLD], data[NEW], rtol=1e-12, atol=1e-12, equal_nan=True
                        )
                        np.testing.assert_allclose(
                            results["legacy_full"], results["recovered_full"], rtol=1e-9, atol=1e-9
                        )
                        checks.append(
                            {
                                **label,
                                "check": sample + "_invariance",
                                "max_abs": float(
                                    np.abs(results["legacy_full"] - results["recovered_full"]).max()
                                ),
                            }
                        )
                    if sample == "core_without_recovery_regions":
                        np.testing.assert_allclose(
                            results["dual_flags_old_values"],
                            results["dual_flags_new_values"],
                            rtol=1e-9,
                            atol=1e-9,
                        )
                        checks.append(
                            {
                                **label,
                                "check": "excluded_region_shared_flags_invariance",
                                "max_abs": float(
                                    np.abs(
                                        results["dual_flags_old_values"]
                                        - results["dual_flags_new_values"]
                                    ).max()
                                ),
                            }
                        )
                    for name, prediction in results.items():
                        rows = data[
                            [
                                *KEY,
                                "region_code",
                                "recovered_any",
                                "recovery_region",
                                "completeness_stratum",
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
        "configs/round5_analysis_v1.json",
        "scripts/round2_model_analysis.py",
        "scripts/round3_model_analysis.py",
        "src/municipal_profiles/enriched_analysis.py",
        "uv.lock",
        "reports/round4_analysis/models/model_panel.csv",
        "reports/data_round5/economy/new_numeric_cells.csv",
        "reports/round4_analysis/models/oof_predictions.parquet",
        "reports/round5_analysis/audit/annual_features.csv",
    ]
    input_hashes = {p: sha(repo / p) for p in inputs if not p.endswith("annual_features.csv")}
    assert sha(audit_root / "annual_features.csv") == audit_sha256
    input_hashes["reports/round5_analysis/audit/annual_features.csv"] = audit_sha256
    lock = json.loads((out / "protocol_lock.json").read_text("utf-8"))
    assert lock["protocol_sha256"] == sha(protocol)
    assert lock["model_code_sha256"] == sha(__file__)
    manifest = {
        "status": "complete",
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "inputs": input_hashes,
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
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo, args.out, args.audit_root)
