"""Fixed-protocol retrospective increments from new municipal monetary activity data."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from round2_model_analysis import FLAGS, KEY, SECTORS, TARGETS, P, design, folds_for, sha
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from municipal_profiles.enriched_analysis import CORE, clr, paired_region_loss_ci, r2_matrix

PREFIXES = ["investment", "shipments", "retail"]
NEW = [f"{p}_log_rubles_per_person" for p in PREFIXES]
NEW_FLAGS = [f"{p}_missing" for p in PREFIXES]
RICH = [*CORE, *FLAGS, *SECTORS]
MODELS = {
    "economic": CORE,
    "economic_newflags": [*CORE, *NEW_FLAGS],
    "economic_newvalues": [*CORE, *NEW_FLAGS, *NEW],
    "economic_sectors": RICH,
    "economic_sectors_newflags": [*RICH, *NEW_FLAGS],
    "economic_sectors_newvalues": [*RICH, *NEW_FLAGS, *NEW],
}
BASELINE = "economic_sectors_newflags"
FULL = "economic_sectors_newvalues"


def shuffle_new(frame):
    result = frame.copy()
    rng = np.random.default_rng(91013)
    for feature in NEW:
        for group in sorted(frame.region_code.unique()):
            mask = frame.region_code.eq(group) & frame[feature].notna()
            result.loc[mask, feature] = rng.permutation(frame.loc[mask, feature].to_numpy())
    return result


def fit_oof(d, y, numeric, folds, regime, label, audit, *, permute=False):
    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if regime == "heldout_regions":
            assert not set(d.loc[train, "region_code"]) & set(d.loc[test, "region_code"])
        a, b = d.loc[train], d.loc[test]
        if permute:
            a, b = shuffle_new(a), shuffle_new(b)
        x, z, log = design(a, b, numeric, regime == "within_region")
        prediction[test] = Ridge(alpha=1.0).fit(x, y[train]).predict(z)
        audit.append(
            {**label, "fold": fold, "train_n": int(train.sum()), "test_n": int(test.sum()), **log}
        )
    assert np.isfinite(prediction).all()
    return prediction


def assert_prior_folds(d, folds, prior, label):
    old = prior
    for key, value in label.items():
        old = old.loc[old[key].eq(value)]
    old = old.loc[old.model.eq("economic"), [*KEY, "fold"]]
    current = d[KEY].copy()
    current["fold"] = folds
    check = current.merge(
        old, on=KEY, how="outer", indicator=True, validate="one_to_one", suffixes=("_new", "_old")
    )
    assert check["_merge"].eq("both").all() and check.fold_new.eq(check.fold_old).all()


def build_panel(repo):
    frozen = pd.read_csv(repo / "reports/round2_analysis/models/model_panel.csv")
    fresh = pd.read_csv(repo / "reports/round3_analysis/audit/annual_features.csv")
    assert len(frozen) == len(fresh) == 4032
    assert not frozen.duplicated(KEY).any() and not fresh.duplicated(KEY).any()
    columns = [*KEY, *NEW, *NEW_FLAGS]
    panel = frozen.merge(fresh[columns], on=KEY, how="left", validate="one_to_one", indicator=True)
    assert panel["_merge"].eq("both").all()
    panel = (
        panel.drop(columns="_merge").sort_values(["year", "territory_id"]).reset_index(drop=True)
    )
    for value, flag in zip(NEW, NEW_FLAGS, strict=True):
        assert (
            panel.loc[panel.core_eligible, value]
            .isna()
            .equals(panel.loc[panel.core_eligible, flag].eq(1))
        )
        assert np.isfinite(panel[value].dropna()).all()
    panel["new_observed_count"] = panel[NEW].notna().sum(axis=1)
    # These ranks define only descriptive OOF error slices; they never enter models or folds.
    panel["population_quartile"] = (
        panel.groupby("year")["population_total"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        .astype(str)
    )
    return panel


def run(repo, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh output directory; preserve every prior run")
    out.mkdir(parents=True, exist_ok=True)
    protocol = repo / "configs/round3_analysis_v1.json"
    cfg = json.loads(protocol.read_text(encoding="utf-8"))
    panel = build_panel(repo)
    panel.to_csv(out / "model_panel.csv", index=False)
    prior = duckdb.read_parquet(
        str(repo / "reports/round2_analysis/models/oof_predictions.parquet")
    ).df()
    metrics, comparisons, saved, preprocess, slices, skipped = [], [], [], [], [], []
    fold_checks = []
    for year in cfg["years"]:
        core = panel.loc[panel.year.eq(year) & panel.core_eligible]
        assert len(core) == cfg["expected_core_n"][str(year)]
        samples = {
            "core": core,
            "stable_code": core.loc[core.conservative_eligible],
            "complete_new3": core.dropna(subset=NEW),
        }
        assert len(samples["stable_code"]) == cfg["expected_stable_n"][str(year)]
        for sample, d in samples.items():
            if len(d) < 100 or d.region_code.nunique() < 10:
                skipped.append(dict(year=year, sample=sample, n=len(d), reason="too_small"))
                continue
            d = d.reset_index(drop=True)
            y = np.column_stack([d.log_total, clr(d[P])])
            groups = d.region_code.to_numpy()
            for regime in cfg["regimes"]:
                seeds = [13, 29, 47] if sample == "core" and regime == "within_region" else [13]
                for seed in seeds:
                    label = dict(year=year, sample=sample, regime=regime, seed=seed)
                    folds = folds_for(d, regime, seed)
                    if sample in {"core", "stable_code"}:
                        assert_prior_folds(d, folds, prior, label)
                        fold_checks.append({**label, "n": len(d), "exact_prior_folds": True})
                    results = {}
                    for name, numeric in MODELS.items():
                        results[name] = fit_oof(
                            d, y, numeric, folds, regime, {**label, "model": name}, preprocess
                        )
                    if sample == "core" and seed == 13:
                        results["permuted_newvalues"] = fit_oof(
                            d,
                            y,
                            MODELS[FULL],
                            folds,
                            regime,
                            {**label, "model": "permuted_newvalues"},
                            preprocess,
                            permute=True,
                        )
                        for prefix, feature in zip(PREFIXES, NEW, strict=True):
                            name = "without_" + prefix
                            numeric = [c for c in MODELS[FULL] if c != feature]
                            results[name] = fit_oof(
                                d, y, numeric, folds, regime, {**label, "model": name}, preprocess
                            )
                    for name, prediction in results.items():
                        row = d[[*KEY, "region_code", "new_observed_count"]].copy()
                        for key, value in {**label, "model": name}.items():
                            row[key] = value
                        row["fold"] = folds
                        for i, target in enumerate(TARGETS):
                            row["actual_" + target], row["predicted_" + target] = (
                                y[:, i],
                                prediction[:, i],
                            )
                        saved.append(row)
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            metrics.append(
                                {
                                    **label,
                                    "model": name,
                                    "target": target,
                                    "n": len(d),
                                    "regions": d.region_code.nunique(),
                                    "mse_oof": np.square(y[:, idx] - prediction[:, idx]).mean(),
                                    "r2_oof": r2_matrix(y[:, idx], prediction[:, idx]),
                                }
                            )
                    pairs = [
                        ("economic", "economic_newflags"),
                        ("economic_newflags", "economic_newvalues"),
                        ("economic", "economic_newvalues"),
                        ("economic_sectors", BASELINE),
                        (BASELINE, FULL),
                        ("economic_sectors", FULL),
                    ]
                    if "permuted_newvalues" in results:
                        pairs += [(BASELINE, "permuted_newvalues")]
                        pairs += [("without_" + p, FULL) for p in PREFIXES]
                    for baseline, alternative in pairs:
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            effect, low, high = paired_region_loss_ci(
                                y[:, idx],
                                results[baseline][:, idx],
                                results[alternative][:, idx],
                                groups,
                                repeats=2000,
                                seed=13,
                            )
                            comparisons.append(
                                {
                                    **label,
                                    "baseline": baseline,
                                    "alternative": alternative,
                                    "target": target,
                                    "n": len(d),
                                    "regions": d.region_code.nunique(),
                                    "relative_mse_reduction": effect,
                                    "ci_low": low,
                                    "ci_high": high,
                                }
                            )
                    if sample == "core" and seed == 13:
                        for scope in [
                            "region_code",
                            "new_observed_count",
                            "municipal_district_type",
                            "population_quartile",
                        ]:
                            for group in sorted(d[scope].unique()):
                                mask = d[scope].eq(group).to_numpy()
                                for target, idx in [
                                    ("log_total", [0]),
                                    ("profile_clr", [1, 2, 3, 4, 5]),
                                ]:
                                    a = np.square(
                                        y[mask][:, idx] - results[BASELINE][mask][:, idx]
                                    ).sum()
                                    b = np.square(
                                        y[mask][:, idx] - results[FULL][mask][:, idx]
                                    ).sum()
                                    slices.append(
                                        {
                                            **label,
                                            "target": target,
                                            "slice": scope,
                                            "group": group,
                                            "n": int(mask.sum()),
                                            "baseline_sse": a,
                                            "alternative_sse": b,
                                            "relative_mse_reduction": 1 - b / a,
                                        }
                                    )
                    print(json.dumps({**label, "n": len(d), "models": len(results)}), flush=True)
    pd.DataFrame(metrics).to_csv(out / "cv_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "cv_comparisons.csv", index=False)
    pd.DataFrame(slices).to_csv(out / "slice_losses.csv", index=False)
    pd.DataFrame(fold_checks).to_csv(out / "prior_fold_checks.csv", index=False)
    duckdb.from_df(pd.concat(saved, ignore_index=True)).write_parquet(
        str(out / "oof_predictions.parquet")
    )
    (out / "preprocessing.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")
    inputs = [
        "configs/round3_analysis_v1.json",
        "scripts/round2_model_analysis.py",
        "src/municipal_profiles/enriched_analysis.py",
        "uv.lock",
        "reports/round3_analysis/audit/annual_features.csv",
        "reports/round2_analysis/models/model_panel.csv",
        "reports/round2_analysis/models/oof_predictions.parquet",
    ]
    manifest = dict(
        status="complete",
        completed_at=datetime.now(UTC).isoformat(),
        inputs={p: sha(repo / p) for p in inputs},
        code_sha256=sha(__file__),
        models=MODELS,
        skipped_samples=skipped,
        tuning_performed=False,
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="complete", metrics=len(metrics), comparisons=len(comparisons))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo, args.out or args.repo / "reports/round3_analysis/models")
