"""Frozen-protocol exploratory incremental industry attributes, separate from prior analyses."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from municipal_profiles.enriched_analysis import (
    CORE,
    clr,
    paired_region_loss_ci,
    r2_matrix,
    within_region_folds,
)

KEY = ["territory_id", "year"]
P = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
SECTIONS = ["A", "B", "C", "D", "H", "O", "P", "Q"]
SOURCE_CODES = ["10000", "50000", "20000", "30000", "49000", "84000", "85000", "86000"]
SECTORS = [f"sector_{s}_share" for s in SECTIONS]
FLAGS = [f"missing_{s}" for s in SECTIONS]
TARGETS = ["log_total", *[f"clr_{p}" for p in P]]
MODEL_FEATURES = {
    "economic": CORE,
    "economic_missingness": [*CORE, *FLAGS],
    "economic_sectors": [*CORE, *FLAGS, *SECTORS],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_features(source_path, panel):
    raw = duckdb.read_parquet(str(source_path)).df()
    for key in KEY:
        raw[key] = raw[key].astype(int)
    raw["source_activity_code"] = raw.source_activity_code.astype(str)
    raw = raw.loc[raw.indicator_id.astype(str).eq("8423005")].copy()
    assert raw.period_code.eq(79).all()
    assert not raw.duplicated([*KEY, "source_activity_code"]).any()
    joined = raw.merge(
        panel[[*KEY, "official_oktmo_8"]],
        on=KEY,
        validate="many_to_one",
        suffixes=("_source", "_panel"),
        how="left",
    )
    assert joined.official_oktmo_8_source.eq(joined.official_oktmo_8_panel).all()
    raw["accepted_value"] = raw.value.where(raw.status.eq("observed") & np.isfinite(raw.value))
    wide = raw.pivot(index=KEY, columns="source_activity_code", values="accepted_value")
    total = wide["101"]
    result = pd.DataFrame(index=wide.index)
    for code, name in zip(SOURCE_CODES, SECTORS, strict=True):
        numerator = wide[code]
        valid = numerator.ge(0) & total.gt(0) & numerator.le(total)
        result[name] = numerator.div(total).where(valid)
    check = panel.loc[panel.core_eligible, [*KEY, "workers"]].merge(
        total.rename("source_workers").reset_index(), on=KEY, validate="one_to_one"
    )
    assert check.workers.eq(check.source_workers).all()
    return result.reset_index()


def design(train, test, numeric, region):
    """Training-only imputation and moments; missing flags have explicit fixed meaning."""
    cats = ["municipal_district_type", *(["region_code"] if region else [])]
    a = pd.get_dummies(train[cats].astype(str), dtype=float)
    b = pd.get_dummies(test[cats].astype(str), dtype=float).reindex(columns=a.columns, fill_value=0)
    values = train[numeric].astype(float)
    empty = values.isna().all()
    medians = values.median().fillna(0.0)
    v, w = values.fillna(medians), test[numeric].astype(float).fillna(medians)
    means, sd = v.mean(), v.std(ddof=0).replace(0, 1)
    x = np.column_stack([a.to_numpy(), ((v - means) / sd).to_numpy()])
    z = np.column_stack([b.to_numpy(), ((w - means) / sd).to_numpy()])
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("Nonfinite design")
    audit = {
        "train_empty_columns": list(empty[empty].index),
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "sd": sd.to_dict(),
        "dummy_columns": list(a.columns),
    }
    return x, z, audit


def folds_for(frame, regime, seed):
    groups = frame.region_code.to_numpy()
    if regime == "within_region":
        return within_region_folds(groups, seed=seed)
    result = np.empty(len(frame), dtype=int)
    for fold, (_, test) in enumerate(GroupKFold(5).split(frame, groups=groups)):
        result[test] = fold
    return result


def shuffle_observed(frame):
    """Retain regional distributions and missingness, destroy row-specific alignment."""
    shuffled = frame.copy()
    rng = np.random.default_rng(71013)
    for sector in SECTORS:
        for region in sorted(frame.region_code.unique()):
            mask = frame.region_code.eq(region) & frame[sector].notna()
            shuffled.loc[mask, sector] = rng.permutation(frame.loc[mask, sector].to_numpy())
    return shuffled


def fit_oof(frame, y, numeric, regime, folds, label, audit, *, permute=False):
    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if regime == "heldout_regions":
            assert not set(frame.loc[train, "region_code"]) & set(frame.loc[test, "region_code"])
        train_frame, test_frame = frame.loc[train], frame.loc[test]
        if permute:
            train_frame = shuffle_observed(train_frame)
            test_frame = shuffle_observed(test_frame)
        x, z, log = design(train_frame, test_frame, numeric, regime == "within_region")
        prediction[test] = Ridge(alpha=1.0).fit(x, y[train]).predict(z)
        audit.append(
            {**label, "fold": fold, "train_n": int(train.sum()), "test_n": int(test.sum()), **log}
        )
    assert np.isfinite(prediction).all()
    return prediction


def correlations(panel, out):
    records = []
    for year in [2023, 2024]:
        for sector in SECTORS:
            d = panel.loc[panel.year.eq(year) & panel.core_eligible].dropna(subset=[sector])
            y = np.column_stack([d.log_total, clr(d[P])])
            cats = pd.get_dummies(
                d[["region_code", "municipal_district_type"]].astype(str),
                drop_first=True,
                dtype=float,
            )
            nuisance = np.column_stack([np.ones(len(d)), d[CORE].to_numpy(), cats.to_numpy()])
            both = np.column_stack([d[sector], y])
            residual = both - nuisance @ np.linalg.lstsq(nuisance, both, rcond=None)[0]
            for i, target in enumerate(TARGETS):
                records.append(
                    {
                        "year": year,
                        "sector": sector,
                        "target": target,
                        "n": len(d),
                        "regions": d.region_code.nunique(),
                        "raw_pearson": pearsonr(both[:, 0], both[:, i + 1]).statistic,
                        "partial_pearson": pearsonr(residual[:, 0], residual[:, i + 1]).statistic,
                        "inference": "descriptive_observed_only_no_causal_claim",
                    }
                )
    pd.DataFrame(records).to_csv(out / "partial_correlations.csv", index=False)


def run(repo, out):
    out.mkdir(parents=True, exist_ok=True)
    inputs = [
        "reports/enriched_analysis/core_v2/annual_panel.csv",
        "reports/data_round2/sector/bulk/annual_candidate_sector.parquet",
        "configs/round2_analysis_v1.json",
        "src/municipal_profiles/enriched_analysis.py",
        "uv.lock",
    ]
    panel = pd.read_csv(repo / inputs[0], dtype={"official_oktmo_8": str})
    features = selected_features(repo / inputs[1], panel)
    features.to_csv(out / "selected_sector_features.csv", index=False)
    assert not panel.duplicated(KEY).any() and not features.duplicated(KEY).any()
    assert len(panel) == len(features) == 4032
    assert set(map(tuple, panel[KEY].to_numpy())) == set(map(tuple, features[KEY].to_numpy()))
    panel = panel.merge(features[[*KEY, *SECTORS]], on=KEY, validate="one_to_one")
    assert len(panel) == 4032
    for sector, flag in zip(SECTORS, FLAGS, strict=True):
        assert panel[sector].dropna().between(0, 1).all()
        panel[flag] = panel[sector].isna().astype(float)
    panel["selected_sector_observed"] = panel[SECTORS].notna().sum(axis=1)
    panel = panel.sort_values(["year", "territory_id"]).reset_index(drop=True)
    panel.to_csv(out / "model_panel.csv", index=False)
    metrics, comparisons, saved, preprocessing, slices = [], [], [], [], []
    skipped = []
    for year in [2023, 2024]:
        base = panel.loc[panel.year.eq(year) & panel.core_eligible]
        samples = {
            "core": base,
            "stable_code": base.loc[base.conservative_eligible],
            "complete_sectors": base.dropna(subset=SECTORS),
        }
        for sample, d in samples.items():
            if len(d) < 100 or d.region_code.nunique() < 10:
                skipped.append(
                    {
                        "year": year,
                        "sample": sample,
                        "n": len(d),
                        "regions": d.region_code.nunique(),
                        "reason": "too_small",
                    }
                )
                continue
            d = d.reset_index(drop=True)
            groups = d.region_code.to_numpy()
            y = np.column_stack([d.log_total, clr(d[P])])
            for regime in ["within_region", "heldout_regions"]:
                seeds = [13, 29, 47] if sample == "core" and regime == "within_region" else [13]
                for seed in seeds:
                    label = {"year": year, "sample": sample, "regime": regime, "seed": seed}
                    folds = folds_for(d, regime, seed)
                    results = {}
                    for name, numeric in MODEL_FEATURES.items():
                        results[name] = fit_oof(
                            d, y, numeric, regime, folds, {**label, "model": name}, preprocessing
                        )
                    if sample == "core" and seed == 13:
                        results["permuted_sectors"] = fit_oof(
                            d,
                            y,
                            MODEL_FEATURES["economic_sectors"],
                            regime,
                            folds,
                            {**label, "model": "permuted_sectors"},
                            preprocessing,
                            permute=True,
                        )
                    for name, pred in results.items():
                        row = d[[*KEY, "region_code", "selected_sector_observed"]].copy()
                        for key, value in {**label, "model": name}.items():
                            row[key] = value
                        row["fold"] = folds
                        for i, target in enumerate(TARGETS):
                            row["actual_" + target], row["predicted_" + target] = (
                                y[:, i],
                                pred[:, i],
                            )
                        saved.append(row)
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            metrics.append(
                                {
                                    **label,
                                    "model": name,
                                    "target": target,
                                    "n": len(d),
                                    "regions": len(set(groups)),
                                    "r2_oof": r2_matrix(y[:, idx], pred[:, idx]),
                                    "mse_oof": np.square(y[:, idx] - pred[:, idx]).mean(),
                                }
                            )
                    pairs = [
                        ("economic", "economic_missingness"),
                        ("economic", "economic_sectors"),
                        ("economic_missingness", "economic_sectors"),
                    ]
                    if "permuted_sectors" in results:
                        pairs.append(("economic_missingness", "permuted_sectors"))
                    for baseline, alternative in pairs:
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            effect, low, high = paired_region_loss_ci(
                                y[:, idx],
                                results[baseline][:, idx],
                                results[alternative][:, idx],
                                groups,
                            )
                            comparisons.append(
                                {
                                    **label,
                                    "target": target,
                                    "baseline": baseline,
                                    "alternative": alternative,
                                    "n": len(d),
                                    "relative_mse_reduction": effect,
                                    "ci_low": low,
                                    "ci_high": high,
                                }
                            )
                    if sample == "core" and seed == 13:
                        for target, idx in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                            for scope in ["region_code", "selected_sector_observed"]:
                                for group in sorted(d[scope].unique()):
                                    mask = d[scope].eq(group).to_numpy()
                                    losses = {
                                        name: np.square(y[mask][:, idx] - pred[mask][:, idx]).sum()
                                        for name, pred in results.items()
                                    }
                                    slices.append(
                                        {
                                            **label,
                                            "target": target,
                                            "slice": scope,
                                            "group": group,
                                            "n": int(mask.sum()),
                                            "baseline_sse": losses["economic_missingness"],
                                            "alternative_sse": losses["economic_sectors"],
                                            "relative_mse_reduction": 1
                                            - losses["economic_sectors"]
                                            / losses["economic_missingness"],
                                        }
                                    )
    pd.DataFrame(metrics).to_csv(out / "cv_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "cv_comparisons.csv", index=False)
    pd.DataFrame(slices).to_csv(out / "slice_losses.csv", index=False)
    duckdb.from_df(pd.concat(saved, ignore_index=True)).write_parquet(
        str(out / "oof_predictions.parquet")
    )
    (out / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    correlations(panel, out)
    manifest = {
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "inputs": {p: sha(repo / p) for p in inputs},
        "code_sha256": sha(Path(__file__)),
        "skipped_samples": skipped,
        "rows": panel.groupby("year")
        .agg(core=("core_eligible", "sum"), stable=("conservative_eligible", "sum"))
        .to_dict("index"),
        "models": MODEL_FEATURES,
        "outputs": {
            p.name: sha(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "metrics": len(metrics), "skipped": skipped}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repo, args.out or args.repo / "reports/round2_analysis/models")
