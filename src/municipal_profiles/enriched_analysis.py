"""Exploratory, explicitly conditional analysis of municipal source enrichment."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from municipal_profiles.data import PUBLISHED_CATEGORIES, TOTAL_CATEGORY, load_panel
from municipal_profiles.dynamics import PROFILE_COLUMNS

CORE = ["log_population", "log_wage", "log_worker_concentration"]
INPUTS = {
    "consumption": "data/raw/consumption.parquet",
    "population": "reports/enrichment_v2/population/municipal_population_wide.csv",
    "economy": "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv",
    "spatial": "reports/enrichment_v3/spatial/sber_yearly_spatial_attributes.csv",
    "market": "reports/enrichment_v3/spatial/sber_static_context_2024.csv",
    "changes": "reports/enrichment_v2/code_bridge/primary_78_code_type_changes.csv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def join_unique(left, right, keys, name, audit):
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Nonunique join key: {name}")
    result = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    audit.append(
        {
            "source": name,
            "left_rows": len(left),
            "right_rows": len(right),
            "result_rows": len(result),
            "unmatched_left": int(result._merge.eq("left_only").sum()),
        }
    )
    return result.drop(columns="_merge")


def build_panel(repo):
    audit = []
    raw = load_panel(repo / INPUTS["consumption"])
    wide = raw.pivot(index=["territory_id", "date"], columns="category", values="value")
    p = wide[list(PUBLISHED_CATEGORIES)].div(wide[list(PUBLISHED_CATEGORIES)].sum(axis=1), axis=0)
    p.columns = PROFILE_COLUMNS
    monthly = p.assign(total=wide[TOTAL_CATEGORY]).reset_index()
    monthly["year"] = monthly.date.str[:4].astype(int)
    annual = monthly.groupby(["territory_id", "year"], as_index=False).agg(
        **{name: (name, "mean") for name in PROFILE_COLUMNS},
        total_mean=("total", "mean"),
        observed_months=("date", "nunique"),
    )
    if len(annual) != 4032 or not annual.observed_months.eq(12).all():
        raise ValueError("Expected2016complete municipalities in each of two years")
    population = pd.read_csv(repo / INPUTS["population"])
    population["year"] = population.reference_date.str[:4].astype(int)
    annual = join_unique(
        annual,
        population.drop(columns="reference_date"),
        ["territory_id", "year"],
        "accepted_population_v2",
        audit,
    )
    spatial = pd.read_csv(repo / INPUTS["spatial"], dtype={"official_oktmo_8": str})
    spatial["year"] = spatial.snapshot_date.str[:4].astype(int)
    columns = [
        "territory_id",
        "year",
        "official_oktmo_8",
        "region_code",
        "region_name",
        "municipal_district_name",
        "municipal_district_type",
    ]
    annual = join_unique(annual, spatial[columns], ["territory_id", "year"], "sber_yearly", audit)
    economy = pd.read_csv(repo / INPUTS["economy"], dtype={"indicator_id": str})
    if economy.duplicated(["territory_id", "year", "indicator_id"]).any():
        raise ValueError("Ambiguous economy observations")
    if not economy.period_code.dropna().eq(79).all():
        raise ValueError("Nonannual economy period")
    observed = economy.cell_status.eq("observed")
    economy["numeric_value"] = pd.to_numeric(economy.value.where(observed), errors="raise")
    ec = economy.pivot(
        index=["territory_id", "year"], columns="indicator_id", values="numeric_value"
    )
    ec = ec.rename(columns={"8423005": "workers", "8423007": "wage"}).reset_index()
    flags = economy.assign(
        calendar_review=economy.temporal_review_required.astype(str).str.lower().eq("true")
    )
    flags = flags.groupby(["territory_id", "year"], as_index=False).calendar_review.max()
    ec = ec.merge(flags, validate="one_to_one", on=["territory_id", "year"])
    annual = join_unique(annual, ec, ["territory_id", "year"], "economy_annual_candidates", audit)
    market = pd.read_csv(repo / INPUTS["market"])
    market["year"] = 2024
    annual = join_unique(
        annual,
        market[
            ["territory_id", "year", "market_access", "road_distance_to_sber_flagged_center_km"]
        ],
        ["territory_id", "year"],
        "market_2024_only",
        audit,
    )
    changed = pd.read_csv(repo / INPUTS["changes"]).territory_id
    annual["code_type_changed"] = annual.territory_id.isin(changed)
    for name, col in [
        ("log_population", "population_total"),
        ("log_wage", "wage"),
        ("log_total", "total_mean"),
        ("log_market_access", "market_access"),
    ]:
        annual[name] = np.log(annual[col].where(annual[col].gt(0)))
    annual["log_worker_concentration"] = np.log(
        annual.workers.where(annual.workers.gt(0))
        / annual.population_total.where(annual.population_total.gt(0))
    )
    if not annual.urban_share.dropna().between(0, 1).all():
        raise ValueError("Invalid urban population fraction")
    if annual.loc[annual.year.eq(2023), "market_access"].notna().any():
        raise ValueError("2024market must not be carried backward")
    annual["core_eligible"] = annual[CORE].notna().all(axis=1) & ~annual.calendar_review
    annual["conservative_eligible"] = annual.core_eligible & ~annual.code_type_changed
    annual["geography_status"] = (
        "conditional_source_code_and_annual_version_not_legal_boundary_proof"
    )
    return annual.sort_values(["year", "territory_id"]).reset_index(drop=True), audit


def clr(profiles):
    p = np.asarray(profiles, dtype=float)
    if (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("CLR requires strictly positive normalized profiles")
    logs = np.log(p)
    return logs - logs.mean(axis=1, keepdims=True)


def within_region_folds(regions, folds=5, seed=13):
    rng = np.random.default_rng(seed)
    result = np.empty(len(regions), dtype=int)
    for group in sorted(set(regions)):
        indices = np.flatnonzero(np.asarray(regions) == group)
        order = rng.permutation(indices)
        result[order] = (np.arange(len(order)) + rng.integers(folds)) % folds
    return result


def design(train, test, numeric, region):
    cats = ["municipal_district_type"] + (["region_code"] if region else [])
    a = pd.get_dummies(train[cats].astype(str), dtype=float)
    b = pd.get_dummies(test[cats].astype(str), dtype=float).reindex(columns=a.columns, fill_value=0)
    x, z = a.to_numpy(), b.to_numpy()
    if numeric:
        mean = train[numeric].mean().to_numpy()
        std = train[numeric].std(ddof=0).to_numpy(copy=True)
        std[std == 0] = 1
        x = np.column_stack((x, (train[numeric].to_numpy() - mean) / std))
        z = np.column_stack((z, (test[numeric].to_numpy() - mean) / std))
    return x, z


def r2_matrix(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    denominator = np.square(truth - truth.mean(axis=0)).sum()
    return float(1 - np.square(truth - prediction).sum() / denominator)


def paired_region_loss_ci(y, baseline, alternative, regions, repeats=2000, seed=13):
    groups = np.unique(regions)
    losses = np.array(
        [
            [
                np.square(y[regions == g] - pred[regions == g]).sum()
                for pred in [baseline, alternative]
            ]
            for g in groups
        ]
    )
    rng = np.random.default_rng(seed)
    draws = losses[rng.integers(len(groups), size=(repeats, len(groups)))].sum(axis=1)
    reduction = 1 - draws[:, 1] / draws[:, 0]
    return float(1 - losses[:, 1].sum() / losses[:, 0].sum()), *np.quantile(
        reduction, [0.025, 0.975]
    ).tolist()


def run_cv(panel, out):
    metrics, predictions, comparisons = [], [], []
    for year in [2023, 2024]:
        annual = panel.loc[panel.year.eq(year) & panel.core_eligible]
        scenarios = [
            ("core", annual, [("context", []), ("economic", CORE)]),
            (
                "urban",
                annual.dropna(subset=["urban_share"]),
                [("context", []), ("economic", CORE), ("economic_urban", [*CORE, "urban_share"])],
            ),
            (
                "stable_code",
                annual.loc[~annual.code_type_changed],
                [("context", []), ("economic", CORE)],
            ),
        ]
        if year == 2024:
            scenarios.append(
                (
                    "market",
                    annual.dropna(subset=["log_market_access"]),
                    [
                        ("context", []),
                        ("economic", CORE),
                        ("economic_market", [*CORE, "log_market_access"]),
                    ],
                )
            )
        for scenario, frame, models in scenarios:
            frame = frame.reset_index(drop=True)
            regions = frame.region_code.to_numpy()
            y = np.column_stack([frame.log_total, clr(frame[list(PROFILE_COLUMNS)])])
            for regime in ["within_region", "heldout_regions"]:
                if regime == "within_region":
                    folds = within_region_folds(regions)
                else:
                    folds = np.empty(len(frame), dtype=int)
                    for fold, (_, test) in enumerate(GroupKFold(5).split(frame, groups=regions)):
                        folds[test] = fold
                results = {}
                for model, numeric in models:
                    pred = np.full_like(y, np.nan)
                    for fold in range(5):
                        train, test = folds != fold, folds == fold
                        x, z = design(
                            frame.loc[train], frame.loc[test], numeric, regime == "within_region"
                        )
                        fitted = Ridge(alpha=1.0).fit(x, y[train])
                        pred[test] = fitted.predict(z)
                        if regime == "heldout_regions":
                            assert not set(regions[train]) & set(regions[test])
                    assert np.isfinite(pred).all()
                    results[model] = pred
                    for target, indices in [("log_total", [0]), ("profile_clr", list(range(1, 6)))]:
                        row = {
                            "year": year,
                            "sample": scenario,
                            "regime": regime,
                            "model": model,
                            "target": target,
                            "n": len(frame),
                            "regions": len(set(regions)),
                            "r2_oof": r2_matrix(y[:, indices], pred[:, indices]),
                            "mse_oof": float(np.square(y[:, indices] - pred[:, indices]).mean()),
                        }
                        metrics.append(row)
                    records = frame[["territory_id", "year", "region_code"]].copy()
                    records["sample"], records["regime"], records["model"], records["fold"] = (
                        scenario,
                        regime,
                        model,
                        folds,
                    )
                    for i, name in enumerate(["log_total", *["clr_" + c for c in PROFILE_COLUMNS]]):
                        records["actual_" + name], records["predicted_" + name] = (
                            y[:, i],
                            pred[:, i],
                        )
                    predictions.append(records)
                for model, _ in models[1:]:
                    baseline = "context" if model == "economic" else "economic"
                    for target, indices in [("log_total", [0]), ("profile_clr", list(range(1, 6)))]:
                        estimate, low, high = paired_region_loss_ci(
                            y[:, indices],
                            results[baseline][:, indices],
                            results[model][:, indices],
                            regions,
                        )
                        comparisons.append(
                            {
                                "year": year,
                                "sample": scenario,
                                "regime": regime,
                                "target": target,
                                "baseline": baseline,
                                "alternative": model,
                                "relative_mse_reduction": estimate,
                                "ci_low": low,
                                "ci_high": high,
                                "bootstrap_scope": "fixed_OOF_predictions_paired_region_resampling",
                            }
                        )
    pd.DataFrame(metrics).to_csv(out / "cv_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "cv_comparisons.csv", index=False)
    duckdb.from_df(pd.concat(predictions)).write_parquet(str(out / "oof_predictions.parquet"))


def cluster_ols(frame, target, numeric, type_effect=True):
    cats = ["region_code"] + (["municipal_district_type"] if type_effect else [])
    dummy = pd.get_dummies(frame[cats].astype(str), drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(frame)), frame[numeric].to_numpy(), dummy.to_numpy()])
    y = frame[target].to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    groups = frame.region_code.to_numpy()
    scores = np.stack([x[groups == g].T @ residual[groups == g] for g in np.unique(groups)])
    g = len(scores)
    covariance = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (len(x) - 1) / (len(x) - rank)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    critical = t.ppf(0.975, g - 1)
    return [
        {
            "variable": col,
            "coefficient": float(beta[i]),
            "se_region_clustered": float(se[i]),
            "ci_low": float(beta[i] - critical * se[i]),
            "ci_high": float(beta[i] + critical * se[i]),
            "n": len(frame),
            "regions": g,
            "rank": int(rank),
        }
        for i, col in enumerate(["intercept", *numeric])
        if i
    ]


def relationships(panel, out):
    correlations, coefficients = [], []
    for year in [2023, 2024]:
        frame = panel.loc[panel.year.eq(year) & ~panel.calendar_review]
        for x in [
            *CORE,
            "urban_share",
            "log_market_access",
            "road_distance_to_sber_flagged_center_km",
        ]:
            for y in ["log_total", *PROFILE_COLUMNS]:
                data = frame.dropna(subset=[x, y])
                if len(data) < 20:
                    continue
                ranks = data[[x, y]].rank()
                centered = ranks - ranks.groupby(data.region_code).transform("mean")
                correlations.append(
                    {
                        "year": year,
                        "feature": x,
                        "outcome": y,
                        "n": len(data),
                        "regions": data.region_code.nunique(),
                        "spearman": float(spearmanr(data[x], data[y]).statistic),
                        "region_centered_rank_correlation": float(centered[x].corr(centered[y])),
                    }
                )
        for sample, mask in [
            ("core", frame.core_eligible),
            ("stable_code", frame.conservative_eligible),
        ]:
            for row in cluster_ols(frame.loc[mask], "log_total", CORE):
                coefficients.append({"year": year, "sample": sample, "outcome": "log_total", **row})
    before = panel.loc[panel.year.eq(2023) & panel.conservative_eligible].set_index("territory_id")
    after = panel.loc[panel.year.eq(2024) & panel.conservative_eligible].set_index("territory_id")
    common = before.index.intersection(after.index)
    delta = after.loc[
        common, ["region_code", "municipal_district_type", "municipal_district_name"]
    ].copy()
    for name in ["log_total", *CORE]:
        delta["delta_" + name] = after.loc[common, name] - before.loc[common, name]
    for row in cluster_ols(
        delta, "delta_log_total", ["delta_" + c for c in CORE], type_effect=False
    ):
        coefficients.append(
            {
                "year": "2024_vs_2023",
                "sample": "stable_code_pairs",
                "outcome": "delta_log_total",
                **row,
            }
        )
    delta.reset_index().to_csv(out / "stable_code_changes.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "regression_coefficients.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "correlations.csv", index=False)


def run(repo, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory; previous analysis is immutable")
    out.mkdir(parents=True, exist_ok=True)
    config = repo / "configs/enriched_analysis_v1.json"
    manifest = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha(config),
        "input_hashes": {p: sha(repo / p) for p in INPUTS.values()},
        "code_sha256": sha(__file__),
        "analysis_status": "exploratory_conditional_geography",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with threadpool_limits(limits=2):
        panel, audit = build_panel(repo)
        duckdb.from_df(panel).write_parquet(str(out / "annual_panel.parquet"))
        panel.to_csv(out / "annual_panel.csv", index=False)
        coverage = panel.groupby("year").agg(
            n=("territory_id", "size"),
            core=("core_eligible", "sum"),
            conservative=("conservative_eligible", "sum"),
            population=("population_total", "count"),
            urban=("urban_share", "count"),
            wages=("wage", "count"),
            workers=("workers", "count"),
            market=("market_access", "count"),
        )
        coverage.to_csv(out / "coverage.csv")
        (out / "join_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        relationships(panel, out)
        run_cv(panel, out)
    manifest.update(
        status="complete",
        finished_at=datetime.now(UTC).isoformat(),
        outputs={p.name: sha(p) for p in out.iterdir() if p.name != "manifest.json"},
        tests_of_causal_effects=False,
        untouched_evaluation_data=False,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "complete", "coverage": coverage.reset_index().to_dict("records")}
