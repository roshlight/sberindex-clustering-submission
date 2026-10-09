"""Read-only independent verification; does not call the production analysis module."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from audit_cli import parse_paths
from scipy import linalg, stats
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

args = parse_paths(
    "Independently verify frozen enriched core results", with_audit=True, with_core=True
)
R, OUT, A, C = args.repo, args.out, args.audit, args.core
KEY = ["territory_id", "year"]
P = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
CATS = ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
XCORE = ["log_population", "log_wage", "log_worker_concentration"]
TARGETS = ["log_total", *["clr_" + c for c in P]]
checks = {}
max_errors = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(name, condition, details=None):
    checks[name] = {"status": "PASS" if condition else "FAIL", "details": details}
    if not condition:
        raise AssertionError(name)


def close(name, a, b, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    error = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
    max_errors[name] = error
    verify(name, np.allclose(a, b, equal_nan=True, atol=atol, rtol=rtol), {"max_abs_error": error})


def read(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def independent_design(train, test, numeric, regime):
    """Dummy levels and numerical moments are obtained only from training rows."""
    features = ["municipal_district_type"]
    if regime == "within_region":
        features += ["region_code"]
    a, b = [], []
    for feature in features:
        train_s = train[feature].astype(str)
        test_s = test[feature].astype(str)
        for level in sorted(train_s.unique()):
            a.append(train_s.eq(level).to_numpy(dtype=float))
            b.append(test_s.eq(level).to_numpy(dtype=float))
    for feature in numeric:
        v, w = train[feature].to_numpy(), test[feature].to_numpy()
        mean, sd = v.mean(), v.std(ddof=0)
        sd = sd if sd else 1.0
        a.append((v - mean) / sd)
        b.append((w - mean) / sd)
    return np.column_stack(a), np.column_stack(b)


def independent_ridge(x, y, test, alpha=1.0):
    # Direct centered normal equations, instead of production sklearn Ridge.fit.
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    beta = linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc, assume_a="pos")
    return ym + (test - xm) @ beta


def independent_ols(frame, target, numeric, with_type):
    columns = [np.ones(len(frame)), *[frame[c].to_numpy() for c in numeric]]
    for feature in ["region_code", *(["municipal_district_type"] if with_type else [])]:
        values = frame[feature].astype(str)
        for level in sorted(values.unique())[1:]:
            columns.append(values.eq(level).to_numpy(dtype=float))
    x, y = np.column_stack(columns), frame[target].to_numpy()
    # SVD on X itself gives an independent, numerically safer sandwich calculation.
    beta, _, rank, _ = linalg.lstsq(x, y, lapack_driver="gelsd")
    inv_x = linalg.pinv(x)
    residual = y - x @ beta
    groups = frame.region_code.to_numpy()
    cluster_influence = np.column_stack(
        [inv_x[:, groups == region] @ residual[groups == region] for region in sorted(set(groups))]
    )
    g, n = len(set(groups)), len(frame)
    cov = cluster_influence @ cluster_influence.T * g / (g - 1) * (n - 1) / (n - rank)
    se = np.sqrt(np.maximum(0, cov.diagonal()))
    crit = stats.t.ppf(0.975, g - 1)
    return pd.DataFrame(
        {
            "variable": numeric,
            "coefficient": beta[1 : len(numeric) + 1],
            "se_region_clustered": se[1 : len(numeric) + 1],
            "ci_low": (beta - crit * se)[1 : len(numeric) + 1],
            "ci_high": (beta + crit * se)[1 : len(numeric) + 1],
            "n": n,
            "regions": g,
            "rank": rank,
        }
    )


def main():
    manifest = json.loads((C / "manifest.json").read_text())
    verify("run_complete", manifest["status"] == "complete")
    hashchecks = {name: sha(C / name) == digest for name, digest in manifest["outputs"].items()}
    verify("all_output_hashes", all(hashchecks.values()), hashchecks)
    inputchecks = {
        name: sha(R / name) == digest for name, digest in manifest["input_hashes"].items()
    }
    verify("all_input_hashes", all(inputchecks.values()), inputchecks)
    verify(
        "code_hash",
        sha(R / "src/municipal_profiles/enriched_analysis.py") == manifest["code_sha256"],
    )
    verify(
        "config_hash", sha(R / "configs/enriched_analysis_v1.json") == manifest["protocol_sha256"]
    )
    panel = (
        duckdb.read_parquet(str(C / "annual_panel.parquet"))
        .df()
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    audit = read(A / "annual_masks_and_diagnostics.csv").sort_values(KEY).reset_index(drop=True)
    verify(
        "annual_grain",
        len(panel) == 4032
        and not panel.duplicated(KEY).any()
        and panel.territory_id.nunique() == 2016
        and panel.groupby("territory_id").year.nunique().eq(2).all()
        and set(panel.year) == {2023, 2024},
    )
    verify("audit_keys", panel[KEY].equals(audit[KEY]))
    verify(
        "same_id_population_each_year",
        set(panel[panel.year == 2023].territory_id) == set(panel[panel.year == 2024].territory_id),
    )
    orig = duckdb.read_parquet(str(R / "reports/initial_study/assignments.parquet")).df()
    verify("same_cohort_as_initial_study", set(orig.territory_id) == set(panel.territory_id))
    raw = duckdb.read_parquet(str(R / "data/raw/consumption.parquet")).df()
    verify("raw_unique_keys", not raw.duplicated(["territory_id", "date", "category"]).any())
    expected_dates = {f"{year}-{month:02d}" for year in [2023, 2024] for month in range(1, 13)}
    complete_ids = raw.groupby("territory_id").date.agg(set).map(lambda d: d == expected_dates)
    verify("complete_raw_cohort", set(complete_ids[complete_ids].index) == set(panel.territory_id))
    raw = raw[raw.territory_id.isin(panel.territory_id)]
    verify(
        "all_six_categories_every_month",
        raw.groupby(["territory_id", "date"])
        .category.agg(set)
        .map(lambda x: x == set(CATS + ["Все категории"]))
        .all(),
    )
    rawwide = raw.pivot(index=["territory_id", "date"], columns="category", values="value")
    normalized = rawwide[CATS].to_numpy() / rawwide[CATS].sum(axis=1).to_numpy()[:, None]
    monthly = pd.DataFrame(normalized, index=rawwide.index, columns=P).reset_index()
    monthly["total_mean"] = rawwide["Все категории"].to_numpy()
    monthly["year"] = monthly.date.str[:4].astype(int)
    means = monthly.groupby(KEY)[[*P, "total_mean"]].mean().reset_index().sort_values(KEY)
    verify(
        "twelve_months_each_year",
        monthly.groupby(KEY).size().eq(12).all() and panel.observed_months.eq(12).all(),
    )
    close("annual_profiles_mean_of_twelve_normalized_months", panel[P], means[P], atol=1e-14)
    close("annual_total_is_monthly_mean_not_sum", panel.total_mean, means.total_mean)
    close("annual_profile_sum_one", panel[P].sum(axis=1), np.ones(4032), atol=1e-14)
    for saved, audited in [
        ("population_total", "population"),
        ("population_urban", "urban"),
        ("workers", "employees"),
        ("wage", "salary"),
        ("market_access", "market_access"),
    ]:
        close("independent_source_join_" + saved, panel[saved], audit[audited])
    urban_ratio = audit.urban.div(audit.population).where(audit.urban_share_ok)
    close("urban_ratio_and_nulls", panel.urban_share, urban_ratio)
    verify(
        "exact_spatial_join",
        panel.official_oktmo_8.astype(str)
        .str.zfill(8)
        .equals(audit.official_oktmo_8.astype(str).str.zfill(8))
        and panel.region_code.equals(audit.region_code),
    )
    expected_core = audit.mask_core_expanded & ~audit.initial_calendar_flag & audit.employees.gt(0)
    verify(
        "core_mask_exact_rows",
        panel.core_eligible.equals(expected_core),
        panel.groupby("year").core_eligible.sum().to_dict(),
    )
    verify(
        "conservative_mask_exact_rows",
        panel.conservative_eligible.equals(audit.mask_core_strict & audit.employees.gt(0)),
        panel.groupby("year").conservative_eligible.sum().to_dict(),
    )
    verify(
        "calendar_flags_preserved_in_economy",
        panel.calendar_review.equals(audit.initial_calendar_flag),
        {"flagged_annual_rows": int(panel.calendar_review.sum())},
    )
    verify(
        "changed_codes_exact_ids",
        panel.code_type_changed.equals(audit.changed_code_or_type),
        {"ids": int(panel.loc[panel.code_type_changed].territory_id.nunique())},
    )
    verify(
        "market_not_carried_to_2023",
        panel.loc[
            panel.year == 2023,
            ["market_access", "log_market_access", "road_distance_to_sber_flagged_center_km"],
        ]
        .isna()
        .all()
        .all(),
    )
    for feature, value in [
        ("log_population", panel.population_total),
        ("log_wage", panel.wage),
        ("log_total", panel.total_mean),
        ("log_market_access", panel.market_access),
    ]:
        close(feature, panel[feature], np.log(value.where(value > 0)))
    close(
        "log_worker_concentration",
        panel.log_worker_concentration,
        np.log(
            panel.workers.where(panel.workers > 0)
            / panel.population_total.where(panel.population_total > 0)
        ),
    )
    # Both saved serialization formats must carry identical analytical values.
    csvpanel = read(C / "annual_panel.csv").sort_values(KEY)
    close(
        "csv_parquet_numeric_agreement",
        csvpanel[[*P, *XCORE, "total_mean", "urban_share", "market_access"]],
        panel[[*P, *XCORE, "total_mean", "urban_share", "market_access"]],
    )
    oof = duckdb.read_parquet(str(C / "oof_predictions.parquet")).df()
    metrics, comparisons = read(C / "cv_metrics.csv"), read(C / "cv_comparisons.csv")
    CVKEY = ["year", "sample", "regime", "model"]
    verify("oof_unique_keys", not oof.duplicated([*CVKEY, "territory_id"]).any())
    verify("metric_unique_keys", not metrics.duplicated([*CVKEY, "target"]).any())
    verify(
        "oof_finite",
        np.isfinite(oof[["actual_" + c for c in TARGETS] + ["predicted_" + c for c in TARGETS]])
        .all()
        .all(),
    )
    metric_errors, pred_errors, truth_errors, fold_details = [], [], [], []
    for key, group in oof.groupby(CVKEY, sort=True):
        year, sample, regime, model = key
        group = group.sort_values("territory_id").reset_index(drop=True)
        frame = panel[(panel.year == year) & panel.core_eligible]
        if sample == "stable_code":
            frame = frame[frame.conservative_eligible]
        if sample == "urban":
            frame = frame[frame.urban_share.notna()]
        if sample == "market":
            frame = frame[frame.log_market_access.notna()]
        frame = frame.sort_values("territory_id").reset_index(drop=True)
        assert frame.territory_id.equals(group.territory_id)
        profiles = frame[P].to_numpy()
        logs = np.log(profiles)
        truth = np.column_stack([frame.log_total.to_numpy(), logs - logs.mean(axis=1)[:, None]])
        y = group[["actual_" + c for c in TARGETS]].to_numpy()
        pred = group[["predicted_" + c for c in TARGETS]].to_numpy()
        truth_errors.append(np.max(np.abs(y - truth)))
        numeric = [] if model == "context" else XCORE.copy()
        if model == "economic_urban":
            numeric += ["urban_share"]
        if model == "economic_market":
            numeric += ["log_market_access"]
        folds = group.fold.to_numpy()
        regions = frame.region_code.to_numpy()
        if regime == "heldout_regions":
            expfold = np.empty(len(frame), dtype=int)
            for k, (_, test) in enumerate(GroupKFold(5).split(frame, groups=regions)):
                expfold[test] = k
            assert np.array_equal(folds, expfold)
            assert group.groupby("region_code").fold.nunique().eq(1).all()
        else:
            rng = np.random.default_rng(13)
            expfold = np.empty(len(frame), dtype=int)
            for region in sorted(set(regions)):
                rows = np.flatnonzero(regions == region)
                shuffled = rng.permutation(rows)
                offset = rng.integers(5)
                expfold[shuffled] = (np.arange(len(rows)) + offset) % 5
            assert np.array_equal(folds, expfold)
        for fold in range(5):
            test, train = folds == fold, folds != fold
            overlap = set(regions[train]) & set(regions[test])
            if regime == "heldout_regions":
                assert not overlap
            x, z = independent_design(frame.loc[train], frame.loc[test], numeric, regime)
            rebuilt = independent_ridge(x, truth[train], z)
            pred_errors.append(np.max(np.abs(pred[test] - rebuilt)))
            fold_details.append(
                {
                    "year": int(year),
                    "sample": sample,
                    "regime": regime,
                    "model": model,
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "region_overlap_count": len(overlap),
                }
            )
        for target, cols in [("log_total", [0]), ("profile_clr", list(range(1, 6)))]:
            a, b = y[:, cols], pred[:, cols]
            sse = np.square(a - b).sum()
            mse = sse / a.size
            r2 = 1 - sse / np.square(a - a.mean(axis=0)).sum()
            m = metrics[
                (metrics.year == year)
                & (metrics["sample"] == sample)
                & (metrics.regime == regime)
                & (metrics.model == model)
                & (metrics.target == target)
            ]
            assert (
                len(m) == 1
                and m.iloc[0]["n"] == len(group)
                and m.iloc[0].regions == frame.region_code.nunique()
            )
            metric_errors += [abs(m.iloc[0].r2_oof - r2), abs(m.iloc[0].mse_oof - mse)]
    verify(
        "all_oof_targets_rebuilt",
        max(truth_errors) < 1e-12,
        {"max_abs_error": float(max(truth_errors))},
    )
    verify(
        "all_ridge_predictions_refit_independently",
        max(pred_errors) < 1e-9,
        {
            "model_folds": len(pred_errors),
            "max_abs_error": float(max(pred_errors)),
            "solver": "centered_normal_equations_alpha1",
            "scaling": "train_only_population_sd",
            "dummy_levels": "train_only",
        },
    )
    verify(
        "all_cv_scores_recalculated",
        max(metric_errors) < 1e-12,
        {
            "metric_rows": len(metrics),
            "max_abs_error": float(max(metric_errors)),
            "profile_r2": "sum_coordinate_SSE_over_sum_coordinate_SST_not_average_of_coordinate_R2",
        },
    )
    foldtable = pd.DataFrame(fold_details)
    foldtable.to_csv(OUT / "core_validation_folds.csv", index=False)
    verify(
        "heldout_regions_disjoint",
        foldtable.loc[foldtable.regime == "heldout_regions", "region_overlap_count"].eq(0).all(),
    )
    verify(
        "same_comparison_folds",
        oof.groupby(["year", "sample", "regime", "territory_id"]).fold.nunique().eq(1).all(),
    )
    boot_errors = []
    for row in comparisons.itertuples(index=False):
        keys = (oof.year == row.year) & (oof["sample"] == row.sample) & (oof.regime == row.regime)
        a = oof[keys & (oof.model == row.baseline)].sort_values("territory_id")
        b = oof[keys & (oof.model == row.alternative)].sort_values("territory_id")
        assert np.array_equal(a.territory_id.to_numpy(), b.territory_id.to_numpy())
        targets = ["log_total"] if row.target == "log_total" else TARGETS[1:]
        y = a[["actual_" + c for c in targets]].to_numpy()
        ya, yb = (
            a[["predicted_" + c for c in targets]].to_numpy(),
            b[["predicted_" + c for c in targets]].to_numpy(),
        )
        losses = (
            pd.DataFrame(
                {
                    "region": a.region_code.to_numpy(),
                    "a": np.square(y - ya).sum(axis=1),
                    "b": np.square(y - yb).sum(axis=1),
                }
            )
            .groupby("region")[["a", "b"]]
            .sum()
            .to_numpy()
        )
        estimate = 1 - losses[:, 1].sum() / losses[:, 0].sum()
        rng = np.random.default_rng(13)
        draws = rng.integers(0, len(losses), size=(2000, len(losses)))
        totals = losses[draws].sum(axis=1)
        ci = np.quantile(1 - totals[:, 1] / totals[:, 0], [0.025, 0.975])
        boot_errors += [
            abs(row.relative_mse_reduction - estimate),
            abs(row.ci_low - ci[0]),
            abs(row.ci_high - ci[1]),
        ]
        assert row.bootstrap_scope == "fixed_OOF_predictions_paired_region_resampling"
    verify(
        "all_fixed_prediction_region_CIs_recalculated",
        max(boot_errors) < 1e-12,
        {
            "comparison_rows": len(comparisons),
            "repeats": 2000,
            "seed": 13,
            "max_abs_error": float(max(boot_errors)),
            "refitting_inside_bootstrap": False,
        },
    )
    savedcoef = read(C / "regression_coefficients.csv", dtype={"year": str})
    ols_error = []
    for year in [2023, 2024]:
        for sample, mask in [
            ("core", panel.core_eligible),
            ("stable_code", panel.conservative_eligible),
        ]:
            frame = panel[(panel.year == year) & mask]
            rebuilt = independent_ols(frame, "log_total", XCORE, True)
            actual = savedcoef[
                (savedcoef.year == str(year)) & (savedcoef["sample"] == sample)
            ].reset_index(drop=True)
            assert actual.variable.equals(rebuilt.variable)
            cols = [
                "coefficient",
                "se_region_clustered",
                "ci_low",
                "ci_high",
                "n",
                "regions",
                "rank",
            ]
            ols_error += np.abs(actual[cols].to_numpy() - rebuilt[cols].to_numpy()).ravel().tolist()
    before = panel[(panel.year == 2023) & panel.conservative_eligible].set_index("territory_id")
    after = panel[(panel.year == 2024) & panel.conservative_eligible].set_index("territory_id")
    common = before.index.intersection(after.index)
    delta = after.loc[common, ["region_code", "municipal_district_type"]].copy()
    for c in ["log_total", *XCORE]:
        delta["delta_" + c] = after.loc[common, c] - before.loc[common, c]
    saveddelta = read(C / "stable_code_changes.csv").set_index("territory_id").sort_index()
    delta = delta.sort_index()
    verify(
        "stable_pairs_exact_keys", saveddelta.index.equals(delta.index), {"paired_ids": len(common)}
    )
    cols = ["delta_" + c for c in ["log_total", *XCORE]]
    close("stable_pair_log_changes", saveddelta[cols], delta[cols])
    rebuilt = independent_ols(delta, "delta_log_total", ["delta_" + c for c in XCORE], False)
    actual = savedcoef[savedcoef.year == "2024_vs_2023"].reset_index(drop=True)
    cols = ["coefficient", "se_region_clustered", "ci_low", "ci_high", "n", "regions", "rank"]
    ols_error += np.abs(actual[cols].to_numpy() - rebuilt[cols].to_numpy()).ravel().tolist()
    verify(
        "all_OLS_coefficients_cluster_SE_and_CI_recalculated",
        max(ols_error) < 1e-8,
        {
            "coefficient_rows": len(savedcoef),
            "max_abs_error": float(max(ols_error)),
            "covariance": "CR1_with_actual_matrix_rank_and_t_G_minus_1",
            "independent_solver": "scipy_lstsq_and_pinv_of_X",
        },
    )
    corr = read(C / "correlations.csv")
    corr_errors = []
    for row in corr.itertuples(index=False):
        frame = panel[(panel.year == row.year) & ~panel.calendar_review].dropna(
            subset=[row.feature, row.outcome]
        )
        a, b = stats.rankdata(frame[row.feature]), stats.rankdata(frame[row.outcome])
        rankframe = pd.DataFrame({"a": a, "b": b, "region": frame.region_code.to_numpy()})
        within = rankframe[["a", "b"]] - rankframe.groupby("region")[["a", "b"]].transform("mean")
        corr_errors += [
            abs(np.corrcoef(a, b)[0, 1] - row.spearman),
            abs(np.corrcoef(within.a, within.b)[0, 1] - row.region_centered_rank_correlation),
        ]
        assert row.n == len(frame) and row.regions == frame.region_code.nunique()
    verify(
        "all_correlations_recalculated",
        max(corr_errors) < 1e-12,
        {"rows": len(corr), "max_abs_error": float(max(corr_errors))},
    )
    coverage = (
        panel.groupby("year")
        .agg(
            n=("territory_id", "size"),
            core=("core_eligible", "sum"),
            conservative=("conservative_eligible", "sum"),
            population=("population_total", "count"),
            urban=("urban_share", "count"),
            wages=("wage", "count"),
            workers=("workers", "count"),
            market=("market_access", "count"),
        )
        .reset_index()
    )
    verify("coverage_table_recalculated", coverage.equals(read(C / "coverage.csv")))
    coverage.to_csv(OUT / "core_validation_coverage.csv", index=False)
    return {
        "status": "PASS_with_interpretation_conditions",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validated_run": str(C.relative_to(R)) if C.is_relative_to(R) else str(C),
        "run_manifest_sha256": sha(C / "manifest.json"),
        "validator_sha256": sha(Path(__file__)),
        "checks": checks,
        "counts": {
            "annual_rows": len(panel),
            "municipalities": panel.territory_id.nunique(),
            "oof_rows": len(oof),
            "cv_metrics": len(metrics),
            "cv_comparisons": len(comparisons),
            "OLS_rows": len(savedcoef),
            "stable_pairs": len(common),
        },
        "core_cv_metrics": metrics[metrics["sample"] == "core"].to_dict("records"),
        "added_feature_comparisons": comparisons[
            comparisons.alternative.isin(["economic_urban", "economic_market"])
        ].to_dict("records"),
        "interpretation_conditions": [
            (
                "within_region baseline is region plus municipal type; heldout_regions "
                "baseline is municipal type only"
            ),
            (
                "all cross-validation is municipality/region resampling within "
                "observed year, not future time validation"
            ),
            (
                "bootstrap confidence intervals condition on fixed OOF predictions and "
                "do not include model refitting uncertainty"
            ),
            (
                "employees divided by January resident population is a workplace "
                "concentration proxy, not employment rate"
            ),
            "cohort selected retrospectively from complete 2023-2024 observations",
            (
                "geographic joins remain conditional on source codes and annual "
                "versions; no independent boundary proof"
            ),
            (
                "correlations use pairwise available samples, whereas model "
                "comparisons use the same complete sample within each scenario"
            ),
            "no new demography or secondary candidates enter the validated model",
        ],
        "new_external_requests": 0,
        "source_or_pipeline_modifications": 0,
    }


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        try:
            result = main()
        except Exception as exc:
            result = {
                "status": "FAIL",
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "checks": checks,
                "error": str(exc),
                "type": type(exc).__name__,
            }
            (OUT / "core_validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise
    (OUT / "core_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": result["status"], "counts": result["counts"], "checks": len(checks)},
            ensure_ascii=False,
        )
    )
