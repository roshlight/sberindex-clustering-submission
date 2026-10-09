"""Independent read-only validation of fourth-round monetary recovery experiments.

No production fitting, preprocessing, target or bootstrap functions are imported.
Reference Ridge predictions solve centered normal equations directly with NumPy.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

KEY = ["territory_id", "year"]
PROFILE = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
TARGETS = ["log_total", *["clr_" + c for c in PROFILE]]
ACTUAL = ["actual_" + c for c in TARGETS]
PREDICTED = ["predicted_" + c for c in TARGETS]
CORE = ["log_population", "log_wage", "log_worker_concentration"]
SECTORS = [f"sector_{s}_share" for s in "ABCDHOPQ"]
FLAGS = [f"missing_{s}" for s in "ABCDHOPQ"]
SPECS = {
    "investment": ("8109001", "17"),
    "shipments": ("8401011", "79"),
    "retail": ("8401003", "79"),
}
NEW = [f"{p}_log_rubles_per_person" for p in SPECS]
NEW_FLAGS = [f"{p}_missing" for p in SPECS]
RICH = CORE + FLAGS + SECTORS
OLD = ["old_" + c for c in NEW]
OLD_FLAGS = ["old_" + c for c in NEW_FLAGS]
MODELS = {
    "rich_baseline": RICH,
    "new_flags": RICH + NEW_FLAGS,
    "recovered_full": RICH + NEW_FLAGS + NEW,
    "legacy_flags": RICH + OLD_FLAGS,
    "legacy_full": RICH + OLD_FLAGS + OLD,
    "dual_flags_old_values": RICH + OLD_FLAGS + NEW_FLAGS + OLD,
    "dual_flags_new_values": RICH + OLD_FLAGS + NEW_FLAGS + NEW,
}
PAIRS = {
    "primary": ("new_flags", "recovered_full"),
    "state_update": ("legacy_full", "recovered_full"),
    "shared_flags_recovery": ("dual_flags_old_values", "dual_flags_new_values"),
    "missingness": ("rich_baseline", "new_flags"),
    "whole_block": ("rich_baseline", "recovered_full"),
    "legacy_increment": ("legacy_flags", "legacy_full"),
}
SCENARIO = ["year", "sample", "regime", "seed"]
GROUP = SCENARIO + ["model"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    good = np.isfinite(a)
    return float(np.max(np.abs(a[good] - b[good]))) if good.any() else 0.0


def select(frame, **conditions):
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    return frame.loc[mask]


def reference_design(train, test, numeric, region):
    """Train-only category vocabulary, imputation and population SD."""
    left, right, names = [], [], []
    for field in ["municipal_district_type"] + (["region_code"] if region else []):
        a, b = train[field].astype(str), test[field].astype(str)
        for value in sorted(a.unique()):
            left.append(a.eq(value).to_numpy(dtype=float))
            right.append(b.eq(value).to_numpy(dtype=float))
            names.append(f"{field}_{value}")
    a, b = train[numeric].to_numpy(float), test[numeric].to_numpy(float)
    empty = np.isnan(a).all(axis=0)
    medians = np.array(
        [0.0 if empty[i] else np.median(a[np.isfinite(a[:, i]), i]) for i in range(a.shape[1])]
    )
    a, b = np.where(np.isnan(a), medians, a), np.where(np.isnan(b), medians, b)
    means, sd = a.mean(axis=0), a.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    left.extend(((a - means) / sd).T)
    right.extend(((b - means) / sd).T)
    audit = dict(
        medians=dict(zip(numeric, medians, strict=True)),
        means=dict(zip(numeric, means, strict=True)),
        sd=dict(zip(numeric, sd, strict=True)),
        dummy_columns=names,
        train_empty_columns=[c for c, v in zip(numeric, empty, strict=True) if v],
    )
    return np.column_stack(left), np.column_stack(right), audit


def reference_ridge(x, y, z):
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    centered = x - xm
    beta = np.linalg.solve(centered.T @ centered + np.eye(x.shape[1]), centered.T @ (y - ym))
    return (z - xm) @ beta + ym


def reference_shuffle(frame, seed):
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for column in NEW:
        for region in sorted(frame.region_code.unique()):
            mask = frame.region_code.eq(region) & frame[column].notna()
            result.loc[mask, column] = rng.permutation(frame.loc[mask, column].to_numpy())
    assert result[NEW].isna().equals(frame[NEW].isna())
    pd.testing.assert_frame_equal(result[NEW_FLAGS], frame[NEW_FLAGS])
    return result


def bootstrap(y, first, second, regions):
    """Paired region resampling conditional on the same fixed OOF predictions."""
    losses = np.column_stack([np.square(y - first).sum(axis=1), np.square(y - second).sum(axis=1)])
    unique, inverse = np.unique(regions, return_inverse=True)
    totals = np.zeros((len(unique), 2))
    np.add.at(totals, inverse, losses)
    rng = np.random.default_rng(13)
    drawn = totals[rng.integers(0, len(unique), size=(2000, len(unique)))].sum(axis=1)
    interval = np.percentile(1 - drawn[:, 1] / drawn[:, 0], [2.5, 97.5])
    return np.r_[1 - totals[:, 1].sum() / totals[:, 0].sum(), interval]


def run(repo, record):
    def check(name, condition, **details):
        record["checks"].append(dict(check=name, status="PASS" if condition else "FAIL", **details))
        if not condition:
            raise AssertionError(name)

    directory = repo / "reports/round4_analysis/models"
    cfg = json.loads((repo / "configs/round4_analysis_v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "protocol_lock.json").read_text(encoding="utf-8"))
    check(
        "manifest_hashes",
        all(sha(repo / p) == v for p, v in manifest["inputs"].items())
        and all(sha(directory / p) == v for p, v in manifest["outputs"].items())
        and sha(repo / "scripts/round4_model_analysis.py") == manifest["code_sha256"],
    )
    check(
        "protocol_lock",
        lock["protocol_sha256"] == sha(repo / "configs/round4_analysis_v1.json")
        and lock["model_code_sha256"] == manifest["code_sha256"]
        and lock["started_at"] == manifest["started_at"]
        and lock["source_base_commit"] == cfg["base_commit"],
    )
    check(
        "specified_protocol",
        cfg["years"] == [2023, 2024]
        and cfg["primary_seed"] == 13
        and cfg["within_region_core_seeds"] == [13, 29, 47]
        and cfg["samples"]
        == ["core", "stable_code", "common_complete_new3", "expanded_complete_new3"]
        and cfg["regimes"] == ["within_region", "heldout_regions"]
        and cfg["primary_comparison"] == ["new_flags", "recovered_full"]
        and cfg["paired_flags_recovery_comparison"]
        == ["dual_flags_old_values", "dual_flags_new_values"]
        and manifest["models"] == MODELS
        and not manifest["tuning_performed"],
    )

    def read(path):
        return pd.read_csv(path).sort_values(KEY).reset_index(drop=True)

    panel = read(directory / "model_panel.csv")
    old = read(repo / "reports/round3_analysis/models/model_panel.csv")
    frozen = read(repo / "reports/enriched_analysis/core_v2/annual_panel.csv")
    features = read(repo / "reports/round4_analysis/audit/annual_features.csv")
    for label, frame in [
        ("panel", panel),
        ("old", old),
        ("frozen", frozen),
        ("features", features),
    ]:
        check(
            label + "_unique_keys",
            len(frame) == 4032
            and not frame.duplicated(KEY).any()
            and frame[KEY].equals(frozen[KEY]),
        )
    retained = [c for c in old if c not in NEW + NEW_FLAGS + ["new_observed_count"]]
    pd.testing.assert_frame_equal(panel[retained], old[retained], rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(features[frozen.columns], frozen, rtol=1e-13, atol=1e-13)
    check("frozen_values_eligibility_and_geography", True)
    worst = dict(
        feature=0.0,
        truth=0.0,
        metric=0.0,
        bootstrap=0.0,
        prior_prediction=0.0,
        common_invariance=0.0,
        ridge=0.0,
        preprocessing=0.0,
        slices=0.0,
    )
    data_inputs = {}
    for vintage, subdir in [("new", "data_round4"), ("old", "data_round3")]:
        path = repo / f"reports/{subdir}/economy/annual_candidate_panel.csv"
        data_inputs[str(path.relative_to(repo))] = sha(path)
        source = pd.read_csv(
            path, dtype={"indicator_id": str, "period_code": str, "activity_code": str}
        )
        check(
            vintage + "_source_grid",
            len(source) == 12096 and not source.duplicated(KEY + ["indicator_id"]).any(),
        )
        for prefix, (indicator, period) in SPECS.items():
            rows = source.loc[source.indicator_id.eq(indicator)].sort_values(KEY)
            rows = rows.reset_index(drop=True)
            sourced = rows.source_sha256.notna()
            check(
                vintage + "_" + prefix + "_source_semantics",
                rows[KEY].equals(frozen[KEY])
                and rows.official_oktmo_8.equals(frozen.official_oktmo_8)
                and rows.code_reference_date.eq(rows.year.astype(str) + "-01-01").all()
                and rows.loc[sourced, "unit"].eq("thousand_rubles").all()
                and rows.loc[sourced, "period_code"].eq(period).all()
                and rows.loc[sourced, "activity_code"].eq("101").all(),
            )
            raw, pop = rows.value.to_numpy(float), frozen.population_total.to_numpy(float)
            valid = (
                rows.value_status.eq("observed").to_numpy()
                & np.isfinite(raw)
                & (raw > 0)
                & np.isfinite(pop)
                & (pop > 0)
            )
            expected = np.full(len(rows), np.nan)
            expected[valid] = np.log(raw[valid] * 1000.0 / pop[valid])
            stem = ("old_" if vintage == "old" else "") + prefix
            field, flag = stem + "_log_rubles_per_person", stem + "_missing"
            worst["feature"] = max(worst["feature"], error(panel[field], expected))
            check(
                vintage + "_" + prefix + "_feature_flags",
                panel.loc[panel.core_eligible, flag].eq(~valid[panel.core_eligible]).all(),
            )
            if vintage == "old":
                worst["feature"] = max(
                    worst["feature"], error(panel[field], old[prefix + "_log_rubles_per_person"])
                )
            else:
                worst["feature"] = max(
                    worst["feature"],
                    error(features[field], expected),
                    error(features[prefix + "_thousand_rubles"], raw),
                )
                check(
                    prefix + "_audit_source_status",
                    features[prefix + "_value_status"].equals(rows.value_status),
                )
    old_count = panel[OLD].notna().sum(axis=1)
    new_count = panel[NEW].notna().sum(axis=1)
    check(
        "availability_and_complete_sample_membership",
        panel.old_observed_count.equals(old_count)
        and panel.new_observed_count.equals(new_count)
        and panel.recovered_feature_count.eq(new_count - old_count).all()
        and panel.recovered_any.eq(new_count.gt(old_count)).all()
        and panel.common_complete_new3.eq(old_count.eq(3) & new_count.eq(3)).all()
        and panel.expanded_complete_new3.eq(new_count.eq(3)).all()
        and (new_count >= old_count).all(),
    )
    for current, previous in zip(NEW, OLD, strict=True):
        check(
            current + "_no_revision",
            error(panel.loc[panel[previous].notna(), current], panel[previous].dropna()) < 1e-12,
        )

    oof = duckdb.read_parquet(str(directory / "oof_predictions.parquet")).df()
    prior = duckdb.read_parquet(
        str(repo / "reports/round3_analysis/models/oof_predictions.parquet")
    ).df()
    metrics = pd.read_csv(directory / "cv_metrics.csv")
    comparisons = pd.read_csv(directory / "cv_comparisons.csv")
    slices = pd.read_csv(directory / "slice_losses.csv", dtype={"group": str})
    preprocessing = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    prep = {tuple(row[c] for c in GROUP + ["fold"]): row for row in preprocessing}
    comparison_key = SCENARIO + ["comparison", "baseline", "alternative", "target"]
    slice_key = SCENARIO + ["comparison", "target", "slice", "group"]
    check(
        "unique_result_keys",
        not oof.duplicated(GROUP + ["territory_id"]).any()
        and not metrics.duplicated(GROUP + ["target"]).any()
        and not comparisons.duplicated(comparison_key).any()
        and not slices.duplicated(slice_key).any()
        and len(prep) == len(preprocessing),
    )
    check("finite_predictions_and_truth", np.isfinite(oof[ACTUAL + PREDICTED]).all().all())
    expected_groups, expected_metrics, expected_pairs, expected_slices, expected_prep = (
        set() for _ in range(5)
    )
    counts, fit_count, prior_checks, common_checks = [], 0, 0, 0
    for year in [2023, 2024]:
        core = panel.loc[panel.year.eq(year) & panel.core_eligible]
        samples = {
            "core": core,
            "stable_code": core.loc[core.conservative_eligible],
            "common_complete_new3": core.loc[old_count.eq(3) & new_count.eq(3)],
            "expanded_complete_new3": core.loc[new_count.eq(3)],
        }
        check(
            f"frozen_population_{year}",
            len(core) == cfg["core_n"][str(year)]
            and len(samples["stable_code"]) == cfg["stable_n"][str(year)],
        )
        for sample, frame in samples.items():
            d = frame.sort_values(KEY).reset_index(drop=True)
            counts.append(
                dict(year=year, sample=sample, n=len(d), regions=int(d.region_code.nunique()))
            )
            logp = np.log(d[PROFILE].to_numpy(float))
            truth = np.column_stack([d.log_total, logp - logp.mean(axis=1, keepdims=True)])
            for regime in ["within_region", "heldout_regions"]:
                for seed in (
                    [13, 29, 47] if sample == "core" and regime == "within_region" else [13]
                ):
                    context = dict(year=year, sample=sample, regime=regime, seed=seed)
                    scenario = tuple(context.values())
                    models = (
                        dict(MODELS)
                        if sample in {"core", "stable_code"}
                        else {
                            name: MODELS[name]
                            for name in [
                                "rich_baseline",
                                "new_flags",
                                "recovered_full",
                                "legacy_full",
                            ]
                        }
                    )
                    pairs = {
                        name: pair
                        for name, pair in PAIRS.items()
                        if all(model in models for model in pair)
                    }
                    if sample == "core" and seed == 13:
                        for permutation in [91013, 91029, 91047]:
                            name = f"permuted_{permutation}"
                            models[name] = MODELS["recovered_full"]
                            pairs[name] = ("new_flags", name)
                        for prefix, field in zip(SPECS, NEW, strict=True):
                            name = "without_" + prefix
                            models[name] = [c for c in MODELS["recovered_full"] if c != field]
                            pairs[name] = (name, "recovered_full")
                    old_sample = "stable_code" if sample == "stable_code" else "core"
                    old_rows = select(
                        prior,
                        year=year,
                        sample=old_sample,
                        regime=regime,
                        seed=seed,
                        model="economic_sectors_newvalues",
                    )
                    old_rows = d[KEY].merge(old_rows, on=KEY, how="left", validate="one_to_one")
                    folds = old_rows.fold.to_numpy(int)
                    check(
                        "retained_prior_fold_coverage",
                        set(folds) == set(range(5)) and old_rows[PREDICTED].notna().all().all(),
                        scenario=context,
                    )
                    predictions = {}
                    for model, numeric in models.items():
                        gk = (*scenario, model)
                        expected_groups.add(gk)
                        rows = select(oof, **context, model=model).sort_values(KEY)
                        rows = rows.reset_index(drop=True)
                        check(
                            "oof_population_folds",
                            rows[KEY].equals(d[KEY])
                            and rows.region_code.equals(d.region_code)
                            and np.array_equal(rows.fold, folds),
                            group=list(gk),
                        )
                        for info in [
                            "recovered_any",
                            "recovered_feature_count",
                            "old_observed_count",
                            "new_observed_count",
                        ]:
                            check("oof_annotation_" + info, rows[info].equals(d[info]))
                        predicted = rows[PREDICTED].to_numpy(float)
                        predictions[model] = predicted
                        worst["truth"] = max(worst["truth"], error(rows[ACTUAL], truth))
                        if sample in {"core", "stable_code"} and model == "legacy_full":
                            worst["prior_prediction"] = max(
                                worst["prior_prediction"], error(predicted, old_rows[PREDICTED])
                            )
                            prior_checks += 1
                        for fold in range(5):
                            train_mask, test_mask = folds != fold, folds == fold
                            train, test = d.loc[train_mask], d.loc[test_mask]
                            if regime == "heldout_regions":
                                check(
                                    "regional_disjointness",
                                    not set(train.region_code) & set(test.region_code),
                                )
                            permutation = (
                                int(model.split("_")[1]) if model.startswith("permuted_") else None
                            )
                            if permutation is not None:
                                train = reference_shuffle(train, permutation)
                                test = reference_shuffle(test, permutation)
                            x, z, parameters = reference_design(
                                train, test, numeric, regime == "within_region"
                            )
                            pk = (*gk, fold)
                            expected_prep.add(pk)
                            logged = prep[pk]
                            check(
                                "train_only_preprocessing_schema",
                                logged["dummy_columns"] == parameters["dummy_columns"]
                                and logged["train_empty_columns"]
                                == parameters["train_empty_columns"]
                                and logged["train_n"] == len(train)
                                and logged["test_n"] == len(test)
                                and logged["numeric"] == numeric
                                and logged["permutation_seed"] == permutation,
                            )
                            for field in ["medians", "means", "sd"]:
                                check(
                                    "preprocessing_feature_names",
                                    set(logged[field]) == set(numeric),
                                )
                                worst["preprocessing"] = max(
                                    worst["preprocessing"],
                                    error(
                                        [logged[field][c] for c in numeric],
                                        [parameters[field][c] for c in numeric],
                                    ),
                                )
                            reference = reference_ridge(x, truth[train_mask], z)
                            worst["ridge"] = max(
                                worst["ridge"], error(predicted[test_mask], reference)
                            )
                            fit_count += 1
                        for target, columns in [
                            ("log_total", [0]),
                            ("profile_clr", [1, 2, 3, 4, 5]),
                        ]:
                            expected_metrics.add((*gk, target))
                            saved = select(metrics, **context, model=model, target=target)
                            check(
                                "metric_row_counts",
                                len(saved) == 1
                                and saved.iloc[0].n == len(d)
                                and saved.iloc[0].regions == d.region_code.nunique(),
                            )
                            y, p = truth[:, columns], predicted[:, columns]
                            expected = [
                                np.square(y - p).mean(),
                                1 - np.square(y - p).sum() / np.square(y - y.mean(axis=0)).sum(),
                            ]
                            worst["metric"] = max(
                                worst["metric"],
                                error(saved[["mse_oof", "r2_oof"]].iloc[0], expected),
                            )
                    if sample == "common_complete_new3":
                        worst["common_invariance"] = max(
                            worst["common_invariance"],
                            error(predictions["legacy_full"], predictions["recovered_full"]),
                        )
                        common_checks += 1
                    for comparison, (baseline, alternative) in pairs.items():
                        for target, columns in [
                            ("log_total", [0]),
                            ("profile_clr", [1, 2, 3, 4, 5]),
                        ]:
                            expected_pairs.add(
                                (*scenario, comparison, baseline, alternative, target)
                            )
                            saved = select(
                                comparisons,
                                **context,
                                comparison=comparison,
                                baseline=baseline,
                                alternative=alternative,
                                target=target,
                            )
                            check(
                                "comparison_row_counts",
                                len(saved) == 1
                                and saved.iloc[0].n == len(d)
                                and saved.iloc[0].regions == d.region_code.nunique(),
                            )
                            expected = bootstrap(
                                truth[:, columns],
                                predictions[baseline][:, columns],
                                predictions[alternative][:, columns],
                                d.region_code.to_numpy(),
                            )
                            worst["bootstrap"] = max(
                                worst["bootstrap"],
                                error(
                                    saved[["relative_mse_reduction", "ci_low", "ci_high"]].iloc[0],
                                    expected,
                                ),
                            )
                            if (
                                sample != "core"
                                or seed != 13
                                or comparison
                                not in {"primary", "state_update", "shared_flags_recovery"}
                            ):
                                continue
                            for scope in [
                                "recovered_any",
                                "region_code",
                                "municipal_district_type",
                                "population_quartile",
                                "new_observed_count",
                            ]:
                                for value in sorted(d[scope].unique()):
                                    mask = d[scope].eq(value).to_numpy()
                                    expected_slices.add(
                                        (*scenario, comparison, target, scope, str(value))
                                    )
                                    part = select(
                                        slices,
                                        **context,
                                        comparison=comparison,
                                        target=target,
                                        slice=scope,
                                        group=str(value),
                                    )
                                    check(
                                        "slice_row_counts",
                                        len(part) == 1
                                        and part.iloc[0].n == mask.sum()
                                        and part.iloc[0].regions
                                        == d.loc[mask, "region_code"].nunique(),
                                    )
                                    a = np.square(
                                        truth[mask][:, columns]
                                        - predictions[baseline][mask][:, columns]
                                    ).sum()
                                    b = np.square(
                                        truth[mask][:, columns]
                                        - predictions[alternative][mask][:, columns]
                                    ).sum()
                                    expected = [a, b, 1 - b / a if a else np.nan]
                                    worst["slices"] = max(
                                        worst["slices"],
                                        error(
                                            part[
                                                [
                                                    "baseline_sse",
                                                    "alternative_sse",
                                                    "relative_mse_reduction",
                                                ]
                                            ].iloc[0],
                                            expected,
                                        ),
                                    )
    check(
        "complete_model_grid", expected_groups == set(oof[GROUP].itertuples(index=False, name=None))
    )
    check(
        "complete_metric_grid",
        expected_metrics == set(metrics[GROUP + ["target"]].itertuples(index=False, name=None)),
    )
    check(
        "complete_comparison_grid",
        expected_pairs == set(comparisons[comparison_key].itertuples(index=False, name=None)),
    )
    check(
        "complete_slice_grid",
        expected_slices == set(slices[slice_key].itertuples(index=False, name=None)),
    )
    check(
        "complete_preprocessing_grid", expected_prep == set(prep) and fit_count == manifest["fits"]
    )
    for name, value in worst.items():
        check(
            "maximum_error_" + name,
            value < (1e-8 if name in {"ridge", "slices"} else 1e-10),
            maximum_absolute_error=value,
        )
    primary = select(
        comparisons, sample="core", seed=13, comparison="primary", target="profile_clr"
    )
    stable = select(
        comparisons, sample="stable_code", seed=13, comparison="primary", target="profile_clr"
    )
    adoption = (
        len(primary) == 4
        and len(stable) == 4
        and primary.relative_mse_reduction.ge(0.01).all()
        and primary.ci_low.gt(0).all()
        and stable.relative_mse_reduction.ge(0).all()
    )
    record.update(
        sample_counts=counts,
        reference_ridge_fits=fit_count,
        model_groups=len(expected_groups),
        prior_legacy_groups=prior_checks,
        common_invariance_groups=common_checks,
        metric_rows=len(metrics),
        comparison_rows=len(comparisons),
        slice_rows=len(slices),
        oof_rows=len(oof),
        maximum_absolute_errors=worst,
        independent_adoption_criteria_met=bool(adoption),
        primary_evidence=primary.to_dict("records"),
        stable_evidence=stable.to_dict("records"),
        independent_data_inputs=data_inputs,
        production_code_sha256=manifest["code_sha256"],
        protocol_sha256=sha(repo / "configs/round4_analysis_v1.json"),
        model_manifest_sha256=sha(directory / "manifest.json"),
        verifier_sha256=sha(__file__),
        inference_limits=[
            "Exploratory retrospective attribution, not forecasting",
            "Conditional fixed-OOF regional bootstrap; no pipeline refit",
            "No multiplicity adjustment; secondary results descriptive",
            "Separate train/test permutations are diagnostics, not p-values",
            "No proof of historical real-time feature availability",
            "Independent calculations do not independently validate source truth",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.repo / "reports/round4_analysis/independent_QA.json"
    record = dict(status="RUNNING", checked_at=datetime.now(UTC).isoformat(), checks=[])
    try:
        with threadpool_limits(limits=1):
            run(args.repo, record)
        record["status"] = "PASS"
    except Exception as exc:
        record.update(status="FAIL", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in record.items() if k != "checks"}))


if __name__ == "__main__":
    main()
