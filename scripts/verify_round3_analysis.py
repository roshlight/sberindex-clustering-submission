"""Independent offline validation of round-three features, OOF losses and Ridge fits.

No production model/feature functions are imported. Reference fits solve centered
normal equations with NumPy; source, model and input artifacts remain read-only.
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
MODELS = {
    "economic": CORE,
    "economic_newflags": CORE + NEW_FLAGS,
    "economic_newvalues": CORE + NEW_FLAGS + NEW,
    "economic_sectors": RICH,
    "economic_sectors_newflags": RICH + NEW_FLAGS,
    "economic_sectors_newvalues": RICH + NEW_FLAGS + NEW,
}
BASE = "economic_sectors_newflags"
FULL = "economic_sectors_newvalues"
SCENARIO = ["year", "sample", "regime", "seed"]
GROUP = SCENARIO + ["model"]
PAIRS = [
    ("economic", "economic_newflags"),
    ("economic_newflags", "economic_newvalues"),
    ("economic", "economic_newvalues"),
    ("economic_sectors", BASE),
    (BASE, FULL),
    ("economic_sectors", FULL),
]


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


def reference_design(train, test, numeric, region):
    """Explicit category comparisons, training medians and population SD."""
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
    a = x - xm
    beta = np.linalg.solve(a.T @ a + np.eye(x.shape[1]), a.T @ (y - ym))
    return (z - xm) @ beta + ym


def reference_shuffle(frame):
    result = frame.copy()
    rng = np.random.default_rng(91013)
    for column in NEW:
        for region in sorted(frame.region_code.unique()):
            mask = frame.region_code.eq(region) & frame[column].notna()
            result.loc[mask, column] = rng.permutation(frame.loc[mask, column].to_numpy())
    assert result[NEW].isna().equals(frame[NEW].isna())
    return result


def bootstrap(y, first, second, regions):
    losses = np.column_stack([np.square(y - first).sum(axis=1), np.square(y - second).sum(axis=1)])
    unique, inverse = np.unique(regions, return_inverse=True)
    totals = np.zeros((len(unique), 2))
    np.add.at(totals, inverse, losses)
    rng = np.random.default_rng(13)
    sampled = rng.integers(0, len(unique), size=(2000, len(unique)))
    drawn = totals[sampled].sum(axis=1)
    intervals = np.percentile(1 - drawn[:, 1] / drawn[:, 0], [2.5, 97.5])
    return np.r_[1 - totals[:, 1].sum() / totals[:, 0].sum(), intervals]


def select(frame, **conditions):
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    return frame.loc[mask]


def run(repo, record):
    def check(name, condition, **details):
        record["checks"].append(dict(check=name, status="PASS" if condition else "FAIL", **details))
        if not condition:
            raise AssertionError(name)

    directory = repo / "reports/round3_analysis/models"
    data = repo / "reports/data_round3/economy/annual_candidate_panel.csv"
    audit_path = repo / "reports/round3_analysis/audit/annual_features.csv"
    cfg = json.loads((repo / "configs/round3_analysis_v1.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    check(
        "manifest_hashes",
        all(sha(repo / p) == v for p, v in manifest["inputs"].items())
        and all(sha(directory / p) == v for p, v in manifest["outputs"].items())
        and sha(repo / "scripts/round3_model_analysis.py") == manifest["code_sha256"],
    )
    check(
        "specified_constants",
        cfg["years"] == [2023, 2024]
        and cfg["primary_seed"] == 13
        and cfg["within_region_core_sensitivity_seeds"] == [29, 47]
        and manifest["models"] == MODELS
        and not manifest["tuning_performed"]
        and not manifest["skipped_samples"],
    )
    panel = pd.read_csv(directory / "model_panel.csv").sort_values(KEY).reset_index(drop=True)
    old = (
        pd.read_csv(repo / "reports/round2_analysis/models/model_panel.csv")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    frozen = (
        pd.read_csv(
            repo / "reports/enriched_analysis/core_v2/annual_panel.csv",
            dtype={"official_oktmo_8": str},
        )
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    features = (
        pd.read_csv(
            audit_path,
            dtype={"official_oktmo_8": str, **{f"{p}_official_oktmo_8": str for p in SPECS}},
        )
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    source = pd.read_csv(
        data,
        dtype={
            "indicator_id": str,
            "official_oktmo_8": str,
            "period_code": str,
            "activity_code": str,
        },
    )
    for name, frame in [
        ("panel", panel),
        ("prior", old),
        ("frozen", frozen),
        ("features", features),
    ]:
        check(
            name + "_unique_keys",
            len(frame) == 4032
            and not frame.duplicated(KEY).any()
            and frame[KEY].equals(frozen[KEY]),
        )
    pd.testing.assert_frame_equal(
        panel[old.columns], old, check_exact=False, rtol=1e-13, atol=1e-13
    )
    pd.testing.assert_frame_equal(
        features[frozen.columns], frozen, check_exact=False, rtol=1e-13, atol=1e-13
    )
    check("frozen_values_and_eligibility", True)
    check(
        "source_full_grid",
        len(source) == 12096 and not source.duplicated(KEY + ["indicator_id"]).any(),
    )
    max_feature_error = 0.0
    for prefix, (indicator, period) in SPECS.items():
        s = source.loc[source.indicator_id.eq(indicator)].sort_values(KEY).reset_index(drop=True)
        check(
            prefix + "_source_semantics",
            s[KEY].equals(frozen[KEY])
            and s.official_oktmo_8.equals(frozen.official_oktmo_8)
            and s.code_reference_date.eq(s.year.astype(str) + "-01-01").all()
            and s.loc[s.source_sha256.notna(), "unit"].eq("thousand_rubles").all()
            and s.loc[s.source_sha256.notna(), "period_code"].eq(period).all()
            and s.loc[s.source_sha256.notna(), "activity_code"].eq("101").all(),
        )
        raw = s.value.to_numpy(float)
        pop = frozen.population_total.to_numpy(float)
        valid = (
            s.value_status.eq("observed").to_numpy()
            & np.isfinite(raw)
            & (raw > 0)
            & np.isfinite(pop)
            & (pop > 0)
        )
        expected = np.full(len(s), np.nan)
        expected[valid] = np.log(raw[valid] * 1000.0 / pop[valid])
        field = prefix + "_log_rubles_per_person"
        max_feature_error = max(
            max_feature_error, error(features[field], expected), error(panel[field], expected)
        )
        check(
            prefix + "_source_value_and_flags",
            error(features[prefix + "_thousand_rubles"], raw) < 1e-10
            and features[prefix + "_value_status"].equals(s.value_status)
            and features[prefix + "_missing"].eq(s.value.isna()).all()
            and panel.loc[panel.core_eligible, prefix + "_missing"]
            .eq(~valid[panel.core_eligible])
            .all(),
        )
        for field in [
            "source_sha256",
            "request_key",
            "source_oktmo",
            "source_region_url",
            "source_oktmo_name",
            "code_reference_date",
            "temporal_review_required",
            "retrieved_at",
            "source_csv_row",
        ]:
            left = features[prefix + "_" + field].fillna("").astype(str)
            right = s[field].fillna("").astype(str)
            check(prefix + "_provenance_" + field, left.equals(right))
    check(
        "all_new_log_features", max_feature_error < 1e-12, maximum_absolute_error=max_feature_error
    )
    oof = duckdb.read_parquet(str(directory / "oof_predictions.parquet")).df()
    prior = duckdb.read_parquet(
        str(repo / "reports/round2_analysis/models/oof_predictions.parquet")
    ).df()
    metrics = pd.read_csv(directory / "cv_metrics.csv")
    comparisons = pd.read_csv(directory / "cv_comparisons.csv")
    slices = pd.read_csv(directory / "slice_losses.csv")
    preprocess = json.loads((directory / "preprocessing.json").read_text())
    prep = {tuple(r[c] for c in GROUP + ["fold"]): r for r in preprocess}
    check(
        "unique_output_keys",
        not oof.duplicated(GROUP + ["territory_id"]).any()
        and not metrics.duplicated(GROUP + ["target"]).any()
        and not comparisons.duplicated(SCENARIO + ["baseline", "alternative", "target"]).any()
        and len(prep) == len(preprocess),
    )
    check("finite_predictions", np.isfinite(oof[ACTUAL + PREDICTED]).all().all())
    model_groups, metric_keys, comparison_keys, prep_keys = set(), set(), set(), set()
    worst = dict(
        truth=0.0,
        metric=0.0,
        bootstrap=0.0,
        prior_prediction=0.0,
        ridge=0.0,
        preprocessing=0.0,
        slices=0.0,
    )
    counts, prior_checks, fit_count = [], 0, 0
    for year in [2023, 2024]:
        core = panel.loc[panel.year.eq(year) & panel.core_eligible]
        samples = {
            "core": core,
            "stable_code": core.loc[core.conservative_eligible],
            "complete_new3": core.dropna(subset=NEW),
        }
        check(
            f"frozen_counts_{year}",
            len(core) == cfg["expected_core_n"][str(year)]
            and len(samples["stable_code"]) == cfg["expected_stable_n"][str(year)],
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
                    models = dict(MODELS)
                    pairs = list(PAIRS)
                    if sample == "core" and seed == 13:
                        models["permuted_newvalues"] = MODELS[FULL]
                        pairs.append((BASE, "permuted_newvalues"))
                        for prefix, feature in zip(SPECS, NEW, strict=True):
                            models["without_" + prefix] = [c for c in MODELS[FULL] if c != feature]
                            pairs.append(("without_" + prefix, FULL))
                    reference_folds, predictions = None, {}
                    for model, numeric in models.items():
                        group_key = (*scenario, model)
                        model_groups.add(group_key)
                        rows = (
                            select(oof, **context, model=model)
                            .sort_values(KEY)
                            .reset_index(drop=True)
                        )
                        check(
                            "oof_rows_" + "/".join(map(str, group_key)),
                            rows[KEY].equals(d[KEY]) and rows.region_code.equals(d.region_code),
                        )
                        worst["truth"] = max(worst["truth"], error(rows[ACTUAL], truth))
                        folds = rows.fold.to_numpy()
                        if reference_folds is None:
                            reference_folds = folds
                        check(
                            "paired_folds_" + "/".join(map(str, group_key)),
                            np.array_equal(folds, reference_folds) and set(folds) == set(range(5)),
                        )
                        predicted = rows[PREDICTED].to_numpy()
                        predictions[model] = predicted
                        if sample != "complete_new3" and model in {"economic", "economic_sectors"}:
                            previous = (
                                select(prior, **context, model=model)
                                .sort_values(KEY)
                                .reset_index(drop=True)
                            )
                            check(
                                "prior_rows_" + "/".join(map(str, group_key)),
                                previous[KEY].equals(rows[KEY])
                                and np.array_equal(previous.fold, folds),
                            )
                            worst["prior_prediction"] = max(
                                worst["prior_prediction"], error(previous[PREDICTED], predicted)
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
                            if model == "permuted_newvalues":
                                train, test = reference_shuffle(train), reference_shuffle(test)
                            x, z, parameters = reference_design(
                                train, test, numeric, regime == "within_region"
                            )
                            pk = (*group_key, fold)
                            prep_keys.add(pk)
                            logged = prep[pk]
                            check(
                                "preprocessing_schema",
                                logged["dummy_columns"] == parameters["dummy_columns"]
                                and logged["train_empty_columns"]
                                == parameters["train_empty_columns"]
                                and logged["train_n"] == len(train)
                                and logged["test_n"] == len(test),
                            )
                            for field in ["medians", "means", "sd"]:
                                check(
                                    "preprocessing_feature_set", set(logged[field]) == set(numeric)
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
                            metric_keys.add((*group_key, target))
                            m = select(metrics, **context, model=model, target=target)
                            check(
                                "one_metric_row",
                                len(m) == 1
                                and m.iloc[0].n == len(d)
                                and m.iloc[0].regions == d.region_code.nunique(),
                            )
                            y, p = truth[:, columns], predicted[:, columns]
                            mse = np.square(y - p).mean()
                            r2 = 1 - np.square(y - p).sum() / np.square(y - y.mean(axis=0)).sum()
                            worst["metric"] = max(
                                worst["metric"], error(m[["mse_oof", "r2_oof"]].iloc[0], [mse, r2])
                            )
                    for baseline, alternative in pairs:
                        for target, columns in [
                            ("log_total", [0]),
                            ("profile_clr", [1, 2, 3, 4, 5]),
                        ]:
                            comparison_keys.add((*scenario, baseline, alternative, target))
                            c = select(
                                comparisons,
                                **context,
                                baseline=baseline,
                                alternative=alternative,
                                target=target,
                            )
                            check(
                                "one_comparison_row",
                                len(c) == 1
                                and c.iloc[0].n == len(d)
                                and c.iloc[0].regions == d.region_code.nunique(),
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
                                    c[["relative_mse_reduction", "ci_low", "ci_high"]].iloc[0],
                                    expected,
                                ),
                            )
                    if sample == "core" and seed == 13:
                        saved_slices = select(slices, **context)
                        expected_slices = 0
                        for scope in [
                            "region_code",
                            "new_observed_count",
                            "municipal_district_type",
                            "population_quartile",
                        ]:
                            for value in sorted(d[scope].unique()):
                                mask = d[scope].eq(value).to_numpy()
                                for target, columns in [
                                    ("log_total", [0]),
                                    ("profile_clr", [1, 2, 3, 4, 5]),
                                ]:
                                    part = saved_slices.loc[
                                        saved_slices["slice"].eq(scope)
                                        & saved_slices.group.astype(str).eq(str(value))
                                        & saved_slices.target.eq(target)
                                    ]
                                    check(
                                        "one_slice_row",
                                        len(part) == 1 and part.iloc[0].n == mask.sum(),
                                    )
                                    a = np.square(
                                        truth[mask][:, columns]
                                        - predictions[BASE][mask][:, columns]
                                    ).sum()
                                    b = np.square(
                                        truth[mask][:, columns]
                                        - predictions[FULL][mask][:, columns]
                                    ).sum()
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
                                            [a, b, 1 - b / a],
                                        ),
                                    )
                                    expected_slices += 1
                        check("complete_slice_grid", expected_slices == len(saved_slices))
    check(
        "all_prespecified_arms", model_groups == set(oof[GROUP].itertuples(index=False, name=None))
    )
    check(
        "all_prespecified_metrics",
        len(metrics) == 224
        and metric_keys == set(metrics[GROUP + ["target"]].itertuples(index=False, name=None)),
    )
    check(
        "all_prespecified_comparisons",
        len(comparisons) == 224
        and comparison_keys
        == set(
            comparisons[SCENARIO + ["baseline", "alternative", "target"]].itertuples(
                index=False, name=None
            )
        ),
    )
    check("all_preprocessing_records", prep_keys == set(prep))
    for name, value in worst.items():
        check(
            "maximum_error_" + name,
            value < (1e-8 if name in {"ridge", "slices"} else 1e-10),
            maximum_absolute_error=value,
        )
    record.update(
        sample_counts=counts,
        reference_ridge_fits=fit_count,
        prior_baseline_groups=prior_checks,
        metric_rows=len(metrics),
        comparison_rows=len(comparisons),
        slice_rows=len(slices),
        oof_rows=len(oof),
        maximum_absolute_errors=worst,
        source_feature_maximum_error=max_feature_error,
        production_code_sha256=manifest["code_sha256"],
        model_manifest_sha256=sha(directory / "manifest.json"),
        verifier_sha256=sha(__file__),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.repo / "reports/round3_analysis/independent_QA.json"
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
