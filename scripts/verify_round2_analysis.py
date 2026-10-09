"""Independent, offline checks of the frozen round-two model outputs.

Does not import the production model builder or mutate model/source files.
Reference ridge fits use NumPy linear algebra, not sklearn's Ridge.
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
TARGETS = ["log_total", *["clr_" + x for x in PROFILE]]
CORE = ["log_population", "log_wage", "log_worker_concentration"]
SOURCE_CODES = dict(
    A="10000", B="50000", C="20000", D="30000", H="49000", O="84000", P="85000", Q="86000"
)
SECTORS = [f"sector_{s}_share" for s in SOURCE_CODES]
FLAGS = [f"missing_{s}" for s in SOURCE_CODES]
NUMERIC = {
    "economic": CORE,
    "economic_missingness": CORE + FLAGS,
    "economic_sectors": CORE + FLAGS + SECTORS,
    "permuted_sectors": CORE + FLAGS + SECTORS,
}
GROUP = ["year", "sample", "regime", "seed", "model"]
ACTUAL = ["actual_" + t for t in TARGETS]
PREDICTED = ["predicted_" + t for t in TARGETS]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_parquet(path):
    return duckdb.read_parquet(str(path)).df()


def normalized_keys(frame):
    frame = frame.copy()
    frame[KEY] = frame[KEY].astype(int)
    return frame


def index_keys(frame):
    return set(map(tuple, frame[KEY].to_numpy()))


def maximum_error(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return 1e300
    good = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return 1e300
    return float(np.max(np.abs(a[good] - b[good]))) if good.any() else 0.0


def reference_design(train, test, numeric, region):
    """Explicit category equality and NumPy train-only median/scale reference."""
    categorical = ["municipal_district_type"] + (["region_code"] if region else [])
    train_parts, test_parts, dummy_names = [], [], []
    for field in categorical:
        left, right = train[field].astype(str), test[field].astype(str)
        for category in sorted(left.unique()):
            train_parts.append(left.eq(category).to_numpy(dtype=float))
            test_parts.append(right.eq(category).to_numpy(dtype=float))
            dummy_names.append(f"{field}_{category}")
    a = train[numeric].to_numpy(dtype=float, copy=True)
    b = test[numeric].to_numpy(dtype=float, copy=True)
    empty = np.isnan(a).all(axis=0)
    median = np.array(
        [0.0 if empty[i] else np.median(a[np.isfinite(a[:, i]), i]) for i in range(a.shape[1])]
    )
    a = np.where(np.isnan(a), median, a)
    b = np.where(np.isnan(b), median, b)
    mean = a.mean(axis=0)
    sd = a.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    train_parts.extend(((a - mean) / sd).T)
    test_parts.extend(((b - mean) / sd).T)
    return (
        np.column_stack(train_parts),
        np.column_stack(test_parts),
        {
            "train_empty_columns": [
                name for name, flag in zip(numeric, empty, strict=True) if flag
            ],
            "medians": dict(zip(numeric, median, strict=True)),
            "means": dict(zip(numeric, mean, strict=True)),
            "sd": dict(zip(numeric, sd, strict=True)),
            "dummy_columns": dummy_names,
        },
    )


def reference_shuffle(frame):
    result = frame.copy()
    rng = np.random.default_rng(71013)
    for column in SECTORS:
        for region in sorted(frame.region_code.unique()):
            positions = frame.region_code.eq(region) & frame[column].notna()
            result.loc[positions, column] = rng.permutation(frame.loc[positions, column].to_numpy())
    return result


def reference_ridge(x, y, z):
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x - x_mean, y - y_mean
    gram = centered_x.T @ centered_x + np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)
    return (z - x_mean) @ coefficients + y_mean


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **evidence):
        self.checks.append({"check": name, "status": "PASS" if passed else "FAIL", **evidence})

    def require(self, name, passed, **evidence):
        self.check(name, passed, **evidence)
        if not passed:
            raise ValueError(f"Cannot continue safely: {name}")


def validate(repo, audit):
    directory = repo / "reports/round2_analysis/models"
    frozen = normalized_keys(
        read_parquet(repo / "reports/enriched_analysis/core_v2/annual_panel.parquet")
    )
    panel = normalized_keys(
        pd.read_csv(directory / "model_panel.csv", dtype={"official_oktmo_8": str})
    )
    features = normalized_keys(
        pd.read_csv(repo / "reports/round2_analysis/sector/feature_panel.csv")
    )
    source = normalized_keys(
        read_parquet(repo / "reports/data_round2/sector/bulk/annual_candidate_sector.parquet")
    )
    oof = normalized_keys(read_parquet(directory / "oof_predictions.parquet"))
    metrics = pd.read_csv(directory / "cv_metrics.csv")
    comparisons = pd.read_csv(directory / "cv_comparisons.csv")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((repo / "configs/round2_analysis_v1.json").read_text(encoding="utf-8"))
    audit.check(
        "frozen_design_constants",
        config["sector_sections"] == list(SOURCE_CODES)
        and config["ridge_alpha"] == 1
        and config["folds"] == 5
        and config["primary_seed"] == 13
        and config["additional_within_region_seeds"] == [29, 47],
    )
    output_mismatches = [
        p for p, value in manifest["outputs"].items() if digest(directory / p) != value
    ]
    input_mismatches = [p for p, value in manifest["inputs"].items() if digest(repo / p) != value]
    audit.check(
        "manifest_hashes",
        not output_mismatches
        and not input_mismatches
        and digest(repo / "scripts/round2_model_analysis.py") == manifest["code_sha256"],
        output_mismatches=output_mismatches,
        input_mismatches=input_mismatches,
    )
    for name, frame in [("frozen", frozen), ("model_panel", panel), ("features", features)]:
        audit.require(
            name + "_keys",
            len(frame) == 4032
            and not frame.duplicated(KEY).any()
            and index_keys(frame) == index_keys(frozen),
            rows=len(frame),
        )
    frozen = frozen.sort_values(KEY).reset_index(drop=True)
    panel = panel.sort_values(KEY).reset_index(drop=True)
    features = features.sort_values(KEY).reset_index(drop=True)
    selected = normalized_keys(pd.read_csv(directory / "selected_sector_features.csv"))
    selected = selected.sort_values(KEY).reset_index(drop=True)
    audit.require(
        "selected_feature_keys",
        len(selected) == 4032
        and not selected.duplicated(KEY).any()
        and index_keys(selected) == index_keys(frozen),
    )
    feature_error = max(
        maximum_error(panel[SECTORS], features[SECTORS]),
        maximum_error(panel[SECTORS], selected[SECTORS]),
    )
    audit.check(
        "root_agent_feature_implementations_agree",
        feature_error < 1e-12,
        cells=4032 * 8,
        max_abs_error=feature_error,
        agent_feature_sha256=digest(repo / "reports/round2_analysis/sector/feature_panel.csv"),
    )
    frozen_max = {}
    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            error = maximum_error(panel[column], frozen[column])
            if error > 1e-9:
                frozen_max[column] = error
        else:
            left, right = (
                panel[column].astype("string").fillna("<NULL>"),
                frozen[column].astype("string").fillna("<NULL>"),
            )
            if not left.equals(right):
                frozen_max[column] = int(left.ne(right).sum())
    audit.check("all_frozen_panel_fields_preserved", not frozen_max, mismatches=frozen_max)
    expected_core = {2023: 1975, 2024: 1998}
    expected_stable = {2023: 1897, 2024: 1930}
    counts = {
        "core": panel.groupby("year").core_eligible.sum().to_dict(),
        "conservative": panel.groupby("year").conservative_eligible.sum().to_dict(),
    }
    audit.check(
        "frozen_masks",
        counts["core"] == expected_core and counts["conservative"] == expected_stable,
        **counts,
    )
    audit.check(
        "new_totals_do_not_expand_sample",
        not panel.loc[panel.territory_id.eq(2710), ["core_eligible", "conservative_eligible"]]
        .any()
        .any(),
    )

    # Check all documented source codes, keeping a separately preserved 7-digit UI literal.
    joined = source.merge(
        frozen[KEY + ["official_oktmo_8"]],
        on=KEY,
        validate="many_to_one",
        suffixes=("_source", "_frozen"),
    )
    code_bad = joined.official_oktmo_8_source.astype(str).ne(
        joined.official_oktmo_8_frozen.astype(str)
    )
    audit.check(
        "source_documented_official_code_year",
        len(joined) == 162210 and not code_bad.any(),
        source_rows=len(joined),
        conflicts=int(code_bad.sum()),
    )
    workers = source.loc[source.indicator_id.eq("8423005")].copy()
    audit.check(
        "worker_source_unit_and_annual_period",
        workers.unit.eq("человек").all()
        and workers.period_code.eq(79).all()
        and workers.period_start.eq(workers.year.astype(str) + "-01-01").all()
        and workers.period_end.eq(workers.year.astype(str) + "-12-31").all(),
        rows=len(workers),
    )
    audit.require(
        "worker_activity_key_unique", not workers.duplicated(KEY + ["source_activity_code"]).any()
    )
    totals = workers.loc[workers.source_activity_code.eq("101"), KEY + ["value", "status"]].rename(
        columns={"value": "denominator", "status": "denominator_status"}
    )
    denominators = panel[KEY + ["workers", "core_eligible"]].merge(
        totals, on=KEY, how="left", validate="one_to_one"
    )
    eligible_den = denominators.loc[denominators.core_eligible]
    audit.check(
        "eligible_official_denominators",
        len(eligible_den) == 3973
        and eligible_den.denominator_status.eq("observed").all()
        and maximum_error(eligible_den.workers, eligible_den.denominator) == 0,
        checked=len(eligible_den),
    )
    source_ratio_error = {}
    for section, code in SOURCE_CODES.items():
        numerator = workers.loc[workers.source_activity_code.eq(code), KEY + ["value", "status"]]
        expected = (
            panel[KEY]
            .merge(numerator, on=KEY, how="left", validate="one_to_one")
            .merge(totals, on=KEY, how="left", validate="one_to_one")
        )
        good = (
            expected.status.eq("observed")
            & expected.denominator_status.eq("observed")
            & expected.value.ge(0)
            & expected.denominator.gt(0)
            & expected.value.le(expected.denominator)
        )
        ratio = (expected.value / expected.denominator).where(good)
        actual = panel[f"sector_{section}_share"]
        source_ratio_error[section] = maximum_error(actual, ratio)
        audit.check(
            f"source_ratio_{section}",
            source_ratio_error[section] < 1e-12,
            max_abs_error=source_ratio_error[section],
            observed=int(actual.notna().sum()),
            missing=int(actual.isna().sum()),
        )
        audit.check(
            f"missing_flag_{section}",
            np.array_equal(
                panel[f"missing_{section}"].to_numpy(), actual.isna().astype(float).to_numpy()
            ),
        )
    audit.check(
        "selected_sector_count",
        np.array_equal(
            panel.selected_sector_observed.to_numpy(), panel[SECTORS].notna().sum(axis=1).to_numpy()
        ),
    )

    audit.require(
        "unique_oof_key", not oof.duplicated(GROUP + ["territory_id"]).any(), rows=len(oof)
    )
    audit.require("finite_oof", np.isfinite(oof[ACTUAL + PREDICTED].to_numpy()).all())
    grouped = {
        k: v.sort_values("territory_id").reset_index(drop=True)
        for k, v in oof.groupby(GROUP, sort=True)
    }
    expected_groups = set()
    for year in (2023, 2024):
        for sample in ("core", "stable_code", "complete_sectors"):
            for regime in ("within_region", "heldout_regions"):
                seeds = (13, 29, 47) if sample == "core" and regime == "within_region" else (13,)
                for seed in seeds:
                    models = ["economic", "economic_missingness", "economic_sectors"]
                    if sample == "core" and seed == 13:
                        models.append("permuted_sectors")
                    expected_groups.update((year, sample, regime, seed, m) for m in models)
    audit.require(
        "all_predeclared_model_groups_present",
        set(grouped) == expected_groups,
        expected=len(expected_groups),
        actual=len(grouped),
    )
    coverage_bad, truth_errors, split_bad = [], [], []
    reference_folds = {}
    direct_refits = []
    preprocessing = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    prep = {tuple(item[k] for k in GROUP) + (item["fold"],): item for item in preprocessing}
    audit.require(
        "preprocessing_records_unique", len(prep) == len(preprocessing) == len(grouped) * 5
    )
    preprocessing_errors, empty_train = [], []
    frozen_oof = read_parquet(repo / "reports/enriched_analysis/core_v2/oof_predictions.parquet")
    old = frozen_oof.loc[frozen_oof["sample"].eq("core") & frozen_oof.model.eq("economic")]
    baseline_errors = []
    for label, predictions in grouped.items():
        year, sample, regime, seed, model = label
        frame = panel.loc[panel.year.eq(year) & panel.core_eligible]
        if sample == "stable_code":
            frame = frame.loc[frame.conservative_eligible]
        elif sample == "complete_sectors":
            frame = frame.dropna(subset=SECTORS)
        frame = frame.sort_values("territory_id").reset_index(drop=True)
        if list(frame.territory_id) != list(predictions.territory_id):
            coverage_bad.append(list(label))
            continue
        log_profiles = np.log(frame[PROFILE].to_numpy())
        y = np.column_stack(
            [frame.log_total, log_profiles - log_profiles.mean(axis=1, keepdims=True)]
        )
        error = maximum_error(predictions[ACTUAL], y)
        if error > 1e-12:
            truth_errors.append({"label": list(label), "error": error})
        f = predictions.fold.to_numpy()
        key = label[:-1]
        if key in reference_folds:
            if not np.array_equal(f, reference_folds[key]):
                split_bad.append({"label": list(label), "reason": "unpaired_models"})
        else:
            reference_folds[key] = f
        if regime == "within_region":
            rng = np.random.default_rng(seed)
            expected_folds = np.empty(len(frame), dtype=int)
            for region in sorted(frame.region_code.unique()):
                positions = np.flatnonzero(frame.region_code.to_numpy() == region)
                shuffled = rng.permutation(positions)
                expected_folds[shuffled] = (np.arange(len(shuffled)) + rng.integers(5)) % 5
            if not np.array_equal(f, expected_folds):
                split_bad.append({"label": list(label), "reason": "within_seed_mismatch"})
        if sample == "core" and seed == 13 and model == "economic":
            prior = old.loc[old.year.eq(year) & old.regime.eq(regime)].sort_values("territory_id")
            if list(prior.territory_id) != list(predictions.territory_id) or not np.array_equal(
                prior.fold.to_numpy(), f
            ):
                split_bad.append({"label": list(label), "reason": "frozen_baseline_fold_mismatch"})
            baseline_errors.append(maximum_error(prior[PREDICTED], predictions[PREDICTED]))
        for fold in range(5):
            training, validation = f != fold, f == fold
            train, test = frame.loc[training], frame.loc[validation]
            if regime == "heldout_regions" and set(train.region_code) & set(test.region_code):
                split_bad.append({"label": list(label), "reason": "shared_test_region"})
            if model == "permuted_sectors":
                train, test = reference_shuffle(train), reference_shuffle(test)
            x, z, expected_prep = reference_design(
                train, test, NUMERIC[model], regime == "within_region"
            )
            recorded = prep[label + (fold,)]
            mismatch = recorded["train_n"] != len(train) or recorded["test_n"] != len(test)
            for field in ["dummy_columns", "train_empty_columns"]:
                mismatch |= recorded[field] != expected_prep[field]
            for field in ["medians", "means", "sd"]:
                a = np.array([recorded[field][c] for c in NUMERIC[model]])
                b = np.array([expected_prep[field][c] for c in NUMERIC[model]])
                mismatch |= not np.allclose(a, b, atol=1e-12, rtol=1e-10)
            if mismatch:
                preprocessing_errors.append({"label": list(label), "fold": fold})
            if expected_prep["train_empty_columns"]:
                empty_train.append(
                    {
                        "label": list(label),
                        "fold": fold,
                        "columns": expected_prep["train_empty_columns"],
                    }
                )
            if (
                sample == "core"
                and seed == 13
                and model in ("economic", "economic_sectors")
                and fold in (0, 4)
            ):
                refit = reference_ridge(x, y[training], z)
                direct_refits.append(
                    {
                        "label": list(label),
                        "fold": fold,
                        "train_n": len(train),
                        "test_n": len(test),
                        "max_abs_error": maximum_error(
                            refit, predictions.loc[validation, PREDICTED]
                        ),
                    }
                )
    audit.check(
        "oof_exact_sample_coverage",
        not coverage_bad,
        model_groups=len(grouped),
        failures=coverage_bad,
    )
    audit.check("oof_targets_rebuilt_from_frozen_panel", not truth_errors, failures=truth_errors)
    audit.check(
        "paired_folds_and_region_separation",
        not split_bad,
        split_groups=len(reference_folds),
        failures=split_bad,
    )
    audit.check(
        "frozen_economic_oof_replicated",
        max(baseline_errors, default=1e300) < 1e-9,
        max_abs_error=max(baseline_errors, default=1e300),
    )
    audit.check(
        "all_training_preprocessing_recomputed",
        not preprocessing_errors,
        records=len(preprocessing),
        failures=preprocessing_errors,
        train_empty=empty_train,
    )
    audit.check(
        "direct_numpy_ridge_refits",
        len(direct_refits) == 16 and max(x["max_abs_error"] for x in direct_refits) < 1e-9,
        refits=direct_refits,
    )

    metric_errors = []
    for row in metrics.to_dict("records"):
        label = tuple(row[k] for k in GROUP)
        group = grouped[label]
        columns = [0] if row["target"] == "log_total" else [1, 2, 3, 4, 5]
        y, prediction = (
            group[ACTUAL].to_numpy()[:, columns],
            group[PREDICTED].to_numpy()[:, columns],
        )
        square = np.square(y - prediction)
        r2 = 1 - square.sum() / np.square(y - y.mean(axis=0)).sum()
        mse = square.mean()
        error = max(abs(r2 - row["r2_oof"]), abs(mse - row["mse_oof"]))
        if error > 1e-12 or row["n"] != len(group) or row["regions"] != group.region_code.nunique():
            metric_errors.append({"label": list(label), "target": row["target"], "error": error})
    audit.check(
        "all_oof_metrics",
        not metric_errors and len(metrics) == len(grouped) * 2,
        rows=len(metrics),
        failures=metric_errors,
    )

    bootstrap_errors, primary_results = [], []
    for row in comparisons.to_dict("records"):
        common = tuple(row[k] for k in GROUP[:-1])
        baseline, alternative = (
            grouped[common + (row["baseline"],)],
            grouped[common + (row["alternative"],)],
        )
        audit.require(
            "comparison_pair_identity_"
            + "_".join(map(str, common))
            + "_"
            + row["baseline"]
            + "_"
            + row["alternative"]
            + "_"
            + row["target"],
            list(baseline.territory_id) == list(alternative.territory_id)
            and np.array_equal(baseline.fold, alternative.fold),
        )
        columns = [0] if row["target"] == "log_total" else [1, 2, 3, 4, 5]
        y = baseline[ACTUAL].to_numpy()[:, columns]
        b, a = (
            baseline[PREDICTED].to_numpy()[:, columns],
            alternative[PREDICTED].to_numpy()[:, columns],
        )
        residuals = pd.DataFrame(
            {
                "region": baseline.region_code,
                "baseline": np.square(y - b).sum(axis=1),
                "alternative": np.square(y - a).sum(axis=1),
            }
        )
        losses = (
            residuals.groupby("region", sort=True)[["baseline", "alternative"]].sum().to_numpy()
        )
        draws = np.random.default_rng(13).integers(len(losses), size=(2000, len(losses)))
        bootstrap_sums = losses[draws].sum(axis=1)
        estimates = 1 - bootstrap_sums[:, 1] / bootstrap_sums[:, 0]
        low, high = np.quantile(estimates, [0.025, 0.975])
        effect = 1 - losses[:, 1].sum() / losses[:, 0].sum()
        error = max(
            abs(effect - row["relative_mse_reduction"]),
            abs(low - row["ci_low"]),
            abs(high - row["ci_high"]),
        )
        if error > 1e-12 or row["n"] != len(baseline):
            bootstrap_errors.append(
                {"label": list(common), "target": row["target"], "error": error}
            )
        if (
            row["seed"] == 13
            and row["baseline"] == "economic_missingness"
            and row["alternative"] == "economic_sectors"
        ):
            primary_results.append(row)
    audit.check(
        "all_paired_fixed_oof_region_bootstraps",
        not bootstrap_errors,
        rows=len(comparisons),
        repetitions=2000,
        seed=13,
        failures=bootstrap_errors,
    )

    return {
        "source_ratio_max_errors": source_ratio_error,
        "primary_comparisons_recomputed": primary_results,
        "manifest_sha256": digest(directory / "manifest.json"),
        "source_sha256": digest(
            repo / "reports/data_round2/sector/bulk/annual_candidate_sector.parquet"
        ),
        "model_code_sha256": digest(repo / "scripts/round2_model_analysis.py"),
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "No independent reacquisition or legal boundary proof; "
            "documented source code/year fields checked against frozen panel.",
            "Both years were already observed; "
            "this is not temporal validation or causal identification.",
            "Bootstrap resamples regions for fixed OOF errors; "
            "models are not refitted in the bootstrap.",
            "Worker ratios refer to covered organizations excluding small businesses, "
            "not all resident employment.",
            "Partial correlations and the price branch are outside "
            "this numerical model verification.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.repo / "reports/round2_analysis/independent_review.json"
    audit = Audit()
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    try:
        with threadpool_limits(limits=1):
            numerical = validate(args.repo, audit)
    except Exception as error:
        numerical = {"execution_error": f"{type(error).__name__}: {error}"}
        audit.check("verification_completed", False)
    numerical["checks"] = audit.checks
    numerical["verified_at_utc"] = datetime.now(UTC).isoformat()
    numerical["passed"] = sum(x["status"] == "PASS" for x in audit.checks)
    numerical["failed"] = sum(x["status"] == "FAIL" for x in audit.checks)
    report["stage"] = "independent_numerical_verification"
    report["status"] = "PASS_WITH_SCOPE_LIMITATIONS" if numerical["failed"] == 0 else "FAIL"
    for check in report.get("design_checks", []):
        if check["id"] == "D08":
            check["details"] = (
                "Eight sections A/B/C/D/H/O/P/Q fixed before model results. "
                "73000 excluded.50000 absent as an option in174 municipality-years, "
                "not a zero. Missing flags and all4032 exact keys independently checked."
            )
        if numerical["failed"] == 0 and check["status"] in (
            "required_before_model_pass",
            "supported_at_input_stage",
        ):
            check["status"] = "verified"
    report["model_output_checks"] = numerical
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": numerical["passed"],
                "failed": numerical["failed"],
                "execution_error": numerical.get("execution_error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if numerical["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
