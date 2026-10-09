"""Independent read-only validation of fifth-round monetary recovery experiments.

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


def log_feature(values, population):
    values, population = np.asarray(values, float), np.asarray(population, float)
    valid = np.isfinite(values) & (values > 0) & np.isfinite(population) & (population > 0)
    result = np.full(len(values), np.nan)
    result[valid] = np.log(values[valid] * 1000 / population[valid])
    return result


def data_checks(repo, record, root):
    """Reconstruct features from the old/new candidate snapshots and frozen denominator."""
    directory = root / "audit"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert all(
        sha(repo / row["repo_path"]) == row["sha256"] for row in manifest["input_files"].values()
    )
    assert all(sha(directory / name) == value for name, value in manifest["output_hashes"].items())
    assert sha(repo / "scripts/round5_data_audit.py") == manifest["script_sha256"]
    assert sha(repo / "scripts/round3_data_audit.py") == manifest["helper_sha256"]
    assert sha(repo / "uv.lock") == manifest["lock_sha256"]

    def read(path, keys=KEY):
        codes = {"source_oktmo": str, "official_oktmo_8": str}
        codes.update(
            {
                prefix + "_" + field: str
                for prefix in SPECS
                for field in ["source_oktmo", "official_oktmo_8"]
            }
        )
        return pd.read_csv(path, dtype=codes).sort_values(keys).reset_index(drop=True)

    base = read(repo / "reports/enriched_analysis/core_v2/annual_panel.csv")
    features = read(directory / "annual_features.csv")
    prior = read(repo / "reports/round4_analysis/audit/annual_features.csv")
    old_path = repo / "reports/data_round4/economy/annual_candidate_panel.csv"
    new_path = repo / "reports/data_round5/economy/annual_candidate_panel.csv"
    cell_key = KEY + ["indicator_id"]
    old, new = read(old_path, cell_key), read(new_path, cell_key)
    assert len(old) == len(new) == 12096
    assert not old.duplicated(cell_key).any() and not new.duplicated(cell_key).any()
    assert old[cell_key].equals(new[cell_key])
    assert len(features) == len(base) == 4032 and not features.duplicated(KEY).any()
    # CSV read/write round trips move some stored floats by one ULP; masks/keys stay exact.
    pd.testing.assert_frame_equal(features[base.columns], base, rtol=1e-14, atol=1e-14)
    old_observed = old.value.notna()
    pd.testing.assert_frame_equal(
        old.loc[old_observed], new.loc[old_observed, old.columns], rtol=0, atol=0, check_dtype=False
    )
    gains = old.value.isna() & new.value.notna()
    assert old_observed.sum() == 11608 and gains.sum() == 64
    assert new.value.notna().sum() == 11672 and new.value.isna().sum() == 424
    identities = cell_key + [
        "source_region_url",
        "source_oktmo",
        "official_oktmo_8",
        "code_reference_date",
        "temporal_review_required",
    ]
    pd.testing.assert_frame_equal(old[identities], new[identities], check_dtype=False)
    changed = new.loc[gains, cell_key].merge(base, on=KEY, validate="many_to_one")
    assert changed.territory_id.nunique() == 32 and changed[KEY].drop_duplicates().shape[0] == 64
    assert changed.core_eligible.all() and changed.conservative_eligible.all()
    assert set(changed.region_code) == {25, 59}
    assert changed.groupby(["region_code", "indicator_id"]).size().to_dict() == {
        (25, 8401003): 22,
        (59, 8401011): 42,
    }
    feature_error = 0.0
    expected_features = {}
    for vintage, source in [("old", old), ("new", new)]:
        prefix_vintage = "old_" if vintage == "old" else ""
        matrix = []
        for prefix, (indicator, period) in SPECS.items():
            source_rows = source.loc[source.indicator_id.eq(int(indicator))].sort_values(KEY)
            source_rows = source_rows.reset_index(drop=True)
            assert source_rows[KEY].equals(base[KEY])
            expected = log_feature(source_rows.value, base.population_total)
            stem = prefix_vintage + prefix
            field = stem + "_log_rubles_per_person"
            feature_error = max(feature_error, error(features[field], expected))
            feature_error = max(
                feature_error, error(features[stem + "_thousand_rubles"], source_rows.value)
            )
            assert features[stem + "_missing"].equals(source_rows.value.isna())
            expected_status = np.select(
                [
                    source_rows.value.isna(),
                    source_rows.value.le(0),
                    ~np.isfinite(base.population_total) | base.population_total.le(0),
                ],
                [
                    "source_value_missing",
                    "source_value_nonpositive",
                    "population_missing_or_nonpositive",
                ],
                default="eligible_positive_value_and_population",
            )
            assert np.array_equal(features[stem + "_feature_status"], expected_status)
            if vintage == "old":
                feature_error = max(
                    feature_error, error(prior[prefix + "_log_rubles_per_person"], expected)
                )
            else:
                for source_column in [
                    "value_status",
                    "source_sha256",
                    "request_key",
                    "source_csv_row",
                    "source_oktmo",
                    "official_oktmo_8",
                    "source_region_url",
                    "code_reference_date",
                    "temporal_review_required",
                    "retrieved_at",
                ]:
                    pd.testing.assert_series_equal(
                        features[prefix + "_" + source_column],
                        source_rows[source_column],
                        check_names=False,
                        check_dtype=False,
                        rtol=0,
                        atol=0,
                    )
            supplied = source_rows.source_sha256.notna()
            assert source_rows.value_status.eq("observed").equals(source_rows.value.notna())
            assert source_rows.loc[supplied, "period_code"].eq(int(period)).all()
            assert source_rows.loc[supplied, "activity_code"].eq(101).all()
            assert source_rows.loc[supplied, "unit"].eq("thousand_rubles").all()
            assert source_rows.official_oktmo_8.equals(base.official_oktmo_8)
            assert source_rows.code_reference_date.eq(base.year.astype(str) + "-01-01").all()
            matrix.append(expected)
        expected_features[vintage] = np.column_stack(matrix)
    assert feature_error < 1e-12
    before, after = expected_features["old"], expected_features["new"]
    recovered = np.isnan(before) & np.isfinite(after)
    assert recovered.sum() == 64
    assert np.array_equal(features.activity_new_feature_count, recovered.sum(axis=1))
    assert np.array_equal(features.activity_new_numeric_count, recovered.sum(axis=1))
    assert np.array_equal(features.activity_new_numeric_any, recovered.any(axis=1))
    old_count, new_count = np.isfinite(before).sum(axis=1), np.isfinite(after).sum(axis=1)
    for column, expected in {
        "activity_feature_count_old": old_count,
        "activity_feature_count_new": new_count,
        "activity_all3_features_available_old": old_count == 3,
        "activity_all3_features_available": new_count == 3,
        "activity_all3_features_common": (old_count == 3) & (new_count == 3),
        "activity_all3_features_newly_complete": (old_count != 3) & (new_count == 3),
    }.items():
        assert np.array_equal(features[column], expected), column
    joint = pd.read_csv(directory / "joint_coverage.csv")
    for row in joint.to_dict("records"):
        mask = base.year.eq(row["year"])
        if row["sample"] != "whole":
            mask &= base["core_eligible" if row["sample"] == "core" else "conservative_eligible"]
        expected = dict(
            denominator=int(mask.sum()),
            old_all3=int(((old_count == 3) & mask).sum()),
            new_all3=int(((new_count == 3) & mask).sum()),
            common_all3=int(((old_count == 3) & (new_count == 3) & mask).sum()),
            newly_complete=int(((old_count != 3) & (new_count == 3) & mask).sum()),
            affected_rows=int((recovered.any(axis=1) & mask).sum()),
            gained_numeric_cells=int(recovered[mask].sum()),
            gained_features=int(recovered[mask].sum()),
        )
        assert all(row[k] == v for k, v in expected.items()), row
    assert features.activity_boundary_equivalence_proven.isna().all()
    qa = json.loads((directory / "QA.json").read_text(encoding="utf-8"))
    assert not qa["point_in_time_claim"] and not qa["independent_2024_holdout"]
    record["checks"].append(dict(check="independent_source_feature_audit", status="PASS"))
    record["data_audit"] = dict(
        old_numeric=11608,
        new_numeric=11672,
        gains=64,
        remaining_missing=424,
        affected_municipality_years=64,
        distinct_municipalities=32,
        gained_log_features=64,
        maximum_feature_error=feature_error,
        joint_coverage_rows=len(joint),
        frozen_core_rows=int(base.core_eligible.sum()),
        frozen_stable_rows=int(base.conservative_eligible.sum()),
        old_candidate_sha256=sha(old_path),
        new_candidate_sha256=sha(new_path),
        audit_manifest_sha256=sha(directory / "manifest.json"),
    )


def run(repo, record, root):
    def check(name, condition, **details):
        record["checks"].append(dict(check=name, status="PASS" if condition else "FAIL", **details))
        if not condition:
            raise AssertionError(name)

    directory = root / "models"
    cfg = json.loads((repo / "configs/round5_analysis_v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "protocol_lock.json").read_text(encoding="utf-8"))
    check(
        "manifest_hashes",
        all(
            sha(root / p.removeprefix("reports/round5_analysis/")) == v
            if p.startswith("reports/round5_analysis/")
            else sha(repo / p) == v
            for p, v in manifest["inputs"].items()
        )
        and all(sha(directory / p) == v for p, v in manifest["outputs"].items())
        and sha(repo / "scripts/round5_model_analysis.py") == manifest["code_sha256"],
    )
    check(
        "protocol_lock",
        lock["protocol_sha256"] == sha(repo / "configs/round5_analysis_v1.json")
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
        == ["core", "stable_code", "common_complete_new3", "core_without_recovery_regions"]
        and cfg["regimes"] == ["within_region", "heldout_regions"]
        and cfg["primary_comparison"] == ["new_flags", "recovered_full"]
        and cfg["paired_flags_recovery_comparison"]
        == ["dual_flags_old_values", "dual_flags_new_values"]
        and cfg["base_commit"] == "932afe503e11d248f39360c6f8f23df75d7819cc"
        and cfg["locked_before_new_model_results"]
        and cfg["recovery_regions"] == [25, 59]
        and cfg["slices"]
        == ["recovered_any", "recovery_region", "region_code", "completeness_stratum"]
        and manifest["models"] == MODELS
        and not manifest["tuning_performed"],
    )

    def read(path):
        return pd.read_csv(path).sort_values(KEY).reset_index(drop=True)

    panel = read(directory / "model_panel.csv")
    old = read(repo / "reports/round4_analysis/models/model_panel.csv")
    frozen = read(repo / "reports/enriched_analysis/core_v2/annual_panel.csv")
    features = read(root / "audit/annual_features.csv")
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
    derived = (
        NEW
        + NEW_FLAGS
        + OLD
        + OLD_FLAGS
        + [
            "old_observed_count",
            "new_observed_count",
            "recovered_feature_count",
            "recovered_any",
            "common_complete_new3",
            "expanded_complete_new3",
        ]
    )
    retained = [c for c in old if c not in derived]
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
    for vintage, subdir in [("new", "data_round5"), ("old", "data_round4")]:
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
        and panel.recovery_region.equals(panel.region_code.isin([25, 59]))
        and panel.completeness_stratum.eq(
            np.select(
                [old_count.eq(3) & new_count.eq(3), new_count.eq(3)],
                ["common_complete", "newly_complete"],
                default="still_incomplete",
            )
        ).all()
        and panel.recovered_feature_count.sum() == 64
        and panel.recovered_any.sum() == 64
        and (new_count >= old_count).all(),
    )
    for current, previous in zip(NEW, OLD, strict=True):
        check(
            current + "_no_revision",
            error(panel.loc[panel[previous].notna(), current], panel[previous].dropna()) < 1e-12,
        )

    oof = duckdb.read_parquet(str(directory / "oof_predictions.parquet")).df()
    prior = duckdb.read_parquet(
        str(repo / "reports/round4_analysis/models/oof_predictions.parquet")
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
            "core_without_recovery_regions": core.loc[~core.region_code.isin([25, 59])],
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
                        if sample in {"core", "stable_code", "core_without_recovery_regions"}
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
                    old_sample = "stable_code" if sample == "stable_code" else "core"
                    old_rows = select(
                        prior,
                        year=year,
                        sample=old_sample,
                        regime=regime,
                        seed=seed,
                        model="recovered_full",
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
                            "recovery_region",
                            "completeness_stratum",
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
                                and logged["numeric"] == numeric,
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
                            if (
                                sample == "core"
                                and seed == 13
                                and model
                                in {
                                    "new_flags",
                                    "recovered_full",
                                    "legacy_full",
                                    "dual_flags_old_values",
                                    "dual_flags_new_values",
                                }
                            ):
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
                    if sample in {"common_complete_new3", "core_without_recovery_regions"}:
                        worst["common_invariance"] = max(
                            worst["common_invariance"],
                            error(predictions["legacy_full"], predictions["recovered_full"]),
                        )
                        common_checks += 1
                        if sample == "core_without_recovery_regions":
                            worst["common_invariance"] = max(
                                worst["common_invariance"],
                                error(
                                    predictions["dual_flags_old_values"],
                                    predictions["dual_flags_new_values"],
                                ),
                            )
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
                                "recovery_region",
                                "completeness_stratum",
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
        "complete_preprocessing_grid",
        expected_prep == set(prep) and len(expected_prep) == manifest["fits"],
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
        production_preprocessing_fits_checked=len(expected_prep),
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
        protocol_sha256=sha(repo / "configs/round5_analysis_v1.json"),
        model_manifest_sha256=sha(directory / "manifest.json"),
        verifier_sha256=sha(__file__),
        inference_limits=[
            "Exploratory retrospective attribution, not forecasting",
            "Conditional fixed-OOF regional bootstrap; no pipeline refit",
            "No multiplicity adjustment; secondary results descriptive",
            "Prior round4 negative controls not rerun; no new negative-control evidence",
            "No proof of historical real-time feature availability",
            "Independent calculations do not independently validate source truth",
        ],
    )


def recovered_checks(repo, record, root):
    """Direct amount/population medians and ranks, independent of stored log intensities."""
    directory = root / "recovered"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == sha(root / "audit/annual_features.csv")
    assert manifest["code_sha256"] == sha(repo / "scripts/round5_recovered_profiles.py")
    assert all(sha(directory / p) == v for p, v in manifest["outputs"].items())
    codes = {prefix + "_source_oktmo": str for prefix in SPECS}
    features = pd.read_csv(root / "audit/annual_features.csv", dtype=codes)
    profiles = pd.read_csv(
        directory / "recovered_municipality_profiles.csv", dtype={"source_oktmo": str}
    )
    pairs = pd.read_csv(
        directory / "recovered_two_year_pairs.csv",
        dtype={"source_oktmo_2023": str, "source_oktmo_2024": str},
    )
    summaries = pd.read_csv(directory / "regional_selection_summary.csv")
    assert len(profiles) == 64 and not profiles.duplicated(KEY + ["prefix"]).any()
    assert len(pairs) == 32 and not pairs.duplicated(["territory_id", "prefix"]).any()
    assert len(summaries) == 4 and not summaries.duplicated(["year", "prefix", "region_code"]).any()
    checked, ratios = set(), []
    worst_relative = 0.0
    for summary in summaries.to_dict("records"):
        year, prefix, region = (summary[k] for k in ["year", "prefix", "region_code"])
        assert (prefix, region) in {("shipments", 59), ("retail", 25)}
        d = features.loc[
            features.year.eq(year) & features.region_code.eq(region) & features.core_eligible
        ]
        raw = d[prefix + "_thousand_rubles"].to_numpy(float)
        prior_raw = d["old_" + prefix + "_thousand_rubles"].to_numpy(float)
        pop = d.population_total.to_numpy(float)
        valid = np.isfinite(raw) & (raw > 0) & np.isfinite(pop) & (pop > 0)
        recovered = np.isnan(prior_raw) & valid
        previous = np.isfinite(prior_raw) & valid
        intensities = raw * 1000 / pop
        old_median, new_median = np.median(intensities[previous]), np.median(intensities[recovered])
        expected = dict(
            core_n=len(d),
            recovered_n=int(recovered.sum()),
            old_observed_n=int(previous.sum()),
            new_observed_n=int(valid.sum()),
            remaining_missing_n=int((~valid).sum()),
            recovered_median_rubles_per_person=new_median,
            previously_observed_median_rubles_per_person=old_median,
            median_ratio=new_median / old_median,
            recovered_median_population=np.median(pop[recovered]),
            previously_observed_median_population=np.median(pop[previous]),
        )
        for field, value in expected.items():
            np.testing.assert_allclose(summary[field], value, rtol=1e-12, atol=1e-8)
            worst_relative = max(worst_relative, abs(summary[field] - value) / max(1, abs(value)))
        sorted_values = np.sort(intensities[valid])
        for index in np.flatnonzero(recovered):
            item = d.iloc[index]
            matches = profiles.loc[
                profiles.territory_id.eq(item.territory_id)
                & profiles.year.eq(year)
                & profiles.prefix.eq(prefix)
            ]
            assert len(matches) == 1
            row = matches.iloc[0]
            checked.add((int(item.territory_id), year, prefix))
            assert row.source_oktmo == item[prefix + "_source_oktmo"]
            assert row.source_sha256 == item[prefix + "_source_sha256"]
            assert row.municipality_name == item.municipal_district_name
            assert row.region_code == region and row.rank_denominator == valid.sum()
            np.testing.assert_allclose(
                [row.thousand_rubles, row.population_jan1, row.rubles_per_person],
                [raw[index], pop[index], intensities[index]],
                rtol=1e-14,
                atol=1e-8,
            )
            value = intensities[index]
            rank = (
                np.searchsorted(sorted_values, value, "left")
                + np.searchsorted(sorted_values, value, "right")
                + 1
            ) / (2 * len(sorted_values))
            assert abs(row.within_region_available_percentile - rank) < 1e-12
        ratios.append(dict(year=year, prefix=prefix, region_code=region, **expected))
    assert len(checked) == 64
    for pair in pairs.to_dict("records"):
        a = profiles.loc[
            profiles.territory_id.eq(pair["territory_id"]) & profiles.prefix.eq(pair["prefix"])
        ].set_index("year")
        assert set(a.index) == {2023, 2024}
        for year in [2023, 2024]:
            assert pair[f"source_oktmo_{year}"] == a.loc[year, "source_oktmo"]
            assert pair[f"source_sha256_{year}"] == a.loc[year, "source_sha256"]
            for field in [
                "thousand_rubles",
                "population_jan1",
                "rubles_per_person",
                "within_region_available_percentile",
                "rank_denominator",
            ]:
                np.testing.assert_allclose(
                    pair[f"{field}_{year}"], a.loc[year, field], rtol=1e-14, atol=1e-10
                )
        aggregate = a.loc[2024, "thousand_rubles"] / a.loc[2023, "thousand_rubles"] - 1
        intensity = a.loc[2024, "rubles_per_person"] / a.loc[2023, "rubles_per_person"] - 1
        np.testing.assert_allclose(
            [pair["nominal_aggregate_change_fraction"], pair["nominal_per_person_change_fraction"]],
            [aggregate, intensity],
            rtol=1e-13,
            atol=1e-13,
        )
    record["checks"].append(
        dict(check="recovered_medians_ranks_pairs_and_literal_source_codes", status="PASS")
    )
    record["recovered_description"] = dict(
        rows=64,
        pairs=32,
        summary_rows=ratios,
        maximum_summary_relative_error=worst_relative,
        manifest_sha256=sha(directory / "manifest.json"),
    )


def readout_checks(repo, record, root):
    directory = root / "readout"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha(root / "models" / p) == v for p, v in manifest["inputs"].items())
    assert all(sha(directory / p) == v for p, v in manifest["outputs"].items())
    assert manifest["code_sha256"] == sha(repo / "scripts/round5_model_readout.py")
    prior_path = repo / "reports/round4_analysis/models/cv_comparisons.csv"
    assert manifest["prior_comparisons_sha256"] == sha(prior_path)
    comparisons = pd.read_csv(root / "models/cv_comparisons.csv")
    primary = comparisons.loc[comparisons.comparison.eq("primary")].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(directory / "primary_all_samples.csv"), primary)
    sensitivity = comparisons.loc[
        comparisons["sample"].eq("core_without_recovery_regions")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        pd.read_csv(directory / "excluded_regions_sensitivity.csv"), sensitivity
    )
    contributions = pd.read_csv(directory / "state_update_contributions.csv", dtype={"group": str})
    identifier = ["year", "regime", "target", "slice", "group"]
    assert not contributions.duplicated(identifier).any()
    panel = pd.read_csv(root / "models/model_panel.csv").sort_values(KEY).reset_index(drop=True)
    oof = duckdb.read_parquet(str(root / "models/oof_predictions.parquet")).df()
    expected_keys, maximum = set(), 0.0
    for year in [2023, 2024]:
        d = (
            panel.loc[panel.year.eq(year) & panel.core_eligible]
            .sort_values(KEY)
            .reset_index(drop=True)
        )
        for regime in ["within_region", "heldout_regions"]:
            arms = {}
            for model in ["legacy_full", "recovered_full"]:
                arm = select(
                    oof, year=year, sample="core", regime=regime, seed=13, model=model
                ).sort_values(KEY)
                assert arm[KEY].reset_index(drop=True).equals(d[KEY])
                arms[model] = np.square(arm[ACTUAL].to_numpy() - arm[PREDICTED].to_numpy())
            for target, cols in [("log_total", [0]), ("profile_clr", [1, 2, 3, 4, 5])]:
                a, b = (
                    arms[name][:, cols].sum(axis=1) for name in ["legacy_full", "recovered_full"]
                )
                denominator = a.sum()
                overall = 1 - b.sum() / denominator
                for scope in [
                    "region_code",
                    "recovered_any",
                    "recovery_region",
                    "completeness_stratum",
                ]:
                    total = 0.0
                    for group in sorted(d[scope].unique()):
                        expected_keys.add((year, regime, target, scope, str(group)))
                        mask = d[scope].eq(group).to_numpy()
                        found = select(
                            contributions,
                            year=year,
                            regime=regime,
                            target=target,
                            slice=scope,
                            group=str(group),
                        )
                        assert len(found) == 1
                        found = found.iloc[0]
                        assert (
                            found.n == mask.sum()
                            and found.regions == d.loc[mask, "region_code"].nunique()
                        )
                        old, new = a[mask].sum(), b[mask].sum()
                        expected = [
                            old,
                            new,
                            new - old,
                            denominator,
                            (old - new) / old,
                            (old - new) / denominator,
                        ]
                        fields = [
                            "legacy_sse",
                            "recovered_sse",
                            "sse_change",
                            "legacy_total_sse",
                            "local_relative_reduction",
                            "contribution_to_total_relative_reduction",
                        ]
                        maximum = max(maximum, error(found[fields], expected))
                        total += found.contribution_to_total_relative_reduction
                    assert abs(total - overall) < 1e-12
    assert expected_keys == set(contributions[identifier].itertuples(index=False, name=None))
    assert maximum < 1e-10
    qa = json.loads((directory / "QA.json").read_text(encoding="utf-8"))
    assert qa["adoption_passed"] == record["independent_adoption_criteria_met"]
    assert not qa["adoption_passed"]
    prior = select(pd.read_csv(prior_path), comparison="primary", seed=13, target="profile_clr")
    old_core, old_stable = (select(prior, sample=sample) for sample in ["core", "stable_code"])
    old_primary_pass = (
        old_core.relative_mse_reduction.ge(0.01).all() and old_core.ci_low.gt(0).all()
    )
    assert qa["prior_round4_adoption_stable_strict_positive"] == bool(
        old_primary_pass and old_stable.relative_mse_reduction.gt(0).all()
    )
    assert qa["prior_round4_adoption_stable_nonnegative"] == bool(
        old_primary_pass and old_stable.relative_mse_reduction.ge(0).all()
    )
    intervals = select(primary, seed=13, target="profile_clr")
    assert qa["all_primary_seed13_intervals_include_zero"] == bool(
        (intervals.ci_low.le(0) & intervals.ci_high.ge(0)).all()
    )
    assert qa["affected_municipalities"] == 32 and qa["affected_municipality_years"] == 64
    record["checks"].append(
        dict(check="independent_postfit_contribution_denominators_and_adoption", status="PASS")
    )
    record["readout"] = dict(
        contribution_rows=len(contributions),
        maximum_absolute_error=maximum,
        prior_and_current_adoption=False,
        manifest_sha256=sha(directory / "manifest.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, help="Alternate round5 output root for fresh replay")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    root = args.root or args.repo / "reports/round5_analysis"
    out = args.out or root / "independent_QA.json"
    record = dict(status="RUNNING", checked_at=datetime.now(UTC).isoformat(), checks=[])
    try:
        with threadpool_limits(limits=1):
            data_checks(args.repo, record, root)
            if not args.data_only:
                run(args.repo, record, root)
                recovered_checks(args.repo, record, root)
                readout_checks(args.repo, record, root)
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
