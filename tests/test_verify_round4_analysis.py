"""Meaningful independent-reference checks for the fourth-round validator."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "round4_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_round4_analysis.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_analytic_ridge_shrinkage_with_unpenalized_intercept():
    train = np.array([[-1.0], [0.0], [1.0]])
    prediction = V.reference_ridge(train, 3 + 2 * train, np.array([[-2.0], [2.0]]))
    np.testing.assert_allclose(prediction.ravel(), [1 / 3, 17 / 3], atol=1e-14)


def test_no_test_learning_for_imputation_scale_or_categories():
    train = pd.DataFrame({"municipal_district_type": ["a"] * 3, "v": [1, np.nan, 3]})
    test = pd.DataFrame({"municipal_district_type": ["unknown", "a"], "v": [np.nan, 999]})
    a, b, info = V.reference_design(train, test, ["v"], False)
    a2, _, info2 = V.reference_design(train, test.assign(v=[-1e9, 1e9]), ["v"], False)
    np.testing.assert_array_equal(a, a2)
    assert info == info2 and info["medians"]["v"] == 2
    assert info["dummy_columns"] == ["municipal_district_type_a"]
    np.testing.assert_array_equal(b[0], [0, 0])


def test_paired_region_bootstrap_constant_effect_and_row_order_invariance():
    y = np.zeros((5, 2))
    baseline = np.arange(1, 11).reshape(5, 2)
    alternative = baseline / 2
    regions = np.array([1, 1, 1, 2, 3])
    result = V.bootstrap(y, baseline, alternative, regions)
    np.testing.assert_allclose(result, [0.75, 0.75, 0.75], atol=1e-14)
    order = [4, 1, 3, 2, 0]
    np.testing.assert_array_equal(
        result, V.bootstrap(y[order], baseline[order], alternative[order], regions[order])
    )


def test_permutation_respects_regions_missing_flags_and_split_donors():
    train = pd.DataFrame({"region_code": [1, 1, 1, 2, 2, 2]})
    for column, flag in zip(V.NEW, V.NEW_FLAGS, strict=True):
        train[column] = [1, np.nan, 2, 3, 4, 5]
        train[flag] = train[column].isna()
    test = train.copy()
    test[V.NEW] += 100
    for seed in [91013, 91029, 91047]:
        for original in [train, test]:
            shuffled = V.reference_shuffle(original, seed)
            pd.testing.assert_frame_equal(shuffled[V.NEW].isna(), original[V.NEW].isna())
            pd.testing.assert_frame_equal(shuffled[V.NEW_FLAGS], original[V.NEW_FLAGS])
            for region in [1, 2]:
                for column in V.NEW:
                    mask = original.region_code.eq(region)
                    np.testing.assert_array_equal(
                        np.sort(shuffled.loc[mask, column].dropna()),
                        np.sort(original.loc[mask, column].dropna()),
                    )
        assert V.reference_shuffle(train, seed)[V.NEW].max().max() < 100
        assert V.reference_shuffle(test, seed)[V.NEW].min().min() > 100


def test_all_empty_training_feature_has_fixed_zero_fill_and_constant_scale():
    train = pd.DataFrame({"municipal_district_type": ["a", "a"], "v": [np.nan, np.nan]})
    test = pd.DataFrame({"municipal_district_type": ["a", "b"], "v": [np.nan, 100]})
    a, b, info = V.reference_design(train, test, ["v"], False)
    np.testing.assert_array_equal(a[:, 1], [0, 0])
    np.testing.assert_array_equal(b[:, 1], [0, 100])
    assert info["train_empty_columns"] == ["v"]
    assert info["sd"]["v"] == 1
