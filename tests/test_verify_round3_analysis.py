"""Small independent-reference checks, without re-running or tuning models."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "round3_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_round3_analysis.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_reference_ridge_analytic_shrinkage_and_unpenalized_intercept():
    train = np.array([[-1.0], [0.0], [1.0]])
    target = 3 + 2 * train
    prediction = V.reference_ridge(train, target, np.array([[-2.0], [2.0]]))
    np.testing.assert_allclose(prediction.ravel(), [1 / 3, 17 / 3], atol=1e-14)


def test_reference_preprocessing_cannot_learn_test_extremes_or_categories():
    train = pd.DataFrame({"municipal_district_type": ["a"] * 3, "v": [1, np.nan, 3]})
    test = pd.DataFrame({"municipal_district_type": ["unseen", "a"], "v": [np.nan, 999]})
    a, b, info = V.reference_design(train, test, ["v"], False)
    a2, _, info2 = V.reference_design(train, test.assign(v=[-1e9, 1e9]), ["v"], False)
    np.testing.assert_array_equal(a, a2)
    assert info == info2 and info["medians"]["v"] == 2
    assert info["dummy_columns"] == ["municipal_district_type_a"]
    np.testing.assert_array_equal(b[0], [0, 0])


def test_bootstrap_pairs_regions_and_is_invariant_to_row_order():
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


def test_negative_control_keeps_split_donors_missing_positions_and_region_values():
    train = pd.DataFrame({"region_code": [1, 1, 1, 2, 2, 2]})
    for column in V.NEW:
        train[column] = [1, np.nan, 2, 3, 4, 5]
    test = train.copy()
    test[V.NEW] = test[V.NEW] + 100
    for original in [train, test]:
        shuffled = V.reference_shuffle(original)
        pd.testing.assert_frame_equal(shuffled[V.NEW].isna(), original[V.NEW].isna())
        for region in [1, 2]:
            for column in V.NEW:
                mask = original.region_code.eq(region)
                np.testing.assert_array_equal(
                    np.sort(shuffled.loc[mask, column].dropna()),
                    np.sort(original.loc[mask, column].dropna()),
                )
    assert V.reference_shuffle(train)[V.NEW].max().max() < 100
    assert V.reference_shuffle(test)[V.NEW].min().min() > 100
