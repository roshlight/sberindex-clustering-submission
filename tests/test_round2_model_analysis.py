"""Protect the incremental comparison against preprocessing and spatial leakage."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "round2_model", Path(__file__).resolve().parents[1] / "scripts/round2_model_analysis.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_validation_values_do_not_change_training_imputation_or_scale():
    train = pd.DataFrame(
        {
            "municipal_district_type": ["x"] * 3,
            "region_code": [1, 1, 2],
            "share": [0.2, np.nan, 0.8],
        }
    )
    test = pd.DataFrame(
        {
            "municipal_district_type": ["unseen", "x"],
            "region_code": [99, 1],
            "share": [np.nan, 1000.0],
        }
    )
    x, z, audit = M.design(train, test, ["share"], True)
    changed = test.assign(share=[-1000.0, 2000.0])
    x2, _, audit2 = M.design(train, changed, ["share"], True)
    np.testing.assert_array_equal(x, x2)
    assert audit == audit2
    assert audit["medians"]["share"] == 0.5
    assert z[0, -1] == 0.0
    assert not any("unseen" in c or "99" in c for c in audit["dummy_columns"])


def test_empty_train_feature_is_audited_and_keeps_finite_design():
    train = pd.DataFrame({"municipal_district_type": ["x", "x"], "share": [np.nan, np.nan]})
    test = pd.DataFrame({"municipal_district_type": ["x"], "share": [0.7]})
    x, z, audit = M.design(train, test, ["share"], False)
    assert audit["train_empty_columns"] == ["share"]
    np.testing.assert_array_equal(x[:, -1], [0.0, 0.0])
    assert z[0, -1] == 0.7 and np.isfinite(z).all()


def test_region_holdout_has_no_shared_region():
    frame = pd.DataFrame({"region_code": np.repeat(np.arange(10), 7)})
    folds = M.folds_for(frame, "heldout_regions", 13)
    for fold in range(5):
        assert not set(frame.loc[folds == fold, "region_code"]) & set(
            frame.loc[folds != fold, "region_code"]
        )


def test_negative_control_preserves_missingness_and_region_distributions():
    frame = pd.DataFrame({"region_code": [1, 1, 1, 1, 2, 2, 2, 2]})
    for column in M.SECTORS:
        frame[column] = [0.1, np.nan, 0.2, 0.3, np.nan, 0.6, 0.8, 0.9]
    result = M.shuffle_observed(frame)
    for column in M.SECTORS:
        assert frame[column].isna().equals(result[column].isna())
        for region in [1, 2]:
            mask = frame.region_code == region
            np.testing.assert_array_equal(
                np.sort(frame.loc[mask, column]), np.sort(result.loc[mask, column])
            )


def test_constant_profile_r2_is_not_averaged_over_unidentifiable_coordinates():
    y = np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0], [3.0, -3.0, 0.0]])
    p = np.zeros_like(y)
    expected = 1 - np.square(y - p).sum() / np.square(y - y.mean(axis=0)).sum()
    assert M.r2_matrix(y, p) == expected
