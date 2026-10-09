"""Guard fixed folds and the new block's falsification control."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("round3_model", SCRIPTS / "round3_model_analysis.py")
M = importlib.util.module_from_spec(SPEC)
with patch("sys.path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(M)


def test_negative_control_keeps_missing_flags_and_regional_distributions():
    frame = pd.DataFrame({"region_code": [1] * 4 + [2] * 4, "unchanged": range(8)})
    for feature, flag in zip(M.NEW, M.NEW_FLAGS, strict=True):
        frame[feature] = [1.0, np.nan, 3.0, 4.0, 101.0, 102.0, np.nan, 104.0]
        frame[flag] = frame[feature].isna().astype(float)
    result = M.shuffle_new(frame)
    pd.testing.assert_frame_equal(frame.drop(columns=M.NEW), result.drop(columns=M.NEW))
    for feature in M.NEW:
        assert result[feature].isna().equals(frame[feature].isna())
        for region in [1, 2]:
            mask = frame.region_code.eq(region)
            np.testing.assert_array_equal(
                np.sort(frame.loc[mask, feature]), np.sort(result.loc[mask, feature])
            )


def test_permutation_cannot_draw_from_other_partition():
    train = pd.DataFrame({"region_code": [1] * 7})
    test = pd.DataFrame({"region_code": [1] * 7})
    for feature in M.NEW:
        train[feature] = range(7)
        test[feature] = range(100, 107)
    assert M.shuffle_new(train)[M.NEW].max().max() < 100
    assert M.shuffle_new(test)[M.NEW].min().min() >= 100


def test_prior_fold_check_rejects_missing_extra_and_changed_rows():
    d = pd.DataFrame({"territory_id": [10, 20, 30], "year": [2024] * 3})
    label = dict(year=2024, sample="core", regime="within_region", seed=13)
    prior = d.assign(
        model="economic", fold=[0, 1, 2], **{k: v for k, v in label.items() if k != "year"}
    )
    M.assert_prior_folds(d, np.array([0, 1, 2]), prior, label)
    for changed in [
        prior.iloc[:-1],
        prior.assign(territory_id=[10, 20, 99]),
        prior.assign(fold=[0, 1, 4]),
    ]:
        with pytest.raises(AssertionError):
            M.assert_prior_folds(d, np.array([0, 1, 2]), changed, label)
