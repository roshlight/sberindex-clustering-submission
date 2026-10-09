"""Protect the vintage transition and unchanged municipality-key fold assignments."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "round5_model_analysis",
    Path(__file__).resolve().parents[1] / "scripts/round5_model_analysis.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def panels():
    old = pd.DataFrame(
        {
            "territory_id": [1, 2],
            "year": [2023, 2023],
            "region_code": [25, 12],
            "core_eligible": [True, True],
            "log_total": [2.0, 4.0],
            "recovered_any": [True, False],
            "old_observed_count": [0, 0],
            "new_observed_count": [0, 0],
            "common_complete_new3": [False, False],
        }
    )
    for feature, flag in zip(module.NEW, module.NEW_FLAGS, strict=True):
        old[feature] = [np.nan, 3.0]
        old[flag] = [1, 0]
        old["old_" + feature] = [-99.0, -99.0]
        old["old_" + flag] = [0, 1]
    new = old[module.KEY + module.NEW + module.NEW_FLAGS].copy()
    new.loc[0, module.NEW[1]] = 5.0
    new.loc[0, module.NEW_FLAGS[1]] = 0
    return old, new


def test_old_vintage_is_round4_not_nested_round3_and_counts_are_rebuilt():
    old, new = panels()
    panel = module.combine_panels(old, new)
    assert panel.columns.is_unique
    assert panel[module.OLD].iloc[0].isna().all()
    assert panel[module.OLD].iloc[1].eq(3).all()
    assert panel.old_observed_count.tolist() == [0, 3]
    assert panel.new_observed_count.tolist() == [1, 3]
    assert panel.recovered_feature_count.tolist() == [1, 0]
    assert panel.common_complete_new3.tolist() == [False, True]
    assert panel.log_total.equals(old.log_total)
    assert panel.recovery_region.tolist() == [True, False]


def test_missing_new_key_is_not_silently_dropped():
    old, new = panels()
    with pytest.raises(AssertionError):
        module.combine_panels(old, new.iloc[:1])


def test_recovery_must_not_revise_an_existing_value():
    old, new = panels()
    new.loc[1, module.NEW[0]] = 30
    with pytest.raises(AssertionError):
        module.combine_panels(old, new)


def test_missingness_flag_cannot_disagree_with_value():
    old, new = panels()
    new.loc[0, module.NEW_FLAGS[0]] = 0
    with pytest.raises(AssertionError):
        module.combine_panels(old, new)


def test_subset_keeps_keyed_prior_core_folds_even_when_row_order_changes():
    prior = pd.DataFrame(
        {
            "territory_id": list(range(10)),
            "year": 2023,
            "sample": "core",
            "regime": "heldout_regions",
            "seed": 13,
            "model": "recovered_full",
            "fold": np.arange(10) % 5,
        }
    )
    data = prior.loc[[9, 6, 3, 0, 7], module.KEY]
    actual = module.frozen_folds(
        data, prior, 2023, "core_without_recovery_regions", "heldout_regions", 13
    )
    np.testing.assert_array_equal(actual, [4, 1, 3, 0, 2])
    with pytest.raises(AssertionError):
        module.frozen_folds(data, prior, 2023, "stable_code", "heldout_regions", 13)
