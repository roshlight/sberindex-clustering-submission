"""Reject snapshot revision, key drift and promotion of unusable monetary values."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("round4_audit", SCRIPTS / "round4_data_audit.py")
M = importlib.util.module_from_spec(SPEC)
with patch("sys.path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(M)


def cells():
    return pd.DataFrame(
        {
            "territory_id": [1, 2],
            "year": [2023, 2023],
            "indicator_id": ["8109001"] * 2,
            "source_region_url": ["https://example.test"] * 2,
            "source_oktmo": ["01101000", "01102000"],
            "official_oktmo_8": ["01101000", "01102000"],
            "code_reference_date": ["2023-01-01"] * 2,
            "temporal_review_required": [False, True],
            "value": [100.0, np.nan],
            "value_status": ["observed", "not_returned_or_unaccepted"],
            "source_sha256": ["old-hash", None],
        }
    )


def test_only_missing_numeric_is_gain_and_zero_is_preserved():
    old = cells()
    new = old.copy()
    new.loc[1, ["value", "value_status", "source_sha256"]] = [0.0, "observed", "new-hash"]
    delta = M.compare_candidates(old, new)
    assert delta.numeric_gain.tolist() == [False, True]
    assert delta.old_numeric_preserved.tolist() == [True, False]


@pytest.mark.parametrize("replacement", [101.0, np.nan])
def test_reject_old_numeric_revision_or_loss(replacement):
    old = cells()
    new = old.copy()
    new.loc[0, "value"] = replacement
    with pytest.raises(ValueError, match="changed or removed"):
        M.compare_candidates(old, new)


@pytest.mark.parametrize(
    "column,value",
    [
        ("source_oktmo", "09999000"),
        ("year", 2024),
        ("temporal_review_required", False),
        ("code_reference_date", "2024-01-01"),
    ],
)
def test_reject_annual_identity_drift(column, value):
    old = cells()
    new = old.copy()
    new.loc[1, column] = value
    with pytest.raises(AssertionError):
        M.compare_candidates(old, new)


def test_reject_old_numeric_provenance_change():
    old = cells()
    new = old.copy()
    new.loc[0, "source_sha256"] = "new-hash"
    with pytest.raises(AssertionError):
        M.compare_candidates(old, new)


def test_duplicate_identity_cannot_be_counted_as_gain():
    old = cells()
    with pytest.raises(ValueError, match="duplicate"):
        M.compare_candidates(old, pd.concat([old, old.iloc[[0]]], ignore_index=True))


def feature_pair():
    old = pd.DataFrame(
        {
            "territory_id": [1, 2, 3],
            "year": [2023] * 3,
            "core_eligible": [True, True, False],
            "conservative_eligible": [True, False, False],
            "code_type_changed": [False, True, False],
            "calendar_review": [False] * 3,
            "population_total": [1000.0, 2000.0, np.nan],
            "activity_all3_features_available": [True, False, False],
        }
    )
    for prefix in M.PREFIXES:
        old[f"{prefix}_thousand_rubles"] = [1.0, np.nan, np.nan]
        old[f"{prefix}_log_rubles_per_person"] = [0.0, np.nan, np.nan]
        old[f"{prefix}_missing"] = [False, True, True]
        old[f"{prefix}_feature_status"] = [
            "eligible",
            "source_value_missing",
            "source_value_missing",
        ]
    new = old.copy()
    for prefix in M.PREFIXES:
        new[f"{prefix}_thousand_rubles"] = [1.0, 2.0, 3.0]
        new[f"{prefix}_log_rubles_per_person"] = [0.0, 0.0, np.nan]
        new[f"{prefix}_missing"] = False
    new["activity_all3_features_available"] = [True, True, False]
    return old, new


def test_common_complete_excludes_newly_complete_and_invalid_denominator():
    old, new = feature_pair()
    result = M.add_comparison_flags(old, new)
    assert result.activity_new_numeric_count.tolist() == [0, 3, 3]
    assert result.activity_new_feature_count.tolist() == [0, 3, 0]
    assert result.activity_all3_features_common.tolist() == [True, False, False]
    assert result.activity_all3_features_newly_complete.tolist() == [False, True, False]
    assert result.core_eligible.tolist() == old.core_eligible.tolist()
    assert result.conservative_eligible.tolist() == old.conservative_eligible.tolist()
    assert result.old_investment_log_rubles_per_person.isna().tolist() == [False, True, True]


def test_new_data_cannot_promote_frozen_mask():
    old, new = feature_pair()
    new.loc[2, "core_eligible"] = True
    with pytest.raises(AssertionError):
        M.add_comparison_flags(old, new)
