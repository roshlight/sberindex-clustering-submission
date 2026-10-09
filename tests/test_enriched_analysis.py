import numpy as np
import pandas as pd
import pytest

from municipal_profiles.enriched_analysis import (
    clr,
    design,
    join_unique,
    paired_region_loss_ci,
    r2_matrix,
    within_region_folds,
)


def test_ambiguous_join_fails_instead_of_multiplying_rows():
    left = pd.DataFrame({"id": [1, 2]})
    right = pd.DataFrame({"id": [1, 1], "value": [3, 4]})
    with pytest.raises(ValueError, match="Nonunique"):
        join_unique(left, right, ["id"], "duplicate_source", [])


def test_missing_join_is_preserved_and_counted():
    audit = []
    result = join_unique(
        pd.DataFrame({"id": [1, 2]}),
        pd.DataFrame({"id": [1], "value": [0]}),
        ["id"],
        "source",
        audit,
    )
    assert len(result) == 2 and result.value.iloc[0] == 0 and np.isnan(result.value.iloc[1])
    assert audit[0]["unmatched_left"] == 1


def test_clr_recovers_log_ratios_and_rejects_zero():
    result = clr([[0.2, 0.3, 0.5]])
    assert result.sum() == pytest.approx(0)
    assert result[0, 0] - result[0, 1] == pytest.approx(np.log(2 / 3))
    with pytest.raises(ValueError):
        clr([[0, 0.5, 0.5]])


def test_scaling_and_categories_fit_on_training_rows_only():
    train = pd.DataFrame(
        {"region_code": [1, 1], "municipal_district_type": ["a", "a"], "feature": [1.0, 3.0]}
    )
    test = pd.DataFrame(
        {"region_code": [99], "municipal_district_type": ["unseen"], "feature": [101.0]}
    )
    x, z = design(train, test, ["feature"], region=True)
    assert x[:, -1].tolist() == [-1, 1]
    assert z[0, -1] == 99
    assert np.all(z[:, :-1] == 0)


def test_balanced_folds_and_paired_loss_reduction():
    groups = np.repeat(np.arange(8), 10)
    folds = within_region_folds(groups)
    for g in range(8):
        assert np.bincount(folds[groups == g], minlength=5).tolist() == [2] * 5
    y = np.arange(80)[:, None]
    point, low, high = paired_region_loss_ci(y, y + 2, y + 1, groups, repeats=100)
    assert (
        point == pytest.approx(0.75) and low == pytest.approx(0.75) and high == pytest.approx(0.75)
    )
    assert r2_matrix(y, y) == 1
    assert r2_matrix(y, np.full_like(y, y.mean(), dtype=float)) == 0
