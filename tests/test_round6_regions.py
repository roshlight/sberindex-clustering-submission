import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

SPEC = importlib.util.spec_from_file_location(
    "round6_regions_test", Path(__file__).parents[1] / "scripts/analyze_round6_regions.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_rank_statistics_agree_with_spearman_and_independent_projection():
    values = np.array([[1, 4, 2], [3, 1, 2], [2, 2, 4], [3, 5, 1], [5, 2, 3]])
    rho, partial = M.rank_statistics(values)
    assert rho == pytest.approx(spearmanr(values[:, 0], values[:, 1]).statistic)
    ranks = pd.DataFrame(values).rank().to_numpy()
    z = np.column_stack([np.ones(len(values)), ranks[:, 2]])
    residuals = ranks[:, :2] - z @ np.linalg.lstsq(z, ranks[:, :2], rcond=None)[0]
    assert partial == pytest.approx(np.corrcoef(residuals.T)[0, 1])


def test_batch_resampling_recomputes_ranks_for_repeated_rows():
    rng = np.random.default_rng(32)
    values = rng.normal(size=(20, 4))
    draws = rng.integers(0, 20, size=(5, 20))
    got = np.array(M.rank_statistics(values[draws])).T
    expected = np.array([M.rank_statistics(values[indices]) for indices in draws])
    np.testing.assert_allclose(got, expected)
    assert not np.allclose(got[:, 0], float(M.rank_statistics(values)[0]))


def test_constant_attribute_is_undefined_not_zero():
    values = np.column_stack([np.ones(12), np.arange(12), np.arange(12)[::-1]])
    rho, partial = M.rank_statistics(values)
    assert np.isnan(rho) and np.isnan(partial)


def test_partial_adjustment_does_not_impute_absent_controls():
    values = np.arange(36, dtype=float).reshape(12, 3)
    values[0, 2] = np.nan
    with pytest.raises(np.linalg.LinAlgError):
        M.rank_statistics(values)


def test_cluster_labels_are_permutation_invariant_not_ordered_numbers():
    values = np.array([1, 2, 3, 8, 9, 10])
    first = M.rank_eta_squared(values, np.array([1, 1, 1, 2, 2, 2]))
    second = M.rank_eta_squared(values, np.array([900, 900, 900, -4, -4, -4]))
    assert first == second and 0 < first < 1
    with pytest.raises(ValueError, match="No attribute variation"):
        M.rank_eta_squared(np.ones(6), np.arange(6))


def test_bh_adjust_preserves_original_order():
    np.testing.assert_allclose(
        M.bh_adjust([0.2, 0.01, 0.03, 0.9]), [0.2666666666667, 0.04, 0.06, 0.9]
    )


def test_two_year_sample_requires_both_years_and_unique_keys():
    frame = pd.DataFrame({"territory_id": [1, 1, 2, 3], "year": [2023, 2024, 2023, 2024]})
    assert M.paired_ids(frame) == [1]
    with pytest.raises(ValueError, match="Duplicate"):
        M.paired_ids(pd.concat([frame, frame.iloc[:1]]))
