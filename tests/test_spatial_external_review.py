"""Spatial identity, pair weighting, exact thresholds and blocked-null checks."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "external_review", SCRIPTS / "review_spatial_external.py"
)
review = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(review)


def roads(rows):
    return pd.DataFrame(rows, columns=["territory_id_x", "territory_id_y", "distance", "type"])


def test_reverse_pairs_count_once_without_mixing_transport_types():
    result, audit = review.canonical_roads(
        roads(
            [
                (1, 2, 10.0, "highway"),
                (2, 1, 10.0, "highway"),
                (1, 2, 7.0, "railway"),
            ]
        )
    )
    assert result.to_dict("records") == [{"a": 1, "b": 2, "distance": 10.0}]
    assert audit["consistent_reverse_rows_collapsed"] == 1


def test_conflicting_reverse_distances_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        review.canonical_roads(roads([(1, 2, 10.0, "highway"), (2, 1, 11.0, "highway")]))


def test_duplicate_direction_rejected_even_if_values_equal():
    with pytest.raises(ValueError, match="duplicate"):
        review.canonical_roads(roads([(1, 2, 10.0, "highway")] * 2))


@pytest.mark.parametrize(
    "row",
    [
        (1, 1, 0.0, "highway"),
        (1, 2, -1.0, "highway"),
        (1, 2, np.inf, "highway"),
        (1.5, 2, 1.0, "highway"),
    ],
)
def test_invalid_road_rows_rejected(row):
    with pytest.raises(ValueError):
        review.canonical_roads(roads([row]))


def test_hand_calculated_pair_mean_and_inclusive_cdf():
    labels = np.array([0, 0, 1, 1])
    a, b = np.array([0, 0, 2]), np.array([1, 2, 3])
    distance = np.array([10.0, 999.0, 30.0])
    thresholds = np.array([10.0, 20.0, 30.0])
    bins = np.searchsorted(thresholds, distance, side="left")
    value, cdf, count = review.road_statistic(labels, a, b, distance, bins, len(thresholds))
    assert value == 20 and count == 2
    np.testing.assert_allclose(cdf, [0.5, 0.5, 1.0])
    # Renaming groups does not alter equality or any pair statistic.
    renamed = 9 - labels
    assert review.road_statistic(renamed, a, b, distance, bins, 3)[0] == value


def test_missing_distance_is_not_an_invented_zero():
    labels = np.array([0, 0, 0])
    value, _, count = review.road_statistic(
        labels, np.array([0]), np.array([1]), np.array([12.0]), np.array([0]), 1
    )
    assert value == 12 and count == 1


def test_between_share_hand_example_and_constant_guard():
    assert review.between_share([0, 0, 2, 2], [0, 0, 1, 1]) == 1
    assert review.between_share([0, 0, 2, 2], [0, 1, 0, 1]) == 0
    with pytest.raises(ValueError, match="Constant"):
        review.between_share([1, 1], [0, 1])


def test_regional_group_sizes_preserved():
    labels = np.array([0, 0, 1, 1, 2, 2])
    blocks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    rng = np.random.default_rng(1905)
    for _ in range(20):
        shuffled = review.shuffle_labels(labels, blocks, rng)
        for block in blocks:
            np.testing.assert_array_equal(np.sort(shuffled[block]), np.sort(labels[block]))


def test_null_tails_and_degenerate_case():
    null = np.array([1.0, 2.0, 3.0])
    assert review.null_summary(0.0, null, True)["p_raw"] == 0.25
    assert review.null_summary(0.0, null, False)["p_raw"] == 1.0
    assert review.null_summary(1.0, np.ones(3), True)["null_status"] == "degenerate"


def test_coverage_floor_is_enforced():
    review.check_coverage(95, 100, 0.95, "test")
    with pytest.raises(ValueError, match="Insufficient"):
        review.check_coverage(94, 100, 0.95, "test")
