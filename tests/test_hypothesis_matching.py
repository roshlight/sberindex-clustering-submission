import itertools
import json

import numpy as np
import pandas as pd
import pytest

from municipal_profiles.dynamics import PROFILE_COLUMNS
from municipal_profiles.enriched_analysis import CORE
from municipal_profiles.hypothesis_matching import (
    MODEL_PANEL,
    SECTOR_COLUMNS,
    _h3_pairs,
    _h4_pairs,
    _hellinger_pair_mean,
    _minimum_cost_matching,
    _profile_permutation_null,
    _sign_flip_null,
    _standardize,
    _tail_p,
    run_matching_tests,
)


def _panel():
    rows = []
    for region in range(4):
        for level, share in enumerate([0.1, 0.2, 0.4, 0.5]):
            high = level >= 2
            profile = [0.3, 0.1, 0.4, 0.1, 0.1] if high else [0.3, 0.1, 0.2, 0.1, 0.3]
            rows.append(
                {
                    "territory_id": 4 * region + level,
                    "year": 2024,
                    "region_code": region,
                    "municipal_district_type": "district",
                    "core_eligible": True,
                    **dict.fromkeys(CORE, 1.0),
                    **dict.fromkeys(SECTOR_COLUMNS, 0.02),
                    "sector_C_share": share,
                    **dict(zip(PROFILE_COLUMNS, profile, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def _write_panel(tmp_path, frame):
    source = tmp_path / MODEL_PANEL
    source.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)


def _brute_matching(edges, costs):
    candidates = []
    for mask in itertools.product([False, True], repeat=len(edges)):
        selected = np.flatnonzero(mask)
        if len(np.unique(edges[selected])) == 2 * len(selected):
            candidates.append((len(selected), float(costs[selected].sum())))
    cardinality = max(size for size, _ in candidates)
    return cardinality, min(cost for size, cost in candidates if size == cardinality)


@pytest.mark.parametrize(
    ("edges", "costs"),
    [
        # The cheapest edge blocks a two-pair solution: cardinality must win.
        ([(0, 1), (0, 2), (1, 3)], [0.0, 100.0, 100.0]),
        # An odd cycle verifies integer matching rather than fractional packing.
        ([(0, 1), (1, 2), (0, 2), (2, 3)], [0.7, 0.2, 0.1, 0.3]),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], [1.0, 0.2, 0.3, 1.0]),
    ],
)
def test_matching_lexicographic_optimum_agrees_with_exhaustive_search(edges, costs):
    edges, costs = np.asarray(edges), np.asarray(costs)
    selected = _minimum_cost_matching(4, edges, costs)
    size, cost = _brute_matching(edges, costs)
    assert len(selected) == size
    assert costs[selected].sum() == pytest.approx(cost)
    assert len(np.unique(edges[selected])) == 2 * size
    assert np.array_equal(selected, _minimum_cost_matching(4, edges, costs))


def test_matching_rejects_bad_edges_and_costs():
    with pytest.raises(ValueError, match="duplicate"):
        _minimum_cost_matching(2, [(0, 1), (1, 0)], [1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        _minimum_cost_matching(2, [(0, 1)], [-1])


def test_h3_matching_is_outcome_blind_and_respects_strata_and_calipers():
    frame = _panel()
    frame.loc[0, "log_population"] = 20
    # A different municipal type has its own median and cannot share pairs.
    frame.loc[[8, 9], "municipal_district_type"] = "city"
    cohort, pairs, _, _ = _h3_pairs(frame)
    changed = frame.copy()
    changed[list(PROFILE_COLUMNS)] = np.random.default_rng(22).dirichlet(np.ones(5), len(frame))
    pd.testing.assert_frame_equal(pairs, _h3_pairs(changed)[1])
    assert pairs.left_region_code.equals(pairs.right_region_code)
    assert pairs.left_municipal_district_type.equals(pairs.right_municipal_district_type)
    assert (pairs.core_max_abs_z <= 0.5).all()
    assert (pairs.share_C_difference > 0).all()
    assert not pairs[["left_territory_id", "right_territory_id"]].eq(0).any().any()
    assert len(np.unique(pairs[["left_index", "right_index"]])) == 2 * len(pairs)
    assert cohort.high_manufacturing.equals(cohort.sector_C_share.gt(cohort.share_C_median))


def test_h4_requires_sector_similarity_not_only_low_weighted_cost():
    frame = _panel().iloc[:2].copy()
    frame.region_code = [1, 2]
    frame.sector_A_share = [0.01, 0.9]
    frame.sector_C_share = 0.1
    # Two independent differing sectors give RMS sqrt(2*2^2/8)=1: admitted.
    # Three give sqrt(3*2^2/8)>1: excluded even with identical economic data.
    frame.sector_B_share = [0.01, 0.9]
    assert len(_h4_pairs(frame)[1]) == 1
    frame.sector_D_share = [0.01, 0.9]
    assert _h4_pairs(frame)[1].empty


def test_h4_pairs_different_regions_without_using_profiles_or_type():
    frame = _panel()
    frame.loc[frame.region_code.eq(1), "municipal_district_type"] = "city"
    _, pairs, _, _ = _h4_pairs(frame)
    changed = frame.copy()
    changed[list(PROFILE_COLUMNS)] = 0.2
    pd.testing.assert_frame_equal(pairs, _h4_pairs(changed)[1])
    assert (pairs.left_region_code != pairs.right_region_code).all()
    assert (pairs.core_max_abs_z <= 0.5).all()
    assert (pairs.sector_rms_z <= 1.0).all()
    assert len(np.unique(pairs[["left_index", "right_index"]])) == 2 * len(pairs)


def test_zero_variance_standardization_uses_unit_scale():
    values, _, scale = _standardize(pd.DataFrame({"a": [2.0, 2.0]}), ["a"])
    np.testing.assert_array_equal(values, [[0], [0]])
    np.testing.assert_array_equal(scale, [1])


def test_sign_flips_are_shared_by_region_and_by_outcome():
    differences = np.array([[2.0, -4.0], [3.0, -6.0], [1.0, -2.0]])
    null = _sign_flip_null(differences, np.array([1, 1, 2]), 99, 13)
    assert set(np.round(null[:, 0], 10)).issubset(set(np.round([-2, -4 / 3, 4 / 3, 2], 10)))
    np.testing.assert_allclose(null[:, 1], -2 * null[:, 0])
    np.testing.assert_array_equal(null, _sign_flip_null(differences, [1, 1, 2], 99, 13))


def test_profile_permutations_keep_whole_vectors_and_singletons_fixed():
    profiles = np.array([[0.8, 0.05, 0.05, 0.05, 0.05], [0.2] * 5, [0.05] * 4 + [0.8]])
    left, right = np.array([0]), np.array([2])
    null = _profile_permutation_null(profiles, [1, 1, 2], left, right, 99, 13)
    expected = [_hellinger_pair_mean(profiles, np.array([source]), right) for source in [0, 1]]
    assert set(np.round(null, 10)) == set(np.round(expected, 10))
    singleton = _profile_permutation_null(profiles, [1, 2, 3], left, right, 9, 13)
    np.testing.assert_allclose(singleton, expected[0])


def test_directional_plus_one_p_values_include_ties():
    null = np.array([-1.0, 0.0, 1.0])
    assert _tail_p(1, null, "positive") == 0.5
    assert _tail_p(-1, null, "negative") == 0.5
    assert _tail_p(2, null, "positive") == 0.25
    assert _tail_p(0, np.zeros(3), "negative") == 1


def test_runner_writes_audit_and_keeps_missing_sectors_out_of_h4_only(tmp_path):
    frame = _panel()
    frame.loc[0, "sector_A_share"] = np.nan
    _write_panel(tmp_path, frame)
    output = tmp_path / "matching"
    result = run_matching_tests(tmp_path, output, n_resamples=99)
    assert result.test_id.tolist() == [
        "H3_manufacturing_marketplaces",
        "H3_manufacturing_transport",
        "H4_cross_region_similarity",
    ]
    assert result.n_eligible.tolist() == [16, 16, 15]
    assert result.available.all()
    assert result.estimate.iloc[0] > 0
    assert result.estimate.iloc[1] < 0
    assert result.estimate.iloc[2] == pytest.approx(
        result.observed.iloc[2] - result.null_mean.iloc[2]
    )
    assert result.improvement.iloc[2] == pytest.approx(-result.estimate.iloc[2])
    assert result.p_raw.between(0.01, 1).all()
    assert not any("holm" in column.lower() for column in result.columns)
    assert len(pd.read_csv(output / "nulls_H3.csv")) == 99
    assert len(pd.read_csv(output / "nulls_H4.csv")) == 99
    assert {"H3", "H4"} == set(pd.read_csv(output / "balance.csv").hypothesis)
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["H4"]["eligible_entities"] == 15
    hashes = json.loads((output / "source_hashes.json").read_text())
    assert hashes[0]["source"] == MODEL_PANEL and len(hashes[0]["sha256"]) == 64
    repeat = run_matching_tests(tmp_path, tmp_path / "repeat", n_resamples=99)
    pd.testing.assert_frame_equal(result, repeat)


def test_unavailable_tests_keep_p_one_and_the_declared_three_rows(tmp_path):
    frame = _panel().iloc[:1]
    _write_panel(tmp_path, frame)
    result = run_matching_tests(tmp_path, tmp_path / "matching", n_resamples=9)
    assert len(result) == 3
    assert not result.available.any()
    assert result.p_raw.eq(1).all()
    assert result.n_pairs.eq(0).all()
    assert result.estimate.isna().all()


@pytest.mark.parametrize("problem", ["duplicate", "bad_profile", "bad_sector", "bad_flag"])
def test_runner_rejects_invalid_source_instead_of_silent_selection(tmp_path, problem):
    frame = _panel()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "bad_profile":
        frame.loc[0, "p_food"] = 0
    elif problem == "bad_sector":
        frame.loc[0, "sector_C_share"] = 1.5
    else:
        frame["core_eligible"] = frame.core_eligible.astype(object)
        frame.loc[0, "core_eligible"] = "unknown"
    _write_panel(tmp_path, frame)
    with pytest.raises(ValueError):
        run_matching_tests(tmp_path, tmp_path / "matching", n_resamples=9)
