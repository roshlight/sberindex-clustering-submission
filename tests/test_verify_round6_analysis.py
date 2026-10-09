"""Independent validator arithmetic and rejection tests for material round-six claims."""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "independent_round6", REPO / "scripts/verify_round6_analysis.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_partial_rank_correlation_matches_three_correlation_identity():
    data = np.array([[1, 1, 5], [2, 3, 2], [3, 2, 4], [4, 5, 1], [5, 4, 3]])
    rho, partial = V.rank_pair(data)
    # Hand-computed Spearman correlations: xy=.8, xz=-.5, yz=-.9.
    assert rho == pytest.approx(0.8)
    assert partial == pytest.approx((0.8 - 0.5 * 0.9) / np.sqrt((1 - 0.5**2) * (1 - 0.9**2)))


def test_duplicate_ids_are_ranked_again_inside_resample():
    data = np.array([[1, 1, 5], [2, 3, 2], [3, 2, 4], [4, 5, 1], [5, 4, 3]])
    draw = np.array([0, 0, 1, 2, 4])
    rank_x = np.array([1.5, 1.5, 3, 4, 5])
    rank_y = np.array([1.5, 1.5, 4, 3, 5])
    assert V.rank_pair(data[draw])[0] == pytest.approx(np.corrcoef(rank_x, rank_y)[0, 1])
    assert not np.isclose(V.rank_pair(data[draw])[0], np.corrcoef(data[draw, :2].T)[0, 1])


def test_rank_eta_ignores_arbitrary_cluster_numbers():
    ranks = np.array([1.0, 2.0, 3.0, 4.0])
    assert V.independent_eta(ranks, np.array([0, 0, 1, 1])) == pytest.approx(0.8)
    assert V.independent_eta(ranks, np.array([99, 99, -7, -7])) == pytest.approx(0.8)
    assert V.independent_eta(ranks, np.array([5, 5, 5, 5])) == 0


def test_bh_adjustment_uses_complete_family():
    np.testing.assert_allclose(
        V.independent_bh([0.01, 0.04, 0.03, 0.2]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2]
    )


def test_rounding_intervals_use_printed_precision():
    assert V.printed_half_step("1,20") == V.Decimal("0.005")
    assert V.printed_half_step("120") == V.Decimal("0.5")
    assert V.independent_rounding("100.0", "120.0", "120.0")["status"] == "rounding_compatible"
    result = V.independent_rounding("100.0", "120.0", "140.0")
    assert result["status"] == "different_beyond_rounding"
    assert result["difference_pp"] == -20


def test_table_comparison_rejects_denominator_and_missingness_changes():
    expected = pd.DataFrame({"year": [2023, 2024], "value": [1.0, np.nan]})
    with pytest.raises(ValueError, match="Missingness"):
        V.compare_table(expected, expected.fillna(0), ["year"], "cash")
    with pytest.raises(ValueError, match="Row count"):
        V.compare_table(expected, expected.iloc[:1], ["year"], "cash")
    changed = expected.copy()
    changed.loc[0, "value"] = 1.01
    with pytest.raises(ValueError, match="Numeric cells"):
        V.compare_table(expected, changed, ["year"], "cash")


def test_source_cohorts_preserve_urban_missingness_and_equal_years():
    panel, groups = V.regional_expected_panel(REPO)
    assert len(groups[(26, "paired_available")][0]) == 33
    assert len(groups[(26, "paired_urban_complete")][0]) == 19
    assert len(groups[(26, "paired_stable_urban_complete")][0]) == 9
    assert len(groups[(64, "paired_available")][0]) == 38
    assert len(groups[(64, "paired_urban_complete")][0]) == 32
    assert not panel.loc[panel.territory_id == 1947, "paired_available"].any()
    assert panel.loc[panel.territory_id == 826, "urban_share"].isna().any()


def test_changed_source_amount_is_rejected_without_relying_on_manifest(tmp_path):
    shutil.copytree(REPO / "reports/round6_analysis/admission", tmp_path / "admission")
    path = tmp_path / "admission/row_admission.csv"
    rows = V.read_strings(path)
    rows.loc[0, "source_value_literal"] = "1.0"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Admission source cells"):
        V.verify_admission(REPO, tmp_path, tmp_path / "unused_raw", [])


def test_zero_for_absent_2024_cash_is_rejected_without_manifest(tmp_path):
    shutil.copytree(REPO / "reports/round6_analysis/context", tmp_path / "context")
    path = tmp_path / "context/airport_comparison.csv"
    rows = pd.read_csv(path)
    rows.loc[rows.year == 2024, "airport_cash_million_rubles"] = 0
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missingness.*airport_cash"):
        V.verify_context(REPO, tmp_path, [])
