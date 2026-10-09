import numpy as np
import pandas as pd
import pytest

from municipal_profiles.hypothesis_summary import TEST_IDS, holm_adjust, summarize_tests


def example():
    return pd.DataFrame(
        {
            "test_id": TEST_IDS,
            "hypothesis": ["H1", "H2", "H3", "H3", "H4"],
            "year": 2024,
            "estimate": [0.2, 0.3, 0.04, -0.05, -0.1],
            "expected_direction": ["positive", "positive", "positive", "negative", "negative"],
            "p_raw": [0.001, 0.002, 0.01, 0.2, 0.003],
        }
    )


def test_holm_preserves_input_order_and_known_step_down_values():
    np.testing.assert_allclose(
        holm_adjust([0.04, 0.01, 0.05, 0.03, 0.02]), [0.09, 0.05, 0.09, 0.09, 0.08]
    )
    np.testing.assert_allclose(holm_adjust([0, 1, 1]), [0, 1, 1])


def test_one_significant_component_does_not_support_joint_h3():
    result = summarize_tests(example().iloc[::-1]).set_index("test_id")
    assert result.loc[TEST_IDS[2], "supported"]
    assert not result.loc[TEST_IDS[3], "supported"]
    assert not result.loc[TEST_IDS[2], "hypothesis_supported"]
    assert not result.loc[TEST_IDS[3], "hypothesis_supported"]


def test_wrong_direction_and_unavailable_test_cannot_pass():
    raw = example()
    raw.loc[0, "estimate"] = -0.2
    raw["available"] = True
    raw.loc[4, ["available", "estimate", "p_raw"]] = [False, np.nan, np.nan]
    result = summarize_tests(raw)
    assert not result.loc[0, "supported"]
    assert result.loc[4, "p_raw"] == 1
    assert result.loc[4, "decision"] == "unavailable"
    assert result.loc[0, "p_holm"] == pytest.approx(0.005)


@pytest.mark.parametrize("pvalues", [[-0.1], [1.1], [np.nan], [[0.1]]])
def test_invalid_pvalues_are_rejected(pvalues):
    with pytest.raises(ValueError):
        holm_adjust(pvalues)


def test_omitting_a_failed_contrast_is_rejected():
    with pytest.raises(ValueError, match="five"):
        summarize_tests(example().iloc[:-1])


@pytest.mark.parametrize("column,value", [("hypothesis", "H5"), ("expected_direction", "negative")])
def test_changing_a_declared_group_or_direction_is_rejected(column, value):
    raw = example()
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match="Declared"):
        summarize_tests(raw)
