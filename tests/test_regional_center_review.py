import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_regional_centers import classify  # noqa: E402


def test_city_status_is_not_transferred_to_district_or_unknown_region():
    cohort = pd.DataFrame(
        {
            "territory_id": ["2338", "2339", "1397", "9000"],
            "region_code_2024": ["79", "79", "47", "50"],
        }
    )
    reference = pd.DataFrame({"region_code": ["79"], "center_territory_id": ["2338"]})
    result = classify(cohort, reference).set_index("territory_id")
    assert result.loc["2338", "matches_reference_center_municipality"]
    assert not result.loc["2339", "matches_reference_center_municipality"]
    assert pd.isna(result.loc["1397", "matches_reference_center_municipality"])
    assert pd.isna(result.loc["9000", "matches_reference_center_municipality"])
    assert result.loc["1397", "center_review_status"] == "city_verified_municipal_grain_unresolved"


def test_two_centers_in_one_region_rejected():
    cohort = pd.DataFrame({"territory_id": ["1"], "region_code_2024": ["79"]})
    reference = pd.DataFrame({"region_code": ["79", "79"], "center_territory_id": ["2338", "2339"]})
    with pytest.raises(ValueError, match="Nonunique"):
        classify(cohort, reference)


def test_repeated_cohort_id_rejected():
    cohort = pd.DataFrame({"territory_id": ["1", "1"], "region_code_2024": ["79", "79"]})
    reference = pd.DataFrame({"region_code": ["79"], "center_territory_id": ["2338"]})
    with pytest.raises(ValueError, match="Nonunique"):
        classify(cohort, reference)
