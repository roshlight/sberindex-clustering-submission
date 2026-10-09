"""Attribution guards for the new, isolated city-price extraction."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "round2_price_extract", SCRIPTS / "extract_round2_prices.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bridge_spec = importlib.util.spec_from_file_location(
    "round2_price_bridge", SCRIPTS / "prepare_round2_prices.py"
)
bridge = importlib.util.module_from_spec(bridge_spec)
bridge_spec.loader.exec_module(bridge)


@pytest.mark.parametrize("label", ["пгт.Селенгинск", "пгт. Селенгинск", "пгт Селенгинск"])
def test_type_token_only(label):
    assert bridge.city_literal(label) == "Селенгинск"
    assert bridge.city_literal("Орел") != bridge.city_literal("Орёл")
    assert bridge.city_literal("Старый Оскол") == "Старый Оскол"


def test_exact_region_is_required_for_city_name():
    official = pd.DataFrame(
        [
            {"city_name_key": "Советск", "region_name": "Калининградская область"},
        ]
    )
    assert module.identify_city("27430000000", "Советск", "Калининградская область", official) == (
        "Советск",
        "Калининградская область",
    )
    with pytest.raises(ValueError, match="Unattributed"):
        module.identify_city("33430000000", "Советск", "Кировская область", official)
    with pytest.raises(ValueError, match="Unattributed"):
        module.identify_city("27430000000", "Совет", "Калининградская область", official)


def test_duplicate_identity_is_not_resolved_arbitrarily():
    official = pd.DataFrame([{"city_name_key": "Советск", "region_name": "R"}] * 2)
    with pytest.raises(ValueError, match="Unattributed"):
        module.identify_city("27430000000", "Советск", "R", official)


def test_federal_city_caption_guard():
    official = pd.DataFrame()
    name = module.FEDERAL_CITIES["45000000000"][0]
    assert module.identify_city("45000000000", name, None, official) == ("Москва", "Москва")
    with pytest.raises(ValueError, match="Changed"):
        module.identify_city("45000000000", "Любой район Москвы", None, official)


@pytest.mark.parametrize("value", ["...", "…", "-", "—"])
def test_missing_markers_stay_null(value):
    assert module.source_value(value) == (None, "source_marker", value)
    assert module.source_value(0) == (0.0, "observed_zero", "0")
    assert module.source_value(None) == (None, "source_blank", "")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "125 рублей", "12,34"])
def test_unexpected_number_or_annotation_fails_closed(value):
    with pytest.raises(ValueError):
        module.source_value(value)


def test_rural_homonym_does_not_become_city(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text(
        '"15";"638";"151";"051";"5";"2";"п Навля";;;"000";"0";14.06.2013;01.01.2014\n'
        '"15";"638";"453";"101";"0";"2";"п Навля";;;"000";"0";14.06.2013;01.01.2014\n',
        encoding="utf8",
    )
    rows = bridge.load_registry(path)
    assert rows.city_name_key.tolist() == ["Навля", "Навля"]
    assert rows.urban_settlement_code.tolist() == [True, False]
    assert rows.official_oktmo_11.tolist() == ["15638151051", "15638453101"]


@pytest.mark.parametrize("role", ["city", "parent", "immediate_parent"])
def test_later_date_in_each_role_requires_its_own_temporal_evidence(role):
    dates = dict.fromkeys(["city", "parent", "immediate_parent"], "01.01.2014")
    dates[role] = "16.01.2023"
    assert bridge.temporal_review_roles(dates, 2023) == [role]
    assert bridge.temporal_review_roles(dates, 2024) == []


@pytest.mark.parametrize("date", ["", "unknown", None])
def test_unknown_date_is_not_approved(date):
    assert bridge.temporal_review_roles({"city": date}, 2023) == ["city"]
