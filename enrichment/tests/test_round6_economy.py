import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_round6_economy.py"
SPEC = importlib.util.spec_from_file_location("r6_economy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("token", ["...", "…", "-", "–", "Х"])
def test_markers_are_not_zero(token):
    assert MODULE.number(token) is None


def test_russian_number_retains_scale():
    assert MODULE.number("101 571 624,8") == Decimal("101571624.8")
    assert MODULE.number("0.0") == Decimal("0.0")
    with pytest.raises(ValueError):
        MODULE.number("101 млн")


def test_columns_and_reporting_year_are_guarded():
    text = (
        "1.1. ОТГРУЖЕНО БЕЗ СУБЪЕКТОВ МАЛОГО ПРЕДПРИНИМАТЕЛЬСТВА "
        "ЗА ЯНВАРЬ-ДЕКАБРЬ 2023 ГОДА (тыс. рублей) "
        "По хозяйственным видам По чистым видам Всего 100.5 110.0 90.0 105.0 "
        "1.2. ПРОДАНО Всего 200.4 111.0"
    )
    first, second = MODULE.shipment_totals(text, 2023)
    assert first[0] == "100.5" and first[2] == "90.0"
    assert second[0] == "200.4"
    with pytest.raises(ValueError, match="semantics"):
        MODULE.shipment_totals(text, 2024)
    with pytest.raises(ValueError, match="semantics"):
        MODULE.shipment_totals(text.replace("тыс. рублей", "млн. рублей"), 2023)


def test_legal_type_normalization_does_not_merge_different_places():
    assert MODULE.normalized_name("Будённовский муниципальный округ") == "буденновский"
    assert MODULE.normalized_name("городской округ город Ставрополь") == "ставрополь"
    assert MODULE.normalized_name("Саратовский муниципальный район") != "саратов"
