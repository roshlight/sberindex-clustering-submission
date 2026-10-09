"""Financial semantic guards for round-six documentary context."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round6_cases_test", Path(__file__).parents[1] / "scripts/collect_round6_cases.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
amount_sum = MODULE.amount_sum
check_financing_layout = MODULE.check_financing_layout
extract_snapshot = MODULE.extract_snapshot


@pytest.mark.parametrize("values", [[None], [3, None], [None, None]])
def test_blank_is_not_zero(values):
    assert amount_sum(values) == ""


def test_two_distinct_budget_codes_are_added_without_float_roundoff():
    assert amount_sum([731890.8, 56652]) == "788542.8"
    assert amount_sum([0]) == "0"


def test_no_component_is_not_an_observed_zero():
    with pytest.raises(ValueError, match="at least one"):
        amount_sum([])


def test_windows_newline_replay_preserves_html_text(tmp_path):
    (tmp_path / "raw.html").write_bytes(b"<html><body>Execution\r\n2023</body></html>")
    text = extract_snapshot({"path": "raw.html", "id": "test", "url": "https://test"}, tmp_path)
    assert "\r" not in text
    output = tmp_path / "text.txt"
    output.write_text(text, encoding="utf-8")
    assert output.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("values", [["=Z56+Z57"], ["731890.8"], [False]])
def test_nonliteral_amount_is_rejected(values):
    with pytest.raises(ValueError, match="literal numeric"):
        amount_sum(values)


class Sheet(dict):
    merged_cells = ["E56:E57", "D56:D57", "AH56:AH57"]

    def __getitem__(self, key):
        return SimpleNamespace(value=super().__getitem__(key))


def sheet():
    return Sheet(
        A4="(тыс. рублей)",
        N5="2022 год",
        V5="2023 год",
        AD5="2024 год",
        R6="кассовое исполнение",
        Z6="кассовое исполнение",
        AH6="кассовое исполнение",
        D56="00000000209970",
        C56="Республика Саха (Якутия), пос.Черский",
        E56="Реконструкция аэропорта Черский, Республика Саха (Якутия)",
        B56="0408 241V7 24100 414",
        B57="0408 241V7 2410F 414",
    )


@pytest.mark.parametrize(
    ("cell", "value"),
    [("V5", "2024 год"), ("A4", "рубли"), ("Z6", "ЛБО"), ("D56", "00000000209921")],
)
def test_wrong_year_unit_metric_or_neighbor_object_is_rejected(cell, value):
    source = sheet()
    check_financing_layout(source)
    source[cell] = value
    with pytest.raises(ValueError, match="row/header changed"):
        check_financing_layout(source)


def test_lost_merged_project_identity_is_rejected():
    source = sheet()
    source.merged_cells = ["AH56:AH57"]
    with pytest.raises(ValueError, match="row spans"):
        check_financing_layout(source)
