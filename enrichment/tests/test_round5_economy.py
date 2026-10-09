"""Reviewed source-type variants must not alter identity, units or scope."""

import pandas as pd
import pytest
from test_round4_economy import csv_source, html, options

from sber_enrichment.round4_economy import parse_response, payload_for
from sber_enrichment.round5_economy import (
    control_code,
    is_internal_file_error,
    merge_candidate,
    parse_form,
)


@pytest.mark.parametrize(
    "code,caption",
    [
        ("7", "Городской округ"),
        ("7", "Городские округа, городской округ с внутригородским делением"),
        ("10", "Муниципальные районы"),
        ("3", "Внутригородские территории городов федерального значения"),
        (
            "3",
            "Внутригородская территория\n(внутригородское муниципальное образование) "
            "города федерального значения",
        ),
    ],
)
def test_reviewed_literal_municipal_type_is_requested_and_checked(code, caption):
    values = options()
    values["tippos"] = {code: caption, "1": "Городское поселение"}
    form = parse_form(html(values), "8109001")
    assert form["selected_types"] == [code]
    assert f"tippos:{code};" in payload_for(form, "8109001", ["00123456"])["Qry"]
    source = csv_source().replace("Муниципальный район", caption)
    result = parse_response(source, form, "8109001", ["00123456"], region="munst57")
    assert result[0]["value"] == "123.5" and result[0]["source_oktmo"] == "00123456"


def test_same_type_code_with_unknown_caption_is_not_accepted():
    values = options()
    values["tippos"] = {"7": "Новое неизвестное территориальное образование"}
    with pytest.raises(ValueError, match="No verified"):
        parse_form(html(values), "8109001")


@pytest.mark.parametrize(
    "transform,error",
    [
        (lambda s: s.replace("тыс.рублей", "-"), "monetary unit"),
        (lambda s: s.replace(" (без субъектов малого предпринимательства)", ""), "exclusion"),
        (lambda s: s.replace(";2023;2024;", ";2024;2024;"), "annual header"),
    ],
)
def test_type_fix_never_relaxes_unit_scope_or_year(transform, error):
    form = parse_form(html(options()), "8109001")
    with pytest.raises(ValueError, match=error):
        parse_response(transform(csv_source()), form, "8109001", ["00123456"], region="munst07")


def test_recaptured_revision_is_recorded_without_overwriting_old_number():
    old = pd.DataFrame(
        {
            "territory_id": [1, 2],
            "year": ["2023"] * 2,
            "indicator_id": ["8109001"] * 2,
            "source_region_url": ["https://rosstat.gov.ru/dbscripts/munst/munst57/"] * 2,
            "source_oktmo": ["57701000", "57502000"],
            "official_oktmo_8": ["57701000", "57502000"],
            "value": ["100", ""],
            "value_status": ["observed", "not_returned_or_unaccepted"],
            "source_sha256": ["old-sha", ""],
            "publication_vintage": ["old", "old"],
        }
    )
    source = old.drop(columns=["territory_id", "official_oktmo_8", "publication_vintage"]).copy()
    source["value"] = ["101", "22"]
    source["value_status"] = "observed"
    source["source_sha256"] = "new-sha"
    candidate, delta = merge_candidate(old, source)
    assert candidate.value.tolist() == ["100", "22"]
    assert candidate.source_sha256.tolist() == ["old-sha", "new-sha"]
    assert delta.numeric_revision_conflict.tolist() == [True, False]
    assert delta.numeric_gain.tolist() == [False, True]


def test_logical_500_is_recognized_even_when_http_status_is_200():
    assert is_internal_file_error(b"Status: 500 Cannot open file\nException: EFOpenError")
    assert not is_internal_file_error(b"Status: 500 Cannot open file")
    assert not is_internal_file_error(b"Other HTML response")


def test_control_requires_observed_code_and_unique_full_caption():
    old = pd.DataFrame(
        {
            "source_region_url": ["official"] * 3,
            "indicator_id": ["8109001"] * 3,
            "source_oktmo": ["001", "002", "003"],
            "value": ["10", "20", ""],
        }
    )
    form = {"options": {"oktmo": {"001": "Truncated...", "002": "Unique", "003": "Missing"}}}
    assert control_code(old, form, "official", "8109001", {"003"}) == "002"
    with pytest.raises(ValueError, match="no unambiguous observed control"):
        control_code(old, form, "official", "8109001", {"002", "003"})
