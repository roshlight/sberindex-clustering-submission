import json

import httpx
import pytest

from sber_enrichment.economy_bulk import parse_form, parse_response, payload_for, run, safe_groups

TITLE = "Среднесписочная численность работников организаций, человек"
AGGREGATE = "Всего по обследуемым видам экономической деятельности"
OPTIONS = {
    "Pokazateli": {"8423005": TITLE},
    "munr": {"79700000": "Городские округа"},
    "tippos": {"7": "Городской округ"},
    "oktmo": {"79701000": "Город Майкоп", "79703000": "Город Адыгейск"},
    "okved2": {"101": AGGREGATE, "10000": "Раздел А"},
    "god": {"2023": "2023", "2024": "2024"},
    "period": {"79": "январь-декабрь"},
}
DOCUMENT = (
    f"{TITLE}, Городские округа, Городской округ, {AGGREGATE}, январь-декабрь\n"
    ";2023;2024;\nГород Майкоп;35318;34686;\nГород Адыгейск;1620;1592;\n"
)


def form_html(options):
    return "".join(
        "".join(f'p_{name}[{i}]="{code}";' for i, code in enumerate(values))
        + f'<SELECT NAME="{name}">'
        + "".join(f"<OPTION>{value}" for value in values.values())
        + "</SELECT>"
        for name, values in options.items()
    )


def test_coded_form_selects_verified_annual_aggregate_only():
    options = parse_form(form_html(OPTIONS), "8423005")
    payload = payload_for(options, list(options["oktmo"]))
    assert "okved2:101;" in payload["Qry"]
    assert "period:79;" in payload["Qry"]
    assert "oktmo:79701000,79703000;" in payload["Qry"]
    with pytest.raises(ValueError, match="units"):
        parse_form(form_html(OPTIONS).replace(", человек", ", тысяча человек"), "8423005")
    with pytest.raises(ValueError, match="meaning"):
        parse_form(form_html(OPTIONS).replace(AGGREGATE, "Другая деятельность"), "8423005")


def test_real_two_municipality_layout_keeps_year_value_code_and_source_row():
    rows = parse_response(DOCUMENT, OPTIONS, list(OPTIONS["oktmo"]))
    assert [(r["source_oktmo"], r["year"], r["value"]) for r in rows] == [
        ("79701000", "2023", "35318"),
        ("79701000", "2024", "34686"),
        ("79703000", "2023", "1620"),
        ("79703000", "2024", "1592"),
    ]
    assert rows[-1]["source_csv_row"] == 4
    assert rows[-1]["period_end"] == "2024-12-31"


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace(";2023;2024;", ";2023;;2024;"),
        lambda s: s.replace(";2023;2024;", ";2023;2023;"),
        lambda s: s.replace("Город Майкоп;", "Чужой город;"),
        lambda s: s.replace("январь-декабрь", "январь-март"),
        lambda s: s + "Город Майкоп;1;2;\n",
    ],
)
def test_ambiguous_attribution_or_context_is_rejected(change):
    with pytest.raises(ValueError):
        parse_response(change(DOCUMENT), OPTIONS, list(OPTIONS["oktmo"]))


def test_partial_and_explicit_missing_cells_remain_distinct():
    text = (
        DOCUMENT.replace(";2023;2024;", ";2024;")
        .replace("Город Майкоп;35318;34686;", "Город Майкоп;34686,5;")
        .replace("Город Адыгейск;1620;1592;\n", "")
    )
    rows = parse_response(text, OPTIONS, list(OPTIONS["oktmo"]))
    assert len(rows) == 1 and rows[0]["value"] == "34686.5"
    rows = parse_response(text.replace("34686,5", "-"), OPTIONS, list(OPTIONS["oktmo"]))
    assert rows[0]["value"] is None and rows[0]["value_raw"] == "-"


def test_source_uppercase_aggregate_and_period_keep_literal_csv_scope():
    options = {
        **OPTIONS,
        "okved2": {"101": AGGREGATE.replace("Всего", "ВСЕГО")},
        "period": {"79": "Январь-декабрь"},
    }
    parsed = parse_form(form_html(options), "8423005")
    document = DOCUMENT.replace(AGGREGATE, options["okved2"]["101"]).replace(
        "январь-декабрь", "Январь-декабрь"
    )
    rows = parse_response(document, parsed, list(options["oktmo"]))
    assert rows[0]["value"] == "35318"
    assert rows[0]["okved2_label"].startswith("ВСЕГО")
    assert rows[0]["period_label"] == "Январь-декабрь"
    with pytest.raises(ValueError, match="context"):
        parse_response(DOCUMENT, parsed, list(options["oktmo"]))


def test_source_ruble_plural_is_not_thousand_rubles():
    title = "Среднемесячная заработная плата работников организаций, рублей"
    options = {**OPTIONS, "Pokazateli": {"8423007": title}}
    parsed = parse_form(form_html(options), "8423007")
    rows = parse_response(DOCUMENT.replace(TITLE, title), parsed, list(options["oktmo"]))
    assert rows[0]["unit"] == "рубль" and rows[0]["source_unit_label"] == "рублей"
    with pytest.raises(ValueError, match="units"):
        parse_form(form_html(options).replace(", рублей", ", тысяча рублей"), "8423007")


def test_source_aggregate_code_annotation_must_be_exact_and_retained():
    caption = AGGREGATE + " (101.АГ)"
    options = {**OPTIONS, "okved2": {"101": caption}}
    parsed = parse_form(form_html(options), "8423005")
    rows = parse_response(DOCUMENT.replace(AGGREGATE, caption), parsed, list(options["oktmo"]))
    assert rows[0]["okved2_label"] == caption
    with pytest.raises(ValueError, match="meaning"):
        parse_form(form_html(options).replace("101.АГ", "10000"), "8423005")


def test_wrapped_indicator_and_explicit_annual_preposition_require_full_caption():
    options = {**OPTIONS, "period": {"79": "за январь-декабрь"}}
    parsed = parse_form(form_html(options), "8423005")
    text = DOCUMENT.replace("работников организаций", "работников\nорганизаций").replace(
        "январь-декабрь", "за январь-декабрь"
    )
    rows = parse_response(text, parsed, list(options["oktmo"]))
    assert rows[0]["value"] == "35318" and rows[0]["source_csv_row"] == 4
    with pytest.raises(ValueError, match="context"):
        parse_response(
            text.replace("организаций, человек", "чужих лиц, человек"),
            parsed,
            list(options["oktmo"]),
        )


def test_truncated_caption_is_isolated_without_authorizing_prefix_attribution():
    options = {**OPTIONS, "oktmo": {**OPTIONS["oktmo"], "28714000": "Вышневолоцкий..."}}
    assert safe_groups(options, list(options["oktmo"]), 100) == [
        ["79701000", "79703000"],
        ["28714000"],
    ]
    text = DOCUMENT.split(";2023;2024;")[0] + ";2023;2024;\nВышневолоцкий городской округ;1;2;\n"
    with pytest.raises(ValueError, match="attribution"):
        parse_response(text, options, ["28714000"])


def test_duplicate_global_name_requires_single_code_request():
    options = {**OPTIONS, "oktmo": {**OPTIONS["oktmo"], "79999999": "Город Майкоп"}}
    assert safe_groups(options, ["79701000", "79703000"], 100) == [["79703000"], ["79701000"]]
    with pytest.raises(ValueError, match="unique"):
        payload_for(options, ["79701000", "79703000"])
    rows = parse_response(
        DOCUMENT.replace("Город Адыгейск;1620;1592;\n", ""), options, ["79701000"]
    )
    assert rows[0]["source_oktmo"] == "79701000"


def test_unescaped_semicolon_in_exact_label_and_wrapped_title_do_not_shift_numbers():
    options = {**OPTIONS, "oktmo": {"79701000": "Город; точное название"}}
    text = (
        DOCUMENT.replace("Город Майкоп", "Город; точное название")
        .replace("Город Адыгейск;1620;1592;\n", "")
        .replace("Городской округ,", "Городской\nокруг,")
    )
    rows = parse_response(text, options, ["79701000"])
    assert rows[0]["value"] == "35318" and rows[0]["source_csv_row"] == 4


def test_network_pause_preserves_completed_indicator_and_resume_does_not_refetch_it(tmp_path):
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def request(self, method, url, payload):
            calls.append(payload)
            if payload.get("pl") == "8423007":
                httpx.Response(403, request=httpx.Request(method, url)).raise_for_status()
            raw = (form_html(OPTIONS) if "pl" in payload else DOCUMENT).encode("cp1251")
            return raw, {
                "url": url,
                "raw_sha256": "rawsha",
                "request_key": "key",
                "retrieved_at": "2026-09-15T00:00:00+00:00",
            }

    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            [
                {
                    "region_url": "https://rosstat.gov.ru/dbscripts/munst/munst79/",
                    "oktmo": code,
                    "reference_links": [{"territory_id": i, "reference_date": "2023-01-01"}],
                }
                for i, code in enumerate(OPTIONS["oktmo"])
            ]
        )
    )
    args = (targets, tmp_path / "output", tmp_path / "cache")
    result = run(*args, client_factory=Client)
    assert result["status"] == "paused_network" and result["record_count"] == 4
    assert result["processed_target_indicator_pairs"] == 2
    before = len(calls)
    result = run(*args, client_factory=Client, resume=True)
    assert result["record_count"] == 4 and len(calls) == before + 1
