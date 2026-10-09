import pytest

from sber_enrichment.round3_economy import parse_form, parse_response, payload_for


@pytest.fixture
def options():
    return {
        "Pokazateli": {"8109001": "Инвестиции в основной капитал, осуществляемые организациями"},
        "munr": {"1": "Районы"},
        "tippos": {"2": "Муниципальный район"},
        "oktmo": {"00123456": "Полное название района"},
        "okved2": {"101": "Всего по обследуемым видам экономической деятельности"},
        "istinv": {"8": "Всего", "43": "Средства местного бюджета"},
        "period": {"17": "значение показателя за год"},
        "god": {"2023": "2023", "2024": "2024", "2025": "2025"},
    }


def source(a="123,5", b="х", caption="Полное название района"):
    return (
        "Инвестиции в основной капитал, осуществляемые организациями, находящимися "
        "на территории муниципального образования (без субъектов малого предпринимательства), "
        "тысяча рублей, Районы, Муниципальный район, "
        "Всего по обследуемым видам экономической деятельности, Всего, "
        "значение показателя за год\n;2023;2024;\n"
        f"{caption};{a};{b};\n"
    )


def test_full_source_values_and_missing_are_distinct(options):
    rows = parse_response(source(), options, "8109001", ["00123456"])
    assert [(r["year"], r["value"]) for r in rows] == [("2023", "123.5"), ("2024", None)]
    assert rows[1]["value_status"] == "unresolved_source_marker"
    assert rows[0]["source_oktmo"] == "00123456"
    assert rows[0]["unit"] == "thousand_rubles"
    assert rows[0]["source_csv_row"] == 3


def test_source_zero_is_preserved(options):
    rows = parse_response(source("0", ""), options, "8109001", ["00123456"])
    assert rows[0]["value"] == "0"
    assert rows[1]["value"] is None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("тысяча рублей", "рубль"),
        ("значение показателя за год", "январь-сентябрь"),
        ("без субъектов малого предпринимательства", "все организации"),
        (";2023;2024;", ";2024;2024;"),
        (";2023;2024;", ";2024;2025;"),
        ("Всего, значение", "Средства местного бюджета, значение"),
    ],
)
def test_reject_changed_semantics(options, old, new):
    with pytest.raises(ValueError):
        parse_response(source().replace(old, new), options, "8109001", ["00123456"])


def test_no_truncated_or_fuzzy_name_join(options):
    with pytest.raises(ValueError, match="caption"):
        parse_response(source(caption="Полное название"), options, "8109001", ["00123456"])


def test_duplicate_cells_are_rejected(options):
    with pytest.raises(ValueError, match="Duplicate"):
        parse_response(source() + source(), options, "8109001", ["00123456"])


def test_unknown_markers_rejected(options):
    with pytest.raises(ValueError, match="numeric"):
        parse_response(source("неизвестно"), options, "8109001", ["00123456"])


def test_payload_excludes_nested_financing_and_future(options):
    payload = payload_for(options, "8109001", ["00123456"])
    assert "istinv:8;" in payload["Qry"]
    assert "okved2:101;" in payload["Qry"]
    assert "god:2023,2024;" in payload["Qry"]
    assert "00123456" in payload["Qry"]


def test_ambiguous_names_must_be_single_code(options):
    options["oktmo"]["00765432"] = "Полное название района"
    with pytest.raises(ValueError, match="Ambiguous"):
        payload_for(options, "8109001", ["00123456", "00765432"])


def test_unknown_source_code_rejected(options):
    with pytest.raises(ValueError, match="exact"):
        payload_for(options, "8109001", ["123456"])


def test_verified_source_unit_abbreviation(options):
    rows = parse_response(
        source().replace("тысяча рублей", "тыс. рублей"), options, "8109001", ["00123456"]
    )
    assert rows[0]["value"] == "123.5"
    assert rows[0]["unit"] == "thousand_rubles"


def test_source_specific_dimension_order(options):
    reordered = {
        k: options[k]
        for k in ["Pokazateli", "munr", "tippos", "oktmo", "period", "okved2", "istinv", "god"]
    }
    old = "Всего по обследуемым видам экономической деятельности, Всего, значение показателя за год"
    new = "значение показателя за год, Всего по обследуемым видам экономической деятельности, Всего"
    rows = parse_response(source().replace(old, new), reordered, "8109001", ["00123456"])
    assert rows[0]["value"] == "123.5"


def form_html(options):
    chunks = []
    for name, values in options.items():
        chunks.append(f"p_{name}=new Array;\n")
        chunks.extend(f'p_{name}[{i}]="{code}";\n' for i, code in enumerate(values))
        chunks.append(f'<SELECT NAME="{name}">')
        chunks.extend(f"<OPTION>{label}\n" for label in values.values())
        chunks.append("</SELECT>")
    return "".join(chunks)


def test_verified_annual_label_alias_and_case(options):
    options["period"]["17"] = "Значение за год"
    parsed = parse_form(form_html(options), "8109001")
    assert parsed["period"]["17"] == "Значение за год"


def test_period_code_alone_cannot_authorize_quarterly_data(options):
    options["period"]["17"] = "январь-сентябрь"
    with pytest.raises(ValueError, match="Annual"):
        parse_form(form_html(options), "8109001")
