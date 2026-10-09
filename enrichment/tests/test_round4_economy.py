from copy import deepcopy

import pytest

from sber_enrichment.round3_economy import SPECS
from sber_enrichment.round4_economy import parse_form, parse_response, payload_for


def options(indicator="8109001"):
    title = SPECS[indicator]["prefix"]
    values = {
        "Pokazateli": {indicator: title},
        "munr": {"1": "Районы"},
        "tippos": {"10": "Муниципальный район", "1": "Городское поселение"},
        "oktmo": {"00123456": "Полное название района"},
        "okved2": {"101": "Всего по обследуемым видам экономической деятельности"},
    }
    if indicator == "8109001":
        values["istinv"] = {"8": "Всего", "43": "Средства местного бюджета"}
    values["period"] = (
        {"17": "значение показателя за год"} if indicator == "8109001" else {"79": "январь-декабрь"}
    )
    values["god"] = {"2023": "2023", "2024": "2024"}
    return values


def html(values):
    chunks = []
    for name, entries in values.items():
        items = list(entries.items()) if isinstance(entries, dict) else entries
        chunks.append(f"p_{name}=new Array;\n")
        chunks.extend(f'p_{name}[{i}]="{code}";\n' for i, (code, _) in enumerate(items))
        chunks.append(f'<SELECT NAME="{name}">')
        chunks.extend(f"<OPTION>{label}\n" for _, label in items)
        chunks.append("</SELECT>")
    return "".join(chunks)


def csv_source(value="123,5", second="х", caption="Полное название района", indicator="8109001"):
    title = SPECS[indicator]["prefix"]
    suffix = "Всего, значение показателя за год" if indicator == "8109001" else "январь-декабрь"
    return (
        f"{title} (без субъектов малого предпринимательства), тыс.рублей, Районы, "
        "Муниципальный район, Всего по обследуемым видам экономической деятельности, "
        f"{suffix}\n;2023;2024;\n{caption};{value};{second};\n"
    )


def test_samara_unit_spacing_does_not_change_scale_or_markers():
    form = parse_form(html(options()), "8109001")
    rows = parse_response(csv_source(), form, "8109001", ["00123456"], region="munst36")
    assert rows[0]["value"] == "123.5" and rows[0]["unit"] == "thousand_rubles"
    assert rows[0]["source_unit_literal"] == "тыс.рублей"
    assert rows[1]["value"] is None and rows[1]["value_raw"] == "х"


def test_explicit_ui_ellipsis_requires_single_code_request():
    values = options()
    values["oktmo"] = {"00123456": "Полное название района (с 2024...", "00987654": "Другой район"}
    form = parse_form(html(values), "8109001")
    with pytest.raises(ValueError, match="single-code"):
        payload_for(form, "8109001", ["00123456", "00987654"])
    rows = parse_response(
        csv_source(caption="Полное название района (с 2024 года округ)"),
        form,
        "8109001",
        ["00123456"],
        region="munst28",
    )
    assert rows[0]["source_oktmo"] == "00123456"
    assert rows[0]["caption_evidence_rule"].startswith("explicit_UI_ellipsis")
    with pytest.raises(ValueError, match="Unknown source caption"):
        parse_response(
            csv_source(caption="Полное название района (с 2025 года округ)"),
            form,
            "8109001",
            ["00123456"],
            region="munst28",
        )


def test_nontruncated_caption_never_prefix_matched():
    form = parse_form(html(options()), "8109001")
    with pytest.raises(ValueError, match="Unknown source caption"):
        parse_response(
            csv_source(caption="Полное название района другое"),
            form,
            "8109001",
            ["00123456"],
            region="munst28",
        )


def test_auxiliary_footnote_defines_confidential_numeric_sentinel():
    values = options("8401003")
    values["footnotes"] = {
        "8401003_1": "999999999 - условное обозначение конфиденциальности первичных данных"
    }
    form = parse_form(html(values), "8401003")
    payload = payload_for(form, "8401003", ["00123456"])
    assert "footnotes" not in payload["Qry"] and form["footnotes"]
    rows = parse_response(
        csv_source("999999999,0", "0", indicator="8401003"),
        form,
        "8401003",
        ["00123456"],
        region="munst84",
    )
    assert rows[0]["value"] is None
    assert rows[0]["value_status"] == "suppressed_explicit_source_footnote"
    assert rows[1]["value"] == "0" and rows[1]["value_status"] == "observed"


def test_conflicting_unselected_rural_code_is_quarantined():
    values = options()
    values["oktmo"] = [
        ("00123456", "Полное название района"),
        ("78640415", "Волжское (с 2014 года)"),
        ("78640415", "Глебовское (до 2014 года)"),
    ]
    form = parse_form(html(values), "8109001")
    assert "78640415" in form["blocked_oktmo"]
    assert "78640415" not in form["options"]["oktmo"]
    payload_for(form, "8109001", ["00123456"])
    with pytest.raises(ValueError, match="nonconflicting"):
        payload_for(form, "8109001", ["78640415"])


def test_financing_source_name_and_case_are_preserved():
    values = options()
    values = {("p_istinv" if k == "istinv" else k): v for k, v in values.items()}
    values["p_istinv"]["8"] = "всего"
    form = parse_form(html(values), "8109001")
    assert "p_istinv:8;" in payload_for(form, "8109001", ["00123456"])["Qry"]
    assert "istinv:43;" not in payload_for(form, "8109001", ["00123456"])["Qry"]


def test_lower_settlement_type_not_requested_or_accepted():
    form = parse_form(html(options()), "8109001")
    assert "tippos:10;" in payload_for(form, "8109001", ["00123456"])["Qry"]
    with pytest.raises(ValueError, match="upper municipality"):
        parse_response(
            csv_source().replace("Муниципальный район", "Городское поселение"),
            form,
            "8109001",
            ["00123456"],
            region="munst90",
        )


def test_scope_typo_is_a_reviewed_region_specific_rule():
    form = parse_form(html(options("8401003")), "8401003")
    source = csv_source(indicator="8401003").replace(
        "малого предпринимательства", "малого редпринимательства"
    )
    assert parse_response(source, form, "8401003", ["00123456"], region="munst05")
    with pytest.raises(ValueError, match="exclusion"):
        parse_response(source, form, "8401003", ["00123456"], region="munst36")


def test_samara_shipments_explicit_scope_typo_requires_reviewed_indicator_and_region():
    form = parse_form(html(options("8401011")), "8401011")
    source = csv_source(indicator="8401011").replace("предпринимательства", "предприниматетельства")
    rows = parse_response(source, form, "8401011", ["00123456"], region="munst36")
    assert rows[0]["value"] == "123.5"
    with pytest.raises(ValueError, match="exclusion"):
        parse_response(source, form, "8401011", ["00123456"], region="munst07")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("тыс.рублей", "рублей"),
        ("тыс.рублей", "-"),
        ("значение показателя за год", "январь-сентябрь"),
        ("без субъектов малого предпринимательства", "все организации"),
        (";2023;2024;", ";2024;2024;"),
        (";2023;2024;", ";2024;2025;"),
    ],
)
def test_unknown_units_scope_and_period_remain_rejected(old, new):
    form = parse_form(html(options()), "8109001")
    with pytest.raises(ValueError):
        parse_response(
            csv_source().replace(old, new), form, "8109001", ["00123456"], region="munst36"
        )


def test_duplicate_cells_not_silently_collapsed():
    form = parse_form(html(options()), "8109001")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_response(csv_source() + csv_source(), form, "8109001", ["00123456"], region="munst90")


def test_unknown_auxiliary_dimension_is_not_ignored():
    values = deepcopy(options())
    values["new_dimension"] = {"1": "Неизвестно"}
    with pytest.raises(ValueError, match="Unreviewed"):
        parse_form(html(values), "8109001")
