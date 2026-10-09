import io
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from sber_enrichment import office_text as office


def package(files, compression=0):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=compression) as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def docx(text="Residents", extra=""):
    return package(
        {
            "word/document.xml": f'<w:document xmlns:w="{office.WORD_NS}"><w:body>'
            f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>{extra}</w:body></w:document>",
        }
    )


def xlsx(rows, name="Data", rel_target="worksheets/sheet1.xml", rel_mode=""):
    return package(
        {
            "xl/workbook.xml": f'<workbook xmlns="{office.SHEET_NS}" xmlns:r="{office.REL_NS}">'
            f'<sheets><sheet name="{name}" sheetId="1" r:id="r1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": "<Relationships>"
            f'<Relationship Id="r1" Target="{rel_target}" TargetMode="{rel_mode}"/>'
            "</Relationships>",
            "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{office.SHEET_NS}">'
            f"<sheetData>{rows}</sheetData></worksheet>",
            "xl/sharedStrings.xml": f'<sst xmlns="{office.SHEET_NS}">'
            "<si><t>Population on January 1</t></si></sst>",
        }
    )


def test_docx_preserves_paragraphs_table_and_hyperlink_text_without_fetching():
    extra = "<w:tbl><w:tr><w:tc><w:p><w:hyperlink><w:r><w:t>Rubles</w:t></w:r>"
    extra += "</w:hyperlink></w:p></w:tc></w:tr></w:tbl>"
    parsed = office.parse_office(docx(extra=extra), ".docx")
    assert parsed["kind"] == "docx_main_text"
    assert "[paragraph 1] Residents" in parsed["text"]
    assert "[paragraph 2] Rubles" in parsed["text"]
    assert parsed["links"] == []
    assert parsed["truncated"] is False


def test_docx_character_limit_is_explicit():
    parsed = office.parse_office(docx("x" * 1000), "DOCX", max_chars=150)
    assert len(parsed["text"]) == 150
    assert parsed["truncated"] is True


def test_xlsx_reports_coordinates_and_omits_later_data_and_formula_cache():
    rows = '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
    rows += '<row r="2"><c r="B2"><f>1+1</f><v>987654321</v></c></row>'
    rows += '<row r="13"><c r="A13" t="inlineStr"><is><t>LATER DATA</t></is></c></row>'
    parsed = office.parse_office(xlsx(rows), ".xlsx")
    assert parsed["kind"] == "xlsx_header_preview"
    assert "[sheet='Data'; row=1] A1=Population on January 1" in parsed["text"]
    assert "formula omitted" in parsed["text"]
    assert "987654321" not in parsed["text"]
    assert "LATER DATA" not in parsed["text"]
    assert parsed["truncated"] is True


def test_named_legend_includes_later_definition_within_its_bound():
    rows = '<row r="20"><c r="A20" t="inlineStr"><is><t>Definition</t></is></c></row>'
    parsed = office.parse_office(xlsx(rows, name="Предисловие"), "xlsx")
    assert "row=20] A20=Definition" in parsed["text"]
    assert parsed["truncated"] is False


@pytest.mark.parametrize(
    "target,mode",
    [
        ("https://example.com/sheet.xml", "External"),
        ("../external.xml", ""),
    ],
)
def test_rejects_external_or_traversing_worksheet_relationship(target, mode):
    with pytest.raises(ValueError, match="external|Unsafe"):
        office.parse_office(xlsx("", rel_target=target, rel_mode=mode), "xlsx")


def test_rejects_unsafe_archive_member_even_if_not_requested():
    body = package({"../outside.txt": "bad", "word/document.xml": "<document/>"})
    with pytest.raises(ValueError, match="Unsafe"):
        office.parse_office(body, "docx")


def test_checks_uncompressed_size_before_xml_parsing(monkeypatch):
    monkeypatch.setattr(office, "MAX_MEMBER_BYTES", 128)
    with pytest.raises(ValueError, match="uncompressed limit"):
        office.parse_office(docx("x" * 200), "docx")


def test_rejects_high_compression_ratio():
    body = package({"word/document.xml": "a" * 1_000_000}, ZIP_DEFLATED)
    with pytest.raises(ValueError, match="compression ratio"):
        office.parse_office(body, "docx")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_rejects_entity_declarations_in_multiple_encodings(encoding):
    declaration = '<?xml version="1.0" encoding="' + encoding + '"?>'
    xml = declaration + '<!DOCTYPE document [<!ENTITY x "EXPANDED">]>'
    xml += f'<w:document xmlns:w="{office.WORD_NS}"><w:p><w:r><w:t>&x;</w:t>'
    xml += "</w:r></w:p></w:document>"
    with pytest.raises(ValueError, match="DTD/entity"):
        office.parse_office(package({"word/document.xml": xml.encode(encoding)}), "docx")


def test_rejects_duplicate_member_name():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "one")
        with pytest.warns(UserWarning):
            archive.writestr("word/document.xml", "two")
    with pytest.raises(ValueError, match="Duplicate"):
        office.parse_office(buffer.getvalue(), "docx")


def test_rejects_corrupt_or_unsupported_input():
    with pytest.raises(ValueError, match="Invalid Office package"):
        office.parse_office(b"not a zip", "docx")
    with pytest.raises(ValueError, match="Only DOCX and XLSX"):
        office.parse_office(b"", "xls")
