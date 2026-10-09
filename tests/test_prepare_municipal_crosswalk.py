"""Coded joins must reject ambiguity and must not extrapolate source validity."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_municipal_crosswalk",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_municipal_crosswalk.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(**changes):
    value = dict.fromkeys(module.REQUIRED_COLUMNS)
    value.update(
        territory_id="1",
        oktmo="01-701-000-000",
        region_code="22",
        region_name="Region",
        municipal_district_name="Same name",
        municipal_district_name_short="Same name",
        municipal_district_type="городской округ",
        shape="1",
        source_rosstat="official.csv",
        year_from="2018",
        year_to="9999",
    )
    return value | changes


def test_preserves_leading_zeroes_and_uses_code_not_names():
    data = module.validate_crosswalk(
        pd.DataFrame(
            [
                row(),
                row(territory_id="2", oktmo="01-703-000-000"),
            ]
        )
    )
    assert data["oktmo_digits"].tolist() == ["01701000000", "01703000000"]
    assert data["region_code"].tolist() == [22, 22]
    assert data["municipal_district_status"].isna().all()
    snapshot = module.snapshot_crosswalk(data, 2024)
    assert module.coverage_record(snapshot, {1, 3}) == {
        "requested_ids": 2,
        "matched_ids": 1,
        "unmatched_ids": [3],
    }


def test_version_boundary_is_exclusive_and_code_change_is_retained():
    data = module.validate_crosswalk(
        pd.DataFrame(
            [
                row(year_to="2024"),
                row(year_from="2024", oktmo="01-705-000-000"),
            ]
        )
    )
    assert module.snapshot_crosswalk(data, 2023)["oktmo_digits"].tolist() == ["01701000000"]
    assert module.snapshot_crosswalk(data, 2024)["oktmo_digits"].tolist() == ["01705000000"]
    with pytest.raises(ValueError, match="2018 through 2024"):
        module.snapshot_crosswalk(data, 2025)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([row(), row()], "Duplicate"),
        ([row(year_to="2024"), row(year_from="2023", oktmo="01-705-000-000")], "Overlapping"),
        ([row(), row(territory_id="2")], "Overlapping.*oktmo"),
        ([row(year_to="2018")], "outside documented|zero-length"),
        ([row(oktmo=1701000000)], "original.*text"),
        ([row(municipal_district_center_lat="50")], "Unpaired"),
    ],
)
def test_rejects_ambiguous_or_malformed_source(rows, match):
    with pytest.raises(ValueError, match=match):
        module.validate_crosswalk(pd.DataFrame(rows))


@pytest.mark.parametrize("formula", [False, True])
def test_xlsx_retains_text_and_empty_column_but_rejects_formula(tmp_path, formula):
    path = tmp_path / "source.xlsx"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    formula_xml = "<f>1+1</f>" if formula else ""
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{relns}"><sheets>'
            '<sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{ns}"><si><t>oktmo</t></si><si><t>status</t></si>'
            "<si><t>01-701-000-000</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData><row r="1">'
            '<c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            f'<row r="2"><c r="A2" t="s">{formula_xml}<v>2</v></c></row>'
            "</sheetData></worksheet>",
        )
    if formula:
        with pytest.raises(ValueError, match="Formula"):
            module.read_source_xlsx(path)
    else:
        result = module.read_source_xlsx(path)
        assert result["oktmo"].tolist() == ["01-701-000-000"]
        assert result["status"].isna().all()
