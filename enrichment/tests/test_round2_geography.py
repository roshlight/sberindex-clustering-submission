"""Guard against silent truncation or code reassignment in geography evidence."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_round2_geography.py"
spec = importlib.util.spec_from_file_location("round2_geo", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_docx(tmp_path, rows):
    path = tmp_path / "pairs.docx"
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        "<w:tr>"
        + "".join(f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row)
        + "</w:tr>"
        for row in rows
    )
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml", f'<w:document xmlns:w="{ns}"><w:tbl>{body}</w:tbl></w:document>'
        )
    return path


def test_full_codes_preserved_and_never_joined_to_municipality(tmp_path):
    path = make_docx(
        tmp_path, [["Аннулированный код", "Действующий код"], ["86 604 101 001", "86 504 000 001"]]
    )
    [row] = module.recoding_rows(path)
    assert row["old_settlement_oktmo_11"] == "86604101001"
    assert row["new_settlement_oktmo_11"] == "86504000001"
    assert row["municipal_territory_id"] == ""
    assert row["source_table_row"] == 2
    assert row["municipal_join_status"] == "not_joined"


def test_partial_pair_refused_instead_of_truncating(tmp_path):
    path = make_docx(tmp_path, [["86 604 101 001", "86504000"]])
    with pytest.raises(ValueError, match="Partial recoding"):
        module.recoding_rows(path)


def test_duplicate_old_code_refused(tmp_path):
    path = make_docx(tmp_path, [["86604101001", "86504000001"], ["86604101001", "86504000002"]])
    with pytest.raises(ValueError, match="Duplicate old"):
        module.recoding_rows(path)


def test_explicit_quarantine_preserves_source_defect_and_valid_pairs(tmp_path):
    path = make_docx(tmp_path, [["68 610 450 131", "п. 1"], ["68 610 450 136", "68 510 000 192"]])
    rejected = []
    rows = module.recoding_rows(path, rejected)
    assert len(rows) == len(rejected) == 1
    assert rejected[0]["new_code_verbatim"] == "п. 1"
    assert rejected[0]["accepted_pair"] is False
    assert rows[0]["old_settlement_oktmo_11"] == "68610450136"


def test_registry_same_code_in_other_section_is_not_municipality(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text(
        "86;604;101;001;0;2;settlement;;;634;1;27.07.2023;01.01.2024\n"
        "86;604;000;000;0;1;district;;;000;0;14.06.2013;01.01.2014\n",
        encoding="utf-8",
    )
    rows = module.registry_rows(path)
    assert set(rows) == {"86604000000"}
    assert rows["86604000000"]["source_csv_line"] == 2


def test_registry_duplicate_full_code_refused(tmp_path):
    path = tmp_path / "registry.csv"
    line = "86;604;000;000;0;1;district;;;000;0;14.06.2013;01.01.2014\n"
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate section-1 code"):
        module.registry_rows(path)
