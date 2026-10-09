"""Regression guards for the bounded historical-geography evidence collector."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round3_geography",
    Path(__file__).parents[1] / "scripts/collect_round3_geography.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(code, section, name):
    return {
        "code": code,
        "RAZDEL": section,
        "NAME1": name,
        "TER": code[:2],
        "KOD1": code[2:5],
        "KOD2": code[5:8],
        "KOD3": code[8:11],
    }


def test_exact_source_block_stops_before_next_municipality():
    rows = [
        row("86604000000", "1", "district"),
        row("86604101000", "1", "lower municipality"),
        row("86604101000", "2", "source heading"),
        row("86604101001", "2", "same name"),
        row("86633000000", "1", "other district"),
        row("86633101001", "2", "same name"),
    ]
    got = MODULE.settlement_block(rows, "86604000000", "district")
    assert [x["code"] for x in got] == ["86604101001"]


def test_duplicate_exact_root_refused():
    rows = [row("86604000000", "1", "district")] * 2
    with pytest.raises(ValueError, match="exactly once"):
        MODULE.settlement_block(rows, "86604000000", "district")


def write_pairs(path, pairs):
    ns = MODULE.NS["w"]
    body = "".join(
        "<w:tr>"
        + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in p)
        + "</w:tr>"
        for p in pairs
    )
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml", f'<w:document xmlns:w="{ns}"><w:tbl>{body}</w:tbl></w:document>'
        )


def test_recoding_cannot_reuse_successor(tmp_path):
    path = tmp_path / "pairs.docx"
    write_pairs(path, [("86604101001", "86504000001"), ("86604101002", "86504000001")])
    with pytest.raises(ValueError, match="one-to-one"):
        MODULE.recoding(path)


def test_recoding_missing_successor_is_not_a_match(tmp_path):
    path = tmp_path / "pairs.docx"
    write_pairs(path, [("86604101001", "see note")])
    with pytest.raises(ValueError, match="Incomplete"):
        MODULE.recoding(path)


def test_publication_date_is_not_law_signature_date(tmp_path):
    path = tmp_path / "card.html"
    path.write_text(
        "Закон от 28.04.2023 Номер опубликования: 1000202305020006 Дата опубликования: 02.05.2023",
        encoding="utf-8",
    )
    MODULE.publication(path, "1000202305020006", "02.05.2023")
    with pytest.raises(ValueError, match="date mismatch"):
        MODULE.publication(path, "1000202305020006", "28.04.2023")
