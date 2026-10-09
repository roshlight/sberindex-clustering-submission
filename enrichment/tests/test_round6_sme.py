"""Semantic regression checks: publication dates, lexical codes and immutable inputs."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_round6_sme import (  # noqa: E402
    address_contracts,
    code_state,
    inspect_workers,
    verify_sources,
    worker_state,
)
from collect_round6_sme import digest, freeze_inventory  # noqa: E402


def sample_xml(*, declared=2):
    return f'''<Файл КолДок="{declared}">
    <Документ ДатаСост="31.12.2023" ДатаДок="25.12.2024">
      <СведНП ИННЮЛ="0000000001"/><СведССЧР КолРаб="0"/>
    </Документ>
    <Документ ДатаСост="31.12.2024" ДатаДок="25.08.2025">
      <СведНП ИННЮЛ="0000000002"/><СведССЧР КолРаб="7"/>
    </Документ></Файл>'''.encode()


def test_data_year_is_distinct_from_publication_year():
    rows, summary = inspect_workers(sample_xml())
    assert [(x["data_year"], x["document_year"]) for x in rows] == [(2023, 2024), (2024, 2025)]
    assert summary["records_2023_or_2024"] == 2
    assert sum(x["records"] for x in rows) == summary["records"]
    assert not summary["historical_municipal_join_approved"]


def test_declared_xml_count_cannot_hide_lost_record():
    with pytest.raises(ValueError, match="document count"):
        inspect_workers(sample_xml(declared=3))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "missing_attribute"),
        ("", "empty_attribute"),
        ("0", "zero"),
        ("7", "positive"),
        ("-1", "negative"),
        ("1.0", "invalid_integer"),
    ],
)
def test_workers_states_are_not_imputed(value, expected):
    assert worker_state(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "empty"),
        ("45000000", "exact_8_or_11_digits"),
        ("01703000001", "exact_8_or_11_digits"),
        ("1703000001", "other_digit_length"),
        ("4.5e+07", "scientific_notation"),
        ("45000000.0", "other_syntax"),
    ],
)
def test_code_lexemes_are_not_silently_repaired(value, expected):
    assert code_state(value) == expected


def test_source_guard_rejects_modified_bytes(tmp_path):
    (tmp_path / "source.xml").write_bytes(sample_xml())
    freeze_inventory(tmp_path)
    assert len(verify_sources(tmp_path)) == 1
    (tmp_path / "source.xml").write_bytes(sample_xml(declared=3))
    with pytest.raises(ValueError, match="SHA/size"):
        verify_sources(tmp_path)


def test_source_guard_rejects_untracked_extra_file(tmp_path):
    (tmp_path / "source.xml").write_bytes(sample_xml())
    freeze_inventory(tmp_path)
    (tmp_path / "extra.xml").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="inventory"):
        verify_sources(tmp_path)


def test_source_guard_rejects_path_traversal(tmp_path):
    row = dict(path="../outside", bytes=0, sha256=digest(b""))
    (tmp_path / "SOURCE_MANIFEST.json").write_text(json.dumps({"files": [row]}))
    with pytest.raises(ValueError, match="Nonportable"):
        verify_sources(tmp_path)


def test_temporal_address_contract_change_blocks_old_claim(tmp_path):
    (tmp_path / "generator_geocode.py").write_text(
        'class LocalGeocoder:\n    ADDR_COLS = ["region", "year"]\n'
        'class DaDataGeocoder:\n    ADDR_COLS = ["region"]\n'
    )
    with pytest.raises(ValueError, match="date contract"):
        address_contracts(tmp_path)
