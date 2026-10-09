"""The code bridge must not invent effective dates, leading zeros, or unique matches."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_historical_oktmo_bridge.py"
SPEC = importlib.util.spec_from_file_location("historical_bridge", SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def registry(tmp_path, rows):
    path = tmp_path / "source.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def row(code="03;701;000;000", date="10.02.2023"):
    return f"{code};0;1;Example;;;598;3;07.02.2023;{date}"


def test_preserves_leading_zero_and_does_not_backdate_late_registry_record(tmp_path):
    _, frame = bridge.read_registry(registry(tmp_path, [row()]))
    evidence = bridge.code_evidence(
        "03701000000",
        "2023-01-01",
        SimpleNamespace(
            territory_id="7",
            year_from="2023",
            year_to="9999",
            source_rosstat="explicit-sber-source.csv",
        ),
        {"2023-02-08": frame},
        ("2023-02-08",),
    )
    assert evidence["official_oktmo_8"] == "03701000"
    assert evidence["sber_oktmo_11"] == "03701000000"
    assert evidence["reference_date"] == "2023-01-01"
    assert evidence["registry_DateVved"] == "10.02.2023"
    assert evidence["temporal_review_required"] is True
    assert evidence["fetch_eligible"] is True
    assert evidence["independent_legal_effectivity_confirmed"] is None
    assert evidence["independent_boundary_equivalence_confirmed"] is None


def test_rejects_duplicate_official_code_instead_of_deduplicating(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        bridge.read_registry(registry(tmp_path, [row(), row()]))


def test_rejects_bad_component_width(tmp_path):
    with pytest.raises(ValueError, match="TER"):
        bridge.read_registry(registry(tmp_path, [row(code="3;701;000;000")]))


def test_missing_code_is_not_replaced_by_similar_or_old_code(tmp_path):
    _, frame = bridge.read_registry(registry(tmp_path, [row()]))
    with pytest.raises(ValueError, match="No official code evidence"):
        bridge.code_evidence(
            "03501000000",
            "2023-01-01",
            None,
            {"2023-02-08": frame},
            ("2023-02-08",),
        )
