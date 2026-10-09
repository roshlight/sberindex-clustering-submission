"""Admission guards for printed precision and distinct economic concepts."""

import importlib.util
import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_round6_admission.py"
SPEC = importlib.util.spec_from_file_location("round6_admission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_published_precision_uses_amount_intervals_too():
    # Point ratio 110.049 lies outside 110.1 +/- .05, but the printed
    # amounts permit a ratio above 110.05, so point-only rounding is wrong.
    result = MODULE.rounding_comparison("100.00", "110.049", "110.1")
    assert result["status"] == "rounding_compatible"
    assert result["ratio_percent"] < 110.05
    assert result["same_definition_confirmed"] is False


def test_source_example_rejects_unexplained_percentage_difference():
    result = MODULE.rounding_comparison("3192279.2", "3524891.7", "123.6")
    assert result["status"] == "different_beyond_rounding"
    assert result["difference_cause"] == "unknown"
    assert result["difference_pp"] < -13


@pytest.mark.parametrize("literal", ["…", "...", "—", ""])
def test_missing_percentage_is_not_zero(literal):
    result = MODULE.rounding_comparison("100.0", "110.0", literal)
    assert result["status"] == "uncomputable"
    assert result["ratio_percent"] is None


def test_source_literal_and_zero_denominator_guards():
    assert MODULE.numeric("1\u00a0234,50") == Decimal("1234.50")
    assert MODULE.rounding_comparison("0.0", "1.0", "100.0")["status"] == "uncomputable"
    with pytest.raises(ValueError, match="Unreviewed"):
        MODULE.numeric("123.4*")


def test_related_concepts_do_not_inherit_model_or_retail_admission():
    for metric in MODULE.SCOPES:
        decision = MODULE.admission_for(metric)
        assert decision["model_admission"] is False
        assert decision["missing_cell_fill"] is False
        assert decision["frozen_value_replacement"] is False
        assert decision["historical_boundary_equivalence"] == "unknown"
    with pytest.raises(ValueError, match="Unknown metric"):
        MODULE.admission_for("total_retail")


def test_ambiguous_word_table_and_existing_output_are_rejected(tmp_path):
    path = tmp_path / "fake.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "<document><body/></document>")
    with pytest.raises(ValueError, match="Ambiguous shipment table"):
        MODULE.read_docx_totals(path)
    with pytest.raises(FileExistsError, match="fresh"):
        MODULE.run(tmp_path, tmp_path, tmp_path)
