"""Semantic guards for the context-only SME and rounded-investment analyses."""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round4_context_analysis", Path(__file__).parents[1] / "scripts/round4_context_analysis.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rounding_bounds_surround_point_and_preserve_fall():
    result = MODULE.rounded_ratio("31.8", "24.8")
    assert result["nominal_ratio_rounding_lower"] < 24.8 / 31.8
    assert result["nominal_ratio_rounding_upper"] > 24.8 / 31.8
    assert result["nominal_growth_rounding_upper_pct"] < 0
    assert result["uncertainty_kind"] == "arithmetic_rounding_bounds_not_confidence_interval"


@pytest.mark.parametrize("old,new", [("0", "12"), ("0.05", "1"), ("1", "NaN"), ("-1", "2")])
def test_invalid_ratio_inputs_are_not_repaired(old, new):
    with pytest.raises(ValueError):
        MODULE.rounded_ratio(old, new)


def test_repeated_records_do_not_become_multiple_entities_or_worker_observations():
    rows = pd.DataFrame(
        [
            ("A", 0, "12345678", 1, "47.11", 10),
            ("A", 0, "12345678", 1, "47.12", 10),
            ("B", 0, "12345678", 1, "47.11", 0),
            ("B", 0, "87654321", 1, "47.11", 0),
            ("C", 1, "12345678", 1, "47.11", None),
            ("D", 0, None, 1, "invalid", -1),
        ],
        columns=[
            "tax_number",
            "is_sole_trader",
            "municipality_code",
            "sme_category",
            "main_nace_code",
            "employees_count",
        ],
    )
    with duckdb.connect() as connection:
        connection.register("sme_source", rows)
        result, sector = MODULE.entity_audit(connection, "sme_source")
    legal = result.set_index("is_sole_trader").loc[0]
    assert legal.tax_ids == 3 and legal.source_rows == 5
    assert legal.tax_ids_valid_known_workers == 2
    assert legal.tax_ids_zero_workers == 1
    assert legal.tax_ids_negative_workers == 1
    assert legal.tax_ids_multiple_full_nace == 1 and legal.tax_ids_multiple_nace2 == 0
    assert legal.record_candidate_known_workers == 1
    sole = result.set_index("is_sole_trader").loc[1]
    assert sole.tax_ids_valid_known_workers == 0
    assert sole.tax_ids_no_valid_workers == 1
    assert not result.historical_municipal_join_ready.any()
    assert sector.legal_entity_tax_ids.sum() == 3


def test_conflicting_published_values_refuse_collapse():
    evidence = pd.read_csv(
        Path(__file__).parents[1] / "reports/data_round4/cases/evidence_ledger.csv",
        dtype=str,
        keep_default_na=False,
    )
    evidence.loc[0, "value"] = "12345.6"
    with pytest.raises(ValueError, match="Conflicting"):
        MODULE.collapse_case_evidence(evidence)


def test_changed_scope_refuses_comparison():
    evidence = pd.read_csv(
        Path(__file__).parents[1] / "reports/data_round4/cases/evidence_ledger.csv",
        dtype=str,
        keep_default_na=False,
    )
    evidence.loc[0, "scope"] = "all_businesses"
    with pytest.raises(ValueError, match="scope"):
        MODULE.collapse_case_evidence(evidence)
