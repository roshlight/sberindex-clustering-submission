"""Protect denominators, noninvertible code transitions, units and time semantics."""

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round6_context", Path(__file__).parents[1] / "scripts/analyze_round6_context.py"
)
context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context)


def edge(old="01111111001", parent="01111111", count="7", predecessors="2"):
    return {
        "year": "2023",
        "source_code": "01222222001",
        "source_rows": count,
        "region": "test_region",
        "new_code_predecessors_all_regions": predecessors,
        "address_date_proven": "False",
        "parent_code": parent,
        "old_code": old,
        "territory_id": parent,
    }


def test_multiple_predecessors_one_parent_counts_source_rows_once():
    rows = [edge(), edge(old="01111111002")]
    groups = context.code_groups(rows)
    assert len(groups) == 1
    assert groups[0]["ambiguity_class"] == "multiple_settlements_same_candidate_parent"
    assert groups[0]["candidate_parent_codes"] == 1
    inventory = [
        {
            "year": "2023",
            "source_code": "01222222001",
            "source_rows": "7",
            "source_code_is_null": "False",
        },
        {"year": "2023", "source_code": "", "source_rows": "3", "source_code_is_null": "True"},
    ]
    summary = context.queue_summary(groups, inventory)[-1]
    assert summary["source_rows_once"] == 7
    assert summary["naive_edge_weighted_rows"] == 14
    assert summary["denominator_all_sme_year_rows"] == 10
    assert summary["share_all_sme_year_rows_pct"] == 70
    assert summary["historical_assignments_approved"] == 0


def test_single_local_edge_does_not_hide_other_crosswalk_predecessor():
    row = context.code_groups([edge()])[0]
    assert row["ambiguity_class"] == "additional_predecessor_outside_candidate_set"
    assert row["candidate_parent_codes"] == 1
    assert row["all_crosswalk_predecessor_codes"] == 2
    assert not row["historical_assignment_approved"]


def test_conflicting_code_group_counts_fail_instead_of_arbitrary_deduplication():
    with pytest.raises(ValueError, match="Inconsistent source_rows"):
        context.code_groups([edge(), edge(old="01111111002", count="8")])


def test_multiple_candidate_parents_stay_ambiguous():
    group = context.code_groups([edge(), edge(old="01111111002", parent="01999999")])[0]
    assert group["ambiguity_class"] == "multiple_candidate_parents"
    assert group["candidate_parent_codes"] == 2


def finance(cash="", thousand="", status="blank_not_zero"):
    return {
        "year": "2024",
        "financial_kind": "annual_cash_execution",
        "value_million_rubles": cash,
        "value_thousand_rubles": thousand,
        "observation_status": status,
        "scope": "object_cash",
        "publication_date": "2025-12-03",
        "source_cells": "AH56",
        "source_sha256": "a",
        "source_url": "https://example.test/cash",
    }


def municipal():
    return {
        "year": "2024",
        "value_million_rubles": "24.8",
        "rounding_half_step_million_rubles": "0.05",
        "municipality": "case",
        "scope": "statistical_investment",
        "source_sha256s": "b",
        "source_urls": "https://example.test/investment",
    }


def test_blank_cash_does_not_become_zero_or_negative_change():
    row = context.airport_comparison([finance()], [municipal()])[0]
    assert row["airport_cash_million_rubles"] is None
    assert row["cash_divided_by_municipal_investment"] is None
    zero = context.airport_comparison([finance("0", "0", "observed")], [municipal()])[0]
    assert zero["cash_divided_by_municipal_investment"] == 0


def test_unit_error_blocks_comparison_and_ratio_is_not_share():
    with pytest.raises(ValueError, match="Cash units disagree"):
        context.airport_comparison([finance("10", "10", "observed")], [municipal()])
    row = context.airport_comparison([finance("50", "50000", "observed")], [municipal()])[0]
    assert row["cash_divided_by_municipal_investment"] > 1
    assert not row["ratio_is_project_share"]
    assert row["ratio_lower_from_municipal_rounding_only"] < Decimal(50) / Decimal("24.8")


def test_asset_syntax_states_must_partition_rows():
    rows = [
        {"asset": "x", "code_state": state, "rows": str(n), "denominator_rows": "10"}
        for state, n in [
            ("empty", 1),
            ("exact_8_or_11_digits", 4),
            ("scientific_notation", 2),
            ("other_digit_length", 2),
            ("other_syntax", 1),
        ]
    ]
    result = context.asset_readiness(rows)[0]
    assert result["syntax_review_rows"] == 6
    assert result["syntax_review_pct"] == 60
    rows[-1]["rows"] = "2"
    with pytest.raises(ValueError, match="partition"):
        context.asset_readiness(rows)


def test_workers_measurement_year_and_publication_year_remain_separate():
    rows = [
        {
            "data_date": "31.12.2025",
            "document_date": "25.08.2026",
            "workers_state": "positive",
            "records": "764",
            "denominator_records": "900",
        }
    ]
    result = context.worker_temporal(rows)
    assert {r["target_year"] for r in result} == {2023, 2024}
    assert all(not r["measurement_year_matches_target"] for r in result)
    assert all(r["document_lag_days_from_data_date"] == 237 for r in result)
    assert not any(r["eligible_historical_input"] for r in result)


def test_ambiguous_edges_partition_without_adding_node_counts():
    rows = [
        {
            "old_code": "a",
            "new_code": "b",
            "predecessor_code_count": "2",
            "successor_code_count": "1",
            "region": "x",
        },
        {
            "old_code": "c",
            "new_code": "b",
            "predecessor_code_count": "2",
            "successor_code_count": "1",
            "region": "x",
        },
    ]
    result = context.ambiguity_summary(rows)[0]
    assert result["edges"] == 2
    assert result["distinct_old_codes"] == 2
    assert result["distinct_new_codes"] == 1
    assert result["edge_share_pct"] == 100
