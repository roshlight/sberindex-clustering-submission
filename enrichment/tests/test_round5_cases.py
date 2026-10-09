"""Fail closed when documentary context is promoted into numerical or causal proof."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round5_cases", Path(__file__).parents[1] / "scripts" / "collect_round5_cases.py"
)
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


@pytest.fixture
def evidence(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(
        "PAGE 1\n(тыс. рублей)\nPAGE 2\nЧерский 370152,2 126838 - 496990,2\n"
        "Оренбург 130000 79799 - 209799",
        encoding="utf-8",
    )
    source = {
        "status": "downloaded",
        "sha256": "abc",
        "text_path": "source.txt",
        "text_sha256": case.sha(path),
        "url": "http://example.test/report.pdf",
        "retrieved_at_utc": "2026-09-16T00:00:00Z",
        "role": "primary_budget_plan",
    }
    row = {
        "source_id": "source",
        "evidence_id": "airport_2023",
        "source_sha256": "abc",
        "needle": "Черский 370152,2 126838 - 496990,2",
        "physical_page": 2,
        "unit_physical_page": 1,
        "unit_needle": "(тыс. рублей)",
        "value_kind": "contract_cost_increase",
        "model_join_accepted": False,
        "causal_attribution_confirmed": False,
        "annual_bdpm_investment_reconciled": False,
    }
    return row, {"source": source}, tmp_path


def test_exact_row_and_units_are_accepted_only_as_context(evidence):
    row, sources, raw = evidence
    result = case.validate_evidence(row, sources, raw)
    assert result["model_join_accepted"] is False
    assert result["annual_bdpm_investment_reconciled"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("physical_page", 1),
        ("unit_physical_page", 2),
        ("source_sha256", "changed"),
        ("needle", "Черский 130000 79799 - 209799"),
        ("unit_needle", "(млн рублей)"),
    ],
)
def test_wrong_page_units_snapshot_or_adjacent_city_cannot_pass(evidence, key, value):
    row, sources, raw = evidence
    row[key] = value
    with pytest.raises(ValueError):
        case.validate_evidence(row, sources, raw)


@pytest.mark.parametrize(
    "key",
    ["model_join_accepted", "causal_attribution_confirmed", "annual_bdpm_investment_reconciled"],
)
@pytest.mark.parametrize("value", [True, "False", None])
def test_context_cannot_be_promoted_or_pass_string_boolean(evidence, key, value):
    row, sources, raw = evidence
    row[key] = value
    with pytest.raises(ValueError):
        case.validate_evidence(row, sources, raw)


def test_failed_source_is_not_accepted(evidence):
    row, sources, raw = evidence
    sources["source"]["status"] = "failed"
    with pytest.raises(ValueError):
        case.validate_evidence(row, sources, raw)


def test_replay_does_not_overwrite_a_previous_result(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.csv").write_text("prior result", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        case.replay(tmp_path / "raw", tmp_path / "recipe.json", out)


def test_public_json_is_parsed_as_json_and_html_fields(tmp_path):
    path = tmp_path / "article.json"
    case.save(
        path,
        {
            "name": "Title",
            "publishAt": "2025-02-05",
            "contentTop": "<p>ТЛЦ &laquo;Черский&raquo;</p>",
        },
    )
    text = case.extract_snapshot(
        {"id": "article", "path": "article.json", "content_type": "application/json"},
        tmp_path,
    )
    assert "ТЛЦ «Черский»" in text
    assert "2025-02-05" in text
    assert "contentTop" not in text


def test_replay_rejects_changed_source_before_deriving(tmp_path):
    (tmp_path / "raw.html").write_text("altered", encoding="utf-8")
    case.save(tmp_path / "retrievals.json", [{"id": "source", "path": "raw.html", "sha256": "old"}])
    case.save(tmp_path / "recipe.json", {})
    with pytest.raises(ValueError, match="Source hash changed"):
        case.replay(tmp_path, tmp_path / "recipe.json", tmp_path / "out")
