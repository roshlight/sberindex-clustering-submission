"""Mutate report copies to prove independent claim guards reject regressions."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/round6_analysis"
SPEC = importlib.util.spec_from_file_location(
    "round6_report_claims", REPO / "scripts/verify_round6_report_claims.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def report_copy(tmp_path):
    for name in ["REPORT.md", "artifact.json", "claim_values.json", "report_design.json"]:
        shutil.copy2(ROOT / name, tmp_path / name)
    return tmp_path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def change_shared_prose(report, block_id, before, after):
    artifact = json.loads((report / "artifact.json").read_text("utf8"))
    block = next(b for b in artifact["manifest"]["blocks"] if b["id"] == block_id)
    assert before in block["body"]
    block["body"] = block["body"].replace(before, after)
    save(report / "artifact.json", artifact)
    text = (report / "REPORT.md").read_text("utf8")
    (report / "REPORT.md").write_text(text.replace(before, after), encoding="utf8")


def test_canonical_report_matches_independent_source_projection(report_copy):
    result = MODULE.verify(REPO, ROOT, report_copy)
    assert result["status"] == "PASS"
    assert len(result["sources"]) == len(result["source_sha256"])
    assert result["sources"]["configs/round6_analysis_v1.json"] == MODULE.sha(
        REPO / "configs/round6_analysis_v1.json"
    )
    assert result["sources"]["reports/round6_analysis/admission/QA.json"] == MODULE.sha(
        ROOT / "admission/QA.json"
    )
    assert all((REPO / path).is_file() for path in result["sources"])


def test_corrupted_interval_is_rejected(report_copy):
    path = report_copy / "artifact.json"
    value = json.loads(path.read_text("utf8"))
    value["snapshot"]["datasets"]["level_coefficients"][0]["rho_ci"] = "[0.100; 0.900]"
    save(path, value)
    assert "dataset:level_coefficients" in MODULE.verify(REPO, ROOT, report_copy)["error"]


def test_wrong_loo_rounding_rejected_even_when_both_formats_agree(report_copy):
    change_shared_prose(report_copy, "influence", "0.518 и 0.788", "0.519 и 0.788")
    assert "loo_range_exact_rounding" in MODULE.verify(REPO, ROOT, report_copy)["error"]


def test_financial_ratio_cannot_be_promoted_to_share(report_copy):
    change_shared_prose(report_copy, "cash", "нельзя объявить долей", "следует объявить долей")
    assert "cash_ratio_not_share" in MODULE.verify(REPO, ROOT, report_copy)["error"]


def test_detached_claim_values_cannot_disagree_with_sources(report_copy):
    path = report_copy / "claim_values.json"
    value = json.loads(path.read_text("utf8"))
    value["admission"][0]["compatible"] += 1
    save(path, value)
    assert "claim_values" in MODULE.verify(REPO, ROOT, report_copy)["error"]
