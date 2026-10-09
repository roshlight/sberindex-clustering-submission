"""Reject reporting-scale, confidence-interval, source and Markdown regressions."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "round5_claims_verifier", REPO / "scripts/verify_round5_report_claims.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


@pytest.fixture
def copied_report(tmp_path):
    source = REPO / "reports/round5_analysis"
    files = [
        *V.SOURCES.values(),
        "models/cv_metrics.csv",
        "models/manifest.json",
        "independent_QA.json",
        "readout/QA.json",
        "headline_metrics.csv",
        "REPORT.md",
        "artifact.json",
    ]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return tmp_path


def test_final_numerical_report_passes(copied_report):
    assert V.verify(REPO, copied_report)["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["percent_scale", "ci_sign", "source_swap", "markdown"])
def test_changed_claim_is_rejected(copied_report, mutation):
    path = copied_report / "artifact.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "percent_scale":
        artifact["snapshot"]["datasets"]["primary_table"][0]["gain"] *= 100
    elif mutation == "ci_sign":
        artifact["snapshot"]["datasets"]["primary_table"][0]["low"] *= -1
    elif mutation == "source_swap":
        artifact["manifest"]["tables"][0]["sourceId"] = "models"
    else:
        report_path = copied_report / "REPORT.md"
        text = report_path.read_text(encoding="utf-8")
        assert "| 2023 | 1975 | 1768 | 1800 |" in text
        report_path.write_text(
            text.replace("| 2023 | 1975 | 1768 | 1800 |", "| 2023 | 1975 | 1768 | 1801 |"),
            encoding="utf-8",
        )
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    result = V.verify(REPO, copied_report)
    assert result["status"] == "FAIL"
    assert any(check["status"] == "FAIL" for check in result["checks"])
