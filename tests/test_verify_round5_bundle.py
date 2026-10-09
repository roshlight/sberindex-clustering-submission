"""Reject stale review evidence, altered files and weakened adoption thresholds."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/verify_round5_bundle.py"
SPEC = importlib.util.spec_from_file_location("verify_round5_bundle", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_absolute_paths_and_parent_escape_are_rejected(tmp_path):
    for name in ["../outside", "/outside", "C:/outside", "C:outside", "\\\\server\\file"]:
        with pytest.raises(ValueError):
            module.inside(tmp_path, tmp_path, name)


def test_hashes_detect_same_length_change_and_duplicate_separator_alias(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    file = folder / "data.csv"
    file.write_bytes(b"12")
    digest = module.sha(file)
    with pytest.raises(ValueError, match="Duplicate"):
        module.hashes(tmp_path, {"folder/data.csv": digest, "folder\\data.csv": digest})
    file.write_bytes(b"21")
    with pytest.raises(ValueError, match="Hash changed"):
        module.hashes(tmp_path, {"folder/data.csv": digest})


def test_report_review_is_invalidated_by_changed_report(tmp_path):
    report, artifact = tmp_path / "REPORT.md", tmp_path / "artifact.json"
    report.write_text("original", encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")
    review = {
        "status": "PASS",
        "checks": [{"status": "PASS"}],
        "model_refits": 0,
        "report_sha256": module.sha(report),
        "artifact_sha256": module.sha(artifact),
        "sources": {"artifact.json": module.sha(artifact)},
    }
    (tmp_path / "report_review.json").write_text(json.dumps(review), encoding="utf-8")
    module.verify_report_review(tmp_path, tmp_path)
    report.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        module.verify_report_review(tmp_path, tmp_path)


def write_comparisons(path, stable_gain=0, lower=0.001, rows_to_remove=0):
    rows = [
        {
            "comparison": "primary",
            "seed": 13,
            "target": "profile_clr",
            "sample": sample,
            "year": year,
            "regime": regime,
            "relative_mse_reduction": 0.01 if sample == "core" else stable_gain,
            "ci_low": lower,
            "ci_high": 0.02,
        }
        for sample in ["core", "stable_code"]
        for year in [2023, 2024]
        for regime in ["within_region", "heldout_regions"]
    ]
    if rows_to_remove:
        rows = rows[:-rows_to_remove]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_adoption_preserves_nonnegative_stable_and_strict_positive_interval(tmp_path):
    path = tmp_path / "comparisons.csv"
    write_comparisons(path)
    assert module.adoption_from_csv(path)
    write_comparisons(path, stable_gain=-1e-12)
    assert not module.adoption_from_csv(path)
    write_comparisons(path, lower=0)
    assert not module.adoption_from_csv(path)


def test_incomplete_adoption_grid_cannot_pass_vacuously(tmp_path):
    path = tmp_path / "comparisons.csv"
    write_comparisons(path, rows_to_remove=1)
    with pytest.raises(ValueError, match="Incomplete"):
        module.adoption_from_csv(path)


@pytest.mark.parametrize("changed_branch", ["audit", "recovered", "readout"])
def test_independent_branch_receipt_rejects_regenerated_manifest(tmp_path, changed_branch):
    qa = {}
    for branch, section, key in [
        ("audit", "data_audit", "audit_manifest_sha256"),
        ("recovered", "recovered_description", "manifest_sha256"),
        ("readout", "readout", "manifest_sha256"),
    ]:
        directory = tmp_path / branch
        directory.mkdir()
        manifest = directory / "manifest.json"
        manifest.write_text('{"script_sha256":"original"}', encoding="utf-8")
        qa[section] = {key: module.sha(manifest)}
    module.verify_independent_branch_receipts(tmp_path, qa)
    (tmp_path / changed_branch / "manifest.json").write_text(
        '{"script_sha256":"reformatted"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=f"Independent {changed_branch} review is stale"):
        module.verify_independent_branch_receipts(tmp_path, qa)
