"""A passing sealed bundle must refer to the exact reviewed evidence and local files."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bundle_round6", Path(__file__).parents[1] / "scripts/verify_round6_bundle.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "C:x", "\\\\host\\x", "x:stream"])
def test_sealed_paths_are_portable_and_contained(tmp_path, name):
    with pytest.raises(ValueError):
        module.inside(tmp_path, name)


def test_same_length_tampering_and_alias_duplicates_fail(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    file = folder / "data"
    file.write_bytes(b"12")
    digest = module.sha(file)
    entries = [{"path": "folder/data", "bytes": 2, "sha256": digest}]
    module.verify_artifacts(tmp_path, entries)
    with pytest.raises(ValueError, match="Duplicate"):
        module.verify_artifacts(tmp_path, entries + [dict(entries[0], path="folder\\data")])
    file.write_bytes(b"21")
    with pytest.raises(ValueError, match="Hash changed"):
        module.verify_artifacts(tmp_path, entries)


def test_byte_length_is_part_of_seal(tmp_path):
    file = tmp_path / "data"
    file.write_bytes(b"a")
    with pytest.raises(ValueError, match="Byte length"):
        module.verify_artifacts(
            tmp_path, [{"path": "data", "bytes": 2, "sha256": module.sha(file)}]
        )


def test_report_review_rejects_changed_report(tmp_path):
    report, artifact = tmp_path / "REPORT.md", tmp_path / "artifact.json"
    report.write_text("original", encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")
    receipt = {
        "status": "PASS",
        "checks": [{"status": "PASS"}],
        "report_sha256": module.sha(report),
        "artifact_sha256": module.sha(artifact),
        "sources": {"artifact.json": module.sha(artifact)},
        "model_refits": 0,
    }
    (tmp_path / "report_review.json").write_text(json.dumps(receipt), encoding="utf-8")
    module.verify_report_review(tmp_path, tmp_path)
    report.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        module.verify_report_review(tmp_path, tmp_path)


def test_a_pass_heading_cannot_hide_failed_checks():
    with pytest.raises(ValueError, match="failed"):
        module.passed_checks({"status": "PASS", "checks": [{"status": "FAIL"}]}, "test")
    with pytest.raises(ValueError, match="did not pass"):
        module.passed_checks({"status": "PASS", "checks": []}, "test")


def test_independent_receipt_becomes_stale_when_a_branch_changes(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    verifier = scripts / "verify_round6_analysis.py"
    verifier.write_text("# fixture", encoding="utf-8")
    root = tmp_path / "reports"
    root.mkdir()
    input_file = tmp_path / "source.csv"
    input_file.write_text("source", encoding="utf-8")
    manifests = {}
    for branch in module.BRANCH_SCRIPTS:
        folder = root / branch
        folder.mkdir()
        path = folder / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        manifests[branch] = module.sha(path)
    receipt = {
        "status": "PASS",
        "checks": [{"status": "PASS"}],
        "model_refits": 0,
        "verifier_sha256": module.sha(verifier),
        "branch_manifests_sha256": manifests,
        "inputs": {"source.csv": module.sha(input_file)},
        "artifact_sha256": {f"{b}/manifest.json": h for b, h in manifests.items()},
    }
    (root / "independent_QA.json").write_text(json.dumps(receipt), encoding="utf-8")
    module.verify_independent(tmp_path, root)
    (root / "context/manifest.json").write_text("{ }", encoding="utf-8")
    with pytest.raises(ValueError, match="Independent branch review is stale: context"):
        module.verify_independent(tmp_path, root)


def test_changed_claim_values_invalidates_claims_receipt(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    code = scripts / "verify_round6_report_claims.py"
    code.write_text("# fixture", encoding="utf-8")
    root = tmp_path / "reports"
    root.mkdir()
    receipt = {
        "status": "PASS",
        "checks": [{"check": "arithmetic", "status": "PASS"}],
        "model_refits": 0,
        "verifier_sha256": module.sha(code),
        "source_sha256": {},
    }
    for filename, key in [
        ("REPORT.md", "report_sha256"),
        ("artifact.json", "artifact_sha256"),
        ("claim_values.json", "claim_values_sha256"),
    ]:
        path = root / filename
        path.write_text("{}", encoding="utf-8")
        receipt[key] = module.sha(path)
    (root / "report_claims_QA.json").write_text(json.dumps(receipt), encoding="utf-8")
    module.verify_report_claims(tmp_path, root)
    (root / "claim_values.json").write_text("{ }", encoding="utf-8")
    with pytest.raises(ValueError, match="stale: claim_values.json"):
        module.verify_report_claims(tmp_path, root)
