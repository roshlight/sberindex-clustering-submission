"""Replay command must not overwrite inputs, skip a branch, or ignore a failed stage."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "round6_replay_test", REPO / "enrichment/scripts/replay_round6.py"
)
R = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(REPO / "enrichment/scripts"))
try:
    SPEC.loader.exec_module(R)
finally:
    sys.path.pop(0)


@pytest.fixture
def environment(tmp_path):
    repo, raw, prior, output = [tmp_path / name for name in ["repo", "raw", "prior", "output"]]
    for path in [repo, prior, *(raw / name for name in R.BRANCH_SCRIPTS)]:
        path.mkdir(parents=True)
    for branch in R.BRANCH_SCRIPTS:
        folder = repo / R.REPORT / branch
        folder.mkdir(parents=True)
        (folder / "fixture.json").write_text(json.dumps({"branch": branch}), encoding="utf-8")
    (repo / R.REPORT / "sme/manifest.json").write_text(
        '{"dependency_hashes": {}}', encoding="utf-8"
    )
    for name in ["evidence_recipe.json", "raw_allowlist.json"]:
        (repo / R.REPORT / "cases" / name).write_text("{}", encoding="utf-8")
    paths = set(R.BRANCH_SCRIPTS.values()) | {
        "enrichment/scripts/replay_round6.py",
        "enrichment/scripts/verify_round6_delivery.py",
        "enrichment/uv.lock",
        "enrichment/pyproject.toml",
        "enrichment/scripts/collect_round5_cases.py",
    }
    for name in paths:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture code", encoding="utf-8")
    return repo, raw, prior, output


@pytest.mark.parametrize("location", ["existing", "repository", "raw", "prior"])
def test_output_must_be_new_and_external(environment, location):
    repo, raw, prior, output = environment
    if location == "existing":
        output.mkdir()
    else:
        output = {"repository": repo, "raw": raw, "prior": prior}[location] / "new"
    with pytest.raises(ValueError):
        R.validate_output(repo, raw, prior, output, raw / "search")


def test_all_five_commands_are_offline_and_prior_is_explicit(environment):
    repo, raw, prior, output = environment
    stages = R.stage_arguments(repo, raw, prior, output, raw / "search")
    assert set(stages) == {"economy", "sme", "geography", "cases", "search"}
    assert stages["geography"][stages["geography"].index("--prior-raw") + 1] == prior
    assert "--derive-only" in stages["cases"]
    assert stages["search"][stages["search"].index("--raw") + 1] == raw / "search"


@pytest.mark.parametrize("failure", [None, "stage", "wrong_output", "raw_mutation"])
def test_replay_checks_output_and_preserves_failure_evidence(environment, monkeypatch, failure):
    repo, raw, prior, output = environment
    expected = {
        branch: {"fixture.json": R.sha(repo / R.REPORT / branch / "fixture.json")}
        for branch in R.BRANCH_SCRIPTS
    }
    monkeypatch.setattr(R, "branch_hashes", lambda _: expected)
    monkeypatch.setattr(R, "verify_raw", lambda *_: {"fixture": 1})

    def subprocess_run(command, **kwargs):
        branch_output = Path(command[command.index("--out") + 1])
        branch = branch_output.name
        branch_output.mkdir()
        shutil.copyfile(repo / R.REPORT / branch / "fixture.json", branch_output / "fixture.json")
        if failure == "wrong_output" and branch == "economy":
            (branch_output / "fixture.json").write_text("bad", encoding="utf-8")
        if failure == "raw_mutation" and branch == "search":
            (raw / "economy/added").write_text("changed", encoding="utf-8")
        return SimpleNamespace(returncode=int(failure == "stage"), stdout=b"stage", stderr=b"")

    monkeypatch.setattr(R.subprocess, "run", subprocess_run)
    if failure is None:
        result = R.replay(repo, raw, prior, output)
        assert result["status"] == "PASS"
        assert result["machine_files_matched"] == 5
        assert result["source_trees_unchanged"] is True
    else:
        with pytest.raises(ValueError):
            R.replay(repo, raw, prior, output)
        qa = json.loads((output / "QA.json").read_text(encoding="utf-8"))
        assert qa["status"] == "FAIL"
        assert qa["stages"]
        assert (output / "economy.stdout.log").is_file()
