"""Delivery integrity rejects drift, incomplete seals and unsafe declared paths."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "round6_delivery_verifier_test", REPO / "enrichment/scripts/verify_round6_delivery.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def record(base, path):
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": V.sha(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture
def package(tmp_path):
    repo, release = tmp_path / "repo", tmp_path / "release"
    paths = set(V.BRANCH_SCRIPTS.values()) | {
        "enrichment/scripts/replay_round6.py",
        "enrichment/scripts/verify_round6_delivery.py",
        "enrichment/uv.lock",
        "enrichment/pyproject.toml",
        "enrichment/scripts/collect_round5_cases.py",
        "inputs/annual.csv",
        "inputs/geography.csv",
    }
    for name in paths:
        put(repo / name, "fixture")
    for branch, names in V.MACHINE_FILES.items():
        folder = repo / V.REPORT / branch
        for name in names - {"manifest.json"}:
            put(folder / name, {"fixture": name})
        if branch == "search":
            continue
        outputs = {name: V.sha(folder / name) for name in names - {"manifest.json"}}
        if branch == "sme":
            manifest = {
                "files": [record(folder, folder / name) for name in outputs],
                "dependency_hashes": {
                    V.BRANCH_SCRIPTS[branch]: V.sha(repo / V.BRANCH_SCRIPTS[branch])
                },
            }
        else:
            manifest = {"outputs": outputs, "script_sha256": V.sha(repo / V.BRANCH_SCRIPTS[branch])}
        if branch == "economy":
            manifest["repo_inputs"] = {"inputs/annual.csv": V.sha(repo / "inputs/annual.csv")}
        elif branch == "geography":
            manifest["inputs"] = {"repo/inputs/geography.csv": V.sha(repo / "inputs/geography.csv")}
        elif branch == "cases":
            put(folder / "evidence_recipe.json", {"manual": True})
            put(folder / "raw_allowlist.json", {"files": []})
            manifest.update(
                helper_script_sha256=V.sha(repo / "enrichment/scripts/collect_round5_cases.py"),
                recipe_sha256=V.sha(folder / "evidence_recipe.json"),
                retrievals_sha256=V.sha(folder / "source_inventory.json"),
            )
        put(folder / "manifest.json", manifest)
    for branch in V.MACHINE_FILES:
        put(release / f"{branch}.zip", "fixture payload; not extracted in this test")
    seal = repo / V.REPORT / "delivery_manifest.json"
    put(
        seal,
        {
            "schema_version": 1,
            "artifacts": [record(repo, p) for p in repo.rglob("*") if p.is_file()],
            "raw_releases": [record(release, p) for p in release.iterdir()],
        },
    )
    return repo, release, seal


def test_complete_seal_and_branches_pass(package):
    repo, release, seal = package
    result = V.verify(repo, seal, release_dir=release)
    assert result["status"] == "PASS"
    assert result["raw_releases_checked"] == 5
    assert sum(result["branch_machine_files"].values()) == 52
    assert result["replay_performed"] is False


@pytest.mark.parametrize(
    "target", ["artifact", "release", "declared_size", "unsealed_report", "missing_code"]
)
def test_changed_or_incomplete_delivery_fails(package, target):
    repo, release, seal = package
    value = V.read_json(seal)
    if target == "artifact":
        put(repo / V.REPORT / "cases/airport_annual_finance.csv", {"changed": True})
    elif target == "release":
        put(release / "sme.zip", {"changed": True})
    elif target == "declared_size":
        value["artifacts"][0]["bytes"] += 1
    elif target == "unsealed_report":
        put(repo / V.REPORT / "extra.json", {"not_sealed": True})
    else:
        value["artifacts"] = [r for r in value["artifacts"] if r["path"] != "inputs/annual.csv"]
    put(seal, value)
    with pytest.raises(ValueError):
        V.verify(repo, seal, release_dir=release)


@pytest.mark.parametrize("path", ["../x", "C:/temp/x", "/tmp/x", "\\\\server\\x", "x:stream"])
def test_external_manifest_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        V.verify_records(tmp_path, [{"path": path, "sha256": "a" * 64, "bytes": 0}])


def test_equivalent_manifest_paths_are_duplicates():
    rows = [{"path": name, "sha256": "a" * 64, "bytes": 0} for name in ["a/b", "a/./b"]]
    with pytest.raises(ValueError, match="Duplicate"):
        V.record_hashes(rows)


def test_conflicting_search_roots_require_selection(tmp_path):
    for name in ["search", "server-complete"]:
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="use --search-raw"):
        V.resolve_search_raw(tmp_path)
    assert V.resolve_search_raw(tmp_path, tmp_path / "search") == (tmp_path / "search").resolve()


def test_canonical_branch_contracts_pass():
    assert {key: len(value) for key, value in V.branch_hashes(REPO).items()} == {
        "economy": 7,
        "sme": 12,
        "geography": 22,
        "cases": 9,
        "search": 2,
    }
