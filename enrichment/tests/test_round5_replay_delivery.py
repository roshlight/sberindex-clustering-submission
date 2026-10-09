"""Guard immutable replay sources and detect incomplete or altered delivery receipts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "verify_round5_delivery", SCRIPTS / "verify_round5_delivery.py"
)
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)
SPEC_REPLAY = importlib.util.spec_from_file_location("replay_round5", SCRIPTS / "replay_round5.py")
replay = importlib.util.module_from_spec(SPEC_REPLAY)
SPEC_REPLAY.loader.exec_module(replay)


def seal_file(root, entries):
    path = root / "seal.json"
    path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "name",
    [
        "../escape.csv",
        "x/../../escape.csv",
        "C:/x.csv",
        "C:x.csv",
        "/x.csv",
        "\\\\host\\x",
        "file:stream",
    ],
)
def test_manifest_paths_cannot_leave_or_alias_filesystem_roots(tmp_path, name):
    with pytest.raises(ValueError):
        verify.inside(tmp_path, name)


def test_portable_and_legacy_separators_resolve_same_file(tmp_path):
    target = tmp_path / "data" / "table.csv"
    target.parent.mkdir()
    target.write_text("12", encoding="utf-8")
    assert verify.inside(tmp_path, "data\\table.csv") == target.resolve()
    with pytest.raises(ValueError, match="Duplicate normalized"):
        verify.verify_hashes(
            tmp_path,
            {
                "data/table.csv": verify.sha(target),
                "data\\table.csv": verify.sha(target),
            },
        )


def test_seal_detects_same_length_numeric_change(tmp_path):
    source = tmp_path / "table.csv"
    source.write_bytes(b"year,value\n2023,12\n")
    seal = seal_file(
        tmp_path,
        {
            "table.csv": {
                "sha256": verify.sha(source),
                "bytes": source.stat().st_size,
            }
        },
    )
    assert verify.verify_seal(tmp_path, seal) == {source.resolve()}
    source.write_bytes(b"year,value\n2023,21\n")
    with pytest.raises(ValueError, match="Hash changed"):
        verify.verify_seal(tmp_path, seal)


def test_seal_rejects_stale_size_even_when_hash_matches(tmp_path):
    source = tmp_path / "table.csv"
    source.write_bytes(b"12")
    seal = seal_file(tmp_path, {"table.csv": {"sha256": verify.sha(source), "bytes": 3}})
    with pytest.raises(ValueError, match="Byte size changed"):
        verify.verify_seal(tmp_path, seal)


@pytest.mark.parametrize("metadata", [{}, {"files": {}}, {"files": {"x": "bad"}}])
def test_empty_or_invalid_seal_is_not_success(tmp_path, metadata):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        verify.verify_seal(tmp_path, path)


def test_replay_refuses_repository_and_raw_output_locations(tmp_path):
    repo, raw, prior = (tmp_path / name for name in ["repo", "raw", "prior"])
    for directory in [repo, raw, prior]:
        directory.mkdir()
    for source in [repo, raw, prior]:
        with pytest.raises(ValueError, match="outside"):
            replay.validate_output(repo, raw, prior, source / "fresh-output")
    replay.validate_output(repo, raw, prior, tmp_path / "fresh-output")
    with pytest.raises(ValueError, match="never overwrites"):
        replay.validate_output(repo, raw, prior, raw)


def test_raw_tree_fingerprint_detects_added_receipt(tmp_path):
    before = replay.tree_state(tmp_path)
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    assert replay.tree_state(tmp_path) != before
