"""Verify delivered bytes and an optional deterministic core replay, without network."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside bundle: {relative}")
    return path


def verify(repo, replay=None):
    root = repo / "reports/enriched_analysis"
    manifest = json.loads((root / "delivery_manifest.json").read_text(encoding="utf-8"))
    checked = {}
    for name, expected in manifest["files"].items():
        path = contained(repo, name)
        checked[name] = path.is_file() and sha(path) == expected
    assert all(checked.values()), [key for key, ok in checked.items() if not ok]
    core = json.loads((root / "core_v2/manifest.json").read_text(encoding="utf-8"))
    inputs = {
        name: sha(contained(repo, name)) == value for name, value in core["input_hashes"].items()
    }
    assert all(inputs.values()), inputs
    comparison = None
    if replay:
        comparison = {name: sha(replay / name) == value for name, value in core["outputs"].items()}
        assert all(comparison.values()), comparison
    return {
        "status": "PASS",
        "delivered_files": len(checked),
        "core_input_hashes": inputs,
        "core_replay_byte_comparison": comparison,
        "network_requests": 0,
        "scope": "Byte identity and core replay; does not validate economic interpretation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--compare-core", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.repo, args.compare_core)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Use a new QA output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
