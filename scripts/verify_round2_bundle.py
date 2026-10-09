"""Verify published round-two analysis bytes and frozen external inputs without networking."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(repo):
    repo = repo.resolve()
    manifest = json.loads((repo / "reports/round2_analysis/bundle_manifest.json").read_text())
    bad = []
    files = {**manifest["files"], **manifest["frozen_inputs"]}
    for relative, expected in files.items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo):
            raise ValueError("Manifest path escapes repository")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(relative)
    print(
        json.dumps(
            {"status": "PASS" if not bad else "FAIL", "files": len(files), "mismatches": bad},
            ensure_ascii=False,
        )
    )
    return bool(bad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    raise SystemExit(verify(parser.parse_args().repo))
