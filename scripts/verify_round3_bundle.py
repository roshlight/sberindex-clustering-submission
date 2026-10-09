"""Verify the immutable third-round analysis delivery without network calls or model fitting."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo):
    root = repo / "reports/round3_analysis"
    seal = json.loads((root / "delivery_manifest.json").read_text("utf-8"))
    for relative, expected in seal["files"].items():
        path = repo / relative
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Delivery file changed: {relative}")
    receipt = json.loads((root / "portable_delivery.json").read_text("utf-8"))
    assert receipt["ok"] and receipt["verification"]["ok"]
    assert receipt["artifact"]["sha256"] == sha(root / "artifact.json")
    assert receipt["output_sha256"] == sha(root / "report.html")
    assert receipt["verification"]["sourceDialog"] == "passed"
    qa = json.loads((root / "independent_QA.json").read_text("utf-8"))
    assert qa["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in qa["checks"])
    claims = json.loads((root / "report_claims_QA.json").read_text("utf-8"))
    assert claims["status"] == "PASS"
    assert claims["report_sha256"] == sha(root / "REPORT.md")
    assert claims["artifact_sha256"] == sha(root / "artifact.json")
    artifact = json.loads((root / "artifact.json").read_text("utf-8"))
    sources = {s["id"]: s for s in artifact["manifest"]["sources"]}
    for item in artifact["manifest"]["charts"] + artifact["manifest"]["tables"]:
        assert item["sourceId"] in sources
        assert item["dataset"] in artifact["snapshot"]["datasets"]
    for source in sources.values():
        assert (repo / source["path"]).is_file()
        for path in source["query"]["tables_used"]:
            assert (repo / path).is_file(), path
    print(
        json.dumps(
            {
                "status": "PASS",
                "hashed_files": len(seal["files"]),
                "html_verified": True,
                "sources_resolve": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().repo)
