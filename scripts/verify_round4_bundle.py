"""Verify the sealed fourth-round delivery offline, without writes or model fitting.

Requires the final delivery_manifest.json and portable_delivery.json. This script
does not create or repair either receipt and does not repeat browser visual QA.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(repo, base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(repo), f"Path leaves repository: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_links(repo, root, text):
    checked = 0
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\s]+)\)", text):
        target = target.strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        inside(repo, root, unquote(parsed.path))
        checked += 1
    return checked


def verify(repo):
    repo = repo.resolve()
    root = repo / "reports/round4_analysis"
    seal = read_json(root / "delivery_manifest.json")
    files = seal["files"]
    require(bool(files), "Empty delivery manifest")
    sealed_paths = set()
    for relative, expected in files.items():
        path = inside(repo, repo, relative)
        require(path not in sealed_paths, f"Duplicate normalized manifest path: {relative}")
        sealed_paths.add(path)
        require(
            sha(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
            f"Delivery file changed: {relative}",
        )
    required = [
        repo / "configs/round4_analysis_v1.json",
        repo / "scripts/verify_round4_bundle.py",
        *[
            root / name
            for name in [
                "REPORT.md",
                "artifact.json",
                "report.html",
                "portable_delivery.json",
                "independent_QA.json",
                "report_claims_QA.json",
                "headline_metrics.csv",
                "readout/QA.json",
                "readout/state_update_contributions.csv",
                "models/manifest.json",
                "models/protocol_lock.json",
            ]
        ],
    ]
    require(
        all(path.resolve() in sealed_paths for path in required),
        "Delivery manifest omits a required final artifact, QA record or protocol",
    )

    receipt = read_json(root / "portable_delivery.json")
    require(
        receipt["ok"] is True and receipt["verification"]["ok"] is True,
        "Portable build or verification did not pass",
    )
    require(
        receipt["artifact"]["sha256"] == sha(root / "artifact.json"),
        "Portable artifact receipt is stale",
    )
    require(receipt["output_sha256"] == sha(root / "report.html"), "Portable HTML receipt is stale")
    require(
        receipt["output_bytes"] == (root / "report.html").stat().st_size,
        "Portable HTML length changed",
    )
    require(
        receipt["verification"]["sourceDialog"] == "passed",
        "Portable source dialog was not verified",
    )

    qa = read_json(root / "independent_QA.json")
    claims = read_json(root / "report_claims_QA.json")
    for name, report in [("independent_QA", qa), ("report_claims_QA", claims)]:
        require(
            report["status"] == "PASS"
            and bool(report["checks"])
            and all(check["status"] == "PASS" for check in report["checks"]),
            f"{name} has incomplete or failed checks",
        )
    require(claims["report_sha256"] == sha(root / "REPORT.md"), "Report claims QA is stale")
    require(
        claims["headline_sha256"] == sha(root / "headline_metrics.csv"),
        "Headline claims QA is stale",
    )
    require(
        claims["contribution_sha256"] == sha(root / "readout/state_update_contributions.csv"),
        "Readout claims QA is stale",
    )
    require(claims["model_refits"] == 0, "Report readout unexpectedly refit models")
    require(
        claims["verifier_sha256"] == sha(repo / "scripts/verify_round4_report_claims.py"),
        "Report verifier code changed after QA",
    )
    require(
        qa["verifier_sha256"] == sha(repo / "scripts/verify_round4_analysis.py"),
        "Model verifier code changed after QA",
    )
    require(
        qa["model_manifest_sha256"] == sha(root / "models/manifest.json"),
        "Independent model QA is stale",
    )

    protocol = repo / "configs/round4_analysis_v1.json"
    cfg, lock = read_json(protocol), read_json(root / "models/protocol_lock.json")
    model_manifest = read_json(root / "models/manifest.json")
    config_hash = sha(protocol)
    require(
        lock["protocol_sha256"] == qa["protocol_sha256"] == config_hash,
        "Protocol changed after fit or independent review",
    )
    require(
        model_manifest["inputs"]["configs/round4_analysis_v1.json"] == config_hash,
        "Model manifest refers to another protocol",
    )
    require(lock["source_base_commit"] == cfg["base_commit"], "Source base commit disagrees")
    require(
        lock["model_code_sha256"]
        == model_manifest["code_sha256"]
        == qa["production_code_sha256"]
        == sha(repo / "scripts/round4_model_analysis.py"),
        "Model code changed after lock or review",
    )
    for relative, digest in model_manifest["inputs"].items():
        require(sha(inside(repo, repo, relative)) == digest, f"Model input changed: {relative}")
    for relative, digest in model_manifest["outputs"].items():
        require(
            sha(inside(repo, root / "models", relative)) == digest,
            f"Model output changed: {relative}",
        )
    readout = read_json(root / "readout/QA.json")
    require(
        readout["status"] == "PASS"
        and readout["adoption_passed"] is False
        and qa["independent_adoption_criteria_met"] is False,
        "Readout and independent adoption decisions disagree",
    )
    require(
        readout["postfit_contribution_sums_match_state_update"] is True,
        "Readout contribution sums were not verified",
    )

    artifact = read_json(root / "artifact.json")
    manifest, snapshot = artifact["manifest"], artifact["snapshot"]
    sources = {source["id"]: source for source in manifest["sources"]}
    require(len(sources) == len(manifest["sources"]), "Duplicate source identifiers")
    require(artifact["sources"] == manifest["sources"], "Artifact source copies disagree")
    charts, tables = manifest["charts"], manifest["tables"]
    for item in charts + tables:
        require(item["sourceId"] in sources, f"Unknown source on {item['id']}")
        require(item["dataset"] in snapshot["datasets"], f"Unknown dataset on {item['id']}")
    for source in sources.values():
        inside(repo, repo, source["path"])
        for path in source["query"]["tables_used"]:
            inside(repo, repo, path)
        for name in ["computation_path", "reproduction_document"]:
            inside(repo, repo, source["query"][name])
    counts = receipt["verification"]["counts"]
    require(
        counts["charts"] == len(charts)
        and counts["tables"] == len(tables)
        and counts["blocks"] == len(manifest["blocks"]),
        "Portable receipt block counts disagree with artifact",
    )
    link_count = markdown_links(repo, root, (root / "REPORT.md").read_text(encoding="utf-8"))
    for block in manifest["blocks"]:
        if block.get("sourceId"):
            require(block["sourceId"] in sources, f"Unknown block source: {block['id']}")
        if block["type"] == "markdown":
            link_count += markdown_links(repo, root, block["body"])
    result = {
        "status": "PASS",
        "hashed_files": len(files),
        "model_refits": 0,
        "portable_receipt_hashes_verified": True,
        "source_dialog_receipt_passed": True,
        "independent_qa_current": True,
        "report_claims_qa_current": True,
        "protocol_hash_verified": True,
        "adoption_passed": False,
        "charts": len(charts),
        "tables": len(tables),
        "sources": len(sources),
        "local_markdown_links_checked": link_count,
        "remote_urls_checked": False,
        "browser_visual_qa_repeated": False,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().repo)))
