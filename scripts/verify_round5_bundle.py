"""Verify the sealed fifth-round analysis without writes, network calls or model refits."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlsplit

BRANCH_SCRIPTS = {
    "audit": "round5_data_audit.py",
    "models": "round5_model_analysis.py",
    "readout": "round5_model_readout.py",
    "recovered": "round5_recovered_profiles.py",
    "context": "round5_context_analysis.py",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(scope, base, relative):
    require(isinstance(relative, str) and bool(relative), "Invalid file path")
    windows = PureWindowsPath(relative)
    require(
        not windows.drive and not windows.root and ":" not in relative,
        f"Absolute or invalid file path: {relative}",
    )
    path = (base / relative.replace("\\", "/")).resolve()
    require(path.is_relative_to(scope.resolve()), f"Path leaves allowed scope: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(base, entries):
    require(isinstance(entries, dict) and bool(entries), "Empty or invalid hash map")
    checked = set()
    for relative, digest in entries.items():
        path = inside(base, base, relative)
        require(path not in checked, f"Duplicate normalized hash path: {relative}")
        checked.add(path)
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"Invalid SHA256: {relative}",
        )
        require(sha(path) == digest, f"Hash changed: {relative}")
    return checked


def verify_seal(repo, root):
    manifest = root / "delivery_manifest.json"
    files = read_json(manifest)["files"]
    require(isinstance(files, dict) and bool(files), "Empty or invalid delivery seal")
    require(all(isinstance(value, dict) for value in files.values()), "Invalid seal metadata")
    checked = hashes(repo, {name: value.get("sha256") for name, value in files.items()})
    for name, value in files.items():
        size = value.get("bytes")
        require(type(size) is int and size >= 0, f"Invalid byte size: {name}")
        require(inside(repo, repo, name).stat().st_size == size, f"Byte size changed: {name}")
    report_files = {p.resolve() for p in root.rglob("*") if p.is_file()}
    require(report_files - {manifest.resolve()} <= checked, "Seal omits analysis report files")
    required = [
        repo / "configs/round5_analysis_v1.json",
        *[
            repo / "scripts" / name
            for name in [
                *BRANCH_SCRIPTS.values(),
                "verify_round5_analysis.py",
                "replay_round5_analysis.py",
                "verify_round5_bundle.py",
            ]
        ],
        *[
            root / name
            for name in [
                "REPORT.md",
                "artifact.json",
                "report.html",
                "portable_delivery.json",
                "independent_QA.json",
                "replay_QA.json",
                "report_review.json",
                "headline_metrics.csv",
                "REPRODUCE.md",
                "QA.md",
                "models/protocol_lock.json",
                "readout/QA.json",
            ]
        ],
        *[root / name / "manifest.json" for name in BRANCH_SCRIPTS],
    ]
    require(
        all(p.resolve() in checked for p in required),
        "Seal omits required protocol, artifact, code or review",
    )
    return len(checked)


def passed_checks(report, name):
    require(
        report.get("status") == "PASS" and bool(report.get("checks")),
        f"{name} has no passing check record",
    )
    require(
        all(item.get("status") == "PASS" for item in report["checks"]),
        f"{name} contains failed or incomplete checks",
    )


def verify_report_review(repo, root):
    review = read_json(root / "report_review.json")
    passed_checks(review, "Report review")
    require(review["report_sha256"] == sha(root / "REPORT.md"), "Report review is stale")
    require(review["artifact_sha256"] == sha(root / "artifact.json"), "Artifact review is stale")
    require(review["model_refits"] == 0, "Report review unexpectedly refit models")
    hashes(repo, review["sources"])


def verify_branches(repo, root):
    manifests = {}
    raw_references = 0
    for branch, script in BRANCH_SCRIPTS.items():
        manifest = read_json(root / branch / "manifest.json")
        manifests[branch] = manifest
        hashes(root / branch, manifest.get("outputs", manifest.get("output_hashes")))
        code_key = "script_sha256" if branch in {"audit", "context"} else "code_sha256"
        hashes(repo, {"scripts/" + script: manifest[code_key]})
        if branch == "audit":
            hashes(
                repo,
                {
                    "scripts/round3_data_audit.py": manifest["helper_sha256"],
                    "uv.lock": manifest["lock_sha256"],
                },
            )
            hashes(repo, {v["repo_path"]: v["sha256"] for v in manifest["input_files"].values()})
        elif branch == "models":
            hashes(repo, manifest["inputs"])
        elif branch == "readout":
            hashes(root / "models", manifest["inputs"])
            hashes(
                repo,
                {
                    "reports/round4_analysis/models/cv_comparisons.csv": manifest[
                        "prior_comparisons_sha256"
                    ]
                },
            )
        elif branch == "recovered":
            hashes(root, {manifest["input"]: manifest["input_sha256"]})
        else:
            repo_inputs = {}
            for item in manifest["inputs"]:
                require(item["root"] in {"repo", "sme_raw"}, "Unknown context input root")
                if item["root"] == "repo":
                    require(item["path"] not in repo_inputs, "Duplicate context repository input")
                    repo_inputs[item["path"]] = item["sha256"]
                else:
                    raw_references += 1
            hashes(repo, repo_inputs)
    return manifests, raw_references


def adoption_from_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        selected = [
            r
            for r in csv.DictReader(stream)
            if r["comparison"] == "primary" and r["seed"] == "13" and r["target"] == "profile_clr"
        ]
    groups = {
        name: [r for r in selected if r["sample"] == name] for name in ["core", "stable_code"]
    }
    expected = {
        (str(y), regime) for y in [2023, 2024] for regime in ["within_region", "heldout_regions"]
    }
    for name, rows in groups.items():
        require(
            len(rows) == 4 and {(r["year"], r["regime"]) for r in rows} == expected,
            f"Incomplete adoption evidence: {name}",
        )
        require(
            all(
                math.isfinite(float(r[k]))
                for r in rows
                for k in ["relative_mse_reduction", "ci_low", "ci_high"]
            ),
            "Nonfinite adoption evidence",
        )
    return all(
        float(r["relative_mse_reduction"]) >= 0.01 and float(r["ci_low"]) > 0
        for r in groups["core"]
    ) and all(float(r["relative_mse_reduction"]) >= 0 for r in groups["stable_code"])


def verify_independent_branch_receipts(root, qa):
    for branch, section, key in [
        ("audit", "data_audit", "audit_manifest_sha256"),
        ("recovered", "recovered_description", "manifest_sha256"),
        ("readout", "readout", "manifest_sha256"),
    ]:
        require(
            qa[section][key] == sha(root / branch / "manifest.json"),
            f"Independent {branch} review is stale",
        )


def verify_model_review(repo, root, manifests):
    qa = read_json(root / "independent_QA.json")
    passed_checks(qa, "Independent QA")
    verify_independent_branch_receipts(root, qa)
    hashes(repo, {"scripts/verify_round5_analysis.py": qa["verifier_sha256"]})
    require(
        qa["model_manifest_sha256"] == sha(root / "models/manifest.json"),
        "Independent model review is stale",
    )
    hashes(repo, qa["independent_data_inputs"])
    protocol = repo / "configs/round5_analysis_v1.json"
    cfg, lock = read_json(protocol), read_json(root / "models/protocol_lock.json")
    require(
        lock["protocol_sha256"] == qa["protocol_sha256"] == sha(protocol),
        "Protocol changed after fit or review",
    )
    require(
        manifests["models"]["inputs"]["configs/round5_analysis_v1.json"] == sha(protocol),
        "Model refers to another protocol",
    )
    require(lock["source_base_commit"] == cfg["base_commit"], "Base commit disagrees")
    require(
        lock["annual_features_sha256"] == sha(root / "audit/annual_features.csv"),
        "Feature audit changed after model lock",
    )
    require(
        lock["model_code_sha256"]
        == manifests["models"]["code_sha256"]
        == qa["production_code_sha256"]
        == sha(repo / "scripts/round5_model_analysis.py"),
        "Model code changed after lock or review",
    )
    adopted = adoption_from_csv(root / "models/cv_comparisons.csv")
    readout = read_json(root / "readout/QA.json")
    require(
        readout["status"] == "PASS"
        and readout["adoption_passed"] is adopted
        and qa["independent_adoption_criteria_met"] is adopted,
        "Recomputed adoption disagrees with independent review or readout",
    )
    require(
        readout["postfit_contribution_sums_match_state_update"] is True,
        "Loss contribution reconciliation did not pass",
    )
    require(
        readout["excluded_regions_recovery_zero_within_1e12"] is True,
        "Excluded-region invariance did not pass",
    )
    return adopted


def verify_replay(repo, root):
    replay = read_json(root / "replay_QA.json")
    require(replay["status"] == "PASS", "Saved-source replay failed")
    require(
        replay["raw_tree_metadata_unchanged"] is True
        and replay["canonical_machine_files_and_code_unchanged"] is True,
        "Replay modified frozen inputs or results",
    )
    require(
        replay["network_requested"] is False and replay["llm_calls_requested"] is False,
        "Saved-source replay requested external computation",
    )
    require(
        len(replay["stages"]) == 6
        and {s["stage"] for s in replay["stages"]} == {*BRANCH_SCRIPTS, "independent"}
        and all(s["returncode"] == 0 for s in replay["stages"]),
        "Incomplete replay stages",
    )
    require(
        {c["branch"] for c in replay["comparisons"]} == set(BRANCH_SCRIPTS),
        "Incomplete branch comparisons",
    )
    hashes(repo, {"scripts/replay_round5_analysis.py": replay["code_sha256"]})
    hashes(repo / "scripts", replay["script_sha256"])
    hashes(
        root,
        {
            branch + "/manifest.json": digest
            for branch, digest in replay["source_manifests_sha256"].items()
        },
    )
    require(
        set(replay["source_manifests_sha256"]) == set(BRANCH_SCRIPTS),
        "Replay omits branch provenance",
    )


def markdown_links(repo, root, text):
    checked = 0
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\s]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        inside(repo, root, unquote(parsed.path))
        checked += 1
    return checked


def verify_portable(repo, root):
    receipt = read_json(root / "portable_delivery.json")
    require(
        receipt["ok"] is True and receipt["verification"]["ok"] is True,
        "Portable build or verification failed",
    )
    require(
        receipt["artifact"]["sha256"] == sha(root / "artifact.json"),
        "Portable artifact receipt is stale",
    )
    require(
        receipt["output_sha256"] == sha(root / "report.html")
        and receipt["output_bytes"] == (root / "report.html").stat().st_size,
        "Portable HTML receipt is stale",
    )
    require(receipt["verification"]["sourceDialog"] == "passed", "Source dialog unverified")
    artifact = read_json(root / "artifact.json")
    manifest, snapshot = artifact["manifest"], artifact["snapshot"]
    sources = {s["id"]: s for s in manifest["sources"]}
    require(len(sources) == len(manifest["sources"]), "Duplicate artifact source IDs")
    require(artifact["sources"] == manifest["sources"], "Artifact source copies disagree")
    for source in sources.values():
        inside(repo, repo, source["path"])
        for path in source["query"]["tables_used"]:
            inside(repo, repo, path)
        for name in ["computation_path", "reproduction_document"]:
            inside(repo, repo, source["query"][name])
    for kind in ["charts", "tables"]:
        items = manifest[kind]
        require(len({item["id"] for item in items}) == len(items), f"Duplicate {kind} IDs")
        for item in items:
            require(item["sourceId"] in sources, "Unknown chart/table source")
            require(item["dataset"] in snapshot["datasets"], "Unknown chart/table dataset")
        require(
            receipt["verification"]["counts"][kind] == len(items),
            f"Portable {kind} count disagrees",
        )
    require(
        receipt["verification"]["counts"]["blocks"] == len(manifest["blocks"]),
        "Portable block count disagrees",
    )
    links = markdown_links(repo, root, (root / "REPORT.md").read_text(encoding="utf-8"))
    for block in manifest["blocks"]:
        if block.get("sourceId"):
            require(block["sourceId"] in sources, "Unknown block source")
        if block["type"] == "markdown":
            links += markdown_links(repo, root, block["body"])
    return {
        "charts": len(manifest["charts"]),
        "tables": len(manifest["tables"]),
        "sources": len(sources),
        "local_markdown_links_checked": links,
    }


def verify(repo):
    repo = repo.resolve()
    root = repo / "reports/round5_analysis"
    count = verify_seal(repo, root)
    manifests, raw_references = verify_branches(repo, root)
    adopted = verify_model_review(repo, root, manifests)
    verify_replay(repo, root)
    verify_report_review(repo, root)
    portable = verify_portable(repo, root)
    return {
        "status": "PASS",
        "hashed_files": count,
        "model_refits": 0,
        "all_report_files_sealed": True,
        "independent_qa_current": True,
        "report_review_current": True,
        "replay_receipt_current": True,
        "protocol_hash_verified": True,
        "adoption_passed": adopted,
        "raw_source_hashes_checked": False,
        "external_raw_references": raw_references,
        "remote_urls_checked": False,
        "browser_visual_qa_repeated": False,
        **portable,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().repo)))
