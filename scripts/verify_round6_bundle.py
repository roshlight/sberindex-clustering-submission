"""Read-only integrity and review checks for the sealed sixth analytical round."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit

ROOT = "reports/round6_analysis"
BRANCH_SCRIPTS = {name: f"analyze_round6_{name}.py" for name in ("admission", "regions", "context")}
REQUIRED_CODE = [
    *BRANCH_SCRIPTS.values(),
    "verify_round6_analysis.py",
    "replay_round6_analysis.py",
    "verify_round6_bundle.py",
    "build_round6_report.py",
    "verify_round6_report_claims.py",
    "build_round2_report.py",
    "build_round2_portable.mjs",
]
REQUIRED_REPORTS = [
    "REPORT.md",
    "artifact.json",
    "report.html",
    "REPRODUCE.md",
    "QA.md",
    "independent_QA.json",
    "replay_QA.json",
    "report_review.json",
    "report_claims_QA.json",
    "portable_delivery.json",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(value):
    require(isinstance(value, str) and bool(value), "Invalid relative path")
    windows = PureWindowsPath(value)
    portable = value.replace("\\", "/")
    require(
        not windows.drive and not windows.root and ":" not in portable,
        f"Absolute or invalid path: {value}",
    )
    require(".." not in PurePosixPath(portable).parts, f"Parent traversal: {value}")
    return portable


def inside(base, relative):
    path = (base / relative_path(relative)).resolve()
    require(path.is_relative_to(base.resolve()), f"Path escapes root: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def hashes(base, entries):
    require(isinstance(entries, dict) and bool(entries), "Empty or invalid hash map")
    checked = set()
    for name, digest in entries.items():
        path = inside(base, name)
        require(path not in checked, f"Duplicate normalized path: {name}")
        checked.add(path)
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"Invalid SHA256: {name}",
        )
        require(sha(path) == digest, f"Hash changed: {name}")
    return checked


def verify_artifacts(repo, artifacts):
    require(isinstance(artifacts, list) and bool(artifacts), "Empty artifact seal")
    checked = set()
    for item in artifacts:
        require(isinstance(item, dict), "Invalid artifact entry")
        path = inside(repo, item["path"])
        require(path not in checked, "Duplicate sealed path")
        checked |= hashes(repo, {item["path"]: item["sha256"]})
        require(type(item["bytes"]) is int and item["bytes"] >= 0, "Invalid byte length")
        require(path.stat().st_size == item["bytes"], f"Byte length changed: {item['path']}")
    return checked


def verify_seal(repo, root):
    path = root / "delivery_manifest.json"
    sealed = verify_artifacts(repo, read_json(path)["artifacts"])
    all_reports = {p.resolve() for p in root.rglob("*") if p.is_file()}
    require(all_reports - {path.resolve()} <= sealed, "Seal omits analysis report files")
    required = [repo / "configs/round6_analysis_v1.json", repo / "uv.lock", repo / "pyproject.toml"]
    required += [repo / "scripts" / name for name in REQUIRED_CODE]
    required += [root / name for name in REQUIRED_REPORTS]
    required += [root / name / "manifest.json" for name in BRANCH_SCRIPTS]
    require(all(p.resolve() in sealed for p in required), "Seal omits required code or report")
    return sealed


def verify_branches(repo, root, sealed):
    raw_references = 0
    for branch, script in BRANCH_SCRIPTS.items():
        folder = root / branch
        manifest = read_json(folder / "manifest.json")
        checked = hashes(folder, manifest["outputs"])
        checked |= hashes(repo, {f"scripts/{script}": manifest["script_sha256"]})
        repo_inputs = {}
        for name, digest in manifest["inputs"].items():
            if branch == "admission":
                label, relative = name.split("/", 1)
                require(label in {"repo", "raw"}, "Unknown admission input root")
                relative_path(relative)
                if label == "raw":
                    raw_references += 1
                    continue
                name = relative
            repo_inputs[name] = digest
        checked |= hashes(repo, repo_inputs)
        require(checked <= sealed, f"Seal omits {branch} inputs, outputs or script")
    return raw_references


def passed_checks(qa, label):
    require(qa.get("status") == "PASS" and bool(qa.get("checks")), f"{label} did not pass")
    require(
        all(r.get("status") == "PASS" for r in qa["checks"]),
        f"{label} contains failed or incomplete checks",
    )


def verify_independent(repo, root):
    qa = read_json(root / "independent_QA.json")
    passed_checks(qa, "Independent QA")
    require(qa["model_refits"] == 0, "Unexpected model fitting")
    hashes(repo, {"scripts/verify_round6_analysis.py": qa["verifier_sha256"]})
    require(
        set(qa["branch_manifests_sha256"]) == set(BRANCH_SCRIPTS),
        "Independent review omits branch provenance",
    )
    for branch, digest in qa["branch_manifests_sha256"].items():
        require(
            sha(root / branch / "manifest.json") == digest,
            f"Independent branch review is stale: {branch}",
        )
    hashes(repo, qa["inputs"])
    hashes(root, qa["artifact_sha256"])
    return qa


def verify_replay(repo, root):
    replay = read_json(root / "replay_QA.json")
    require(replay.get("status") == "PASS", "Replay failed")
    for name in [
        "raw_tree_metadata_unchanged",
        "canonical_machine_files_and_code_unchanged",
        "all_manifest_declared_inputs_hash_checked",
    ]:
        require(replay[name] is True, f"Replay guard failed: {name}")
    for name in ["network_requested", "llm_calls_requested", "clean_environment_created"]:
        require(replay[name] is False, f"Unexpected replay scope: {name}")
    require(
        replay["models_trained"] == 0 and replay["new_clusters"] == 0,
        "Unexpected training during replay",
    )
    require(
        len(replay["stages"]) == 4
        and {r["stage"] for r in replay["stages"]} == {*BRANCH_SCRIPTS, "independent"}
        and all(r["returncode"] == 0 for r in replay["stages"]),
        "Incomplete replay stages",
    )
    require(
        {r["branch"] for r in replay["comparisons"]} == set(BRANCH_SCRIPTS),
        "Incomplete replay comparisons",
    )
    hashes(repo, {"scripts/replay_round6_analysis.py": replay["code_sha256"]})
    hashes(repo / "scripts", replay["script_sha256"])
    require(
        set(replay["source_manifests_sha256"]) == set(BRANCH_SCRIPTS),
        "Replay omits branch provenance",
    )
    hashes(root, {f"{b}/manifest.json": h for b, h in replay["source_manifests_sha256"].items()})


def verify_report_review(repo, root):
    review = read_json(root / "report_review.json")
    passed_checks(review, "Report review")
    require(review["report_sha256"] == sha(root / "REPORT.md"), "Report review is stale")
    require(review["artifact_sha256"] == sha(root / "artifact.json"), "Artifact review is stale")
    require(review.get("model_refits", 0) == 0, "Report review unexpectedly fitted models")
    return hashes(repo, review["sources"])


def verify_report_claims(repo, root):
    claims = read_json(root / "report_claims_QA.json")
    passed_checks(claims, "Report claims")
    require(claims["model_refits"] == 0, "Claims review unexpectedly fitted models")
    checked = hashes(repo, {"scripts/verify_round6_report_claims.py": claims["verifier_sha256"]})
    for filename, key in [
        ("REPORT.md", "report_sha256"),
        ("artifact.json", "artifact_sha256"),
        ("claim_values.json", "claim_values_sha256"),
    ]:
        require(sha(root / filename) == claims[key], f"Report claims receipt is stale: {filename}")
    for name, digest in claims["source_sha256"].items():
        if name.startswith("repo/"):
            checked |= hashes(repo, {name.removeprefix("repo/"): digest})
        else:
            checked |= hashes(root, {name: digest})
    return checked


def local_links(repo, base, text):
    checked = 0
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\s]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (base / unquote(parsed.path)).resolve()
        require(
            path.is_relative_to(repo.resolve()) and path.is_file(),
            f"Broken or nonportable local link: {target}",
        )
        checked += 1
    return checked


def verify_portable(repo, root, sealed):
    receipt = read_json(root / "portable_delivery.json")
    require(
        receipt["ok"] is True and receipt["verification"]["ok"] is True,
        "Portable build/verification failed",
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
    artifact = read_json(root / "artifact.json")
    manifest = artifact["manifest"]
    sources = manifest["sources"]
    require(len({s["id"] for s in sources}) == len(sources), "Duplicate source IDs")
    require(artifact["sources"] == sources, "Artifact source copies disagree")
    paths = set()
    for source in sources:
        paths.add(inside(repo, source["path"]))
        for name in source["query"]["tables_used"]:
            paths.add(inside(repo, name))
        for name in ["computation_path", "reproduction_document"]:
            paths.add(inside(repo, source["query"][name]))
    require(paths <= sealed, "Seal omits artifact source dependencies")
    links = local_links(repo, root, (root / "REPORT.md").read_text(encoding="utf-8"))
    for block in manifest["blocks"]:
        if block["type"] == "markdown":
            links += local_links(repo, root, block["body"])
    return {"sources": len(sources), "local_markdown_links_checked": links}


def verify(repo):
    repo = repo.resolve()
    root = repo / ROOT
    sealed = verify_seal(repo, root)
    raw_references = verify_branches(repo, root, sealed)
    independent = verify_independent(repo, root)
    require(hashes(repo, independent["inputs"]) <= sealed, "Seal omits independent source inputs")
    verify_replay(repo, root)
    require(verify_report_review(repo, root) <= sealed, "Seal omits report review sources")
    require(verify_report_claims(repo, root) <= sealed, "Seal omits report claims sources")
    portable = verify_portable(repo, root, sealed)
    return {
        "status": "PASS",
        "hashed_files": len(sealed),
        "model_refits": 0,
        "all_report_files_sealed": True,
        "independent_qa_current": True,
        "report_review_current": True,
        "report_claims_current": True,
        "replay_receipt_current": True,
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
