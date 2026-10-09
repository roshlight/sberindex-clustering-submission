"""Replay the five fourth-round evidence branches from delivered immutable raw sources."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(repo, raw, output):
    repo, raw, output = repo.resolve(), raw.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = repo / "reports/data_round4"
    scripts = repo / "enrichment/scripts"
    stages = []

    def run(name, arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=repo, capture_output=True, check=False
        )
        (output / f"{name}.stdout.log").write_bytes(result.stdout)
        (output / f"{name}.stderr.log").write_bytes(result.stderr)
        stages.append({"stage": name, "returncode": result.returncode})
        (output / "stages.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Replay failed: {name}; inspect logs in output directory")

    def compare(expected, actual, names):
        for name in names:
            if sha(expected / name) != sha(actual / name):
                raise ValueError(f"Replay bytes differ: {expected.name}/{name}")

    run(
        "economy_collect",
        [
            "-m",
            "sber_enrichment.round4_economy",
            "--repo",
            repo,
            "--output",
            output / "economy",
            "--cache",
            raw / "economy/cache",
            "--cache-only",
        ],
    )
    econ = json.loads((report / "economy/manifest.json").read_text("utf-8"))
    compare(report / "economy", output / "economy", econ["output_hashes"])
    run(
        "economy_independent",
        [
            scripts / "verify_round4_economy.py",
            "--repo",
            repo,
            "--cache",
            raw / "economy/cache",
            "--output",
            output / "economy_independent.json",
        ],
    )
    run(
        "geography",
        [
            scripts / "collect_round4_geography.py",
            "--raw-root",
            raw / "geography",
            "--output",
            report / "geography",
            "--verify-only",
        ],
    )
    run(
        "cases",
        [
            scripts / "collect_round4_cases.py",
            "--plan",
            report / "cases/fetch_plan.json",
            "--raw-dir",
            raw / "cases/raw",
            "--report-dir",
            output / "cases",
            "--derive-only",
            "--recipe",
            report / "cases/evidence_recipe.json",
        ],
    )
    case_manifest = json.loads((report / "cases/manifest.json").read_text("utf-8"))
    # This branch includes a manual evidence recipe; replay verifies it rather than redoing OCR.
    case_names = [x["path"] for x in case_manifest["files"]] + ["manifest.json"]
    if not case_names:
        raise ValueError("No case output hashes")
    compare(report / "cases", output / "cases", case_names)
    run("sme_extract", [scripts / "extract_round4_sme.py", "--raw", raw / "sme"])
    run(
        "sme_audit",
        [
            scripts / "audit_round4_sme.py",
            "--raw",
            raw / "sme",
            "--out",
            output / "sme",
        ],
    )
    sme = json.loads((report / "sme/manifest.json").read_text("utf-8"))
    compare(report / "sme", output / "sme", sme["outputs"])
    run(
        "sme_independent",
        [
            scripts / "verify_round4_sme.py",
            "--raw",
            raw / "sme",
            "--report",
            report / "sme",
            "--out",
            output / "sme_independent.json",
        ],
    )
    run(
        "search",
        [
            scripts / "summarize_round4_search.py",
            "--raw",
            raw / "server-complete",
            "--out",
            output / "search",
        ],
    )
    compare(
        report / "search",
        output / "search",
        [
            "server_summary.json",
            "server_document_inventory.csv",
        ],
    )
    result = {
        "status": "PASS",
        "clean_environment_created": False,
        "network_requests": 0,
        "models_trained": False,
        "scope": "Saved-source replay in the declared existing environment",
        "stages": stages,
    }
    (output / "QA.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.repo, args.raw, args.output)
