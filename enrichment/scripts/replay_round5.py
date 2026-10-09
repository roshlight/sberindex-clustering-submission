"""Replay fifth-round machine outputs from saved sources into a new output directory.

Manual evidence recipes and reviews remain inputs; this does not repeat visual review,
download sources, extract prior SME ZIPs, fit models or create a clean environment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from verify_round5_delivery import (
    BRANCH_SCRIPTS,
    REPORT,
    branch_hashes,
    read_json,
    require,
    sha,
    verify_hashes,
    verify_seal,
)


def tree_state(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def validate_output(repo, raw, prior_sme_raw, output):
    require(not output.exists(), "Choose a new output directory; replay never overwrites")
    for source in [repo, raw, prior_sme_raw]:
        require(source.is_dir(), "A required source directory does not exist")
        require(
            not output.is_relative_to(source) and not source.is_relative_to(output),
            "Replay output must be outside repository and source directories",
        )


def replay(repo, raw, prior_sme_raw, output):
    repo, raw, prior_sme_raw, output = (p.resolve() for p in [repo, raw, prior_sme_raw, output])
    validate_output(repo, raw, prior_sme_raw, output)
    expected = branch_hashes(repo)
    report, scripts = repo / REPORT, repo / "enrichment/scripts"
    seal = report / "delivery_manifest.json"
    seal_checked = seal.exists()
    if seal_checked:
        verify_seal(repo, seal)
    # Fingerprints detect writes by the selected derivation commands, not network activity.
    sources = {
        name: raw / name for name in ["economy", "sme", "cases", "geography", "server-complete"]
    }
    sources["prior_sme"] = prior_sme_raw
    before = {name: tree_state(path) for name, path in sources.items()}
    code = {name: sha(repo / name) for name in BRANCH_SCRIPTS.values()}
    code["enrichment/uv.lock"] = sha(repo / "enrichment/uv.lock")
    code["enrichment/scripts/replay_round5.py"] = sha(Path(__file__))
    code["enrichment/scripts/verify_round5_delivery.py"] = sha(
        scripts / "verify_round5_delivery.py"
    )
    output.mkdir(parents=True)
    stages, comparisons = [], []

    def write(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def run(name, arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        (output / f"{name}.stdout.log").write_bytes(result.stdout)
        (output / f"{name}.stderr.log").write_bytes(result.stderr)
        stages.append({"stage": name, "returncode": result.returncode})
        write("stages.json", stages)
        require(result.returncode == 0, f"Replay failed: {name}; inspect output logs")
        verify_hashes(output / name, expected[name])
        comparisons.append({"branch": name, "files": len(expected[name]), "status": "PASS"})

    try:
        run(
            "economy",
            [
                "-m",
                "sber_enrichment.round5_economy",
                "--repo",
                repo,
                "--output",
                output / "economy",
                "--cache",
                raw / "economy/cache",
                "--cache-only",
            ],
        )
        run(
            "sme",
            [
                scripts / "audit_round5_sme.py",
                "--raw",
                raw / "sme",
                "--prior-raw",
                prior_sme_raw,
                "--out",
                output / "sme",
            ],
        )
        # SME's delivery manifest includes README; its replay manifest contains machine files only.
        replay_sme = read_json(output / "sme/artifact_manifest.json")
        require(
            {x["path"]: x["sha256"] for x in replay_sme} == expected["sme"],
            "SME replay manifest does not match the machine output contract",
        )
        run(
            "cases",
            [
                scripts / "collect_round5_cases.py",
                "--raw-root",
                raw / "cases",
                "--derive-only",
                "--recipe",
                report / "cases/evidence_recipe.json",
                "--out",
                output / "cases",
            ],
        )
        run(
            "geography",
            [
                scripts / "prepare_round5_geography.py",
                "--repo",
                repo,
                "--raw",
                raw / "geography",
                "--prior-raw",
                raw / "geography/reused",
                "--sme-raw",
                prior_sme_raw,
                "--out",
                output / "geography",
            ],
        )
        run(
            "search",
            [
                scripts / "summarize_round5_search.py",
                "--raw",
                raw / "server-complete",
                "--out",
                output / "search",
            ],
        )
        for name, path in sources.items():
            after = tree_state(path)
            changed = sorted(
                key
                for key in before[name].keys() | after.keys()
                if before[name].get(key) != after.get(key)
            )
            require(not changed, f"Source tree changed: {name}: {changed}")
        verify_hashes(repo, code)
        require(branch_hashes(repo) == expected, "Published branch output changed during replay")
        supplemental = {
            branch: sorted(
                p.relative_to(report / branch).as_posix()
                for p in (report / branch).rglob("*")
                if p.is_file() and p.relative_to(report / branch).as_posix() not in expected[branch]
            )
            for branch in expected
        }
        result = {
            "status": "PASS",
            "scope": "Saved-source replay in the existing locked environment",
            "clean_environment_created": False,
            "network_requested": False,
            "source_trees_unchanged": True,
            "models_trained": False,
            "delivery_seal_checked_before_replay": seal_checked,
            "comparisons": comparisons,
            "machine_files_matched": sum(len(x) for x in expected.values()),
            "machine_hashes": expected,
            "code_hashes": code,
            "not_regenerated": supplemental,
            "manual_evidence_review_repeated": False,
            "sme_manifest_comparison": (
                "Machine entries only; delivered README is a manual document"
            ),
            "stages": stages,
        }
        write("QA.json", result)
    except Exception as exc:
        write("QA.json", {"status": "FAIL", "error": str(exc), "stages": stages})
        raise
    print(
        json.dumps({key: result[key] for key in ["status", "machine_files_matched", "comparisons"]})
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prior-sme-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.repo, args.raw, args.prior_sme_raw, args.output)
