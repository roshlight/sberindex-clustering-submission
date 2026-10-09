"""Offline replay of all five round-six branches into a new external directory.

Use the enrichment Python environment. Saved Word conversions and reviewed recipes
are inputs: this command does not repeat conversion, downloads, LLM calls or model fitting.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from verify_round6_delivery import (
    BRANCH_SCRIPTS,
    REPORT,
    branch_hashes,
    read_json,
    require,
    resolve_search_raw,
    sha,
    verify_hashes,
    verify_raw,
    verify_seal,
)


def tree_state(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def validate_output(repo, raw, prior, output, search):
    require(not output.exists(), "Choose a new output directory; replay never overwrites")
    for source in [repo, raw, prior, search]:
        require(source.is_dir(), f"Missing source directory: {source}")
        require(
            not output.is_relative_to(source) and not source.is_relative_to(output),
            "Replay output must be outside repository and source directories",
        )


def stage_arguments(repo, raw, prior, output, search):
    scripts, report = repo / "enrichment/scripts", repo / REPORT
    return {
        "economy": [
            scripts / "prepare_round6_economy.py",
            "--repo",
            repo,
            "--raw",
            raw / "economy",
            "--out",
            output / "economy",
        ],
        "sme": [scripts / "audit_round6_sme.py", "--raw", raw / "sme", "--out", output / "sme"],
        "geography": [
            scripts / "audit_round6_geography.py",
            "--repo",
            repo,
            "--raw",
            raw / "geography",
            "--prior-raw",
            prior,
            "--out",
            output / "geography",
        ],
        "cases": [
            scripts / "collect_round6_cases.py",
            "--raw-root",
            raw / "cases",
            "--derive-only",
            "--recipe",
            report / "cases/evidence_recipe.json",
            "--out",
            output / "cases",
        ],
        "search": [
            scripts / "summarize_round5_search.py",
            "--raw",
            search,
            "--out",
            output / "search",
        ],
    }


def replay(repo, raw, prior_geography_raw, output, search_raw=None):
    repo, raw, prior, output = (p.resolve() for p in [repo, raw, prior_geography_raw, output])
    search = resolve_search_raw(raw, search_raw)
    validate_output(repo, raw, prior, output, search)
    expected = branch_hashes(repo)
    source_checks = verify_raw(repo, raw, prior, search)
    report = repo / REPORT
    seal = report / "delivery_manifest.json"
    seal_checked = seal.exists()
    if seal_checked:
        verify_seal(repo, seal)
    sources = {name: raw / name for name in ["economy", "sme", "geography", "cases"]}
    sources.update(prior_geography=prior, search=search)
    before = {name: tree_state(path) for name, path in sources.items()}
    code_paths = set(BRANCH_SCRIPTS.values()) | {
        "enrichment/scripts/replay_round6.py",
        "enrichment/scripts/verify_round6_delivery.py",
        "enrichment/uv.lock",
        "enrichment/pyproject.toml",
        "enrichment/scripts/collect_round5_cases.py",
    }
    code_paths.update(read_json(report / "sme/manifest.json")["dependency_hashes"])
    code = {name: sha(repo / name) for name in sorted(code_paths)}
    # Report prose may still be edited separately; guard every machine output and recipe.
    published = {
        f"{REPORT}/{branch}/{name}": digest
        for branch, files in expected.items()
        for name, digest in files.items()
    }
    for name in ["cases/evidence_recipe.json", "cases/raw_allowlist.json"]:
        published[f"{REPORT}/{name}"] = sha(report / name)
    output.mkdir(parents=True)
    stages, comparisons = [], []
    started_at = datetime.now(UTC).isoformat()

    def write(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    try:
        for name, arguments in stage_arguments(repo, raw, prior, output, search).items():
            result = subprocess.run(
                [sys.executable, *map(str, arguments)], cwd=repo, capture_output=True, check=False
            )
            (output / f"{name}.stdout.log").write_bytes(result.stdout)
            (output / f"{name}.stderr.log").write_bytes(result.stderr)
            stages.append(dict(stage=name, returncode=result.returncode))
            write("stages.json", stages)
            require(result.returncode == 0, f"Replay failed: {name}; inspect output logs")
            verify_hashes(output / name, expected[name])
            comparisons.append(dict(branch=name, files=len(expected[name]), status="PASS"))
        for name, source in sources.items():
            after = tree_state(source)
            changed = sorted(
                key
                for key in before[name].keys() | after.keys()
                if before[name].get(key) != after.get(key)
            )
            require(not changed, f"Source tree changed: {name}: {changed[:10]}")
        verify_hashes(repo, code)
        verify_hashes(repo, published)
        require(branch_hashes(repo) == expected, "Published branch output changed during replay")
        require(verify_raw(repo, raw, prior, search) == source_checks, "Source inventory changed")
        supplemental = {
            branch: sorted(
                p.relative_to(report / branch).as_posix()
                for p in (report / branch).rglob("*")
                if p.is_file() and p.relative_to(report / branch).as_posix() not in expected[branch]
            )
            for branch in expected
        }
        record = dict(
            status="PASS",
            scope="Saved-source offline replay in existing locked environment",
            clean_environment_created=False,
            network_requested=False,
            models_trained=False,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            runtime={"python": platform.python_version(), "platform": platform.platform()},
            word_conversion_repeated=False,
            manual_evidence_review_repeated=False,
            source_trees_unchanged=True,
            delivery_seal_checked_before_replay=seal_checked,
            raw_source_files_checked=source_checks,
            comparisons=comparisons,
            machine_files_matched=sum(len(value) for value in expected.values()),
            machine_hashes=expected,
            code_hashes=code,
            not_regenerated=supplemental,
            stages=stages,
        )
        write("QA.json", record)
    except Exception as exc:
        write("QA.json", dict(status="FAIL", error=str(exc), stages=stages))
        raise
    print(
        json.dumps({key: record[key] for key in ["status", "machine_files_matched", "comparisons"]})
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prior-geography-raw", type=Path, required=True)
    parser.add_argument("--search-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.repo, args.raw, args.prior_geography_raw, args.output, args.search_raw)
