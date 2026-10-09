"""Rebuild three round-six branches and independently validate their fresh outputs."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

BRANCHES = ("admission", "regions", "context")
SCRIPTS = {name: f"analyze_round6_{name}.py" for name in BRANCHES}
SCRIPTS["independent"] = "verify_round6_analysis.py"
TOLERANCE = 1e-9
IDENTITY_COLUMNS = {"territory_id", "year", "region_code", "cluster", "cluster_id", "community"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inside(base, relative):
    require(isinstance(relative, str) and bool(relative), "Invalid manifest path")
    windows = PureWindowsPath(relative)
    portable = relative.replace("\\", "/")
    require(not windows.drive and not windows.root, "Absolute manifest path")
    require(
        ".." not in PurePosixPath(portable).parts and ":" not in portable, "Unsafe manifest path"
    )
    path = (base / portable).resolve()
    require(path.is_relative_to(base.resolve()), "Manifest file escapes allowed root")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def validate_output(repo, raw, out):
    require(not out.exists(), "Choose a new output directory; no overwrite is allowed")
    for source in [repo, raw]:
        require(source.is_dir(), "Input directory missing")
        require(
            not out.is_relative_to(source) and not source.is_relative_to(out),
            "Replay output must be outside repository and raw source directories",
        )


def tree_state(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def verify_branch(folder):
    manifest = read_json(folder / "manifest.json")
    names = manifest.get("outputs")
    require(isinstance(names, dict) and bool(names), "Missing branch outputs")
    seen = set()
    for name, digest in names.items():
        path = inside(folder, name)
        require(path not in seen, "Duplicate normalized output path")
        seen.add(path)
        require(sha(path) == digest, f"Branch output changed: {folder.name}/{name}")
    return manifest, names


def compare_frame(expected, actual):
    require(list(expected.columns) == list(actual.columns), "Replay table columns changed")
    require(expected.shape == actual.shape, "Replay table shape changed")
    worst = 0.0
    for column in expected:
        a, b = expected[column], actual[column]
        if (
            column not in IDENTITY_COLUMNS
            and pd.api.types.is_float_dtype(a.dtype)
            and pd.api.types.is_numeric_dtype(b.dtype)
        ):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            np.testing.assert_allclose(av, bv, atol=TOLERANCE, rtol=TOLERANCE, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                worst = max(worst, float(np.abs(av[finite] - bv[finite]).max()))
        else:
            pd.testing.assert_series_equal(a, b, check_exact=True)
    return worst


def compare_branch(expected, actual, numeric=False):
    old, files = verify_branch(expected)
    fresh, fresh_files = verify_branch(actual)
    require(files.keys() == fresh_files.keys(), "Machine file set changed")
    if not numeric:
        for name in [*files, "manifest.json"]:
            require(
                sha(expected / name) == sha(actual / name),
                f"Byte replay mismatch: {expected.name}/{name}",
            )
        return {"branch": expected.name, "method": "byte_exact", "files": len(files) + 1}
    tables = {}
    for name in files:
        if name.endswith(".csv"):
            tables[name] = compare_frame(pd.read_csv(expected / name), pd.read_csv(actual / name))
        else:
            require(
                sha(expected / name) == sha(actual / name),
                f"Non-numeric output changed: {expected.name}/{name}",
            )
    require(
        {k: v for k, v in old.items() if k != "outputs"}
        == {k: v for k, v in fresh.items() if k != "outputs"},
        "Region provenance, protocol or code changed",
    )
    return {
        "branch": expected.name,
        "method": "numeric_1e-9_and_exact_structure",
        "files": len(files),
        "tables_max_abs": tables,
        "excluded_byte_equality": ["manifest.json"],
        "exclusion_reason": "Fresh output hashes checked; CSV floats allow 1e-9 tolerance",
    }


def replay(repo, raw, out):
    repo, raw, out = (p.resolve() for p in [repo, raw, out])
    validate_output(repo, raw, out)
    canonical = repo / "reports/round6_analysis"
    frozen = {}
    for branch in BRANCHES:
        manifest, names = verify_branch(canonical / branch)
        for name in [*names, "manifest.json"]:
            path = canonical / branch / name
            frozen[path] = sha(path)
        for name, digest in manifest["inputs"].items():
            if branch == "admission":
                label, relative = name.split("/", 1)
                require(label in {"repo", "raw"}, "Unknown admission source root")
                path = inside(repo if label == "repo" else raw, relative)
            else:
                path = inside(repo, name)
            require(sha(path) == digest, f"Input changed: {name}")
            frozen[path] = digest
    for name in SCRIPTS.values():
        path = repo / "scripts" / name
        frozen[path] = sha(path)
    frozen[Path(__file__).resolve()] = sha(Path(__file__))
    raw_before = tree_state(raw)
    out.mkdir(parents=True)
    stages, comparisons = [], []

    def stage(name, args):
        result = subprocess.run(
            [sys.executable, *map(str, args)], cwd=repo, capture_output=True, check=False
        )
        (out / f"{name}.stdout.log").write_bytes(result.stdout)
        (out / f"{name}.stderr.log").write_bytes(result.stderr)
        stages.append({"stage": name, "returncode": result.returncode})
        write_json(out / "stages.json", stages)
        require(result.returncode == 0, f"Stage failed: {name}; inspect saved logs")
        print(json.dumps(stages[-1]), flush=True)

    try:
        for branch in BRANCHES:
            args = [repo / "scripts" / SCRIPTS[branch], "--repo", repo, "--out", out / branch]
            if branch == "admission":
                args += ["--raw", raw]
            stage(branch, args)
            comparisons.append(
                compare_branch(canonical / branch, out / branch, numeric=branch == "regions")
            )
        stage(
            "independent",
            [
                repo / "scripts" / SCRIPTS["independent"],
                "--repo",
                repo,
                "--root",
                out,
                "--economy-raw",
                raw,
                "--out",
                out / "independent_QA.json",
            ],
        )
        independent = read_json(out / "independent_QA.json")
        require(independent["status"] == "PASS", "Independent fresh-root validation failed")
        require(tree_state(raw) == raw_before, "Raw source tree changed during replay")
        require(
            all(sha(p) == d for p, d in frozen.items()),
            "Frozen source, canonical output or code changed during replay",
        )
        qa = {
            "status": "PASS",
            "scope": "Saved-source replay in existing locked environment",
            "clean_environment_created": False,
            "network_requested": False,
            "llm_calls_requested": False,
            "models_trained": 0,
            "new_clusters": 0,
            "numeric_atol": TOLERANCE,
            "numeric_rtol": TOLERANCE,
            "raw_tree_metadata_unchanged": True,
            "canonical_machine_files_and_code_unchanged": True,
            "all_manifest_declared_inputs_hash_checked": True,
            "independent_validation": "PASS on fresh root",
            "report_generation_replayed": False,
            "visual_review_repeated": False,
            "comparisons": comparisons,
            "stages": stages,
            "independent_QA_sha256": sha(out / "independent_QA.json"),
            "code_sha256": sha(Path(__file__)),
            "source_manifests_sha256": {b: sha(canonical / b / "manifest.json") for b in BRANCHES},
            "script_sha256": {name: sha(repo / "scripts" / name) for name in SCRIPTS.values()},
        }
        write_json(out / "QA.json", qa)
    except Exception as exc:
        write_json(
            out / "QA.json",
            {"status": "FAIL", "error": str(exc), "stages": stages, "comparisons": comparisons},
        )
        raise
    print(json.dumps({"status": "PASS", "stages": len(stages)}))
    return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--economy-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.repo, args.economy_raw, args.out)
