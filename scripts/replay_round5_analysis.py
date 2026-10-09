"""Rebuild five analysis branches in a new directory and verify saved-source agreement."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import duckdb
import numpy as np
import pandas as pd

BRANCHES = ["audit", "models", "readout", "recovered", "context"]
SCRIPTS = {
    "audit": "round5_data_audit.py",
    "models": "round5_model_analysis.py",
    "readout": "round5_model_readout.py",
    "recovered": "round5_recovered_profiles.py",
    "context": "round5_context_analysis.py",
    "independent": "verify_round5_analysis.py",
}
TOLERANCE = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    require(path.is_relative_to(base.resolve()), "Manifest file escapes branch")
    require(path.is_file(), f"Missing manifest file: {relative}")
    return path


def validate_output(repo, sme_raw, output):
    require(not output.exists(), "Choose a new output directory; no overwrite is allowed")
    for source in [repo, sme_raw]:
        require(source.is_dir(), "Required input directory does not exist")
        require(
            not output.is_relative_to(source) and not source.is_relative_to(output),
            "Replay output must be outside repository and raw source directories",
        )


def tree_state(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def verify_branch_manifest(folder):
    manifest = read_json(folder / "manifest.json")
    outputs = manifest.get("output_hashes", manifest.get("outputs"))
    require(isinstance(outputs, dict) and bool(outputs), "Missing machine output hashes")
    normalized = set()
    for name, digest in outputs.items():
        path = inside(folder, name)
        require(path not in normalized, "Duplicate normalized output path")
        normalized.add(path)
        require(sha(path) == digest, f"Branch output changed: {folder.name}/{name}")
    return manifest, outputs


def compare_frame(expected, actual):
    require(list(expected.columns) == list(actual.columns), "Replay table columns changed")
    require(expected.shape == actual.shape, "Replay table shape changed")
    worst = 0.0
    for column in expected:
        a, b = expected[column], actual[column]
        if pd.api.types.is_float_dtype(a.dtype) and pd.api.types.is_numeric_dtype(b.dtype):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            np.testing.assert_allclose(av, bv, atol=TOLERANCE, rtol=TOLERANCE, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                worst = max(worst, float(np.abs(av[finite] - bv[finite]).max()))
        else:
            pd.testing.assert_series_equal(a, b, check_exact=True)
    return worst


def compare_json(expected, actual):
    """Numerical preprocessing can vary within tolerance; all structure remains exact."""
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and expected.keys() == actual.keys(),
            "Replay JSON keys changed",
        )
        for name in expected:
            compare_json(expected[name], actual[name])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), "JSON list changed")
        for a, b in zip(expected, actual, strict=True):
            compare_json(a, b)
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float)) and not isinstance(actual, bool),
            "JSON numeric type changed",
        )
        np.testing.assert_allclose(expected, actual, atol=TOLERANCE, rtol=TOLERANCE)
    else:
        require(type(expected) is type(actual) and expected == actual, "JSON value changed")


def compare_exact(expected, actual):
    _, names = verify_branch_manifest(expected)
    _, fresh_names = verify_branch_manifest(actual)
    require(names.keys() == fresh_names.keys(), "Machine file set changed")
    checked = list(names) + ["manifest.json"]
    for name in checked:
        require(
            sha(expected / name) == sha(actual / name),
            f"Byte replay mismatch: {expected.name}/{name}",
        )
    return {"branch": expected.name, "method": "byte_exact", "files": len(checked)}


def compare_models(expected, actual):
    old, names = verify_branch_manifest(expected)
    fresh, fresh_names = verify_branch_manifest(actual)
    require(names.keys() == fresh_names.keys(), "Model machine file set changed")
    tables = {}
    for name in names:
        if name.endswith(".csv"):
            tables[name] = compare_frame(pd.read_csv(expected / name), pd.read_csv(actual / name))
        elif name.endswith(".parquet"):
            tables[name] = compare_frame(
                duckdb.read_parquet(str(expected / name)).df(),
                duckdb.read_parquet(str(actual / name)).df(),
            )
        elif name == "preprocessing.json":
            compare_json(read_json(expected / name), read_json(actual / name))
        elif name == "protocol_lock.json":
            a, b = read_json(expected / name), read_json(actual / name)
            require(a.pop("started_at") and b.pop("started_at"), "Missing run timestamp")
            require(a == b, "Protocol, code or input lock changed")
        else:
            raise ValueError(f"Undeclared model output comparison: {name}")
    exclusions = {"started_at", "completed_at", "outputs"}
    require(
        {k: v for k, v in old.items() if k not in exclusions}
        == {k: v for k, v in fresh.items() if k not in exclusions},
        "Model manifest provenance or settings changed",
    )
    return {
        "branch": "models",
        "method": "numeric_1e-9_and_exact_structure",
        "tables_max_abs": tables,
        "files": len(names),
        "excluded_byte_equality": ["manifest.json", "protocol_lock.json"],
        "exclusion_reason": "Run timestamps; output byte hashes may differ across runtimes",
        "protocol_code_input_hashes_equal": True,
    }


def compare_readout(expected, actual, models):
    old, names = verify_branch_manifest(expected)
    fresh, fresh_names = verify_branch_manifest(actual)
    require(names.keys() == fresh_names.keys(), "Readout machine file set changed")
    tables = {}
    for name in names:
        if name.endswith(".csv"):
            tables[name] = compare_frame(pd.read_csv(expected / name), pd.read_csv(actual / name))
        else:
            require(read_json(expected / name) == read_json(actual / name), "Readout QA changed")
    for name, digest in fresh["inputs"].items():
        require(sha(inside(models, name)) == digest, "Readout did not use fresh model output")
    exclusions = {"inputs", "outputs"}
    require(
        {k: v for k, v in old.items() if k not in exclusions}
        == {k: v for k, v in fresh.items() if k not in exclusions},
        "Readout code or prior input changed",
    )
    return {
        "branch": "readout",
        "method": "numeric_1e-9_and_exact_QA",
        "files": len(names),
        "tables_max_abs": tables,
        "excluded_byte_equality": ["manifest.json"],
        "exclusion_reason": "Hashes refer to fresh model files, checked against those files",
    }


def replay(repo, sme_raw, out):
    repo, sme_raw, out = (p.resolve() for p in [repo, sme_raw, out])
    validate_output(repo, sme_raw, out)
    canonical = repo / "reports/round5_analysis"
    frozen = {}
    for branch in BRANCHES:
        _, names = verify_branch_manifest(canonical / branch)
        frozen.update(
            {
                canonical / branch / name: sha(canonical / branch / name)
                for name in [*names, "manifest.json"]
            }
        )
    for name in SCRIPTS.values():
        path = repo / "scripts" / name
        frozen[path] = sha(path)
    frozen[Path(__file__).resolve()] = sha(Path(__file__))
    source_before = tree_state(sme_raw)
    out.mkdir(parents=True)
    stages, comparisons = [], []

    def run(name, arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=repo, capture_output=True, check=False
        )
        (out / f"{name}.stdout.log").write_bytes(result.stdout)
        (out / f"{name}.stderr.log").write_bytes(result.stderr)
        stages.append({"stage": name, "returncode": result.returncode})
        write_json(out / "stages.json", stages)
        require(result.returncode == 0, f"Stage failed: {name}; inspect saved logs")
        print(json.dumps(stages[-1]), flush=True)

    try:
        run("audit", [repo / "scripts" / SCRIPTS["audit"], "--repo", repo, "--out", out / "audit"])
        comparisons.append(compare_exact(canonical / "audit", out / "audit"))
        run(
            "models",
            [
                repo / "scripts" / SCRIPTS["models"],
                "--repo",
                repo,
                "--audit-root",
                out / "audit",
                "--out",
                out / "models",
            ],
        )
        comparisons.append(compare_models(canonical / "models", out / "models"))
        run(
            "readout",
            [repo / "scripts" / SCRIPTS["readout"], "--root", out, "--out", out / "readout"],
        )
        comparisons.append(compare_readout(canonical / "readout", out / "readout", out / "models"))
        run(
            "recovered",
            [repo / "scripts" / SCRIPTS["recovered"], "--root", out, "--out", out / "recovered"],
        )
        comparisons.append(compare_exact(canonical / "recovered", out / "recovered"))
        run(
            "context",
            [
                repo / "scripts" / SCRIPTS["context"],
                "--repo",
                repo,
                "--sme-raw",
                sme_raw,
                "--out",
                out / "context",
            ],
        )
        comparisons.append(compare_exact(canonical / "context", out / "context"))
        run(
            "independent",
            [
                repo / "scripts" / SCRIPTS["independent"],
                "--repo",
                repo,
                "--root",
                out,
                "--out",
                out / "independent_QA.json",
            ],
        )
        independent = read_json(out / "independent_QA.json")
        require(independent["status"] == "PASS", "Independent fresh-root validation failed")
        require(tree_state(sme_raw) == source_before, "Raw source tree changed during replay")
        require(
            all(sha(path) == digest for path, digest in frozen.items()),
            "Frozen result or code changed during replay",
        )
        qa = {
            "status": "PASS",
            "scope": "Saved-source replay in existing locked environment",
            "clean_environment_created": False,
            "network_requested": False,
            "llm_calls_requested": False,
            "numeric_atol": TOLERANCE,
            "numeric_rtol": TOLERANCE,
            "raw_tree_metadata_unchanged": True,
            "canonical_machine_files_and_code_unchanged": True,
            "manual_documents_regenerated": False,
            "raw_hash_verification": "Context stage verifies its manifest-declared raw inputs",
            "independent_validation": "PASS on fresh root, including independent Ridge refits",
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sme-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.repo, args.sme_raw, args.out)
