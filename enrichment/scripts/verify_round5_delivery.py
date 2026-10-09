"""Read-only verification of the fifth-round repository seal and branch provenance."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

REPORT = "reports/data_round5"
MACHINE_FILES = {
    "economy": {
        "annual_candidate_panel.csv",
        "annual_candidate_panel.parquet",
        "collection_status.json",
        "coverage_delta.csv",
        "delta_all_cells.csv",
        "form_inventory.json",
        "new_numeric_cells.csv",
        "QA.json",
        "recovered_source_cells.csv",
        "remaining_missing_cells.csv",
        "revision_conflicts.csv",
        "source_only_outside_year_link.csv",
        "target_cell_decisions.csv",
        "execution.json",
        "manifest.json",
    },
    "geography": {
        "registry_versions.csv",
        "sme_code_inventory.csv",
        "sme_code_presence.parquet",
        "code_presence_summary.csv",
        "code_kind_summary.csv",
        "registry_duplicate_codes.csv",
        "june_july_code_changes.csv",
        "QA.json",
        "manifest.json",
    },
    "cases": {
        "evidence_ledger.csv",
        "source_status.csv",
        "hypothesis_status.csv",
        "followup_requests.csv",
        "source_inventory.json",
        "QA.json",
        "manifest.json",
    },
    "sme": {
        "archive_sample_summary.csv",
        "downloaded_sources.csv",
        "evidence_ledger.csv",
        "generator_code_evidence.csv",
        "historical_archive_catalog.csv",
        "municipal_pdf_text.json",
        "municipal_stock_candidates_2024.csv",
        "official_stock_counts.csv",
        "passport_text_evidence.json",
        "prior_source_reference.json",
        "QA.json",
        "raw_inventory.csv",
        "retrieval_attempts.csv",
        "schema_field_evidence.csv",
        "xml_sample_audit.csv",
    },
    "search": {"server_summary.json", "server_document_inventory.csv"},
}
BRANCH_SCRIPTS = {
    "economy": "enrichment/src/sber_enrichment/round5_economy.py",
    "geography": "enrichment/scripts/prepare_round5_geography.py",
    "cases": "enrichment/scripts/collect_round5_cases.py",
    "sme": "enrichment/scripts/audit_round5_sme.py",
    "search": "enrichment/scripts/summarize_round5_search.py",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(base, relative):
    """Accept portable relative files, including older Windows manifest separators."""
    require(isinstance(relative, str) and bool(relative), "Empty or invalid manifest path")
    windows = PureWindowsPath(relative)
    normalized = relative.replace("\\", "/")
    require(
        not windows.drive and not windows.root and not PurePosixPath(normalized).is_absolute(),
        f"Absolute manifest path: {relative}",
    )
    require(".." not in PurePosixPath(normalized).parts, f"Parent path segment: {relative}")
    require(":" not in normalized, f"Invalid manifest path: {relative}")
    path = (base / normalized).resolve()
    require(path.is_relative_to(base.resolve()), f"Path leaves its base: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def verify_hashes(base, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Empty or invalid hash map")
    checked = set()
    for relative, expected in hashes.items():
        path = inside(base, relative)
        require(path not in checked, f"Duplicate normalized path: {relative}")
        checked.add(path)
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"Invalid SHA256: {relative}",
        )
        require(sha(path) == expected, f"Hash changed: {relative}")
    return checked


def verify_seal(repo, manifest):
    seal = read_json(manifest)
    files = seal.get("files")
    require(isinstance(files, dict) and bool(files), "Empty or invalid delivery manifest")
    require(all(isinstance(value, dict) for value in files.values()), "Invalid file metadata")
    hashes = {name: value.get("sha256") for name, value in files.items()}
    checked = verify_hashes(repo, hashes)
    for name, value in files.items():
        size = value.get("bytes")
        require(type(size) is int and size >= 0, f"Invalid byte size: {name}")
        require(inside(repo, name).stat().st_size == size, f"Byte size changed: {name}")
    return checked


def branch_hashes(repo):
    """Verify branch manifests and return only the declared machine replay contract."""
    report = repo / REPORT
    result = {}
    for branch, names in MACHINE_FILES.items():
        folder = report / branch
        manifest_name = "artifact_manifest.json" if branch == "sme" else "manifest.json"
        manifest = read_json(folder / manifest_name)
        if branch == "sme":
            require(isinstance(manifest, list) and bool(manifest), "Invalid SME manifest")
            declared = {item["path"]: item["sha256"] for item in manifest}
            require(len(declared) == len(manifest), "Duplicate SME manifest path")
            for item in manifest:
                require(
                    inside(folder, item["path"]).stat().st_size == item["bytes"],
                    f"SME manifest byte size changed: {item['path']}",
                )
        else:
            key = "output_hashes" if branch == "economy" else "outputs"
            declared = manifest[key]
            script_key = "source_sha256" if branch == "economy" else "script_sha256"
            verify_hashes(repo, {BRANCH_SCRIPTS[branch]: manifest[script_key]})
        verify_hashes(folder, declared)
        required = names - {"manifest.json", "execution.json"}
        require(required <= declared.keys(), f"Branch manifest omits machine outputs: {branch}")
        if branch == "economy":
            verify_hashes(repo, {manifest["input_path"]: manifest["input_sha256"]})
            verify_hashes(repo, manifest["helper_hashes"])
        elif branch == "geography":
            verify_hashes(repo, manifest["repo_inputs"])
        elif branch == "cases":
            verify_hashes(folder, {"evidence_recipe.json": manifest["recipe_sha256"]})
            verify_hashes(folder, {"source_inventory.json": manifest["retrievals_sha256"]})
        result[branch] = {name: sha(inside(folder, name)) for name in sorted(names)}
    return result


def verify(repo, manifest):
    repo, manifest = repo.resolve(), manifest.resolve()
    checked = verify_seal(repo, manifest)
    branches = branch_hashes(repo)
    required = {
        repo / REPORT / branch / name for branch, names in MACHINE_FILES.items() for name in names
    }
    required.update(repo / path for path in BRANCH_SCRIPTS.values())
    required.update(
        repo / "enrichment/scripts" / name
        for name in ["replay_round5.py", "verify_round5_delivery.py"]
    )
    required.update(
        repo / REPORT / branch / name
        for branch, name in [
            ("cases", "evidence_recipe.json"),
            ("sme", "artifact_manifest.json"),
            ("search", "manifest.json"),
        ]
    )
    require(
        all(path.resolve() in checked for path in required),
        "Seal omits required machine outputs, manifests, recipe or replay code",
    )
    # Every delivered document is sealed, although manual documents are not regenerated.
    report_files = {p.resolve() for p in (repo / REPORT).rglob("*") if p.is_file()}
    require(report_files - {manifest} <= checked, "Seal omits files in reports/data_round5")
    return {
        "status": "PASS",
        "sealed_files": len(checked),
        "branch_machine_files": {key: len(value) for key, value in branches.items()},
        "raw_sources_checked": False,
        "replay_performed": False,
        "scope": "Read-only repository seal, branch outputs and declared repository provenance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    path = args.manifest or args.repo / REPORT / "delivery_manifest.json"
    print(json.dumps(verify(args.repo, path), ensure_ascii=False, indent=2))
