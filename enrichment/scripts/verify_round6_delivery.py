"""Read-only round-six delivery seal and branch provenance verification.

Seal schema: artifacts and raw_releases are lists of {path, sha256, bytes}.
Artifact paths are repository-relative; release paths are relative to --release-dir.
Source verification is optional and requires both --raw and --prior-geography-raw.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

REPORT = "reports/data_round6"
BRANCH_SCRIPTS = {
    "economy": "enrichment/scripts/prepare_round6_economy.py",
    "sme": "enrichment/scripts/audit_round6_sme.py",
    "geography": "enrichment/scripts/audit_round6_geography.py",
    "cases": "enrichment/scripts/collect_round6_cases.py",
    "search": "enrichment/scripts/summarize_round5_search.py",
}
MACHINE_FILES = {
    "economy": {
        "catalog_publication_labels.csv",
        "document_inventory.csv",
        "food_reconciliation.csv",
        "pdf_docx_crosscheck.csv",
        "regional_context_candidate.csv",
        "QA.json",
        "manifest.json",
    },
    "sme": {
        "asset_code_syntax.csv",
        "asset_schema_audit.csv",
        "author_source_lineage.csv",
        "code_evidence.csv",
        "federal_city_source_rows.csv",
        "geocoder_address_contracts.csv",
        "QA.json",
        "raw_inventory.csv",
        "retrieval_attempts.csv",
        "worker_dates.csv",
        "worker_sample_summary.csv",
        "manifest.json",
    },
    "geography": {
        "ambiguous_transition_edges.csv",
        "candidate_summary.csv",
        "candidate_transitions_to_panel.csv",
        "classifier_structural_evidence.json",
        "crosswalk_versions.csv",
        "downloaded_sources.csv",
        "evidence_ledger.csv",
        "gar_available_download_catalog.csv",
        "gar_schema_history_fields.csv",
        "official_amendment_document_catalog.csv",
        "panel_2016_classifier_control.csv",
        "perm_reference_control.json",
        "perm_reference_text.json",
        "primorye_643_document_check.json",
        "priority_settlement_parent_candidates.csv",
        "priority_transition_rows.csv",
        "QA.json",
        "retrieval_attempts.json",
        "transition_duplicate_rows.csv",
        "transition_nonlinks.csv",
        "transition_version_changes.csv",
        "manifest.json",
    },
    "cases": {
        "airport_annual_finance.csv",
        "evidence_ledger.csv",
        "financial_semantics.csv",
        "followup_requests.csv",
        "hypothesis_status.csv",
        "QA.json",
        "source_inventory.json",
        "source_status.csv",
        "manifest.json",
    },
    "search": {"server_summary.json", "server_document_inventory.csv"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(relative):
    require(isinstance(relative, str) and bool(relative), "Invalid relative path")
    value = relative.replace("\\", "/")
    windows, posix = PureWindowsPath(relative), PurePosixPath(value)
    require(
        not windows.drive and not windows.root and not posix.is_absolute(),
        f"Absolute path forbidden: {relative}",
    )
    require(".." not in posix.parts and ":" not in value, f"Unsafe path: {relative}")
    require(value not in {"", "."}, f"Invalid file path: {relative}")
    return posix.as_posix()


def inside(base, relative):
    path = (base / relative_path(relative)).resolve()
    require(path.is_relative_to(base.resolve()), f"Path leaves declared root: {relative}")
    require(path.is_file(), f"Missing declared file: {relative}")
    return path


def verify_hashes(base, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Empty or invalid hash map")
    checked = set()
    for relative, expected in hashes.items():
        path = inside(base, relative)
        require(path not in checked, f"Duplicate normalized path: {relative}")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"Invalid SHA256: {relative}",
        )
        require(sha(path) == expected, f"Hash changed: {relative}")
        checked.add(path)
    return checked


def record_hashes(records):
    require(isinstance(records, list) and bool(records), "Empty or invalid file record list")
    hashes, names = {}, set()
    for row in records:
        require(isinstance(row, dict), "Invalid file record")
        name = row.get("path")
        normalized = relative_path(name)
        require(normalized not in names, f"Duplicate normalized file record: {name}")
        names.add(normalized)
        require(
            type(row.get("bytes")) is int and row["bytes"] >= 0, f"Invalid file byte size: {name}"
        )
        require(
            isinstance(row.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            f"Invalid SHA256: {name}",
        )
        hashes[name] = row["sha256"]
    return hashes


def verify_records(base, records):
    hashes = record_hashes(records)
    checked = verify_hashes(base, hashes)
    for row in records:
        require(
            inside(base, row["path"]).stat().st_size == row["bytes"],
            f"Byte size changed: {row['path']}",
        )
    return checked


def branch_hashes(repo):
    """Check four producer manifests; search has its explicit two-file contract."""
    result = {}
    for branch, names in MACHINE_FILES.items():
        folder = repo / REPORT / branch
        if branch != "search":
            manifest = read_json(folder / "manifest.json")
            if branch == "sme":
                declared = record_hashes(manifest["files"])
                verify_records(folder, manifest["files"])
                verify_hashes(repo, manifest["dependency_hashes"])
            else:
                declared = manifest["outputs"]
                verify_hashes(folder, declared)
                verify_hashes(repo, {BRANCH_SCRIPTS[branch]: manifest["script_sha256"]})
            require(
                set(declared) == names - {"manifest.json"},
                f"Unexpected machine output contract: {branch}",
            )
            if branch == "economy":
                verify_hashes(repo, manifest["repo_inputs"])
            elif branch == "geography":
                repository_inputs = {
                    key.removeprefix("repo/"): value
                    for key, value in manifest["inputs"].items()
                    if key.startswith("repo/")
                }
                verify_hashes(repo, repository_inputs)
                require(
                    all(
                        key.startswith(("repo/", "raw/", "prior_raw/"))
                        for key in manifest["inputs"]
                    ),
                    "Unknown geography input namespace",
                )
            elif branch == "cases":
                verify_hashes(
                    repo,
                    {
                        "enrichment/scripts/collect_round5_cases.py": manifest[
                            "helper_script_sha256"
                        ]
                    },
                )
                verify_hashes(
                    folder,
                    {
                        "evidence_recipe.json": manifest["recipe_sha256"],
                        "source_inventory.json": manifest["retrievals_sha256"],
                    },
                )
        result[branch] = {name: sha(inside(folder, name)) for name in sorted(names)}
    return result


def resolve_search_raw(raw, explicit=None):
    if explicit is not None:
        require(explicit.is_dir(), "Missing explicitly selected search directory")
        return explicit.resolve()
    options = [path for path in [raw / "search", raw / "server-complete"] if path.is_dir()]
    require(
        len(options) == 1, "Expected raw/search or raw/server-complete; use --search-raw if both"
    )
    return options[0].resolve()


def verify_raw(repo, raw, prior_geography_raw, search_raw=None):
    """Verify declared source bytes without calling a producer or downloading anything."""
    report = repo / REPORT
    checked = {}
    economy = read_json(report / "economy/manifest.json")
    checked["economy"] = len(verify_hashes(raw / "economy", economy["raw_inputs"]))
    sme = read_json(report / "sme/manifest.json")
    verify_hashes(raw / "sme", {"SOURCE_MANIFEST.json": sme["raw_manifest_sha256"]})
    sources = read_json(raw / "sme/SOURCE_MANIFEST.json")["files"]
    sme_checked = verify_records(raw / "sme", sources)
    actual = {
        p.resolve()
        for p in (raw / "sme").rglob("*")
        if p.is_file() and p.name != "SOURCE_MANIFEST.json"
    }
    require(sme_checked == actual, "SME raw inventory incomplete or has additional files")
    checked["sme"] = len(sme_checked) + 1
    geo = read_json(report / "geography/manifest.json")
    for prefix, source, key in [
        ("raw/", raw / "geography", "geography"),
        ("prior_raw/", prior_geography_raw, "prior_geography"),
    ]:
        hashes = {
            name.removeprefix(prefix): value
            for name, value in geo["inputs"].items()
            if name.startswith(prefix)
        }
        checked[key] = len(verify_hashes(source, hashes))
    case_manifest = read_json(report / "cases/manifest.json")
    verify_hashes(raw / "cases", {"retrievals.json": case_manifest["retrievals_sha256"]})
    cases = read_json(report / "cases/raw_allowlist.json")["files"]
    checked["cases"] = len(verify_records(raw / "cases", cases))
    search = resolve_search_raw(raw, search_raw)
    summary = read_json(report / "search/server_summary.json")
    search_checked = set()
    for run in summary["runs"]:
        path = inside(search, run["run_id"] + "/run_manifest.json")
        require(sha(path) == run["manifest_sha256"], "Search run manifest changed")
        search_checked.add(path)
        manifest = read_json(path)
        search_checked.update(verify_hashes(path.parent, manifest["output_hashes"]))
        for document in read_json(path.parent / "document_inventory.json"):
            if document.get("raw_file"):
                search_checked.update(
                    verify_hashes(path.parent, {document["raw_file"]: document["sha256"]})
                )
    require(bool(search_checked), "Empty search input inventory")
    checked["search"] = len(search_checked)
    return checked


def verify_seal(repo, manifest, release_dir=None):
    seal = read_json(manifest)
    require(seal.get("schema_version") == 1, "Unsupported delivery seal schema")
    checked = verify_records(repo, seal.get("artifacts"))
    releases = seal.get("raw_releases")
    record_hashes(releases)  # Metadata is validated even when the ZIPs are not present locally.
    require(len(releases) == 5, "Expected the five declared raw branch releases")
    release_count = len(verify_records(release_dir, releases)) if release_dir else 0
    return checked, release_count


def verify(repo, manifest, raw=None, prior_geography_raw=None, search_raw=None, release_dir=None):
    repo, manifest = repo.resolve(), manifest.resolve()
    checked, release_count = verify_seal(repo, manifest, release_dir)
    branches = branch_hashes(repo)
    required = {
        repo / REPORT / branch / name for branch, names in MACHINE_FILES.items() for name in names
    }
    required.update(repo / path for path in BRANCH_SCRIPTS.values())
    required.update(
        repo / path
        for path in [
            "enrichment/scripts/replay_round6.py",
            "enrichment/scripts/verify_round6_delivery.py",
            "enrichment/uv.lock",
            "enrichment/pyproject.toml",
            "enrichment/scripts/collect_round5_cases.py",
            REPORT + "/cases/evidence_recipe.json",
            REPORT + "/cases/raw_allowlist.json",
        ]
    )
    required.update(
        repo / path for path in read_json(repo / REPORT / "sme/manifest.json")["dependency_hashes"]
    )
    required.update(
        repo / path for path in read_json(repo / REPORT / "economy/manifest.json")["repo_inputs"]
    )
    required.update(
        repo / name.removeprefix("repo/")
        for name in read_json(repo / REPORT / "geography/manifest.json")["inputs"]
        if name.startswith("repo/")
    )
    require(
        all(path.resolve() in checked for path in required),
        "Seal omits required machine outputs, code, source recipe or lock files",
    )
    report_files = {p.resolve() for p in (repo / REPORT).rglob("*") if p.is_file()}
    require(report_files - {manifest} <= checked, "Seal omits files in reports/data_round6")
    require(
        (raw is None) == (prior_geography_raw is None),
        "Provide --raw and --prior-geography-raw together",
    )
    raw_counts = verify_raw(repo, raw, prior_geography_raw, search_raw) if raw is not None else {}
    return dict(
        status="PASS",
        sealed_artifacts=len(checked),
        raw_releases_checked=release_count,
        branch_machine_files={key: len(value) for key, value in branches.items()},
        raw_source_files_checked=raw_counts,
        replay_performed=False,
        source_conversion_or_manual_review_repeated=False,
        scope="Read-only file integrity and declared provenance; no scientific re-evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--prior-geography-raw", type=Path)
    parser.add_argument("--search-raw", type=Path)
    parser.add_argument("--release-dir", type=Path)
    args = parser.parse_args()
    path = args.manifest or args.repo / REPORT / "delivery_manifest.json"
    print(
        json.dumps(
            verify(
                args.repo,
                path,
                args.raw,
                args.prior_geography_raw,
                args.search_raw,
                args.release_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
