"""Strict round-six packaging with one exact, path-bound audited PDF DHT exception."""

import argparse
import importlib.util
import json
import types
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SPEC = importlib.util.spec_from_file_location(
    "immutable_round5_packager", Path(__file__).with_name("package_round5_delivery.py")
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
SECRETS = previous.SECRETS
sha_bytes = previous.sha_bytes
containing_dht = previous.containing_dht
ALLOWED_PATHS = frozenset(
    {"raw6/cases/raw/ach_rtm_report.pdf", "raw6/cases/raw/ach_rtm_audit_mirror.pdf"}
)
AUDITED_EXCEPTION = {
    "exception_id": "ach-rtm-2025-pdf-jpeg-dht-20260916",
    "file_bytes": 597806,
    "file_sha256": "dd90aa61ad92e695149941bb495cbb88933e8b84f2a947cbac29d23d4ad290ae",
    "offset": 79451,
    "length": 45,
    "match_sha256": "51be0a9642a4cb5bb2c57526b29208cdbc08774fd3c5186aef3a3150db9a42e8",
    "jpeg_offset": 78915,
    "jpeg_bytes": 9838,
    "jpeg_sha256": "a84120f461eae7d378e40bac239aef8939d553eee33c3c1a1316f1b45dc44a37",
    "pdf_object_number": 4,
    "pdf_filter": "/DCTDecode",
    "pdf_subtype": "/Image",
    "dht": {
        "segment_offset": 79157,
        "segment_end_exclusive": 79577,
        "declared_length": 418,
        "table_offset": 79398,
        "table_class": 1,
        "table_id": 1,
        "symbol_count": 162,
        "symbols_offset": 79415,
        "symbols_end_exclusive": 79577,
    },
    "review": "reports/data_round6/delivery/scanner_false_positive_review.json",
}


def scan_bytes(data: bytes, archive_path: str) -> list[dict]:
    if archive_path not in ALLOWED_PATHS:
        return previous.scan_bytes(data)
    # Execute the unchanged prior scanner against a copied globals dictionary.
    # The old module and its exceptions remain immutable; only these two paths
    # can use this one exact source/JPEG/span exception.
    namespace = {
        **previous.__dict__,
        "AUDITED_EXCEPTIONS": (*previous.AUDITED_EXCEPTIONS, AUDITED_EXCEPTION),
    }
    scanner = types.FunctionType(previous.scan_bytes.__code__, namespace)
    return scanner(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite a delivered archive")
    roots = {"work": args.work.resolve(), "repo": args.repo.resolve()}
    entries = json.loads(args.allowlist.read_text(encoding="utf-8"))
    seen, records, scan_exceptions = set(), [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            root = roots[entry["root"]]
            source = (root / entry["path"]).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("Allowlist path is missing or outside its declared root")
            name = PurePosixPath(entry["archive_path"])
            if name.is_absolute() or ".." in name.parts or ":" in str(name):
                raise ValueError("Unsafe archive path")
            if str(name).casefold() in seen:
                raise ValueError("Duplicate archive path")
            seen.add(str(name).casefold())
            data = source.read_bytes()
            if entry.get("sha256") and sha_bytes(data) != entry["sha256"]:
                raise ValueError("Allowlisted source hash changed")
            approved_hits = scan_bytes(data, str(name))
            scan_exceptions.extend({"archive_path": str(name), **hit} for hit in approved_hits)
            z.writestr(str(name), data)
            records.append({"path": str(name), "bytes": len(data), "sha256": sha_bytes(data)})
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": records,
            "source_allowlist_sha256": sha_bytes(args.allowlist.read_bytes()),
            "packager_sha256": sha_bytes(Path(__file__).read_bytes()),
            "immutable_helper_sha256": sha_bytes(Path(previous.__file__).read_bytes()),
            "credential_scan": {
                "policy": "unchanged_regex_and_DHT_parser_with_exact_path_bound_exception",
                "pattern_sha256": sha_bytes(SECRETS.pattern),
                "exceptions_applied": scan_exceptions,
            },
        }
        z.writestr(
            f"MANIFEST.{args.out.stem}.json", json.dumps(manifest, ensure_ascii=False, indent=2)
        )
    with zipfile.ZipFile(args.out) as z:
        assert z.testzip() is None
        for item in records:
            assert sha_bytes(z.read(item["path"])) == item["sha256"]
    validation = {
        "status": "PASS",
        "archive": args.out.name,
        "archive_sha256": sha_bytes(args.out.read_bytes()),
        "archive_bytes": args.out.stat().st_size,
        "verified_files": len(records),
        "uncompressed_bytes": sum(x["bytes"] for x in records),
        "allowlist_sha256": manifest["source_allowlist_sha256"],
        "packager_sha256": manifest["packager_sha256"],
        "immutable_helper_sha256": manifest["immutable_helper_sha256"],
        "credential_scan_policy": manifest["credential_scan"]["policy"],
        "audited_exceptions_applied": len(scan_exceptions),
        "credential_scan_exceptions": scan_exceptions,
    }
    for suffix, value in [(".manifest.json", manifest), (".qa.json", validation)]:
        args.out.with_suffix(suffix).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
