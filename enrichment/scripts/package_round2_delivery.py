"""Package an explicit allowlist of research files and verify every archived byte."""

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
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
    seen = set()
    records = []
    secrets = re.compile(
        rb"(?:sk-or-v1-[A-Za-z0-9]{20,}|\d{8,12}:[A-Za-z0-9_-]{32,}|"
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
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
            if secrets.search(data):
                raise ValueError("Credential-like data detected; contents are not printed")
            z.writestr(str(name), data)
            records.append({"path": str(name), "bytes": len(data), "sha256": sha_bytes(data)})
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": records,
            "source_allowlist_sha256": sha_bytes(args.allowlist.read_bytes()),
            "packager_sha256": sha_bytes(Path(__file__).read_bytes()),
        }
        z.writestr(
            f"MANIFEST.{args.out.stem}.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
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
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.out.with_suffix(".qa.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
