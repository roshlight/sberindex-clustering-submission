"""Verify downloaded source archives without extracting or executing their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for asset in manifest["archives"]:
        name = asset["asset"]
        if Path(name).name != name or not name.endswith(".zip"):
            raise ValueError(f"Invalid asset name: {name}")
        path = directory / name
        if path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        if digest_file(path) != asset["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {name}")
        with ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate archive entry: {name}")
            for entry_name in names:
                entry_path = PurePosixPath(entry_name)
                if entry_path.is_absolute() or ".." in entry_path.parts:
                    raise ValueError(f"Unsafe archive path: {entry_name}")
            contents = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
            expected = {entry["archive_path"] for entry in contents["entries"]}
            if set(names) != expected | {"ARCHIVE_MANIFEST.json"}:
                raise ValueError(f"Unexpected or missing archive entries: {name}")
            if len(contents["entries"]) != asset["archived_files"]:
                raise ValueError(f"Entry count mismatch: {name}")
            for entry in contents["entries"]:
                info = archive.getinfo(entry["archive_path"])
                if info.file_size != entry["bytes"]:
                    raise ValueError(f"Entry size mismatch: {entry['archive_path']}")
                with archive.open(info) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != entry["sha256"]:
                    raise ValueError(f"Entry SHA-256 mismatch: {entry['archive_path']}")
            results.append({"asset": name, "status": "PASS", "files": len(expected)})
    return {"status": "PASS", "archives": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing ZIP assets")
    parser.add_argument(
        "--manifest", type=Path, help="Defaults to directory/archives_manifest.json"
    )
    args = parser.parse_args()
    manifest = args.manifest or args.directory / "archives_manifest.json"
    print(json.dumps(verify(args.directory, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
