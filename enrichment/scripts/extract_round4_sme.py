"""Restore only missing SME Parquet members using immutable saved provenance.

No downloads, timestamp updates, metadata rewrites, or replacement of existing
files. ZIP contents, streamed CRC/SHA and existing outputs must match retrievals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath

CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    """Use archive separators and reject Windows drives/ADS as well as traversal."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} or part.rstrip(" .") != part for part in parts)
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def within(root: Path, relative: PurePosixPath) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Resolved path escapes allowed directory: {relative}")
    return candidate


def expected_file(path: Path, record: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Saved SHA/size mismatch: {path}")


def plan(raw: Path, records: list[dict]) -> list[tuple]:
    jobs = []
    destinations, archives = set(), set()
    for record in records:
        year = str(record["year_in_filename"])
        if len(year) != 4 or not year.isdigit():
            raise ValueError("Invalid year directory")
        archive_relative = safe_relative(record["path"])
        if archive_relative.parts[0] != "raw" or archive_relative.suffix.lower() != ".zip":
            raise ValueError("Archive must be a listed ZIP under raw/")
        archive_path = within(raw, archive_relative)
        archive_key = str(archive_path).casefold()
        if archive_key in archives:
            raise ValueError("Duplicate ZIP record")
        archives.add(archive_key)
        expected_file(archive_path, record)
        entries = {}
        for entry in record["entries"]:
            relative = safe_relative(entry["path"])
            if relative.parts[:2] != ("extracted", year) or relative.suffix != ".parquet":
                raise ValueError("Only listed Parquet paths under extracted/year are permitted")
            final = within(raw, relative)
            extracted = within(raw, PurePosixPath("extracted"))
            if not final.is_relative_to(extracted):
                raise ValueError("Output escaped extracted/")
            destination_key = str(final).casefold()
            if destination_key in destinations:
                raise ValueError("Duplicate or case-colliding output")
            destinations.add(destination_key)
            member = PurePosixPath(*relative.parts[2:]).as_posix()
            entries[member] = (entry, final)
            if final.exists():
                expected_file(final, entry)
        if not entries:
            raise ValueError("No listed extracted members")
        with zipfile.ZipFile(archive_path) as archive:
            members = {}
            for info in archive.infolist():
                relative = safe_relative(
                    info.filename.rstrip("/") if info.is_dir() else info.filename
                )
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Archive symlink rejected")
                if info.is_dir():
                    if not any(relative in PurePosixPath(name).parents for name in entries):
                        raise ValueError("Unlisted archive directory")
                    continue
                name = relative.as_posix()
                if name in members:
                    raise ValueError("Duplicate archive member")
                members[name] = info
            if set(members) != set(entries):
                raise ValueError("Unlisted or missing archive members")
            for name, (entry, final) in entries.items():
                info = members[name]
                if info.file_size != entry["bytes"] or info.CRC != entry["zip_crc32"]:
                    raise ValueError("ZIP directory differs from saved size/CRC")
                jobs.append((archive_path, name, entry, final))
    return jobs


def extract(raw: Path) -> dict:
    raw = raw.resolve()
    metadata_path = raw / "retrievals.json"
    metadata_before = sha256(metadata_path)
    records = json.loads(metadata_path.read_text("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty saved retrieval list")
    jobs = plan(raw, records)  # Preflight all inventories before any extracted write.
    created, existing, streamed_bytes = [], [], 0
    for archive_path, member, entry, final in jobs:
        digest, crc, size = hashlib.sha256(), 0, 0
        temporary = None
        handle = None
        try:
            if not final.exists():
                final.parent.mkdir(parents=True, exist_ok=True)
                handle = tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".sme-", suffix=".partial", dir=final.parent, delete=False
                )
                temporary = Path(handle.name)
            with zipfile.ZipFile(archive_path) as archive, archive.open(member) as stream:
                while block := stream.read(CHUNK_SIZE):
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError("Stream exceeds saved member size")
                    if handle is not None:
                        handle.write(block)
            if handle is not None:
                handle.close()
                handle = None
            if (size, crc, digest.hexdigest()) != (
                entry["bytes"],
                entry["zip_crc32"],
                entry["sha256"],
            ):
                raise ValueError(f"Streamed member CRC/SHA/size differs: {member}")
            streamed_bytes += size
            if temporary is not None:
                try:
                    # An atomic no-replace installation: an existing path cannot be overwritten.
                    os.link(temporary, final)
                    created.append(final.relative_to(raw).as_posix())
                except FileExistsError:
                    expected_file(final, entry)
                    existing.append(final.relative_to(raw).as_posix())
            else:
                expected_file(final, entry)
                existing.append(final.relative_to(raw).as_posix())
        finally:
            if handle is not None:
                handle.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if sha256(metadata_path) != metadata_before:
        raise ValueError("Retrieval metadata changed during extraction")
    return {
        "status": "PASS",
        "archives_verified": len(records),
        "members_crc_sha_verified": len(jobs),
        "streamed_uncompressed_bytes": streamed_bytes,
        "created": created,
        "existing_unchanged": existing,
        "retrievals_sha256_unchanged": metadata_before,
        "metadata_written": False,
        "network_requests": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, required=True, help="SME root containing retrievals.json"
    )
    args = parser.parse_args()
    print(json.dumps(extract(args.raw), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
