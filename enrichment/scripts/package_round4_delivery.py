"""Round4 packager: original scanner plus one exact audited JPEG DHT exception.

No bytes matching the credential pattern are logged. The exception cannot be
configured from the CLI and does not authorize any other file or scanner match.
"""

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SECRETS = re.compile(
    rb"(?:sk-or-v1-[A-Za-z0-9]{20,}|\d{8,12}:[A-Za-z0-9_-]{32,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)

AUDITED_EXCEPTION = {
    "exception_id": "pit-planning-jpeg-dht-20260916",
    "file_bytes": 1055501,
    "file_sha256": "f95741def9af964fbbb1b505cf52cba5b13eb74b197ede960abeabac25d50ea1",
    "offset": 596649,
    "length": 41,
    "match_sha256": "2bd445e4a1c907caa85db7ba09b0aed51556373edb48b41d5e431b38f24416cd",
    "jpeg_offset": 596100,
    "jpeg_bytes": 21478,
    "jpeg_sha256": "c01ee8d56dbd4b76dee8481c41bb2c009d8aae8c1eaa3004129977f875e47cf5",
    "pdf_object_number": 128,
    "pdf_filter": "/DCTDecode",
    "pdf_subtype": "/Image",
    "dht": {
        "segment_offset": 596342,
        "segment_end_exclusive": 596762,
        "declared_length": 418,
        "table_offset": 596583,
        "table_class": 1,
        "table_id": 1,
        "symbol_count": 162,
        "symbols_offset": 596600,
        "symbols_end_exclusive": 596762,
    },
    "review": "reports/data_round4/delivery/scanner_false_positive_review.json",
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def containing_dht(data, jpeg_offset, match_start, match_end):
    """Parse JPEG headers from SOI; accept containment in Huffman symbols only.

    A random marker sequence or a hit within a DHT header/count array is not a
    match. Fail closed at scan data: this review concerns a pre-SOS DHT table.
    """
    if data[jpeg_offset : jpeg_offset + 2] != b"\xff\xd8":
        raise ValueError("Invalid JPEG start")
    position = jpeg_offset + 2
    while position + 4 <= len(data):
        if data[position] != 255:
            raise ValueError("Invalid JPEG marker framing")
        marker = data[position + 1]
        if marker in (0, 216, 217, 218) or 208 <= marker <= 215:
            raise ValueError("No reviewed pre-scan DHT containment")
        size = int.from_bytes(data[position + 2 : position + 4], "big")
        end = position + 2 + size
        if size < 2 or end > len(data):
            raise ValueError("Invalid JPEG segment length")
        found = None
        if marker == 196:
            table = position + 4
            while table < end:
                if table + 17 > end:
                    raise ValueError("Truncated Huffman table")
                info = data[table]
                counts = data[table + 1 : table + 17]
                count = sum(counts)
                symbols = table + 17
                symbols_end = symbols + count
                if info >> 4 not in (0, 1) or info & 15 > 3 or count > 256 or symbols_end > end:
                    raise ValueError("Invalid Huffman table structure")
                slots = 1
                for n in counts:
                    slots = slots * 2 - n
                    if slots < 0:
                        raise ValueError("Oversubscribed Huffman codes")
                if symbols <= match_start < match_end <= symbols_end:
                    found = {
                        "segment_offset": position,
                        "segment_end_exclusive": end,
                        "declared_length": size,
                        "table_offset": table,
                        "table_class": info >> 4,
                        "table_id": info & 15,
                        "symbol_count": count,
                        "symbols_offset": symbols,
                        "symbols_end_exclusive": symbols_end,
                    }
                table = symbols_end
            if found is not None:
                return found
        if position <= match_start < end:
            raise ValueError("Scanner match is outside DHT symbols")
        position = end
    raise ValueError("No DHT containment")


def scan_bytes(data):
    """Return metadata for the one permitted hit; reject every other hit."""
    exceptions = []
    full_sha = None
    for match in SECRETS.finditer(data):
        approved = AUDITED_EXCEPTION
        if full_sha is None:
            full_sha = sha_bytes(data)
        if (
            len(data) != approved["file_bytes"]
            or full_sha != approved["file_sha256"]
            or match.start() != approved["offset"]
            or match.end() - match.start() != approved["length"]
            or sha_bytes(match.group()) != approved["match_sha256"]
        ):
            raise ValueError("Credential-like data detected; contents are not printed")
        jpeg = data[approved["jpeg_offset"] : approved["jpeg_offset"] + approved["jpeg_bytes"]]
        if sha_bytes(jpeg) != approved["jpeg_sha256"] or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("Audited JPEG structure differs")
        dht = containing_dht(data, approved["jpeg_offset"], match.start(), match.end())
        if dht != approved["dht"]:
            raise ValueError("Audited DHT containment differs")
        exceptions.append(
            {**approved, "reason": "JPEG DHT AC-table symbol bytes; not a credential"}
        )
    return exceptions


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
    scan_exceptions = []
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
            approved_hits = scan_bytes(data)
            scan_exceptions.extend({"archive_path": str(name), **hit} for hit in approved_hits)
            z.writestr(str(name), data)
            records.append({"path": str(name), "bytes": len(data), "sha256": sha_bytes(data)})
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": records,
            "source_allowlist_sha256": sha_bytes(args.allowlist.read_bytes()),
            "packager_sha256": sha_bytes(Path(__file__).read_bytes()),
            "credential_scan": {
                "policy": "block_all_matches_except_one_exact_audited_pdf_jpeg_dht",
                "pattern_sha256": sha_bytes(SECRETS.pattern),
                "audited_exception_id": AUDITED_EXCEPTION["exception_id"],
                "exceptions_applied": scan_exceptions,
            },
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
        "credential_scan_policy": manifest["credential_scan"]["policy"],
        "audited_exceptions_applied": len(scan_exceptions),
        "credential_scan_exceptions": scan_exceptions,
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
