"""Round5 packager with exact audited JPEG DHT exceptions and unchanged scanner regex.

The prior round4 exception is retained. Eight further exact byte spans in one
reviewed PDF are permitted only with full-file, match and JPEG digest checks plus
parsed DHT containment. No credential-shaped bytes are logged. The CLI cannot
configure exceptions; all other matches remain blocked.
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


ROUND5_AUDITED_EXCEPTIONS = [
    {
        "exception_id": "mintrans-report2023-jpeg-dht-1-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 640965,
        "length": 50,
        "match_sha256": "993e4804616522490d84eb31308fdc4b49f194772df492a8a049429052b4fa95",
        "jpeg_offset": 640778,
        "jpeg_bytes": 612,
        "jpeg_sha256": "e01f14e09a84751a9343cde17959b3609f728bc339eadfcbda4f4de3f2ee43fd",
        "pdf_object_number": 76,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 640884,
            "segment_end_exclusive": 641096,
            "declared_length": 210,
            "table_offset": 640917,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 640934,
            "symbols_end_exclusive": 641096,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-2-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 1279610,
        "length": 50,
        "match_sha256": "993e4804616522490d84eb31308fdc4b49f194772df492a8a049429052b4fa95",
        "jpeg_offset": 1279423,
        "jpeg_bytes": 569,
        "jpeg_sha256": "4049bc79e56fa84110952f49f91d7638fd3e8df9294395f92dbaa6e52f639c0b",
        "pdf_object_number": 208,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 1279529,
            "segment_end_exclusive": 1279741,
            "declared_length": 210,
            "table_offset": 1279562,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 1279579,
            "symbols_end_exclusive": 1279741,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-3-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 1280581,
        "length": 49,
        "match_sha256": "a667eb841ebc3b147b3d2f0df4785d608f935756ff4d947f111373a3e4825e4b",
        "jpeg_offset": 1280228,
        "jpeg_bytes": 935,
        "jpeg_sha256": "5c250d0a31db8fe6707e7c0f8b3556887c6e273327287698fcbc7fbcc2b7d0b0",
        "pdf_object_number": 209,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 1280470,
            "segment_end_exclusive": 1280890,
            "declared_length": 418,
            "table_offset": 1280532,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 1280549,
            "symbols_end_exclusive": 1280711,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-4-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 1280759,
        "length": 50,
        "match_sha256": "993e4804616522490d84eb31308fdc4b49f194772df492a8a049429052b4fa95",
        "jpeg_offset": 1280228,
        "jpeg_bytes": 935,
        "jpeg_sha256": "5c250d0a31db8fe6707e7c0f8b3556887c6e273327287698fcbc7fbcc2b7d0b0",
        "pdf_object_number": 209,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 1280470,
            "segment_end_exclusive": 1280890,
            "declared_length": 418,
            "table_offset": 1280711,
            "table_class": 1,
            "table_id": 1,
            "symbol_count": 162,
            "symbols_offset": 1280728,
            "symbols_end_exclusive": 1280890,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-5-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 1281560,
        "length": 48,
        "match_sha256": "eafb29da4c7cf08980bd123649096ef3b1555f97e67cf6b80655cea55b277893",
        "jpeg_offset": 1281371,
        "jpeg_bytes": 1023,
        "jpeg_sha256": "84f35ebe01addd28e5aaaded54b09734314650ff0059913ae49a564802698728",
        "pdf_object_number": 210,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 1281477,
            "segment_end_exclusive": 1281689,
            "declared_length": 210,
            "table_offset": 1281510,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 1281527,
            "symbols_end_exclusive": 1281689,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-6-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 2923996,
        "length": 47,
        "match_sha256": "ec5e5a94c3ccbc9bdb7c2b007f07a1c7279d688cc9336d51214507a4119d61fc",
        "jpeg_offset": 2923573,
        "jpeg_bytes": 3761,
        "jpeg_sha256": "63c654e6d0dbd24ae1fef58a6874eb5542e17e462f5ce0245ac9798d7139b42a",
        "pdf_object_number": 391,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 2923883,
            "segment_end_exclusive": 2924303,
            "declared_length": 418,
            "table_offset": 2923945,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 2923962,
            "symbols_end_exclusive": 2924124,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-7-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 2924180,
        "length": 44,
        "match_sha256": "9f18a5f960aab7e9361568100b49c87a89a9ffc7e07bfefab713f9cf2d1627b5",
        "jpeg_offset": 2923573,
        "jpeg_bytes": 3761,
        "jpeg_sha256": "63c654e6d0dbd24ae1fef58a6874eb5542e17e462f5ce0245ac9798d7139b42a",
        "pdf_object_number": 391,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 2923883,
            "segment_end_exclusive": 2924303,
            "declared_length": 418,
            "table_offset": 2924124,
            "table_class": 1,
            "table_id": 1,
            "symbol_count": 162,
            "symbols_offset": 2924141,
            "symbols_end_exclusive": 2924303,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
    {
        "exception_id": "mintrans-report2023-jpeg-dht-8-20260916",
        "file_bytes": 3135381,
        "file_sha256": "25ef09e6feb255f70eeed7c2417a13ecdd7b6705a2d16d03435629704f850e20",
        "offset": 2945975,
        "length": 47,
        "match_sha256": "d0fd8f8d5b78375a1e36fc5c7cb456044c323dcf3ac6b0e33b7fabf912065e3f",
        "jpeg_offset": 2945552,
        "jpeg_bytes": 3684,
        "jpeg_sha256": "8e0b5ee236b030e2e65d4e124eddcb0697c39e0d56f151984af2991871593e7b",
        "pdf_object_number": 403,
        "pdf_filter": "/DCTDecode",
        "pdf_subtype": "/Image",
        "dht": {
            "segment_offset": 2945862,
            "segment_end_exclusive": 2946282,
            "declared_length": 418,
            "table_offset": 2945924,
            "table_class": 1,
            "table_id": 0,
            "symbol_count": 162,
            "symbols_offset": 2945941,
            "symbols_end_exclusive": 2946103,
        },
        "review": "reports/data_round5/delivery/scanner_false_positive_review.json",
    },
]

AUDITED_EXCEPTIONS = (AUDITED_EXCEPTION, *ROUND5_AUDITED_EXCEPTIONS)


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
    """Check every hit independently; never exempt an entire PDF or JPEG stream."""
    exceptions = []
    full_sha = None
    for match in SECRETS.finditer(data):
        if full_sha is None:
            full_sha = sha_bytes(data)
        candidates = [
            approved
            for approved in AUDITED_EXCEPTIONS
            if len(data) == approved["file_bytes"]
            and full_sha == approved["file_sha256"]
            and match.start() == approved["offset"]
            and match.end() - match.start() == approved["length"]
            and sha_bytes(match.group()) == approved["match_sha256"]
        ]
        if len(candidates) != 1:
            raise ValueError("Credential-like data detected; contents are not printed")
        approved = candidates[0]
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
                "policy": "block_all_matches_except_exact_audited_pdf_jpeg_dht",
                "pattern_sha256": sha_bytes(SECRETS.pattern),
                "audited_exception_ids": [item["exception_id"] for item in AUDITED_EXCEPTIONS],
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
