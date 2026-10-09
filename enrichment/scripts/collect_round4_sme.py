"""Preserve two documented SME snapshots; no historical municipality join is implied."""

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE = "https://storage.yandexcloud.net/tochno-st-catalog/FNS/data_sme_153_v20260605"
MAX_BYTES = 500_000_000


def fetch(year, out, recover=False):
    name = f"data_sme_153_year_{year}_v20260605_parquet.zip"
    target = out / "raw" / name
    if target.exists() and not recover:
        raise FileExistsError(target)
    record = {
        "year_in_filename": year,
        "url": f"{BASE}/{name}",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "publisher": "ФНС; обработка «Если быть точным»",
        "landing_url": "https://tochno.st/datasets/sme",
        "license": "CC BY 4.0 as stated on landing page",
        "geography_vintage": "June 2026 per landing page",
        "historical_municipal_join": "NOT_PERFORMED",
    }
    digest, size = hashlib.sha256(), 0
    if recover:
        probes = json.loads((out / "probe.json").read_text("utf-8"))
        probe = next(p for p in probes if p["url"] == record["url"])
        assert target.stat().st_size == int(probe["headers"]["content-length"])
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256")
        size = target.stat().st_size
        record.update(
            status="recovered_complete_download",
            bytes=size,
            sha256=digest.hexdigest(),
            head_probe=probe,
            recovery_reason="Absolute-relative path error after successful download",
            first_download_file_mtime=datetime.fromtimestamp(
                target.stat().st_mtime, UTC
            ).isoformat(),
        )
    else:
        with (
            httpx.Client(timeout=90, follow_redirects=True) as client,
            client.stream("GET", record["url"]) as response,
        ):
            response.raise_for_status()
            expected = int(response.headers["content-length"])
            if expected > MAX_BYTES:
                raise ValueError("Source exceeds documented bound")
            with target.with_suffix(".partial").open("xb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Download exceeds documented bound")
                    digest.update(chunk)
                    handle.write(chunk)
            if size != expected:
                raise ValueError("Content length mismatch")
            record.update(
                status=response.status_code,
                bytes=size,
                sha256=digest.hexdigest(),
                last_modified=response.headers.get("last-modified"),
                etag=response.headers.get("etag"),
                final_url=str(response.url),
            )
        target.with_suffix(".partial").rename(target)
    record["path"] = target.relative_to(out).as_posix()
    entries = []
    dest = (out / "extracted" / str(year)).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            final = (dest / item.filename).resolve()
            if not final.is_relative_to(dest):
                raise ValueError("Unsafe archive path")
            if final.exists() and not recover:
                raise FileExistsError(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            existing = final.exists()
            with archive.open(item) as source:
                handle = None if existing else final.open("xb")
                while block := source.read(1024 * 1024):
                    h.update(block)
                    if handle is not None:
                        handle.write(block)
                if handle is not None:
                    handle.close()
            if existing:
                with final.open("rb") as handle:
                    assert hashlib.file_digest(handle, "sha256").hexdigest() == h.hexdigest()
            assert final.stat().st_size == item.file_size
            entries.append(
                {
                    "path": final.relative_to(out).as_posix(),
                    "bytes": final.stat().st_size,
                    "sha256": h.hexdigest(),
                    "zip_crc32": item.CRC,
                }
            )
    record["entries"] = entries
    (out / f"retrieval_{year}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"year": year, "downloaded_bytes": size, "files": len(entries)}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda year: fetch(year, args.out, args.recover), [2023, 2024]))
    (args.out / "retrievals.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
