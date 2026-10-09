"""Bounded, TLS-verified download of dated municipal-geography source evidence."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sber_enrichment.clients import PublicFetcher

SOURCES = {
    "rosstat_classification.html": "https://rosstat.gov.ru/classification",
    "fias_download_catalog.json": "https://fias.nalog.ru/WebServices/Public/GetAllDownloadFileInfo",
    "fias_updates.html": "https://fias.nalog.ru/Updates",
}


def collect(out, sources, timeout=35):
    if out.exists():
        raise FileExistsError("Choose a fresh output directory; original responses are immutable")
    out.mkdir(parents=True)
    ca_path = os.environ.get("PUBLIC_CA_BUNDLES_JSON")
    ca_bundles = json.loads(Path(ca_path).read_text("utf-8")) if ca_path else None
    fetcher = PublicFetcher(timeout=timeout, max_bytes=40_000_000, ca_bundles=ca_bundles)
    rows = []
    for filename, url in sources.items():
        if Path(filename).name != filename:
            raise ValueError("Use a plain output filename")
        result = fetcher.fetch(url)
        body = result.pop("body")
        result.update(
            retrieved_at=datetime.now(UTC).isoformat(),
            bytes=len(body),
            path=filename if body else None,
            requested_filename=filename,
        )
        if body:
            (out / filename).write_bytes(body)
        rows.append(result)
        (out / "retrievals.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({k: result.get(k) for k in ["path", "status", "bytes", "error"]}), flush=True
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--timeout", type=float, default=35)
    args = parser.parse_args()
    collect(
        args.out,
        json.loads(args.sources.read_text("utf-8")) if args.sources else SOURCES,
        args.timeout,
    )
