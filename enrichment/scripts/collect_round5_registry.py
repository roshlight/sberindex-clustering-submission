"""Fetch the two official classifier versions bracketing June 2026 geocoding."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sber_enrichment.clients import PublicFetcher

BASE = "https://rosstat.gov.ru/opendata/7708234640-oktmo"
SOURCES = {
    "catalog.html": BASE,
    "registry_2026-06-01.csv": BASE + "/data-20260601T1406-structure-20260210T1102.csv",
    "registry_2026-07-01.csv": BASE + "/data-20260701T0007-structure-20260210T1102.csv",
}


def collect(out):
    if out.exists():
        raise FileExistsError("Choose a new download directory")
    ca_path = os.environ.get("PUBLIC_CA_BUNDLES_JSON")
    ca_bundles = json.loads(Path(ca_path).read_text("utf-8")) if ca_path else None
    out.mkdir(parents=True)
    fetcher = PublicFetcher(
        timeout=120,
        max_bytes=32_000_000,
        ca_bundles=ca_bundles,
    )
    rows = []
    for filename, url in SOURCES.items():
        result = fetcher.fetch(url)
        content = result.pop("body")
        result.update(
            retrieved_at=datetime.now(UTC).isoformat(),
            path=filename if content else None,
            bytes=len(content),
            origin="new_network_retrieval",
        )
        if content:
            (out / filename).write_bytes(content)
        rows.append(result)
        (out / "retrievals.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), "utf-8"
        )
        print(json.dumps({k: result[k] for k in ["path", "status", "bytes", "error"]}), flush=True)
    if any(row["error"] or row["status"] != 200 for row in rows):
        raise RuntimeError("One or more registry sources were not downloaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args().out)
