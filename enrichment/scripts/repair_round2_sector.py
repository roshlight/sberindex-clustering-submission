"""Parse saved CSV jobs without network access; preserve successful earlier checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_round2_sector import sha, write_json
from run_round2_sector_bulk import parse_batch

from sber_enrichment.economy_bulk import parse_form


def repair(output, check_all=False):
    records = []
    deferred = {
        row["job_id"]
        for row in json.loads((output / "progress.json").read_text(encoding="utf-8"))
        if row["status"] == "deferred"
    }
    for job in json.loads((output / "jobs.json").read_text(encoding="utf-8")):
        checkpoint = output / "checkpoints" / f"{job['job_id']}.json"
        if not check_all and (checkpoint.exists() or job["job_id"] not in deferred):
            continue
        if check_all and not checkpoint.exists():
            continue
        request = {"method": "POST", "url": job["region"] + "DBInet.cgi", "payload": job["payload"]}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        metadata_path = output / "cache" / f"{key}.json"
        if not metadata_path.exists():
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta["status_code"] != 200:
            continue
        raw = (output / "cache" / f"{key}.raw").read_bytes()
        form_raw = (output / "cache" / f"{job['options_request_key']}.raw").read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["raw_sha256"]:
            raise ValueError("Cached CSV hash mismatch")
        if hashlib.sha256(form_raw).hexdigest() != job["options_sha256"]:
            raise ValueError("Cached source form hash mismatch")
        options = parse_form(form_raw.decode("cp1251"), job["indicator"])
        cells = parse_batch(raw.decode("cp1251"), options, job["codes"])
        if check_all:
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))["records"]
            if previous != cells:
                raise ValueError(f"Fresh parse differs from checkpoint: {job['job_id']}")
        else:
            write_json(checkpoint, {"job": job, "source_metadata": meta, "records": cells})
        records.append(
            {
                "job_id": job["job_id"],
                "cells": len(cells),
                "source_sha256": meta["raw_sha256"],
                "checkpoint_sha256": sha(checkpoint),
            }
        )
    result = {
        "network_requests": 0,
        "mode": "reparse_compare" if check_all else "repair_deferred",
        "repaired": records,
        "script_sha256": sha(__file__),
        "parser_sha256": sha(Path(__file__).with_name("run_round2_sector_bulk.py")),
    }
    # New timestamp-free hash name keeps repeated audits and their evidence separate.
    signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    write_json(output / f"offline-repair-{signature}.json", result)
    print(
        json.dumps(
            {
                "network_requests": 0,
                "mode": result["mode"],
                "checked_jobs": len(records),
                "manifest": f"offline-repair-{signature}.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-dir", type=Path, required=True)
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()
    repair(args.bulk_dir.resolve(), args.check_all)
