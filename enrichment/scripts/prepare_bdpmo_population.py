"""Fetch declared source OKTMO targets, preserving every official request/response."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from sber_enrichment import bdpmo
from sber_enrichment.bdpmo import BdpmoClient


def prepare(
    targets_path: Path,
    output: Path,
    cache: Path,
    interval: float = 0.5,
    limit=None,
    ca_bundle: Path | None = None,
):
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    planned_targets = len(targets)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        targets = targets[:limit]
    keys = [(item["region_url"], item["oktmo"]) for item in targets]
    if not targets or len(keys) != len(set(keys)):
        raise ValueError("Targets must be nonempty and unique by region_url and oktmo")
    if any(not isinstance(item["oktmo"], str) for item in targets):
        raise ValueError("Source OKTMO codes must be explicit strings, preserving leading zeros")
    if output.exists():
        raise FileExistsError("Use a new output directory; the response cache is reusable")
    output.mkdir(parents=True, exist_ok=False)
    client = BdpmoClient(cache, interval=interval, ca_bundle=ca_bundle)
    rows, errors = [], []
    manifest = {
        "status": "running",
        "target_count": len(targets),
        "planned_target_count": planned_targets,
        "limit": limit,
        "ca_bundle": str(ca_bundle.resolve()) if ca_bundle is not None else None,
        "ca_bundle_sha256": hashlib.sha256(ca_bundle.read_bytes()).hexdigest()
        if ca_bundle is not None
        else None,
        "targets_sha256": hashlib.sha256(targets_path.read_bytes()).hexdigest(),
        "targets": targets,
        "raw_cache": str(cache.resolve()),
        "processor_sha256": hashlib.sha256(Path(bdpmo.__file__).read_bytes()).hexdigest(),
        "scope": "Source OKTMO values only; no automatic OKTMO length conversion or SberIndex join",
    }
    manifest_path = output / "bdpmo_manifest.json"
    try:
        for target in targets:
            try:
                result = client.population(target["region_url"], target["oktmo"])
                rows.extend(result)
                print(
                    json.dumps(
                        {"target": target, "status": "complete", "cells": len(result)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as error:
                errors.append({**target, "error": f"{type(error).__name__}: {error}"})
                print(
                    json.dumps(
                        {"target": target, "status": "failed", "error": str(error)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            manifest.update(completed_targets=len(rows) // 4, errors=errors)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["value"] = frame["value"].astype("Int64")
            frame["territory_id"] = frame["territory_id"].astype("Int64")
            unique_key = ["source_region_url", "source_oktmo", "reference_date", "population_type"]
            if frame.duplicated(unique_key).any():
                raise ValueError("Duplicate extracted source observation keys")
            frame.to_csv(output / "population_source_oktmo.csv", index=False, encoding="utf-8")
            with duckdb.connect() as connection:
                connection.register("population_rows", frame)
                connection.execute(
                    "COPY population_rows TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                    [str(output / "population_source_oktmo.parquet")],
                )
        manifest.update(
            status="partial" if errors else "complete_unlinked",
            n_rows=len(frame),
            errors=errors,
            output_hashes={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir())
                if path != manifest_path
            },
        )
    except BaseException as error:
        manifest.update(status="failed", fatal_error=f"{type(error).__name__}: {error}")
        raise
    finally:
        client.close()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", required=True, type=Path, help="JSON array of region_url and exact oktmo"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Maximum number of targets to request")
    parser.add_argument(
        "--ca-bundle", type=Path, help="Additional trusted public CA certificates (PEM)"
    )
    args = parser.parse_args(argv)
    result = prepare(
        args.targets, args.output, args.cache, args.interval, args.limit, args.ca_bundle
    )
    return 0 if result["status"] == "complete_unlinked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
