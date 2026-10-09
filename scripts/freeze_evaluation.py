"""Freeze a completed development implementation before evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from municipal_profiles.study import code_hash, file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--development-run", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("configs/evaluation_lock.json"))
args = parser.parse_args()
manifest = json.loads((args.development_run / "run_manifest.json").read_text(encoding="utf-8"))
if manifest["phase"] != "development" or manifest["status"] != "complete":
    raise ValueError("A completed development run is required")
if code_hash(Path.cwd()) != manifest["code_sha256"]:
    raise ValueError("Code changed since development; rerun or document a new experiment")
if file_hash(Path("configs/robustness_v1.json")) != manifest["config_sha256"]:
    raise ValueError("Configuration changed since development")
if file_hash(Path("data/raw/consumption.parquet")) != manifest["data_sha256"]:
    raise ValueError("Source data changed since development")
lock = {key: manifest[key] for key in ["code_sha256", "config_sha256", "data_sha256"]}
lock.update(
    development_run=manifest["run_id"],
    development_status="complete",
    frozen_at=pd.Timestamp.now(tz="UTC").isoformat(),
)
with args.output.open("x", encoding="utf-8") as handle:
    json.dump(lock, handle, indent=2)
print(json.dumps({"lock_written": str(args.output)}))
