"""Run one development/evaluation year from the locked study configuration."""

import argparse
from pathlib import Path

from municipal_profiles.study import run_study

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", required=True, choices=["development", "evaluation"])
parser.add_argument("--run-id", required=True)
parser.add_argument("--config", type=Path, default=Path("configs/robustness_v1.json"))
parser.add_argument("--lock", type=Path)
args = parser.parse_args()
run_study(Path.cwd(), args.config, args.run_id, args.phase, args.lock)
