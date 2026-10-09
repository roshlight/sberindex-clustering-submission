"""Run the exploratory municipality-year enrichment analysis."""

import argparse
import json
from pathlib import Path

from municipal_profiles.enriched_analysis import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.out), ensure_ascii=False))
