"""Collect official source-code population cells with durable target checkpoints."""

import argparse
import json
from pathlib import Path

from sber_enrichment.population_bulk import run_bulk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("targets-path", "output", "cache"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    result = run_bulk(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"complete", "complete_with_gaps", "checkpointed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
