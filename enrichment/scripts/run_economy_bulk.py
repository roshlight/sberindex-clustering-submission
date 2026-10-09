"""Collect annual total employment/wages from exact regional BDPMO source codes."""

import argparse
import json
from pathlib import Path

from sber_enrichment.economy_bulk import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["targets-path", "output", "cache"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit-regions", type=int)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"complete_with_gaps", "checkpointed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
