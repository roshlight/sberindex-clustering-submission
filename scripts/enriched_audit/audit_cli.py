"""Shared path handling for read-only audit reproduction."""

import argparse
from pathlib import Path


def parse_paths(description, *, with_audit=False, with_core=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--repo", type=Path, required=True, help="Repository root containing reports/"
    )
    parser.add_argument("--out", type=Path, required=True, help="Separate scratch output directory")
    if with_audit:
        parser.add_argument("--audit", type=Path, help="Input audit directory; defaults to --out")
    if with_core:
        parser.add_argument(
            "--core", type=Path, help="Core result directory; defaults to canonical core_v2"
        )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if not (args.repo / "pyproject.toml").is_file():
        parser.error("--repo must contain pyproject.toml")
    for protected in [
        args.repo / "reports",
        args.repo / "src",
        args.repo / "configs",
        args.repo / "data",
    ]:
        if args.out.is_relative_to(protected):
            parser.error(
                "--out must be a scratch directory outside reports/, src/, configs/ and data/"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    if with_audit:
        args.audit = (args.audit or args.out).resolve()
    if with_core:
        args.core = (args.core or args.repo / "reports/enriched_analysis/core_v2").resolve()
    return args
