"""Fresh inventory of official forms for the exact 488 remaining target cells."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from sber_enrichment.bdpmo import BdpmoClient
from sber_enrichment.round4_economy import parse_form, request_once_retry, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    old = pd.read_csv(
        args.repo / "reports/data_round4/economy/annual_candidate_panel.csv",
        dtype=str,
        keep_default_na=False,
    )
    missing = old.loc[old.value.eq("")]
    jobs = sorted(missing.groupby(["source_region_url", "indicator_id"]), key=lambda x: -len(x[1]))
    client = BdpmoClient(args.cache, interval=2.5, timeout=60, cache_only=args.cache_only)
    forms = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for (url, indicator), group in jobs:
            record = dict(region_url=url, indicator=indicator, target_cells=len(group))
            try:
                raw, metadata = request_once_retry(client, url + "DBInet.cgi", {"pl": indicator})
                document = raw.decode("cp1251")
                record["metadata"] = metadata
                try:
                    record["form"] = parse_form(document, indicator)
                    record["status"] = "parsed"
                except ValueError as error:
                    record.update(status="form_review", error=str(error))
                    record["tippos_html"] = re.findall(
                        r'<SELECT\s+NAME="tippos"[^>]*>(.*?)</SELECT>', document, re.I | re.S
                    )
            except Exception as error:
                record.update(status="network_error", error=f"{type(error).__name__}: {error}")
            forms.append(record)
            write_json(args.output / "form_probe.json", forms)
            print(
                json.dumps({k: record[k] for k in ["region_url", "indicator", "status"]}),
                flush=True,
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
