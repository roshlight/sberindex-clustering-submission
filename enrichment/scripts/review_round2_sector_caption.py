"""Two exact-code captures for ID2710; numbers require a capture-specific caption decision."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from run_round2_sector import parse_sector_csv, sector_payload, write_csv, write_json

from sber_enrichment.bdpmo import BdpmoClient
from sber_enrichment.economy_bulk import INDICATORS, parse_form

REGION = "https://rosstat.gov.ru/dbscripts/munst/munst28/"
CODE = "28714000"


def apply_caption_decision(options, body, decision, form_sha, indicator):
    if decision["source_oktmo"] != CODE or decision["indicator_id"] != indicator:
        raise ValueError("Decision has the wrong municipal code or indicator")
    if decision["source_csv_sha256"] != hashlib.sha256(body).hexdigest():
        raise ValueError("Decision is not bound to the exact CSV capture")
    if decision["source_form_sha256"] != form_sha:
        raise ValueError("Decision is not bound to the exact coded form")
    if decision["source_option_caption"] != options["oktmo"][CODE]:
        raise ValueError("Exact original option caption differs")
    if decision["decision"] != "accept_exact_single_code_and_reviewed_full_caption":
        raise ValueError("Decision does not explicitly accept the reviewed caption")
    approved = decision["approved_csv_municipality_caption"]
    if not approved or "..." in approved or "…" in approved:
        raise ValueError("An untruncated complete reviewed caption is required")
    reviewed = copy.deepcopy(options)
    reviewed["oktmo"][CODE] = approved
    return parse_sector_csv(body.decode("cp1251"), reviewed, CODE)


def run(bulk_dir, out, decisions=None):
    out.mkdir(parents=True, exist_ok=True)
    links = []
    with (bulk_dir / "reference_links.csv").open(encoding="utf-8", newline="") as stream:
        links = [row for row in csv.DictReader(stream) if row["territory_id"] == "2710"]
    if len(links) != 2 or any(row["source_oktmo"] != CODE for row in links):
        raise ValueError("Expected frozen year-specific ID2710 links")
    jobs = json.loads((bulk_dir / "jobs.json").read_text(encoding="utf-8"))
    client = BdpmoClient(out / "cache", interval=2.5, timeout=45, cache_only=decisions is not None)
    inventory, candidates, records = [], [], []
    try:
        for indicator in INDICATORS:
            job = next(j for j in jobs if j["region"] == REGION and j["indicator"] == indicator)
            form_key = job["options_request_key"]
            raw_form = (bulk_dir / "cache" / f"{form_key}.raw").read_bytes()
            form_meta = json.loads(
                (bulk_dir / "cache" / f"{form_key}.json").read_text(encoding="utf-8")
            )
            if hashlib.sha256(raw_form).hexdigest() != job["options_sha256"]:
                raise ValueError("Official source form hash mismatch")
            (out / "cache" / f"{form_key}.raw").write_bytes(raw_form)
            write_json(out / "cache" / f"{form_key}.json", form_meta)
            options = parse_form(raw_form.decode("cp1251"), indicator)
            raw, meta = client.request("POST", REGION + "DBInet.cgi", sector_payload(options, CODE))
            inventory.extend(
                [
                    {**form_meta, "purpose": "coded_indicator_form"},
                    {**meta, "purpose": "single_code_caption_review"},
                ]
            )
            text = raw.decode("cp1251")
            title = text.replace("\r\r\n", "\n").split(";2023;2024;", 1)[0].strip()
            candidates.append(
                {
                    "territory_id": 2710,
                    "source_oktmo": CODE,
                    "indicator_id": indicator,
                    "source_option_caption": options["oktmo"][CODE],
                    "source_form_sha256": job["options_sha256"],
                    "source_csv_sha256": meta["raw_sha256"],
                    "source_url": meta["url"],
                    "raw_csv_title": title,
                    "decision": "review_required",
                }
            )
            if decisions is None:
                continue
            decision = next(d for d in decisions if d["indicator_id"] == indicator)
            cells = apply_caption_decision(options, raw, decision, job["options_sha256"], indicator)
            for row in cells:
                link = next(link for link in links if link["year"] == row["year"])
                records.append(
                    {
                        **link,
                        **row,
                        "indicator_id": indicator,
                        "indicator_label": options["Pokazateli"][indicator],
                        "unit": INDICATORS[indicator],
                        "period_code": "79",
                        "period_label": options["period"]["79"],
                        "is_total": row["source_activity_code"] == "101",
                        "classification_dimension": "okved2 (source UI)",
                        "actual_source_oktmo_label": decision["approved_csv_municipality_caption"],
                        "source_url": meta["url"],
                        "source_sha256": meta["raw_sha256"],
                        "request_key": meta["request_key"],
                        "options_request_key": form_key,
                        "options_source_sha256": job["options_sha256"],
                        "retrieved_at": meta["retrieved_at"],
                        "caption_decision": decision["decision"],
                    }
                )
    finally:
        client.close()
    write_json(out / "source_inventory.json", inventory)
    write_json(out / "caption_review_requests.json", candidates)
    if records:
        write_csv(out / "sector_source.csv", records)
        write_json(out / "accepted_caption_decisions.json", decisions)
    qa = {
        "status": "ACCEPTED_CAPTION" if records else "REVIEW_REQUIRED",
        "rows": len(records),
        "observed": sum(row["status"] == "observed" for row in records),
        "network_requests_this_invocation": client.network_request_count,
        "geographic_equivalence_confirmed": None,
        "scope": "Only these indicator/form/CSV hashes, exact code, and complete caption",
    }
    write_json(out / "qa.json", qa)
    print(json.dumps({"qa": qa, "review_requests": candidates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    decisions = json.loads(args.decision.read_text(encoding="utf-8")) if args.decision else None
    run(args.bulk_dir.resolve(), args.out.resolve(), decisions)
