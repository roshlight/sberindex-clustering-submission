"""Collect a bounded, exact-code BDPMO activity pilot without changing earlier studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sber_enrichment.bdpmo import REGION_URL, BdpmoClient
from sber_enrichment.economy_bulk import DIMENSIONS, INDICATORS, YEARS, normalized, parse_form

PILOT_IDS = {331, 318, 2527, 925, 1180, 883}
MARKERS = {"", "-", "…", "...", "х", "x", "X", "к", "К"}
KEY = ("territory_id", "year", "indicator_id", "source_activity_code")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        raise ValueError("Refusing a schema-free empty table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pilot_targets(repo):
    """Preserve frozen year-specific code links; never reconstruct a link from a name."""
    path = Path(repo) / "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv"
    fields = (
        "territory_id",
        "year",
        "source_region_url",
        "source_oktmo",
        "official_oktmo_8",
        "code_reference_date",
        "source_oktmo_name",
        "temporal_review_required",
    )
    targets = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["territory_id"]) not in PILOT_IDS or row["year"] not in YEARS:
                continue
            target = {field: row[field] for field in fields}
            key = row["territory_id"], row["year"]
            if key in targets and targets[key] != target:
                raise ValueError("Conflicting frozen year-specific source links")
            targets[key] = target
    if len(targets) != 12:
        raise ValueError("All six pilot IDs need both frozen year-specific code links")
    return sorted(targets.values(), key=lambda r: (int(r["territory_id"]), r["year"])), path


def sector_payload(options, code):
    if code not in options["oktmo"]:
        raise ValueError("Municipal source code not offered by the exact form")
    labels = list(options["okved2"].values())
    if len(set(labels)) != len(labels):
        raise ValueError("Activity labels do not identify unique source option codes")
    values = {name: list(options[name]) for name in DIMENSIONS}
    years = sorted(YEARS.intersection(options["god"]))
    if not years:
        raise ValueError("No requested years")
    values.update(oktmo=[code], god=years, period=["79"])
    return {
        "Qry": ";".join(name + ":" + ",".join(values[name]) for name in DIMENSIONS) + ";",
        "QryGm": "Pokazateli_z:1;munr_z:2;tippos_z:3;oktmo_z:4;period_z:5;okved2_b:1;god_s:1;",
        "QryFootNotes": ";",
        "YearsList": ";".join(years) + ";",
        "tbl": "1",
        "Format": "CSV",
        "YearFrom": years[0],
        "YearTo": years[-1],
    }


def parse_sector_csv(document, options, code):
    """Match complete source context and labels; preserve absent rows and unresolved markers.

    Source CSV uses unescaped semicolons inside activity labels. Numeric cells are
    therefore bound from the right only after an explicit validated year header.
    The row locator counts CSV records after normalizing the source's CRCRLF.
    Internal p_okved2 values are source keys, not literal OKVED class numbers.
    """
    sector_payload(options, code)
    if "<html" in document.casefold():
        raise ValueError("HTML is not a sector CSV")
    indicator = next(iter(options["Pokazateli"]))
    indicator_label = options["Pokazateli"][indicator]
    allowed_contexts = {
        ", ".join((indicator_label, munr, tippos, options["oktmo"][code], options["period"]["79"]))
        for munr, tippos in itertools.product(options["munr"].values(), options["tippos"].values())
    }
    by_label = {label: key for key, label in options["okved2"].items()}
    header, pending, section = None, None, None
    actual, seen = [], set()
    rows = csv.reader(io.StringIO(document.replace("\r\r\n", "\n")), delimiter=";")
    for number, raw_cells in enumerate(rows, 1):
        cells = [normalized(value) for value in raw_cells]
        if not any(cells):
            continue
        first = cells[0]
        if first.startswith(indicator_label + ",") or (
            len(cells) == 1 and indicator_label.startswith(first)
        ):
            if pending is not None:
                raise ValueError("Section lacks year header")
            pending, header = normalized(";".join(raw_cells)), None
            continue
        if not first and any(cells[1:]):
            if pending not in allowed_contexts:
                raise ValueError("Complete indicator/municipality/period context not confirmed")
            if cells[-1] != "":
                raise ValueError("Year header lacks source terminator")
            header = cells[1:-1]
            if not header or len(set(header)) != len(header) or not set(header) <= YEARS:
                raise ValueError("Unexpected or duplicate year column")
            if not set(header) <= options["god"].keys():
                raise ValueError("Returned year not offered by source")
            section, pending = pending, None
            continue
        if pending is not None:
            if len(cells) != 1:
                raise ValueError("Unexpected title continuation")
            pending = normalized(pending + " " + first)
            continue
        if header is None or cells[-1] != "" or len(cells) < len(header) + 2:
            raise ValueError("Data without a validated year header")
        values = cells[-len(header) - 1 : -1]
        label = normalized(";".join(raw_cells[: -len(header) - 1]))
        if label not in by_label:
            raise ValueError(f"Unknown complete source activity label: {label}")
        for year, raw_value in zip(header, values, strict=True):
            key = by_label[label], year
            if key in seen:
                raise ValueError("Duplicate activity/year cell, including repeated sections")
            seen.add(key)
            if re.fullmatch(r"[0-9]+(?:[,.][0-9]+)?", raw_value):
                value, status = str(Decimal(raw_value.replace(",", "."))), "observed"
            elif raw_value in MARKERS:
                value, status = None, "source_marker_unresolved"
            else:
                raise ValueError(f"Unknown numerical value or marker: {raw_value}")
            actual.append(
                {
                    "source_activity_code": key[0],
                    "source_activity_label": label,
                    "year": year,
                    "value": value,
                    "raw_value": raw_value,
                    "status": status,
                    "csv_record_number": number,
                    "source_section": section,
                }
            )
    if pending is not None:
        raise ValueError("Unfinished source title")
    for activity, label in options["okved2"].items():
        for year in sorted(YEARS):
            if (activity, year) not in seen:
                actual.append(
                    {
                        "source_activity_code": activity,
                        "source_activity_label": label,
                        "year": year,
                        "value": None,
                        "raw_value": None,
                        "status": "not_returned" if year in options["god"] else "year_not_offered",
                        "csv_record_number": None,
                        "source_section": None,
                    }
                )
    return sorted(actual, key=lambda r: (r["year"], int(r["source_activity_code"])))


def run(repo, output, interval=2.5, cache_only=False):
    if interval < 2:
        raise ValueError("This collector requires at least two seconds between requests")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets, target_path = pilot_targets(repo)
    write_csv(output / "pilot_targets.csv", targets)
    client = BdpmoClient(output / "cache", interval=interval, timeout=45, cache_only=cache_only)
    forms, records, inventory = {}, [], {}
    try:
        for target in targets:
            region = target["source_region_url"]
            if not REGION_URL.fullmatch(region):
                raise ValueError("Target is not an official Rosstat municipal URL")
            endpoint, code = region.rstrip("/") + "/DBInet.cgi", target["source_oktmo"]
            for indicator in INDICATORS:
                form_key = region, indicator
                if form_key not in forms:
                    raw, meta = client.request("POST", endpoint, {"pl": indicator})
                    options = parse_form(raw.decode("cp1251"), indicator)
                    forms[form_key] = options, meta
                    inventory[meta["request_key"]] = {**meta, "purpose": "coded_indicator_form"}
                    write_json(
                        output / f"form-{region.rstrip('/').split('/')[-1]}-{indicator}.json",
                        {"options": options, "source_metadata": meta},
                    )
                options, form_meta = forms[form_key]
                raw, meta = client.request("POST", endpoint, sector_payload(options, code))
                inventory[meta["request_key"]] = {
                    **meta,
                    "purpose": "single_municipality_sector_csv",
                }
                for row in parse_sector_csv(raw.decode("cp1251"), options, code):
                    if row["year"] != target["year"]:
                        continue
                    records.append(
                        {
                            **target,
                            **row,
                            "indicator_id": indicator,
                            "indicator_label": options["Pokazateli"][indicator],
                            "unit": INDICATORS[indicator],
                            "period_code": "79",
                            "period_label": options["period"]["79"],
                            "is_total": row["source_activity_code"] == "101",
                            "classification_dimension": "okved2 (source UI)",
                            "actual_source_oktmo_label": options["oktmo"][code],
                            "source_url": meta["url"],
                            "source_sha256": meta["raw_sha256"],
                            "request_key": meta["request_key"],
                            "options_request_key": form_meta["request_key"],
                            "options_source_sha256": form_meta["raw_sha256"],
                            "retrieved_at": meta["retrieved_at"],
                        }
                    )
                print(
                    f"Collected ID={target['territory_id']} year={target['year']} "
                    f"indicator={indicator}",
                    flush=True,
                )
    finally:
        client.close()
    keys = [tuple(row[k] for k in KEY) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate output key")
    records.sort(key=lambda r: tuple(r[k] for k in KEY))
    write_csv(output / "sector_source.csv", records)
    write_json(output / "source_inventory.json", list(inventory.values()))
    qa = {
        "status": "PASS",
        "collected_at": datetime.now(UTC).isoformat(),
        "territories": len({row["territory_id"] for row in records}),
        "municipality_years": len(targets),
        "rows_including_explicit_absences": len(records),
        "unique_key": list(KEY),
        "duplicate_keys": 0,
        "status_counts": dict(Counter(row["status"] for row in records)),
        "zero_observed_cells": sum(
            row["status"] == "observed" and Decimal(row["value"]) == 0 for row in records
        ),
        "network_requests_this_invocation": client.network_request_count,
        "cache_hits_this_invocation": client.cache_hit_count,
        "unique_captured_requests": len(inventory),
        "source_files": {
            "year_specific_links": {
                "path": target_path.relative_to(Path(repo)).as_posix(),
                "sha256": sha(target_path),
            }
        },
        "source_sha256": {
            name: sha(output / name)
            for name in ("sector_source.csv", "pilot_targets.csv", "source_inventory.json")
        },
        "script_sha256": sha(__file__),
        "limits": [
            "Six outcome-selected examples, not a representative municipality sample",
            "Absent rows and source markers are not zero",
            "Workers refer to observed organizations, not employed residents",
            "Internal activity keys are not literal OKVED class codes",
            "Reference-date code evidence does not establish full-year boundary equivalence",
            "Retrieval vintage is current; first publication and revision history unknown",
        ],
    }
    write_json(output / "qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", default=2.5, type=float)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.interval, args.cache_only)


if __name__ == "__main__":
    main()
