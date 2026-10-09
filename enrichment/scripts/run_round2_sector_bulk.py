"""Paced, checkpointed activity exports using already captured official regional forms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from run_round2_sector import sha, write_csv, write_json

from sber_enrichment.bdpmo import BdpmoClient
from sber_enrichment.economy_bulk import (
    DIMENSIONS,
    INDICATORS,
    YEARS,
    normalized,
    parse_form,
    parse_response,
)


class StartPacer:
    """Serialize only request starts; HTTP runs outside this lock."""

    def __init__(self, interval=2.5, clock=time.monotonic, sleeper=time.sleep):
        self.interval, self.clock, self.sleeper = interval, clock, sleeper
        self.next_start = 0.0
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.sleeper(max(0, self.next_start - self.clock()))
            self.next_start = self.clock() + self.interval

    def defer(self, delay):
        with self.lock:
            self.next_start = max(self.next_start, self.clock() + delay)


def payload_for(options, codes):
    if not codes or len(codes) != len(set(codes)) or not set(codes) <= options["oktmo"].keys():
        raise ValueError("Invalid exact municipal source codes")
    counts = Counter(options["oktmo"].values())
    if any(counts[options["oktmo"][code]] != 1 for code in codes):
        raise ValueError("Batch captions must be unique in the complete regional form")
    values = {name: list(options[name]) for name in DIMENSIONS}
    years = sorted(YEARS.intersection(options["god"]))
    values.update(oktmo=codes, god=years, period=["79"])
    return {
        "Qry": ";".join(name + ":" + ",".join(values[name]) for name in DIMENSIONS) + ";",
        "QryGm": "Pokazateli_z:1;munr_z:2;tippos_z:3;okved2_z:4;period_z:5;oktmo_b:1;god_s:1;",
        "QryFootNotes": ";",
        "YearsList": ";".join(years) + ";",
        "tbl": "1",
        "Format": "CSV",
        "YearFrom": years[0],
        "YearTo": years[-1],
    }


def parse_batch(document, options, codes):
    """Audit each complete activity section, then reuse the strict numeric/municipal parser."""
    payload_for(options, codes)
    if "<html" in document.casefold():
        raise ValueError("HTML is not a CSV")
    indicator_label = next(iter(options["Pokazateli"].values()))
    contexts = {}
    for (activity, label), munr, tippos in itertools.product(
        options["okved2"].items(), options["munr"].values(), options["tippos"].values()
    ):
        context = ", ".join((indicator_label, munr, tippos, label, options["period"]["79"]))
        if context in contexts and contexts[context] != activity:
            raise ValueError("Ambiguous source activity section")
        contexts[context] = activity
    lines = document.replace("\r\r\n", "\n").splitlines()
    starts = [
        i
        for i, line in enumerate(lines)
        if normalized(line)
        and (
            normalized(line).startswith(indicator_label + ",")
            or indicator_label.startswith(normalized(line))
        )
    ]
    if any(line.strip() for line in lines[: starts[0] if starts else len(lines)]):
        raise ValueError("Unattributed text before the source section")
    records, seen = [], set()
    for start, stop in zip(starts, [*starts[1:], len(lines)], strict=True):
        block = lines[start:stop]
        header = next((i for i, line in enumerate(block) if line.startswith(";")), None)
        if header is None:
            raise ValueError("Activity section lacks explicit year header")
        context = normalized(" ".join(block[:header]))
        if context not in contexts:
            raise ValueError("Full indicator/parent/type/activity/period context not confirmed")
        activity = contexts[context]
        local_options = {**options, "okved2": {"101": options["okved2"][activity]}}
        # Context was matched in full above. The generic parser expects one title
        # record; collapse only these verified title lines, preserving row locators.
        checked_section = "\n".join([context, *block[header:]])
        for row in parse_response(checked_section, local_options, codes):
            key = row["source_oktmo"], activity, row["year"]
            if key in seen:
                raise ValueError("Duplicate municipality/activity/year across source sections")
            seen.add(key)
            row.update(
                okved2_code=activity, source_csv_row=row["source_csv_row"] + start + header - 1
            )
            records.append(row)
    indicator = next(iter(options["Pokazateli"]))
    for code, activity, year in itertools.product(codes, options["okved2"], sorted(YEARS)):
        if (code, activity, year) in seen:
            continue
        records.append(
            {
                "source_oktmo": code,
                "source_oktmo_name": options["oktmo"][code],
                "indicator_id": indicator,
                "year": year,
                "period_code": "79",
                "period_label": options["period"]["79"],
                "period_start": f"{year}-01-01",
                "period_end": f"{year}-12-31",
                "okved2_code": activity,
                "okved2_label": options["okved2"][activity],
                "value": None,
                "value_raw": None,
                "value_status": "not_returned" if year in options["god"] else "year_not_offered",
                "unit": INDICATORS[indicator],
                "source_unit_label": indicator_label.rsplit(", ", 1)[-1],
                "source_csv_row": None,
                "source_section": None,
            }
        )
    return records


def prepare(repo, archive_path, output, pilot):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    targets, target_rows = {}, []
    input_path = repo / "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv"
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
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            link = {field: row[field] for field in fields}
            key = row["territory_id"], row["year"]
            if key in targets and targets[key] != link:
                raise ValueError("Conflicting original source links")
            targets[key] = link
    target_rows = sorted(targets.values(), key=lambda r: (int(r["territory_id"]), r["year"]))
    if len(target_rows) != 4032:
        raise ValueError("Expected frozen 2016-municipality, two-year links")
    write_csv(output / "reference_links.csv", target_rows)
    by_region = defaultdict(set)
    for row in target_rows:
        by_region[row["source_region_url"]].add(row["source_oktmo"])
    inventory_path = repo / "reports/enrichment_v3/economy/source_capture_inventory.json"
    captures = json.loads(inventory_path.read_text(encoding="utf-8"))["captures"]
    forms = {}
    with zipfile.ZipFile(archive_path) if archive_path else nullcontext() as archive:
        entries = json.loads(archive.read("ARCHIVE_MANIFEST.json"))["entries"] if archive else []
        by_sha = {entry["sha256"]: entry["archive_path"] for entry in entries}
        for capture in captures:
            if set(capture.get("payload", {})) != {"pl"}:
                continue
            region = capture["url"].removesuffix("DBInet.cgi")
            if region not in by_region:
                continue
            indicator, key = capture["payload"]["pl"], capture["request_key"]
            selected_cache = next(
                (
                    base / "cache"
                    for base in (pilot, output)
                    if base
                    and (base / "cache" / f"{key}.json").exists()
                    and (base / "cache" / f"{key}.raw").exists()
                ),
                None,
            )
            if selected_cache is not None:
                metadata = json.loads((selected_cache / f"{key}.json").read_text(encoding="utf-8"))
                raw = (selected_cache / f"{key}.raw").read_bytes()
                if any(metadata[field] != capture[field] for field in ("url", "method", "payload")):
                    raise ValueError("Saved coded form is not the expected official request")
            elif archive:
                raw = archive.read(by_sha[capture["raw_sha256"]])
                metadata = capture
            else:
                raise ValueError(f"Missing saved coded form {key}; supply the original archive")
            if hashlib.sha256(raw).hexdigest() != metadata["raw_sha256"]:
                raise ValueError("Saved official form hash mismatch")
            (cache / f"{key}.raw").write_bytes(raw)
            write_json(cache / f"{key}.json", metadata)
            options = parse_form(raw.decode("cp1251"), indicator)
            forms[region, indicator] = options, metadata
    jobs, queue, activities = [], [], []
    for region, codes in sorted(by_region.items()):
        for indicator in INDICATORS:
            if (region, indicator) not in forms:
                queue.append({"region": region, "indicator": indicator, "reason": "missing_form"})
                continue
            options, meta = forms[region, indicator]
            counts = Counter(options["oktmo"].values())
            eligible = []
            for code in sorted(codes):
                label = options["oktmo"].get(code)
                reason = (
                    "code_not_offered"
                    if label is None
                    else (
                        "ambiguous_or_truncated_full_form_label"
                        if counts[label] != 1 or "..." in label or "…" in label
                        else None
                    )
                )
                if reason:
                    queue.append(
                        {
                            "region": region,
                            "indicator": indicator,
                            "source_oktmo": code,
                            "reason": reason,
                        }
                    )
                else:
                    eligible.append(code)
            for activity, label in options["okved2"].items():
                activities.append(
                    {
                        "source_region_url": region,
                        "indicator_id": indicator,
                        "source_activity_code": activity,
                        "source_activity_label": label,
                        "selected": True,
                        "options_sha256": meta["raw_sha256"],
                    }
                )
            for start in range(0, len(eligible), 100):
                group = eligible[start : start + 100]
                jobs.append(
                    {
                        "job_id": f"{region.rstrip('/').split('/')[-1]}-{indicator}-{start // 100}",
                        "region": region,
                        "indicator": indicator,
                        "codes": group,
                        "options_request_key": meta["request_key"],
                        "options_sha256": meta["raw_sha256"],
                        "payload": payload_for(options, group),
                    }
                )
    if len(jobs) > 250:
        raise ValueError("Plan exceeds the explicit request budget")
    write_json(output / "jobs.json", jobs)
    write_json(output / "deferred_targets.json", queue)
    write_csv(output / "activity_options.csv", activities)
    write_json(
        output / "input_manifest.json",
        {
            "frozen_links_sha256": sha(input_path),
            "capture_inventory_sha256": sha(inventory_path),
            "archive_sha256": sha(archive_path) if archive_path else None,
            "all_form_bytes_available_in_output_cache": True,
            "planned_jobs": len(jobs),
            "deferred_targets": len(queue),
            "reference_links": len(target_rows),
            "regions": len(by_region),
            "scripts": {
                Path(__file__).name: sha(__file__),
                "run_round2_sector.py": sha(Path(__file__).with_name("run_round2_sector.py")),
            },
        },
    )
    return jobs, forms, target_rows


def run(args):
    if args.workers not in (1, 2):
        raise ValueError("Only one or two concurrent regions are permitted")
    output = args.output.resolve()
    jobs, forms, links = prepare(
        args.repo.resolve(), args.archive.resolve() if args.archive else None, output, args.pilot
    )
    if args.prepare_only:
        print(json.dumps({"jobs": len(jobs), "status": "PREPARED"}))
        return
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    statuses, clients = [], []
    status_lock = threading.Lock()
    pacer = StartPacer()
    regional_jobs = defaultdict(list)
    for job in jobs[: args.limit_jobs]:
        regional_jobs[job["region"]].append(job)

    def process_region(region_jobs):
        client = BdpmoClient(output / "cache", interval=0, timeout=45, cache_only=args.cache_only)
        with status_lock:
            clients.append(client)
        for job in region_jobs:
            checkpoint = checkpoints / f"{job['job_id']}.json"
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
                raw = (
                    output / "cache" / f"{result['source_metadata']['request_key']}.raw"
                ).read_bytes()
                if hashlib.sha256(raw).hexdigest() != result["source_metadata"]["raw_sha256"]:
                    raise ValueError("Checkpoint raw hash mismatch")
            else:
                try:
                    request = {
                        "method": "POST",
                        "url": job["region"] + "DBInet.cgi",
                        "payload": job["payload"],
                    }
                    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                    meta_path = output / "cache" / f"{key}.json"
                    cached = (
                        meta_path.exists()
                        and json.loads(meta_path.read_text(encoding="utf-8"))["status_code"] == 200
                    )
                    if not cached:
                        pacer.start()
                    raw, meta = client.request("POST", job["region"] + "DBInet.cgi", job["payload"])
                    options, _ = forms[job["region"], job["indicator"]]
                    records = parse_batch(raw.decode("cp1251"), options, job["codes"])
                    result = {"job": job, "source_metadata": meta, "records": records}
                    write_json(checkpoint, result)
                except Exception as error:
                    pacer.defer(30)
                    with status_lock:
                        statuses.append(
                            {
                                "job_id": job["job_id"],
                                "status": "deferred",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        write_json(output / "progress.pending.json", statuses)
                        (output / "progress.pending.json").replace(output / "progress.json")
                    print(f"DEFERRED {job['job_id']}: {type(error).__name__}: {error}", flush=True)
                    continue
            with status_lock:
                statuses.append(
                    {
                        "job_id": job["job_id"],
                        "status": "parsed",
                        "rows": len(result["records"]),
                        "source_sha256": result["source_metadata"]["raw_sha256"],
                    }
                )
                write_json(output / "progress.pending.json", statuses)
                (output / "progress.pending.json").replace(output / "progress.json")
                print(f"Parsed {len(statuses)}/{len(jobs)} {job['job_id']}", flush=True)
        client.close()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(process_region, regional_jobs.values()):
                pass
    finally:
        for client in clients:
            client.close()
    references = defaultdict(list)
    for link in links:
        references[link["source_region_url"], link["source_oktmo"], link["year"]].append(link)
    rows, panel, inventory, seen = [], [], {}, set()
    for _, meta in forms.values():
        inventory[meta["request_key"]] = {**meta, "purpose": "coded_indicator_form"}
    for checkpoint in sorted(checkpoints.glob("*.json")):
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        job, meta = result["job"], result["source_metadata"]
        inventory[meta["request_key"]] = {**meta, "purpose": "municipality_batch_sector_csv"}
        for record in result["records"]:
            row = {
                **record,
                "source_region_url": job["region"],
                "source_url": meta["url"],
                "source_sha256": meta["raw_sha256"],
                "request_key": meta["request_key"],
                "retrieved_at": meta["retrieved_at"],
                "options_request_key": job["options_request_key"],
                "options_sha256": job["options_sha256"],
            }
            key = (
                job["region"],
                row["source_oktmo"],
                row["indicator_id"],
                row["year"],
                row["okved2_code"],
            )
            if key in seen:
                raise ValueError("Duplicate full source key across checkpoints")
            seen.add(key)
            rows.append(row)
            for link in references[job["region"], row["source_oktmo"], row["year"]]:
                panel.append({**row, **link})
    if rows:
        write_csv(output / "sector_source.csv", rows)
        write_csv(output / "annual_candidate_sector.csv", panel)
    write_json(output / "source_inventory.json", list(inventory.values()))
    qa = {
        "status": "PASS"
        if len(statuses) == len(jobs) and all(row["status"] == "parsed" for row in statuses)
        else "PARTIAL",
        "planned_jobs": len(jobs),
        "parsed_jobs": len(list(checkpoints.glob("*.json"))),
        "network_requests_this_invocation": sum(c.network_request_count for c in clients),
        "max_concurrent_regions": args.workers,
        "global_min_start_interval_seconds": 2.5,
        "raw_source_rows": len(rows),
        "candidate_rows": len(panel),
        "duplicate_source_keys": 0,
        "candidate_territories": len({r["territory_id"] for r in panel}),
        "candidate_municipality_years": len({(r["territory_id"], r["year"]) for r in panel}),
        "source_status_counts": dict(Counter(r["value_status"] for r in rows)),
        "candidate_status_counts": dict(Counter(r["value_status"] for r in panel)),
        "script_sha256": sha(__file__),
    }
    write_json(output / "qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--limit-jobs", type=int, default=250)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    run(parser.parse_args())
