"""Isolated annual BDPMO employment/wage collector, with exact source-code evidence."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from . import bdpmo, population_bulk
from .bdpmo import REGION_URL, BdpmoClient
from .population_bulk import NetworkPause, SourceUnavailable, atomic_json, with_backoff

INDICATORS = {"8423005": "человек", "8423007": "рубль"}
DIMENSIONS = ("Pokazateli", "munr", "tippos", "oktmo", "okved2", "god", "period")
YEARS = {"2023", "2024"}


def normalized(value):
    return " ".join(html.unescape(value).split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_form(document, indicator):
    if indicator not in INDICATORS:
        raise ValueError("Unsupported economy indicator")
    result = {}
    for name in DIMENSIONS:
        codes = re.findall(r"p_" + name + r'\[([0-9]+)\]="([^"\r\n]+)"', document)
        selected = re.search(
            r'<SELECT\s+NAME="' + name + r'"[^>]*>(.*?)</SELECT>', document, re.I | re.S
        )
        if not selected or not codes:
            raise ValueError(f"Missing coded source dimension: {name}")
        labels = re.findall(r"<OPTION\b[^>]*>([^<]*)", selected.group(1), re.I)
        codes = sorted((int(i), code) for i, code in codes)
        if [i for i, _ in codes] != list(range(len(labels))):
            raise ValueError(f"Code/label index mismatch: {name}")
        by_code = {}
        for (_, code), label in zip(codes, labels, strict=True):
            label = normalized(label)
            if code in by_code and by_code[code] != label:
                raise ValueError(f"Conflicting repeated source code: {name}")
            by_code[code] = label
        result[name] = by_code
    if set(result["Pokazateli"]) != {indicator}:
        raise ValueError("Response indicator differs from request")
    units = {"8423005": {"человек"}, "8423007": {"рубль", "рублей"}}
    if result["Pokazateli"][indicator].rsplit(", ", 1)[-1] not in units[indicator]:
        raise ValueError("Exact source indicator units not confirmed")
    if result["period"].get("79", "").casefold() not in {"январь-декабрь", "за январь-декабрь"}:
        raise ValueError("Period 79 is not explicitly January–December")
    aggregate = "всего по обследуемым видам экономической деятельности"
    if result["okved2"].get("101", "").casefold() not in {aggregate, aggregate + " (101.аг)"}:
        raise ValueError("Activity 101 aggregate meaning not confirmed")
    if not YEARS.intersection(result["god"]):
        raise ValueError("No requested years in source form")
    return result


def payload_for(options, codes):
    if not codes or len(set(codes)) != len(codes) or not set(codes) <= options["oktmo"].keys():
        raise ValueError("Codes must exactly match unique source options")
    labels = list(options["oktmo"].values())
    if len(codes) > 1 and any(labels.count(options["oktmo"][code]) != 1 for code in codes):
        raise ValueError("Batch names must be unique in the entire regional form")
    values = {name: list(options[name]) for name in DIMENSIONS}
    years = sorted(YEARS.intersection(options["god"]))
    values.update(oktmo=codes, okved2=["101"], god=years, period=["79"])
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


def parse_response(document, options, codes):
    """Bind rightmost year cells to exact complete source labels, never their prefix."""
    payload_for(options, codes)
    if "<html" in document.casefold():
        raise ValueError("HTML returned instead of an economy CSV")
    indicator = next(iter(options["Pokazateli"]))
    indicator_label = options["Pokazateli"][indicator]
    suffix = ", " + options["okved2"]["101"] + ", " + options["period"]["79"]
    labels = {options["oktmo"][code]: code for code in codes}
    header, section, pending = None, None, None
    records, seen = [], set()
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
                raise ValueError("Section title lacks an explicit year header")
            pending, header = normalized(";".join(raw_cells)), None
            continue
        if not first and any(cells[1:]):
            if (
                pending is None
                or not pending.startswith(indicator_label + ",")
                or not pending.endswith(suffix)
            ):
                raise ValueError("Full source indicator/activity/period context not confirmed")
            if cells[-1] != "":
                raise ValueError("Unexpected year header terminator")
            header = cells[1:-1]
            if not header or len(set(header)) != len(header) or not set(header) <= YEARS:
                raise ValueError("Unknown, blank or duplicate year column")
            if not set(header) <= options["god"].keys():
                raise ValueError("CSV contains a year not offered by the source form")
            section, pending = pending, None
            continue
        if pending is not None:
            if len(cells) != 1:
                raise ValueError("Unknown row before the explicit year header")
            pending = normalized(pending + " " + first)
            continue
        if header is None or cells[-1] != "" or len(cells) < len(header) + 2:
            raise ValueError("Data row has no valid explicit year context")
        values = cells[-len(header) - 1 : -1]
        label = normalized(";".join(raw_cells[: -len(header) - 1]))
        if label not in labels:
            raise ValueError("Unknown or truncated municipality caption; attribution rejected")
        for year, raw in zip(header, values, strict=True):
            code = labels[label]
            key = (code, year)
            if key in seen:
                raise ValueError("Duplicate municipality/year cell")
            seen.add(key)
            compact = raw.replace(" ", "")
            if re.fullmatch(r"[0-9]+(?:[,.][0-9]+)?", compact):
                value = format(Decimal(compact.replace(",", ".")), "f")
                status = "observed"
            elif raw in {"", "-", "…", "...", "х", "x", "X", "к", "К"}:
                value, status = None, "unresolved_source_marker"
            else:
                raise ValueError(f"Unrecognized source numeric cell: {raw!r}")
            records.append(
                {
                    "source_oktmo": code,
                    "source_oktmo_name": label,
                    "indicator_id": indicator,
                    "year": year,
                    "period_code": "79",
                    "period_label": options["period"]["79"],
                    "period_start": f"{year}-01-01",
                    "period_end": f"{year}-12-31",
                    "okved2_code": "101",
                    "okved2_label": options["okved2"]["101"],
                    "value": value,
                    "value_raw": raw,
                    "value_status": status,
                    "unit": INDICATORS[indicator],
                    "source_unit_label": indicator_label.rsplit(", ", 1)[-1],
                    "source_csv_row": number,
                    "source_section": section,
                }
            )
    if pending is not None:
        raise ValueError("Unterminated section title")
    return records


def safe_groups(options, codes, size):
    counts = Counter(options["oktmo"].values())

    def eligible(code):
        label = options["oktmo"][code]
        return counts[label] == 1 and "..." not in label and "…" not in label

    safe = [code for code in codes if eligible(code)]
    ambiguous = [[code] for code in codes if not eligible(code)]
    return [safe[i : i + size] for i in range(0, len(safe), size)] + ambiguous


def run(
    targets_path,
    output,
    cache,
    *,
    ca_bundle=None,
    interval=2.5,
    batch_size=100,
    cache_only=False,
    resume=False,
    limit_regions=None,
    client_factory=BdpmoClient,
):
    targets_path, output, cache = Path(targets_path), Path(output), Path(cache)
    if interval < 2 or batch_size < 1:
        raise ValueError("Use pacing >=2 seconds and a positive batch size")
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    target_map = {}
    for target in targets:
        key = (target["region_url"], target["oktmo"])
        if key in target_map or not REGION_URL.fullmatch(key[0]) or not key[1].isdigit():
            raise ValueError("Invalid or duplicate exact source target")
        target_map[key] = target
    identity = {
        "targets_sha256": sha(targets_path),
        "processor_sha256": sha(__file__),
        "interval": interval,
        "batch_size": batch_size,
        "cache_only": cache_only,
        "bdpmo_sha256": sha(bdpmo.__file__),
        "backoff_helpers_sha256": sha(population_bulk.__file__),
        "ca_bundle_sha256": sha(ca_bundle) if ca_bundle else None,
    }
    if output.exists():
        if not resume:
            raise FileExistsError("Use a new output directory")
        if json.loads((output / "manifest.json").read_text())["identity"] != identity:
            raise ValueError("Resume identity changed")
    output.mkdir(parents=True, exist_ok=True)
    items = output / "items"
    items.mkdir(exist_ok=True)
    snapshots = output / "implementation_snapshot"
    snapshots.mkdir(exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    for source in [
        Path(__file__),
        Path(bdpmo.__file__),
        Path(population_bulk.__file__),
        project / "scripts/run_economy_bulk.py",
    ]:
        destination = snapshots / source.name
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError("Checkpoint implementation snapshot differs from running code")
        destination.write_bytes(source.read_bytes())
    manifest = {
        "status": "running",
        "identity": identity,
        "started_at": datetime.now(UTC).isoformat(),
        "target_count": len(targets),
        "indicator_count": len(INDICATORS),
        "source_cache": str(cache),
        "python_version": sys.version,
        "uv_lock_sha256": sha(project / "uv.lock"),
        "implementation_snapshot_hashes": {p.name: sha(p) for p in snapshots.iterdir()},
        "scope": "Annual source-code observations excluding small businesses; "
        "not a resident-income measure",
    }
    known = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in items.glob("*.json")}
    by_region = defaultdict(list)
    for url, code in target_map:
        by_region[url].append(code)
    regions = list(by_region)
    if limit_regions is not None:
        regions = regions[:limit_regions]

    def item_key(indicator, url, code):
        return hashlib.sha256(json.dumps([indicator, url, code]).encode()).hexdigest()

    def save(indicator, url, code, rows, status, error=None):
        key = item_key(indicator, url, code)
        item = {
            "indicator_id": indicator,
            "source_region_url": url,
            "source_oktmo": code,
            "records": rows,
            "status": status,
            "error": error,
        }
        known[key] = item
        atomic_json(items / f"{key}.json", item)
        manifest["processed_target_indicator_pairs"] = len(known)
        manifest["status_counts"] = dict(Counter(v["status"] for v in known.values()))
        atomic_json(output / "manifest.json", manifest)

    client = client_factory(cache, interval=interval, ca_bundle=ca_bundle, cache_only=cache_only)
    try:
        for url in regions:
            for indicator in INDICATORS:
                pending = [c for c in by_region[url] if item_key(indicator, url, c) not in known]
                if not pending:
                    continue
                try:
                    raw, option_meta = with_backoff(
                        lambda url=url, indicator=indicator: client.request(
                            "POST", url + "DBInet.cgi", {"pl": indicator}
                        )
                    )
                    options = parse_form(raw.decode("cp1251"), indicator)
                except (
                    ValueError,
                    UnicodeError,
                    SourceUnavailable,
                    httpx.HTTPStatusError,
                ) as error:
                    for code in pending:
                        save(indicator, url, code, [], "form_review", str(error))
                    continue
                available = []
                for code in pending:
                    if code not in options["oktmo"]:
                        save(indicator, url, code, [], "code_not_in_source_options")
                    else:
                        available.append(code)
                for codes in safe_groups(options, available, batch_size):
                    try:
                        payload = payload_for(options, codes)
                        raw, metadata = with_backoff(
                            lambda url=url, payload=payload: client.request(
                                "POST", url + "DBInet.cgi", payload
                            )
                        )
                        rows = parse_response(raw.decode("cp1251"), options, codes)
                        for row in rows:
                            row.update(
                                {
                                    "source_region_url": url,
                                    "source_url": metadata["url"],
                                    "source_sha256": metadata["raw_sha256"],
                                    "request_key": metadata["request_key"],
                                    "options_request_key": option_meta["request_key"],
                                    "options_sha256": option_meta["raw_sha256"],
                                    "retrieved_at": metadata["retrieved_at"],
                                    "published_at": None,
                                    "territory_id": None,
                                    "join_status": "source_code_unlinked",
                                }
                            )
                        for code in codes:
                            selected = [r for r in rows if r["source_oktmo"] == code]
                            status = (
                                "complete"
                                if len(selected) == 2
                                else "partial_source"
                                if selected
                                else "not_returned"
                            )
                            save(indicator, url, code, selected, status)
                    except (
                        ValueError,
                        UnicodeError,
                        SourceUnavailable,
                        httpx.HTTPStatusError,
                    ) as error:
                        for code in codes:
                            save(indicator, url, code, [], "response_review", str(error))
                print(
                    json.dumps(
                        {
                            "region": url,
                            "indicator": indicator,
                            "processed_pairs": len(known),
                            "status_counts": manifest.get("status_counts", {}),
                        }
                    ),
                    flush=True,
                )
        manifest["status"] = (
            "complete_with_gaps" if len(known) == len(targets) * 2 else "checkpointed"
        )
    except NetworkPause as error:
        manifest.update(status="paused_network", error=str(error))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        client.close()
        rows = [r for item in known.values() for r in item["records"]]
        if rows:
            frame = pd.DataFrame(rows)
            keys = ["source_region_url", "source_oktmo", "indicator_id", "year"]
            if frame.duplicated(keys).any():
                raise ValueError("Duplicate exported source observations")
            frame = frame.sort_values(keys)
            frame.to_csv(output / "economy_source.csv", index=False)
            with duckdb.connect() as db:
                db.register("rows", frame)
                db.execute(
                    "COPY rows TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                    [str(output / "economy_source.parquet")],
                )
        statuses = [
            {k: v for k, v in item.items() if k != "records"}
            | {"returned_cells": len(item["records"])}
            for item in known.values()
        ]
        pd.DataFrame(statuses).to_csv(output / "target_status.csv", index=False)
        links = [
            {"source_region_url": url, "source_oktmo": code, **link}
            for (url, code), target in target_map.items()
            for link in target["reference_links"]
        ]
        pd.DataFrame(links).to_csv(output / "annual_code_reference_links.csv", index=False)
        manifest.update(
            record_count=len(rows),
            observed_cells=sum(r["value_status"] == "observed" for r in rows),
            network_request_count=getattr(client, "network_request_count", None),
            cache_hit_count=getattr(client, "cache_hit_count", None),
            updated_at=datetime.now(UTC).isoformat(),
            output_hashes={
                p.name: sha(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        )
        atomic_json(output / "manifest.json", manifest)
    return manifest
