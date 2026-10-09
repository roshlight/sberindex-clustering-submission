"""Bounded Rosstat population batches with source-label checks and durable resume."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from . import bdpmo
from .bdpmo import POPULATION_TYPES, YEARS, BdpmoClient


class NetworkPause(RuntimeError):
    """Stop this run after source blocking or exhausted transient retries."""


class SourceUnavailable(RuntimeError):
    """One regional endpoint remained unavailable after bounded transport retries."""


class AttributionReview(ValueError):
    """The returned municipality caption is not the requested source caption."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def target_key(target):
    return hashlib.sha256(json.dumps([target["region_url"], target["oktmo"]]).encode()).hexdigest()


def normalize_targets(rows):
    result = {}
    for row in rows:
        if not isinstance(row.get("oktmo"), str) or not row["oktmo"].isdigit():
            raise ValueError("Source code must be an explicit digit string")
        if not bdpmo.REGION_URL.fullmatch(row.get("region_url", "")):
            raise ValueError("Unexpected regional URL")
        row = {**row, "region_url": row["region_url"].rstrip("/") + "/"}
        key = target_key(row)
        if key in result:
            raise ValueError("Targets must be unique by regional URL and source code")
        for link in row.get("reference_links", []):
            if link.get("reference_date") not in {f"{year}-01-01" for year in YEARS}:
                raise ValueError("Unexpected population reference date")
        result[key] = row
    if not result:
        raise ValueError("No targets supplied")
    return result


def safe_groups(options, codes, batch_size):
    """Use exact labels only when unique among ALL codes in this regional form."""
    by_code = {row["code"]: row["label"] for row in options["oktmo"]}
    population_types = bdpmo.source_population_types(options)
    counts = Counter(by_code.values())
    safe, singletons = [], []
    for code in codes:
        if code not in by_code:
            continue
        if counts[by_code[code]] == 1 and by_code[code] not in population_types.values():
            safe.append(code)
        else:
            singletons.append([code])
    return [safe[i : i + batch_size] for i in range(0, len(safe), batch_size)] + singletons


def batch_payload(options, codes):
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("A batch needs at least two different source codes")
    if not any(group == codes for group in safe_groups(options, codes, len(codes))):
        raise ValueError("Selected labels are not unique in the regional source form")
    payload = bdpmo.population_payload(options, codes[0])
    payload["Qry"] = payload["Qry"].replace(f"oktmo:{codes[0]};", "oktmo:" + ",".join(codes) + ";")
    payload["QryGm"] = "Pokazateli_z:1;munr_z:2;tippos_z:3;period_z:4;oktmo_b:1;mest_b:2;god_s:1;"
    return payload


def parse_batch_csv(document, options, codes):
    """Bind table headers to unique labels from the same official code dimension.

    Unknown headers, duplicate cells and absent selected municipalities reject
    the whole batch. A missing year/type remains absent and is reported in QA.
    """
    batch_payload(options, codes)  # Validate exact codes and regional label uniqueness.
    labels = {row["label"]: row["code"] for row in options["oktmo"] if row["code"] in codes}
    indicator_label = options.get("Pokazateli", [{}])[0].get("label")
    population_types = bdpmo.source_population_types(options)
    buffers, locators = defaultdict(list), defaultdict(dict)
    title, years, current = None, None, None
    for number, cells in bdpmo.csv_context_rows(document, indicator_label):
        if not any(cells):
            continue
        if bdpmo.is_indicator_title(cells[0], indicator_label):
            title, years, current = cells[0], None, None
            continue
        if cells[0] == "" and any(cells[1:]):
            years = bdpmo.year_header(cells)
            current = None
            continue
        if cells[0] in labels and not any(cells[1:]):
            if title is None or years is None:
                raise ValueError("Municipality header has no explicit period context")
            current = labels[cells[0]]
            buffers[current].extend([title, ";" + ";".join(years) + ";"])
            continue
        if cells[0] in population_types.values():
            if current is None:
                raise ValueError("Population cells have no unambiguous municipality header")
            buffers[current].append(";".join(cells))
            kind = next(code for code, label in population_types.items() if label == cells[0])
            for year in years:
                key = (f"{year}-01-01", kind)
                if key in locators[current]:
                    raise ValueError("Duplicate municipality/year/population-type cells")
                locators[current][key] = number
            continue
        raise ValueError(f"Unexpected CSV row {number}: batch attribution rejected")
    if set(buffers) != set(codes):
        raise ValueError("CSV lacks one or more selected municipality headers")
    output = {}
    for code in codes:
        records = bdpmo.parse_population_csv(
            "\n".join(buffers[code]),
            require_complete=False,
            indicator_label=indicator_label,
            population_types=population_types,
        )
        if not records:
            raise ValueError("Selected municipality has no numeric rows")
        for record in records:
            record["source_csv_row"] = locators[code][
                record["reference_date"], record["population_type"]
            ]
        output[code] = records
    return output


def with_backoff(operation, *, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 403:
                raise NetworkPause("HTTP 403; no further source requests in this run") from error
            if status != 429 and status < 500:
                raise
            retry_after = error.response.headers.get("retry-after", "")
            delay = max(5 * (3**attempt), int(retry_after) if retry_after.isdigit() else 0)
            if attempt == 2 or delay > 60 or (retry_after and not retry_after.isdigit()):
                raise NetworkPause(
                    f"HTTP {status}; pause required, Retry-After={retry_after}"
                ) from error
            sleep(delay)
        except httpx.TransportError as error:
            if attempt == 2:
                raise SourceUnavailable(
                    f"Transport failed after three attempts: {error}"
                ) from error
            sleep(5 * (3**attempt))


def source_records(records, code, url, options, option_metadata, metadata, mode):
    label = next(row["label"] for row in options["oktmo"] if row["code"] == code)
    return [
        {
            **row,
            "source_oktmo": code,
            "source_oktmo_name": label,
            "source_region_url": url,
            "indicator_id": bdpmo.INDICATOR,
            "source_url": metadata["url"],
            "source_sha256": metadata["raw_sha256"],
            "request_key": metadata["request_key"],
            "options_request_key": option_metadata["request_key"],
            "retrieved_at": metadata["retrieved_at"],
            "published_at": None,
            "territory_id": None,
            "join_status": "source_oktmo_unlinked_to_sber",
            "extraction_mode": mode,
        }
        for row in records
    ]


def fetch_group(client, url, options, option_metadata, codes, *, refresh=False):
    payload = (
        batch_payload(options, codes)
        if len(codes) > 1
        else bdpmo.population_payload(options, codes[0])
    )
    kwargs = {"refresh": True} if refresh else {}
    raw, metadata = with_backoff(
        lambda: client.request("POST", url + "DBInet.cgi", payload, **kwargs)
    )
    document = raw.decode("cp1251")
    if len(codes) == 1:
        label = next(row["label"] for row in options["oktmo"] if row["code"] == codes[0])
        period = next(
            (row["label"] for row in options.get("period", []) if row["code"] == "208"),
            "на 1 января",
        )
        suffix = f", {label}, {period}"
        indicator = options.get("Pokazateli", [{}])[0].get("label")
        captions = [
            cells[0]
            for _, cells in bdpmo.csv_context_rows(document, indicator)
            if bdpmo.is_indicator_title(cells[0], indicator)
        ]
        if not captions or any(not caption.endswith(suffix) for caption in captions):
            raise AttributionReview(
                "Singleton response does not confirm the exact source municipality caption"
            )
    parsed = (
        parse_batch_csv(document, options, codes)
        if len(codes) > 1
        else {
            codes[0]: bdpmo.parse_population_csv(
                document,
                require_complete=False,
                indicator_label=options.get("Pokazateli", [{}])[0].get("label"),
                population_types=bdpmo.source_population_types(options),
            )
        }
    )
    if any(not records for records in parsed.values()):
        raise ValueError("No source population cells returned")
    return {
        code: source_records(
            records,
            code,
            url,
            options,
            option_metadata,
            metadata,
            "batch_unique_source_label" if len(codes) > 1 else "single_source_code",
        )
        for code, records in parsed.items()
    }


def run_bulk(
    targets_path,
    output,
    cache,
    *,
    ca_bundle=None,
    interval=1,
    batch_size=20,
    limit=None,
    resume=False,
    retry_failed=False,
    cache_only=False,
    client_factory=BdpmoClient,
):
    targets_path, output, cache = Path(targets_path), Path(output), Path(cache)
    if batch_size < 1 or interval < 0 or (limit is not None and limit < 1):
        raise ValueError("Invalid batch size, interval or limit")
    if cache_only and retry_failed:
        raise ValueError("Cache-only replay cannot refresh failed network requests")
    targets = normalize_targets(json.loads(targets_path.read_text(encoding="utf-8")))
    identity = {
        "targets_sha256": digest(targets_path),
        "processor_sha256": digest(__file__),
        "bdpmo_sha256": digest(bdpmo.__file__),
        "batch_size": batch_size,
        "interval": interval,
        "ca_bundle_sha256": digest(ca_bundle) if ca_bundle else None,
        "cache_only": cache_only,
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not resume or not manifest_path.exists():
            raise FileExistsError("Use a fresh output directory or explicit --resume")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise ValueError("Resume inputs, code or request policy changed; use a fresh output")
    else:
        output.mkdir(parents=True)
    items = output / "items"
    items.mkdir(exist_ok=True)
    manifest = {
        "status": "running",
        "identity": identity,
        "target_count": len(targets),
        "source_cache": str(cache.resolve()),
        "started_at": datetime.now(UTC).isoformat(),
        "scope": "Source observations; supplied reference_links are not inferred joins",
        "python_version": sys.version,
    }
    project = Path(__file__).resolve().parents[2]
    snapshot = output / "implementation_snapshot"
    snapshot.mkdir(exist_ok=True)
    snapshot_paths = [
        Path(__file__),
        Path(bdpmo.__file__),
        project / "scripts/run_population_bulk.py",
    ]
    for source in snapshot_paths:
        destination = snapshot / source.name
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError("Implementation snapshot differs from running code")
        destination.write_bytes(source.read_bytes())
    manifest["implementation_snapshot_hashes"] = {
        path.name: digest(path) for path in sorted(snapshot.iterdir())
    }
    manifest["uv_lock_sha256"] = digest(project / "uv.lock")
    atomic_json(manifest_path, manifest)
    known = {
        path.stem: json.loads(path.read_text(encoding="utf-8")) for path in items.glob("*.json")
    }
    if not set(known) <= set(targets):
        raise ValueError("Checkpoint contains unknown targets")
    pending = [
        (key, target)
        for key, target in targets.items()
        if key not in known
        or (retry_failed and known[key]["status"] in {"error", "attribution_review"})
    ]
    if limit is not None:
        pending = pending[:limit]
    grouped = defaultdict(list)
    for key, target in pending:
        grouped[target["region_url"]].append((key, target))
    retry_keys = {key for key, _ in pending if key in known}

    def save(key, status, records=None, error=None):
        records = records or []
        expected = {(f"{year}-01-01", kind) for year in YEARS for kind in POPULATION_TYPES}
        present = {(row["reference_date"], row["population_type"]) for row in records}
        item = {
            "target": targets[key],
            "status": status,
            "records": records,
            "error": error,
            "missing_cells": [list(pair) for pair in sorted(expected - present)],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(items / f"{key}.json", item)
        known[key] = item
        manifest["processed_target_count"] = len(known)
        manifest["status_counts"] = dict(Counter(row["status"] for row in known.values()))
        atomic_json(manifest_path, manifest)

    client = client_factory(
        cache,
        interval=interval,
        ca_bundle=ca_bundle,
        **({"cache_only": True} if cache_only else {}),
    )
    try:
        for url, group in grouped.items():
            try:
                option_kwargs = (
                    {"refresh": True} if any(key in retry_keys for key, _ in group) else {}
                )
                options, option_metadata = with_backoff(
                    lambda url=url, kwargs=option_kwargs: client.options(url, **kwargs)
                )
            except (ValueError, UnicodeError, httpx.HTTPStatusError, SourceUnavailable) as error:
                for key, _ in group:
                    save(key, "error", error=f"Regional options: {type(error).__name__}: {error}")
                continue
            by_code = {target["oktmo"]: key for key, target in group}
            available = {row["code"] for row in options["oktmo"]}
            for code in by_code.keys() - available:
                save(by_code[code], "code_not_in_source_options")
            for codes in safe_groups(options, list(by_code), batch_size):
                try:
                    fetched = fetch_group(
                        client,
                        url,
                        options,
                        option_metadata,
                        codes,
                        refresh=any(by_code[code] in retry_keys for code in codes),
                    )
                except (
                    ValueError,
                    UnicodeError,
                    httpx.HTTPStatusError,
                    SourceUnavailable,
                ) as error:
                    if len(codes) == 1:
                        save(
                            by_code[codes[0]],
                            "attribution_review"
                            if isinstance(error, AttributionReview)
                            else "error",
                            error=f"{type(error).__name__}: {error}",
                        )
                        continue
                    fetched = {}
                    for code in codes:
                        try:
                            single = fetch_group(
                                client,
                                url,
                                options,
                                option_metadata,
                                [code],
                                refresh=by_code[code] in retry_keys,
                            )
                            for single_code, records in single.items():
                                save(
                                    by_code[single_code],
                                    "complete" if len(records) == 4 else "partial_source",
                                    records,
                                )
                        except (
                            ValueError,
                            UnicodeError,
                            httpx.HTTPStatusError,
                            SourceUnavailable,
                        ) as single_error:
                            save(
                                by_code[code],
                                "attribution_review"
                                if isinstance(single_error, AttributionReview)
                                else "error",
                                error=str(single_error),
                            )
                for code, records in fetched.items():
                    save(
                        by_code[code],
                        "complete" if len(records) == 4 else "partial_source",
                        records,
                    )
                print(
                    json.dumps(
                        {
                            "region": url,
                            "processed": len(known),
                            "targets": len(targets),
                            "status_counts": manifest.get("status_counts", {}),
                        }
                    ),
                    flush=True,
                )
        manifest["status"] = (
            (
                "complete"
                if all(item["status"] == "complete" for item in known.values())
                else "complete_with_gaps"
            )
            if len(known) == len(targets)
            else "checkpointed"
        )
    except NetworkPause as error:
        manifest.update(status="paused_network", pause_reason=str(error))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        client.close()
        records = [record for item in known.values() for record in item["records"]]
        if records:
            frame = pd.DataFrame(records)
            frame["value"] = frame["value"].astype("Int64")
            frame["territory_id"] = frame["territory_id"].astype("Int64")
            frame.to_csv(output / "population_source.csv", index=False)
            with duckdb.connect() as connection:
                connection.register("population_rows", frame)
                connection.execute(
                    "COPY population_rows TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                    [str(output / "population_source.parquet")],
                )
        status_rows = [
            {
                "target_key": key,
                "region_url": target["region_url"],
                "source_oktmo": target["oktmo"],
                "status": known.get(key, {}).get("status", "pending"),
                "error": known.get(key, {}).get("error"),
                "returned_cells": len(known.get(key, {}).get("records", [])),
                "missing_cells": json.dumps(known.get(key, {}).get("missing_cells")),
            }
            for key, target in targets.items()
        ]
        pd.DataFrame(status_rows).to_csv(output / "target_status.csv", index=False)
        coverage = []
        for key, target in targets.items():
            cells = {
                (row["reference_date"], row["population_type"]): row
                for row in known.get(key, {}).get("records", [])
            }
            eligible_dates = {link["reference_date"] for link in target.get("reference_links", [])}
            for year in YEARS:
                for kind in POPULATION_TYPES:
                    date = f"{year}-01-01"
                    cell = cells.get((date, kind))
                    coverage.append(
                        {
                            "source_region_url": target["region_url"],
                            "source_oktmo": target["oktmo"],
                            "reference_date": date,
                            "population_type": kind,
                            "expected_by_reference_links": date in eligible_dates,
                            "cell_status": "not_returned"
                            if cell is None
                            else ("returned_missing" if cell["value"] is None else "observed"),
                        }
                    )
        pd.DataFrame(coverage).to_csv(output / "cell_coverage.csv", index=False)
        links = [
            {
                "target_key": key,
                "source_region_url": target["region_url"],
                "source_oktmo": target["oktmo"],
                **link,
            }
            for key, target in targets.items()
            for link in target.get("reference_links", [])
        ]
        if links:
            pd.DataFrame(links).to_csv(output / "target_reference_links.csv", index=False)
        manifest.update(
            network_request_count=getattr(client, "network_request_count", None),
            cache_hit_count=getattr(client, "cache_hit_count", None),
            record_count=len(records),
            processed_target_count=len(known),
            status_counts=dict(Counter(row["status"] for row in known.values())),
            updated_at=datetime.now(UTC).isoformat(),
            output_hashes={
                path.name: digest(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path != manifest_path
            },
        )
        atomic_json(manifest_path, manifest)
    return manifest
