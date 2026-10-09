"""Recover reviewed BDPMO format variants without changing round-three sources.

Only previously refused region/indicator jobs are queried. Annual source codes
remain bound to their existing year-specific links. UI truncation requires an
isolated exact-code request; unknown units, scope, labels, or duplicates fail.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from .bdpmo import BdpmoClient
from .economy_bulk import normalized, sha
from .round3_economy import BASE, SPECS, YEARS

UPPER_TYPES = {
    "10": "муниципальный район",
    "7": "городской округ, городской округ с внутригородским делением",
    "20": "муниципальный округ",
}
REVIEWED_SCOPE_ALIASES = {
    ("munst05", "8401003"): "без субъектов малого редпринимательства",
    ("munst36", "8401011"): "без субъектов малого предприниматетельства",
    ("munst70", "8401011"): "безсубъектов малого предпринимательства",
}
SOURCE_KEY = ["source_region_url", "source_oktmo", "year", "indicator_id"]
KEY = ["territory_id", "year", "indicator_id"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_form(document, indicator):
    """Preserve auxiliary notes and conflicting code occurrences separately."""
    options, blocked, entries = {}, {}, {}
    for name in re.findall(r"p_(\w+)\s*=\s*new Array", document):
        pairs = re.findall(r"p_" + name + r'\[([0-9]+)\]="([^"\r\n]+)"', document)
        selected = re.search(
            r'<SELECT\s+NAME="' + name + r'"[^>]*>(.*?)</SELECT>', document, re.I | re.S
        )
        if not pairs or not selected:
            raise ValueError(f"Missing coded dimension {name}")
        labels = re.findall(r"<OPTION\b[^>]*>([^<]*)", selected.group(1), re.I)
        pairs = sorted((int(i), code) for i, code in pairs)
        if [i for i, _ in pairs] != list(range(len(labels))):
            raise ValueError(f"Dimension indices differ {name}")
        values = {}
        entries[name] = []
        for (index, code), label in zip(pairs, labels, strict=True):
            label = normalized(label)
            entries[name].append(dict(index=index, code=code, label=label))
            values.setdefault(code, set()).add(label)
        conflicts = {code: sorted(v) for code, v in values.items() if len(v) != 1}
        if conflicts and name != "oktmo":
            raise ValueError(f"Conflicting nonterritorial dimension: {name}")
        blocked.update(conflicts)
        options[name] = {code: next(iter(v)) for code, v in values.items() if len(v) == 1}
    notes = options.pop("footnotes", {})
    financing = None
    if indicator == "8109001":
        financing_names = set(options) & {"istinv", "p_istinv"}
        if len(financing_names) != 1:
            raise ValueError("Investment financing dimension must be explicit and unique")
        financing = financing_names.pop()
    expected = BASE | ({financing} if financing else set())
    if set(options) != expected:
        raise ValueError(f"Unreviewed dimensions: {set(options) - expected}")
    spec = SPECS[indicator]
    if set(options["Pokazateli"]) != {indicator}:
        raise ValueError("Wrong source indicator")
    if not options["Pokazateli"][indicator].casefold().startswith(spec["prefix"].casefold()):
        raise ValueError("Indicator title differs")
    periods = (
        {"значение показателя за год", "значение за год"}
        if financing
        else {"январь-декабрь", "за январь-декабрь"}
    )
    if options["period"].get(spec["period"], "").casefold() not in periods:
        raise ValueError("Annual period not verified")
    if options["okved2"].get("101", "").casefold() not in {
        "всего по обследуемым видам экономической деятельности",
        "всего по обследуемым видам экономической деятельности (101.аг)",
    }:
        raise ValueError("Activity total not verified")
    if financing and options[financing].get("8", "").casefold() != "всего":
        raise ValueError("Financing total not verified")
    if not YEARS & options["god"].keys():
        raise ValueError("Requested source years absent")
    types = [
        code
        for code, label in UPPER_TYPES.items()
        if options["tippos"].get(code, "").casefold() == label
    ]
    if not types:
        raise ValueError("No verified upper municipal types")
    return dict(
        options=options,
        footnotes=notes,
        blocked_oktmo=blocked,
        original_entries=entries,
        financing=financing,
        selected_types=types,
    )


def payload_for(form, indicator, codes):
    options = form["options"]
    if not codes or len(codes) != len(set(codes)) or not set(codes) <= options["oktmo"].keys():
        raise ValueError("Codes must be unique exact nonconflicting source options")
    counts = Counter(options["oktmo"].values())
    if len(codes) > 1 and any(
        counts[options["oktmo"][c]] != 1 or options["oktmo"][c].endswith("...") for c in codes
    ):
        raise ValueError("Ambiguous or UI-truncated captions require a single-code query")
    values = {name: list(v) for name, v in options.items()}
    years = sorted(YEARS & options["god"].keys())
    values.update(
        oktmo=codes,
        tippos=form["selected_types"],
        okved2=["101"],
        god=years,
        period=[SPECS[indicator]["period"]],
    )
    if form["financing"]:
        values[form["financing"]] = ["8"]
    z = [name for name in options if name not in {"oktmo", "god"}]
    return {
        "Qry": ";".join(name + ":" + ",".join(v) for name, v in values.items()) + ";",
        "QryGm": ";".join(f"{name}_z:{i}" for i, name in enumerate(z, 1)) + ";oktmo_b:1;god_s:1;",
        # Notes are read from the saved form, including numeric confidentiality codes.
        # They are auxiliary UI metadata, not an extra numeric data dimension.
        "QryFootNotes": ";",
        "YearsList": ";".join(years) + ";",
        "tbl": "1",
        "Format": "CSV",
        "YearFrom": years[0],
        "YearTo": years[-1],
    }


def parse_response(document, form, indicator, codes, *, region):
    payload_for(form, indicator, codes)
    if "<html" in document.casefold():
        raise ValueError("HTML is not a numeric export")
    options = form["options"]
    labels = {options["oktmo"][code]: code for code in codes}
    spec = SPECS[indicator]
    selected = {"okved2": "101", "period": spec["period"]}
    if form["financing"]:
        selected[form["financing"]] = "8"
    suffix = "".join(", " + options[name][selected[name]] for name in options if name in selected)
    # The municipal type is also required in every section, in its exact source wording.
    type_labels = [options["tippos"][code] for code in form["selected_types"]]
    sentinel = any(
        "999999999" in note and "условное обозначение конфиденциальности" in note
        for note in form["footnotes"].values()
    )
    title, header, section, scope_rule, unit_literal = None, None, None, None, None
    rows, seen = [], set()
    for line, raw in enumerate(csv.reader(io.StringIO(document), delimiter=";"), 1):
        cells = [normalized(value) for value in raw]
        if not any(cells):
            continue
        first = cells[0]
        if first.casefold().startswith(spec["prefix"].casefold()):
            if title is not None:
                raise ValueError("Unterminated source section")
            title, header = normalized(";".join(raw)), None
            continue
        if not first and any(cells[1:]):
            if title is None or not title.casefold().endswith(suffix.casefold()) or cells[-1] != "":
                raise ValueError("Section lacks exact selected totals/annual period")
            unit = re.search(r", (тысяч[аи]? рублей|тыс\.\s*рублей),", title, re.I)
            if not unit:
                raise ValueError("Unverified source monetary unit")
            unit_literal = unit.group(1)
            if not any(", " + label + ", " in title for label in type_labels):
                raise ValueError("Section is outside requested upper municipality types")
            normal_scope = "без субъектов малого предпринимательства"
            alias = REVIEWED_SCOPE_ALIASES.get((region, indicator))
            if normal_scope in title.casefold():
                scope_rule = "explicit_normal_scope"
            elif alias and alias in title.casefold():
                scope_rule = "reviewed_literal_source_typo:" + alias
            else:
                raise ValueError("Small-business exclusion not evidenced in this source header")
            header = cells[1:-1]
            if (
                not header
                or len(header) != len(set(header))
                or not set(header) <= YEARS
                or not set(header) <= options["god"].keys()
            ):
                raise ValueError("Invalid explicit annual header")
            section, title = title, None
            continue
        if title is not None:
            if len(cells) != 1:
                raise ValueError("Unknown cells within header")
            title = normalized(title + " " + first)
            continue
        if header is None or cells[-1] != "" or len(cells) < len(header) + 2:
            raise ValueError("Row outside explicit annual header")
        label = normalized(";".join(raw[: -len(header) - 1]))
        code = labels.get(label)
        caption_rule = "exact_full_source_option"
        if code is None and len(codes) == 1:
            ui = options["oktmo"][codes[0]]
            if ui.endswith("...") and label.startswith(ui[:-3].rstrip()):
                code = codes[0]
                caption_rule = "explicit_UI_ellipsis_resolved_by_isolated_exact_code_query"
        if code is None:
            raise ValueError("Unknown source caption; no fuzzy or proximity matching")
        for year, cell in zip(header, cells[-len(header) - 1 : -1], strict=True):
            if (code, year) in seen:
                raise ValueError("Duplicate municipality-year cell; no automatic deduplication")
            seen.add((code, year))
            compact = cell.replace(" ", "")
            if re.fullmatch(r"[0-9]+(?:[,.][0-9]+)?", compact):
                decimal = Decimal(compact.replace(",", "."))
                if sentinel and decimal == Decimal("999999999"):
                    value, status = None, "suppressed_explicit_source_footnote"
                else:
                    value, status = str(decimal), "observed"
            elif cell in {"", "-", "…", "...", "х", "x", "X", "к", "К"}:
                value, status = None, "unresolved_source_marker"
            else:
                raise ValueError(f"Unknown numeric token: {cell!r}")
            rows.append(
                dict(
                    source_oktmo=code,
                    year=year,
                    indicator_id=indicator,
                    value=value,
                    value_raw=cell,
                    value_status=status,
                    unit="thousand_rubles",
                    source_unit_literal=unit_literal,
                    source_csv_row=line,
                    source_section=section,
                    source_oktmo_name=label,
                    source_option_label=options["oktmo"][code],
                    caption_evidence_rule=caption_rule,
                    scope_evidence_rule=scope_rule,
                    source_footnotes=json.dumps(form["footnotes"], ensure_ascii=False),
                    period_code=spec["period"],
                    activity_code="101",
                )
            )
    if title is not None:
        raise ValueError("Unterminated final section")
    return rows


def request_once_retry(client, url, payload):
    try:
        return client.request("POST", url, payload)
    except (httpx.ReadTimeout, httpx.RemoteProtocolError):
        time.sleep(5)
        return client.request("POST", url, payload)


def run(args):
    old_dir = args.repo / "reports/data_round3/economy"
    if args.output.resolve() == old_dir.resolve():
        raise ValueError("Round-three outputs are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    old_path = old_dir / "annual_candidate_panel.csv"
    statuses_path = old_dir / "collection_status.json"
    old_hashes = {p.name: sha(p) for p in old_dir.iterdir() if p.is_file()}
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False)
    jobs = [x for x in json.loads(statuses_path.read_text("utf-8")) if x["status"] != "parsed"]
    jobs.sort(key=lambda x: (0 if x["key"].startswith(("munst28_", "munst36_")) else 1, x["key"]))
    client = BdpmoClient(args.cache, interval=args.interval, timeout=60, cache_only=args.cache_only)
    cells, statuses, forms, blocked_targets = [], [], [], []
    try:
        for job in jobs:
            region, indicator = job["key"].split("_")
            region_url = f"https://rosstat.gov.ru/dbscripts/munst/{region}/"
            url = region_url + "DBInet.cgi"
            group = old.loc[old.source_region_url.eq(region_url) & old.indicator_id.eq(indicator)]
            try:
                body, form_meta = request_once_retry(client, url, {"pl": indicator})
                form = parse_form(body.decode("cp1251"), indicator)
                forms.append(dict(key=job["key"], metadata=form_meta, **form))
                wanted = set(group.source_oktmo)
                for code in sorted(wanted & form["blocked_oktmo"].keys()):
                    blocked_targets.append(
                        dict(
                            key=job["key"],
                            source_oktmo=code,
                            reason="conflicting source option labels",
                        )
                    )
                codes = sorted(wanted & form["options"]["oktmo"].keys())
                counts = Counter(form["options"]["oktmo"].values())
                batch = [
                    c
                    for c in codes
                    if counts[form["options"]["oktmo"][c]] == 1
                    and not form["options"]["oktmo"][c].endswith("...")
                ]
                batches = [batch[i : i + 100] for i in range(0, len(batch), 100)]
                batches += [[c] for c in codes if c not in batch]
                for selected in batches:
                    record = dict(key=job["key"], codes=selected, old_refusal=job["error"])
                    try:
                        payload = payload_for(form, indicator, selected)
                        body, meta = request_once_retry(client, url, payload)
                        record.update(
                            request_key=meta["request_key"], raw_sha256=meta["raw_sha256"]
                        )
                        parsed = parse_response(
                            body.decode("cp1251"), form, indicator, selected, region=region
                        )
                        for row in parsed:
                            row.update(
                                source_region_url=region_url,
                                source_url=url,
                                source_sha256=meta["raw_sha256"],
                                request_key=meta["request_key"],
                                form_request_key=form_meta["request_key"],
                                retrieved_at=meta["retrieved_at"],
                            )
                        cells.extend(parsed)
                        record.update(
                            status="parsed",
                            rows=len(parsed),
                            numeric=sum(x["value"] is not None for x in parsed),
                        )
                    except Exception as error:
                        record.update(
                            status="response_review", error=f"{type(error).__name__}: {error}"
                        )
                    statuses.append(record)
            except Exception as error:
                statuses.append(
                    dict(
                        key=job["key"],
                        status="form_review",
                        old_refusal=job["error"],
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            write_json(args.output / "collection_status.json", statuses)
            write_json(args.output / "form_inventory.json", forms)
            print(
                json.dumps(
                    {
                        "key": job["key"],
                        "source_cells": len(cells),
                        "last_status": statuses[-1]["status"],
                    }
                ),
                flush=True,
            )
    finally:
        client.close()
    new = pd.DataFrame(cells)
    if new.empty or new.duplicated(SOURCE_KEY).any():
        raise ValueError("No cells or duplicate source keys")
    new.to_csv(args.output / "recovered_source_cells.csv", index=False)
    merged = old.merge(
        new, on=SOURCE_KEY, how="left", validate="one_to_one", suffixes=("_old", "_new")
    )
    candidate = old.copy()
    old_value = pd.to_numeric(old.value.replace("", None), errors="raise")
    new_value = pd.to_numeric(merged.value_new, errors="raise")
    conflict = old_value.notna() & new_value.notna() & old_value.ne(new_value)
    incoming = (
        merged.source_sha256_new.notna() & ~conflict & (old_value.isna() | old_value.eq(new_value))
    )
    for column in new:
        if column in SOURCE_KEY:
            continue
        current = merged[column + "_new"] if column in old else merged[column]
        if column not in candidate:
            candidate[column] = ""
        candidate.loc[incoming, column] = current[incoming].fillna("").astype(str).to_numpy()
    candidate["round4_selection"] = "round3_preserved"
    candidate.loc[incoming, "round4_selection"] = "round4_source"
    candidate.loc[conflict, "round4_selection"] = "round3_preserved_numeric_revision_conflict"
    candidate.loc[incoming, "publication_vintage"] = (
        "round4 retrieval; first publication/revisions unknown"
    )
    candidate["geographic_equivalence_proven"] = ""
    candidate.to_csv(args.output / "annual_candidate_panel.csv", index=False)
    typed = candidate.copy()
    for c in ["territory_id", "year"]:
        typed[c] = pd.to_numeric(typed[c], errors="raise").astype(int)
    typed["value"] = pd.to_numeric(typed.value.replace("", None), errors="raise")
    with duckdb.connect() as db:
        db.register("candidate", typed)
        db.execute(
            "COPY candidate TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(args.output / "annual_candidate_panel.parquet")],
        )
    delta = old[KEY + ["source_region_url", "source_oktmo", "official_oktmo_8"]].copy()
    delta["old_value"], delta["new_capture_value"] = old_value, new_value
    delta["candidate_value"] = typed.value
    delta["old_status"], delta["candidate_status"] = old.value_status, candidate.value_status
    delta["old_source_sha256"] = old.source_sha256
    delta["new_source_sha256"] = merged.source_sha256_new
    delta["selection"] = candidate.round4_selection
    delta["numeric_gain"] = old_value.isna() & typed.value.notna()
    delta["numeric_revision_conflict"] = conflict
    delta.to_csv(args.output / "delta_all_cells.csv", index=False)
    delta.loc[delta.numeric_gain | conflict].to_csv(
        args.output / "delta_changed_cells.csv", index=False
    )
    delta.loc[conflict].to_csv(args.output / "revision_conflicts.csv", index=False)
    extra = new.merge(
        old[SOURCE_KEY], on=SOURCE_KEY, how="left", validate="one_to_one", indicator=True
    )
    extra.loc[extra._merge.eq("left_only")].drop(columns="_merge").to_csv(
        args.output / "source_only_outside_year_link.csv", index=False
    )
    pd.DataFrame(blocked_targets, columns=["key", "source_oktmo", "reason"]).to_csv(
        args.output / "blocked_target_codes.csv", index=False
    )
    pd.DataFrame([x for x in statuses if x["status"] != "parsed"]).to_csv(
        args.output / "refusals_remaining.csv", index=False
    )
    delta.groupby(["indicator_id", "year"]).agg(
        denominator=("territory_id", "size"),
        old_numeric=("old_value", "count"),
        new_numeric=("candidate_value", "count"),
        numeric_gain=("numeric_gain", "sum"),
        revision_conflicts=("numeric_revision_conflict", "sum"),
    ).reset_index().to_csv(args.output / "coverage_delta.csv", index=False)
    delta.groupby(["source_region_url", "indicator_id", "year"]).agg(
        denominator=("territory_id", "size"),
        old_numeric=("old_value", "count"),
        new_numeric=("candidate_value", "count"),
        numeric_gain=("numeric_gain", "sum"),
    ).reset_index().to_csv(args.output / "regional_coverage_delta.csv", index=False)
    if len(candidate) != 12096 or candidate.duplicated(KEY).any():
        raise ValueError("Frozen candidate grid changed")
    if {p.name: sha(p) for p in old_dir.iterdir() if p.is_file()} != old_hashes:
        raise ValueError("An immutable round-three input changed")
    qa = dict(
        rows=len(candidate),
        municipalities=candidate.territory_id.nunique(),
        old_numeric=int(old_value.notna().sum()),
        new_numeric=int(typed.value.notna().sum()),
        numeric_gain=int(delta.numeric_gain.sum()),
        numeric_revision_conflicts=int(conflict.sum()),
        recovered_source_rows=len(new),
        recovered_source_numeric=int(new.value.notna().sum()),
        statuses=dict(Counter(x["status"] for x in statuses)),
        recovered_source_value_statuses=new.value_status.value_counts().to_dict(),
        old_files_unchanged=True,
        exact_year_code_join=True,
        missing_imputed=0,
        boundary_equivalence_proven=0,
        frozen_models_or_masks_changed=False,
        requests=client.network_request_count,
        cache_hits=client.cache_hit_count,
    )
    write_json(args.output / "QA.json", qa)
    write_json(
        args.output / "manifest.json",
        dict(
            script_sha256=sha(__file__),
            old_input_hashes=old_hashes,
            output_hashes={
                p.name: sha(p)
                for p in sorted(args.output.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
            cache_files={p.name: sha(p) for p in sorted(args.cache.iterdir()) if p.is_file()},
        ),
    )
    print(json.dumps(qa), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["repo", "output", "cache"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--interval", type=float, default=2.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
