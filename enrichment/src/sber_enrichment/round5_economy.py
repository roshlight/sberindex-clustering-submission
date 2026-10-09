"""Fifth collection: exact remaining cells and reviewed municipal-type captions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from .bdpmo import BdpmoClient
from .economy_bulk import normalized, sha
from .round3_economy import BASE, SPECS, YEARS
from .round4_economy import (
    KEY,
    SOURCE_KEY,
    parse_response,
    payload_for,
    request_once_retry,
    write_json,
)

REVIEWED_TYPES = {
    "10": {"муниципальный район", "муниципальные районы"},
    "7": {
        "городской округ",
        "городской округ, городской округ с внутригородским делением",
        "городские округа, городской округ с внутригородским делением",
    },
    "20": {"муниципальный округ"},
    "3": {
        "внутригородские территории городов федерального значения",
        "внутригородская территория (внутригородское муниципальное образование) "
        "города федерального значения",
    },
}


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
        for code, labels in REVIEWED_TYPES.items()
        if options["tippos"].get(code, "").casefold() in labels
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


def failure_reason(error):
    if isinstance(error, httpx.HTTPError):
        return "network_or_http_failure"
    text = str(error)
    if "Small-business exclusion" in text:
        return "semantic_scope_unverified"
    if "monetary unit" in text:
        return "semantic_unit_unverified"
    return "response_semantics_unresolved"


def is_internal_file_error(body):
    return b"Status: 500 Cannot open file" in body and b"Exception: EFOpenError" in body


def control_code(old, form, region_url, indicator, excluded):
    known = set(
        old.loc[
            old.source_region_url.eq(region_url)
            & old.indicator_id.eq(indicator)
            & old.value.ne(""),
            "source_oktmo",
        ]
    )
    labels = form["options"]["oktmo"]
    counts = Counter(labels.values())
    candidates = sorted(
        code
        for code in known & labels.keys() - set(excluded)
        if counts[labels[code]] == 1 and not labels[code].endswith("...")
    )
    if not candidates:
        raise ValueError("Internal file error; no unambiguous observed control code available")
    return candidates[0]


def merge_candidate(old, source):
    if old.duplicated(KEY).any() or source.duplicated(SOURCE_KEY).any():
        raise ValueError("Duplicate source or annual keys")
    matched = old.merge(
        source, on=SOURCE_KEY, how="left", validate="one_to_one", suffixes=("_old", "_new")
    )
    old_value = pd.to_numeric(old.value.replace("", None), errors="raise")
    new_value = pd.to_numeric(matched.value_new, errors="raise")
    incoming = old_value.isna() & matched.source_sha256_new.notna()
    candidate = old.copy()
    for column in source:
        if column in SOURCE_KEY:
            continue
        current = matched[column + "_new"] if column in old else matched[column]
        if column not in candidate:
            candidate[column] = ""
        candidate.loc[incoming, column] = current[incoming].fillna("").astype(str).to_numpy()
    candidate["round5_selection"] = "round4_preserved"
    candidate.loc[incoming, "round5_selection"] = "round5_source"
    candidate.loc[incoming, "publication_vintage"] = (
        "round5 retrieval; original publication unknown"
    )
    delta = old[KEY + ["source_region_url", "source_oktmo", "official_oktmo_8"]].copy()
    delta["old_value"] = old_value
    delta["new_capture_value"] = new_value
    delta["candidate_value"] = pd.to_numeric(candidate.value.replace("", None), errors="raise")
    delta["numeric_gain"] = old_value.isna() & delta.candidate_value.notna()
    delta["numeric_revision_conflict"] = (
        old_value.notna() & new_value.notna() & old_value.ne(new_value)
    )
    delta["old_status"] = old.value_status
    delta["candidate_status"] = candidate.value_status
    delta["new_source_sha256"] = matched.source_sha256_new
    delta["old_source_sha256"] = old.source_sha256
    pd.testing.assert_frame_equal(
        old.loc[old_value.notna()], candidate.loc[old_value.notna(), old.columns]
    )
    return candidate, delta


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be new or empty; previous snapshots are immutable")
    old_path = args.repo / "reports/data_round4/economy/annual_candidate_panel.csv"
    old_sha = sha(old_path)
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False)
    missing = old.loc[old.value.eq("")]
    if len(missing) != 488 or len(old) != 12096:
        raise ValueError("Wrong frozen candidate snapshot")
    args.output.mkdir(parents=True, exist_ok=True)
    client = BdpmoClient(args.cache, interval=args.interval, timeout=60, cache_only=args.cache_only)
    cells, forms, statuses, decisions = [], [], [], []
    jobs = sorted(
        missing.groupby(["source_region_url", "indicator_id"]),
        key=lambda x: (not x[0][0].endswith(("munst57/", "munst05/")), -len(x[1])),
    )
    try:
        for (region_url, indicator), group in jobs:
            region = region_url.rstrip("/").split("/")[-1]
            url = region_url + "DBInet.cgi"
            job = region + "_" + indicator
            reasons, job_records, response_keys = {}, [], {}
            try:
                body, form_meta = request_once_retry(client, url, {"pl": indicator})
                form = parse_form(body.decode("cp1251"), indicator)
                forms.append(dict(job=job, metadata=form_meta, **form))
                wanted = set(group.source_oktmo)
                for code in wanted - form["options"]["oktmo"].keys():
                    reasons[code] = (
                        "source_code_conflicting_labels"
                        if code in form["blocked_oktmo"]
                        else "exact_source_code_absent_from_current_form"
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
                    record = dict(job=job, codes=selected, selected_types=form["selected_types"])
                    try:
                        body, meta = request_once_retry(
                            client, url, payload_for(form, indicator, selected)
                        )
                        record.update(
                            request_key=meta["request_key"], raw_sha256=meta["raw_sha256"]
                        )
                        response_keys.update({code: meta["request_key"] for code in selected})
                        used_codes = selected
                        control = None
                        if is_internal_file_error(body):
                            control = control_code(old, form, region_url, indicator, wanted)
                            record.update(
                                initial_request_key=meta["request_key"],
                                initial_raw_sha256=meta["raw_sha256"],
                                retry_reason="source_internal_file_open_error",
                                control_code=control,
                            )
                            used_codes = selected + [control]
                            body, meta = request_once_retry(
                                client, url, payload_for(form, indicator, used_codes)
                            )
                            record.update(
                                request_key=meta["request_key"],
                                raw_sha256=meta["raw_sha256"],
                                actual_codes=used_codes,
                            )
                            response_keys.update({code: meta["request_key"] for code in selected})
                        parsed = parse_response(
                            body.decode("cp1251"), form, indicator, used_codes, region=region
                        )
                        if control and not any(
                            row["source_oktmo"] == control and row["value"] is not None
                            for row in parsed
                        ):
                            raise ValueError("The independent control returned no numeric value")
                        for row in parsed:
                            row.update(
                                source_region_url=region_url,
                                source_url=url,
                                source_sha256=meta["raw_sha256"],
                                request_key=meta["request_key"],
                                form_request_key=form_meta["request_key"],
                                retrieved_at=meta["retrieved_at"],
                            )
                        job_records.extend(parsed)
                        record.update(
                            status="parsed",
                            rows=len(parsed),
                            numeric=sum(row["value"] is not None for row in parsed),
                        )
                        reason = (
                            "exact_code_year_not_returned_in_controlled_export"
                            if control
                            else "exact_code_year_not_returned"
                        )
                        reasons.update({code: reason for code in selected})
                    except Exception as error:
                        reason = failure_reason(error)
                        reasons.update({code: reason for code in selected})
                        record.update(status=reason, error=f"{type(error).__name__}: {error}")
                    statuses.append(record)
                cells.extend(job_records)
                returned = {(r["source_oktmo"], str(r["year"])): r for r in job_records}
                for row in group.to_dict("records"):
                    fresh = returned.get((row["source_oktmo"], row["year"]))
                    reason = (
                        "recovered_numeric"
                        if fresh and fresh["value"] is not None
                        else fresh["value_status"]
                        if fresh
                        else reasons[row["source_oktmo"]]
                    )
                    if (
                        reason == "exact_code_year_not_returned"
                        and row["year"] not in form["options"]["god"]
                    ):
                        reason = "observation_year_absent_from_current_form"
                    decisions.append(
                        {k: row[k] for k in KEY + ["source_region_url", "source_oktmo"]}
                        | dict(
                            job=job,
                            reason=reason,
                            form_request_key=form_meta["request_key"],
                            response_request_key=(
                                fresh["request_key"]
                                if fresh
                                else response_keys.get(row["source_oktmo"], "")
                            ),
                        )
                    )
            except Exception as error:
                reason = (
                    "network_or_http_failure"
                    if isinstance(error, httpx.HTTPError)
                    else "form_semantics_unresolved"
                )
                statuses.append(
                    dict(job=job, status=reason, error=f"{type(error).__name__}: {error}")
                )
                for row in group.to_dict("records"):
                    decisions.append(
                        {k: row[k] for k in KEY + ["source_region_url", "source_oktmo"]}
                        | dict(job=job, reason=reason)
                    )
            write_json(args.output / "collection_status.json", statuses)
            write_json(args.output / "form_inventory.json", forms)
            print(
                json.dumps(dict(job=job, source_cells=len(cells), target_cells=len(decisions))),
                flush=True,
            )
    finally:
        client.close()
    source = pd.DataFrame(cells)
    if source.empty:
        raise ValueError("No parsed source cells")
    source.to_csv(args.output / "recovered_source_cells.csv", index=False)
    candidate, delta = merge_candidate(old, source)
    candidate.to_csv(args.output / "annual_candidate_panel.csv", index=False)
    typed = candidate.copy()
    for column in ["territory_id", "year"]:
        typed[column] = pd.to_numeric(typed[column], errors="raise").astype(int)
    typed["value"] = pd.to_numeric(typed.value.replace("", None), errors="raise")
    with duckdb.connect() as db:
        db.register("candidate", typed)
        db.execute(
            "COPY candidate TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(args.output / "annual_candidate_panel.parquet")],
        )
    delta.to_csv(args.output / "delta_all_cells.csv", index=False)
    delta.loc[delta.numeric_gain].to_csv(args.output / "new_numeric_cells.csv", index=False)
    delta.loc[delta.numeric_revision_conflict].to_csv(
        args.output / "revision_conflicts.csv", index=False
    )
    outside = source.merge(
        old[SOURCE_KEY], on=SOURCE_KEY, how="left", validate="one_to_one", indicator=True
    )
    outside.loc[outside._merge.eq("left_only"), source.columns].to_csv(
        args.output / "source_only_outside_year_link.csv", index=False
    )
    reasons = pd.DataFrame(decisions)
    if len(reasons) != 488 or reasons.duplicated(KEY).any():
        raise ValueError("Every original missing cell needs exactly one decision")
    reasons.to_csv(args.output / "target_cell_decisions.csv", index=False)
    reasons.loc[~reasons.reason.eq("recovered_numeric")].to_csv(
        args.output / "remaining_missing_cells.csv", index=False
    )
    delta.groupby(["source_region_url", "indicator_id", "year"]).agg(
        denominator=("territory_id", "size"),
        old_numeric=("old_value", "count"),
        new_numeric=("candidate_value", "count"),
        numeric_gain=("numeric_gain", "sum"),
    ).reset_index().to_csv(args.output / "coverage_delta.csv", index=False)
    if sha(old_path) != old_sha:
        raise ValueError("Immutable input changed")
    qa = dict(
        status="PASS",
        original_missing=488,
        rows=len(candidate),
        numeric_gain=int(delta.numeric_gain.sum()),
        numeric_total=int(typed.value.notna().sum()),
        remaining_missing=int(typed.value.isna().sum()),
        reasons=reasons.reason.value_counts().to_dict(),
        new_source_cells=len(source),
        new_source_numeric=int(source.value.notna().sum()),
        source_only_cells=int(outside._merge.eq("left_only").sum()),
        recaptured_old_numeric=int(
            (delta.old_value.notna() & delta.new_capture_value.notna()).sum()
        ),
        recaptured_numeric_conflicts=int(delta.numeric_revision_conflict.sum()),
        old_numeric_and_provenance_unchanged=True,
        exact_year_codes_unchanged=True,
        missing_imputed=0,
        models_fitted=0,
        boundary_equivalence_proven=0,
    )
    write_json(args.output / "QA.json", qa)
    write_json(
        args.output / "execution.json",
        dict(requests=client.network_request_count, cache_hits=client.cache_hit_count),
    )
    write_json(
        args.output / "manifest.json",
        dict(
            source_sha256=sha(__file__),
            input_path=old_path.relative_to(args.repo).as_posix(),
            input_sha256=old_sha,
            helper_hashes={
                str(p.relative_to(args.repo)): sha(p)
                for p in [
                    args.repo / "enrichment/src/sber_enrichment/round4_economy.py",
                    args.repo / "enrichment/src/sber_enrichment/bdpmo.py",
                    args.repo / "enrichment/uv.lock",
                ]
            },
            output_hashes={
                p.name: sha(p)
                for p in sorted(args.output.iterdir())
                if p.is_file() and p.name not in {"manifest.json", "execution.json"}
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
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    run(args)


if __name__ == "__main__":
    main()
