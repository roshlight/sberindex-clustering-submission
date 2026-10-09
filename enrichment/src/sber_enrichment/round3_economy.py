"""Third-round primary BDPMO totals; exact historical codes, no imputation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

from .bdpmo import BdpmoClient
from .economy_bulk import normalized, sha

SPECS = {
    "8109001": {
        "prefix": "Инвестиции в основной капитал, осуществляемые организациями",
        "period": "17",
        "period_label": "значение показателя за год",
        "extra": {"istinv": "8"},
    },
    "8401011": {
        "prefix": "Отгружено товаров собственного производства",
        "period": "79",
        "period_label": "январь-декабрь",
        "extra": {},
    },
    "8401003": {
        "prefix": "Оборот розничной торговли",
        "period": "79",
        "period_label": "январь-декабрь",
        "extra": {},
    },
}
BASE = {"Pokazateli", "munr", "tippos", "oktmo", "okved2", "god", "period"}
YEARS = {"2023", "2024"}


def parse_form(document, indicator):
    spec = SPECS[indicator]
    options = {}
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
        by_code = {}
        for (_, code), label in zip(pairs, labels, strict=True):
            label = normalized(label)
            if code in by_code and by_code[code] != label:
                raise ValueError(f"Conflicting source labels {name}/{code}")
            by_code[code] = label
        options[name] = by_code
    if set(options) != BASE | set(spec["extra"]):
        raise ValueError(f"Unexpected dimensions {list(options)}")
    if set(options["Pokazateli"]) != {indicator}:
        raise ValueError("Wrong source indicator")
    if not options["Pokazateli"][indicator].casefold().startswith(spec["prefix"].casefold()):
        raise ValueError("Indicator title differs")
    period_labels = {spec["period_label"]}
    if indicator == "8109001":
        period_labels.add("значение за год")
    else:
        period_labels.add("за январь-декабрь")
    if options["period"].get(spec["period"], "").casefold() not in period_labels:
        raise ValueError("Annual period meaning differs")
    total = options["okved2"].get("101", "").casefold()
    if total not in {
        "всего по обследуемым видам экономической деятельности",
        "всего по обследуемым видам экономической деятельности (101.аг)",
    }:
        raise ValueError("Activity total meaning differs")
    if spec["extra"] and options["istinv"].get("8") != "Всего":
        raise ValueError("Investment financing total differs")
    if not YEARS & options["god"].keys():
        raise ValueError("No requested source years")
    return options


def payload_for(options, indicator, codes):
    spec = SPECS[indicator]
    if not codes or len(codes) != len(set(codes)) or not set(codes) <= options["oktmo"].keys():
        raise ValueError("Codes must be unique exact source options")
    counts = Counter(options["oktmo"].values())
    if len(codes) > 1 and any(counts[options["oktmo"][c]] != 1 for c in codes):
        raise ValueError("Ambiguous full source caption")
    values = {k: list(v) for k, v in options.items()}
    years = sorted(YEARS & options["god"].keys())
    values.update(oktmo=codes, okved2=["101"], god=years, period=[spec["period"]])
    values.update({k: [v] for k, v in spec["extra"].items()})
    z = [k for k in options if k not in {"oktmo", "god"}]
    return {
        "Qry": ";".join(k + ":" + ",".join(v) for k, v in values.items()) + ";",
        "QryGm": ";".join(f"{k}_z:{i}" for i, k in enumerate(z, 1)) + ";oktmo_b:1;god_s:1;",
        "QryFootNotes": ";",
        "YearsList": ";".join(years) + ";",
        "tbl": "1",
        "Format": "CSV",
        "YearFrom": years[0],
        "YearTo": years[-1],
    }


def parse_response(document, options, indicator, codes):
    """Only full unambiguous captions with verified units/total/period are accepted."""
    payload_for(options, indicator, codes)
    if "<html" in document.casefold():
        raise ValueError("HTML is not a data export")
    spec = SPECS[indicator]
    labels = {options["oktmo"][code]: code for code in codes}
    selected = {"okved2": "101", "period": spec["period"], **spec["extra"]}
    # Query grouping and CSV section order follow each region's dimension order.
    suffix = "".join(", " + options[k][selected[k]] for k in options if k in selected)
    title, header, section = None, None, None
    records, seen = [], set()
    for line, raw in enumerate(csv.reader(io.StringIO(document), delimiter=";"), 1):
        cells = [normalized(v) for v in raw]
        if not any(cells):
            continue
        first = cells[0]
        if first.casefold().startswith(spec["prefix"].casefold()):
            if title is not None:
                raise ValueError("Unterminated source section")
            title, header = normalized(";".join(raw)), None
            continue
        if not first and any(cells[1:]):
            if (
                title is None
                or not title.endswith(suffix)
                or not re.search(r", (?:тысяч[аи]?|тыс\.) рублей,", title)
                or "без субъектов малого предпринимательства" not in title
                or cells[-1] != ""
            ):
                raise ValueError("Section lacks verified units, coverage, total or annual period")
            header = cells[1:-1]
            if not header or len(header) != len(set(header)) or not set(header) <= YEARS:
                raise ValueError("Invalid year header")
            if not set(header) <= options["god"].keys():
                raise ValueError("CSV year absent from form")
            section, title = title, None
            continue
        if title is not None:
            if len(cells) != 1:
                raise ValueError("Unexpected cells inside source section")
            title = normalized(title + " " + first)
            continue
        if header is None or cells[-1] != "" or len(cells) < len(header) + 2:
            raise ValueError("Row outside an explicit year header")
        label = normalized(";".join(raw[: -len(header) - 1]))
        if label not in labels:
            raise ValueError("Unknown/truncated municipality caption")
        code = labels[label]
        for year, cell in zip(header, cells[-len(header) - 1 : -1], strict=True):
            if (code, year) in seen:
                raise ValueError("Duplicate municipality-year cell")
            seen.add((code, year))
            compact = cell.replace(" ", "")
            if re.fullmatch(r"[0-9]+(?:[,.][0-9]+)?", compact):
                value, status = str(Decimal(compact.replace(",", "."))), "observed"
            elif cell in {"", "-", "…", "...", "х", "x", "X", "к", "К"}:
                value, status = None, "unresolved_source_marker"
            else:
                raise ValueError(f"Unknown source numeric cell {cell!r}")
            records.append(
                dict(
                    source_oktmo=code,
                    year=year,
                    indicator_id=indicator,
                    value=value,
                    value_raw=cell,
                    value_status=status,
                    unit="thousand_rubles",
                    source_csv_row=line,
                    source_section=section,
                    source_oktmo_name=label,
                    period_code=spec["period"],
                    activity_code="101",
                )
            )
    if title is not None:
        raise ValueError("Unterminated section")
    return records


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(args.links, dtype=str, keep_default_na=False)
    if args.limit_regions:
        regions = sorted(links.source_region_url.unique())[: args.limit_regions]
        links = links[links.source_region_url.isin(regions)]
    client = BdpmoClient(
        args.cache, interval=args.interval, ca_bundle=args.ca_bundle, cache_only=args.cache_only
    )
    statuses, rows, forms = [], [], []
    try:
        for region, group in links.groupby("source_region_url", sort=True):
            for indicator in SPECS:
                key = region.split("/")[-2] + "_" + indicator
                try:
                    raw, form_meta = client.request(
                        "POST", region + "DBInet.cgi", {"pl": indicator}
                    )
                    options = parse_form(raw.decode("cp1251"), indicator)
                    forms.append(dict(region=region, indicator_id=indicator, **form_meta))
                except Exception as error:
                    statuses.append(dict(key=key, status="form_unavailable", error=str(error)))
                    continue
                candidates = sorted(set(group.source_oktmo) & options["oktmo"].keys())
                counts = Counter(options["oktmo"].values())
                unique = [c for c in candidates if counts[options["oktmo"][c]] == 1]
                groups = [unique[i : i + 100] for i in range(0, len(unique), 100)]
                groups += [[c] for c in candidates if c not in unique]
                for batch in groups:
                    payload = payload_for(options, indicator, batch)
                    try:
                        raw, meta = client.request("POST", region + "DBInet.cgi", payload)
                        cells = parse_response(raw.decode("cp1251"), options, indicator, batch)
                        for cell in cells:
                            cell.update(
                                source_region_url=region,
                                source_sha256=meta["raw_sha256"],
                                request_key=meta["request_key"],
                                form_request_key=form_meta["request_key"],
                                retrieved_at=meta["retrieved_at"],
                            )
                        rows.extend(cells)
                        statuses.append(
                            dict(
                                key=key,
                                status="parsed",
                                rows=len(cells),
                                codes=batch,
                                request_key=meta["request_key"],
                            )
                        )
                    except Exception as error:
                        statuses.append(
                            dict(key=key, status="response_review", codes=batch, error=str(error))
                        )
                print(
                    json.dumps(dict(key=key, parsed_rows=len(rows)), ensure_ascii=False), flush=True
                )
                (args.output / "collection_status.json").write_text(
                    json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    finally:
        client.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No parsed data; inspect source cache and statuses")
    if frame.duplicated(["source_region_url", "source_oktmo", "year", "indicator_id"]).any():
        raise ValueError("Duplicate primary cells")
    frame.to_csv(args.output / "source_cells.csv", index=False, encoding="utf-8")
    grid = links.merge(pd.DataFrame({"indicator_id": list(SPECS)}), how="cross")
    panel = grid.merge(
        frame,
        on=["source_region_url", "source_oktmo", "year", "indicator_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_prior", ""),
    )
    panel["value_status"] = panel.value_status.fillna("not_returned_or_unaccepted")
    panel["value"] = pd.to_numeric(panel.value, errors="raise")
    panel["geographic_equivalence_proven"] = None
    panel["source_url"] = panel.source_region_url + "DBInet.cgi"
    panel["publication_vintage"] = "retrieved 2026-09-16; first publication/revisions unknown"
    panel.to_csv(args.output / "annual_candidate_panel.csv", index=False, encoding="utf-8")
    with duckdb.connect() as db:
        db.register("panel", panel)
        db.execute(
            "COPY panel TO ? (FORMAT PARQUET)",
            [str(args.output / "annual_candidate_panel.parquet")],
        )
    coverage = panel.groupby(["indicator_id", "year", "value_status"], dropna=False).size()
    coverage.rename("cells").reset_index().to_csv(args.output / "coverage.csv", index=False)
    (args.output / "form_inventory.json").write_text(
        json.dumps(forms, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = dict(
        rows=len(panel),
        numeric=int(panel.value.notna().sum()),
        zeros=int(panel.value.eq(0).sum()),
        source_cells=len(frame),
        municipalities=panel.territory_id.nunique(),
        links_sha256=sha(args.links),
        script_sha256=sha(__file__),
        cache_only=args.cache_only,
        request_count=client.network_request_count,
        cached_count=client.cache_hit_count,
        boundary_equivalence_proven=0,
        values_imputed=0,
    )
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("links", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--limit-regions", type=int)
    parser.add_argument("--interval", type=float, default=0.7)
    parser.add_argument("--cache-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
