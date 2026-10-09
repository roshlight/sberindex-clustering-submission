"""Independent offline source-cell and delta audit; no production parser imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import itertools
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import duckdb

INDICATORS = {"8109001": "17", "8401011": "79", "8401003": "79"}
PREFIXES = {
    "8109001": "Инвестиции в основной капитал, осуществляемые организациями",
    "8401011": "Отгружено товаров собственного производства",
    "8401003": "Оборот розничной торговли",
}
SCOPE_EXCEPTIONS = {
    ("munst05", "8401003"): "без субъектов малого редпринимательства",
    ("munst36", "8401011"): "без субъектов малого предприниматетельства",
    ("munst70", "8401011"): "безсубъектов малого предпринимательства",
}
KEY = ("territory_id", "year", "indicator_id")
SOURCE_KEY = ("source_region_url", "source_oktmo", "year", "indicator_id")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(text):
    return " ".join(html.unescape(text).split())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, columns):
    return tuple(row[c] for c in columns)


def indexed(rows, columns):
    result = {key(row, columns): row for row in rows}
    require(len(result) == len(rows), f"Duplicate key: {columns}")
    return result


def number(value):
    return None if value == "" else Decimal(value)


class Options(HTMLParser):
    """Read actual HTML option boundaries rather than the delivered form inventory."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.name = None
        self.current = None
        self.labels = {}

    def flush(self):
        if self.current is not None:
            self.labels[self.name].append(norm("".join(self.current)))
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "select":
            self.name = dict(attrs).get("name")
            if self.name:
                self.labels[self.name] = []
        elif tag == "option" and self.name:
            self.flush()
            self.current = []

    def handle_endtag(self, tag):
        if tag == "option" and self.name:
            self.flush()
        elif tag == "select" and self.name:
            self.flush()
            self.name = None

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)


def form_dimensions(document):
    parser = Options()
    parser.feed(document)
    code_arrays = defaultdict(dict)
    for name, index, code in re.findall(r'p_(\w+)\[(\d+)\]\s*=\s*"([^"]+)"', document):
        require(index not in code_arrays[name], "Repeated JS array index")
        code_arrays[name][index] = code
    result = {}
    for name, array in code_arrays.items():
        labels = parser.labels[name]
        require(set(array) == {str(i) for i in range(len(labels))}, "HTML/JS index mismatch")
        dimension = defaultdict(set)
        for i, label in enumerate(labels):
            dimension[array[str(i)]].add(label)
        result[name] = dict(dimension)
    return result


def captured(cache, request_key):
    metadata = json.loads((cache / f"{request_key}.json").read_text("utf-8"))
    raw = cache / f"{request_key}.raw"
    request = {name: metadata[name] for name in ("method", "url", "payload")}
    computed = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    require(computed == request_key == metadata["request_key"], "Request fingerprint differs")
    require(metadata["status_code"] == 200, "Non-success source response")
    require(metadata["method"] == "POST", "Unexpected request method")
    require(
        re.fullmatch(
            r"https://rosstat\.gov\.ru/dbscripts/munst/munst\d+/DBInet\.cgi", metadata["url"]
        ),
        "Unexpected source host/endpoint",
    )
    require(metadata["final_url"] == metadata["url"], "Unexpected redirect")
    require(raw.stat().st_size == metadata["bytes"], "Raw byte count differs")
    require(digest(raw) == metadata["raw_sha256"], "Raw fingerprint differs")
    return raw.read_bytes().decode("cp1251"), metadata


def independent_cells(cache, request_key, form_key, indicator):
    document, metadata = captured(cache, request_key)
    form_document, form_metadata = captured(cache, form_key)
    require(form_metadata["url"] == metadata["url"], "Different form/response endpoints")
    require(form_metadata["payload"] == {"pl": indicator}, "Wrong form indicator request")
    dimensions = form_dimensions(form_document)
    payload = metadata["payload"]
    query = {}
    for chunk in payload["Qry"].split(";"):
        if chunk:
            name, values = chunk.split(":", 1)
            require(name not in query, "Duplicate request dimension")
            query[name] = values.split(",")
    required_dims = {"Pokazateli", "munr", "tippos", "oktmo", "okved2", "god", "period"}
    finance = set(dimensions) & {"istinv", "p_istinv"}
    require(len(finance) == (1 if indicator == "8109001" else 0), "Financing dimension differs")
    require(set(query) == required_dims | finance, "Unexpected requested dimension")
    require(set(dimensions) - {"footnotes"} == set(query), "Unknown form dimension")
    for name, codes in query.items():
        require(len(codes) == len(set(codes)), "Duplicate request code")
        for code in codes:
            require(
                code in dimensions[name] and len(dimensions[name][code]) == 1, "Ambiguous option"
            )
    require(query["Pokazateli"] == [indicator], "Wrong indicator")
    require(query["okved2"] == ["101"], "Wrong activity aggregate")
    require(query["period"] == [INDICATORS[indicator]], "Wrong annual period")
    require(set(query["god"]) <= {"2023", "2024"}, "Year outside protocol")
    require(set(query["tippos"]) <= {"10", "7", "20"}, "Lower-level settlement requested")
    for name in finance:
        require(query[name] == ["8"], "Non-total investment financing")
    require(payload["Format"] == "CSV", "Wrong export format")
    require(payload["YearsList"].split(";")[:-1] == query["god"], "YearsList differs")
    require(payload["YearFrom"] == min(query["god"]), "YearFrom differs")
    require(payload["YearTo"] == max(query["god"]), "YearTo differs")
    groups = dict(chunk.split(":") for chunk in payload["QryGm"].split(";") if chunk)
    require(groups.pop("oktmo_b") == "1" and groups.pop("god_s") == "1", "Wrong table axes")
    require(set(groups) == {n + "_z" for n in query if n not in {"oktmo", "god"}}, "Extra axis")

    single = {
        name: {c: next(iter(labels)) for c, labels in values.items() if len(labels) == 1}
        for name, values in dimensions.items()
    }
    require(
        single["Pokazateli"][indicator].casefold().startswith(PREFIXES[indicator].casefold()),
        "Form title differs",
    )
    require(
        single["okved2"]["101"].casefold()
        in {
            "всего по обследуемым видам экономической деятельности",
            "всего по обследуемым видам экономической деятельности (101.аг)",
        },
        "Aggregate label differs",
    )
    if finance:
        require(single[next(iter(finance))]["8"].casefold() == "всего", "Financing label differs")
    allowed_periods = (
        {"значение показателя за год", "значение за год"}
        if finance
        else {"январь-декабрь", "за январь-декабрь"}
    )
    require(
        single["period"][INDICATORS[indicator]].casefold() in allowed_periods,
        "Period label differs",
    )
    upper = {
        "10": "муниципальный район",
        "7": "городской округ, городской округ с внутригородским делением",
        "20": "муниципальный округ",
    }
    require(
        all(single["tippos"][c].casefold() == upper[c] for c in query["tippos"]),
        "Type label differs",
    )
    labels = {single["oktmo"][code]: code for code in query["oktmo"]}
    require(len(labels) == len(query["oktmo"]), "Duplicate requested full label")
    notes = single.get("footnotes", {})
    sentinel = any(
        "999999999" in n and "условное обозначение конфиденциальности" in n for n in notes.values()
    )
    region = metadata["url"].split("/")[-2]
    section, years, result = None, None, []
    for row_index, row in enumerate(csv.reader(io.StringIO(document), delimiter=";"), 1):
        row = [norm(item) for item in row]
        if not any(row):
            continue
        if row[0].casefold().startswith(PREFIXES[indicator].casefold()):
            require(years is not None or section is None, "Unterminated source header")
            section, years = norm(";".join(row)), None
            continue
        if row[0] == "":
            require(section is not None and row[-1] == "", "Missing section/trailing delimiter")
            years = row[1:-1]
            require(
                years and len(set(years)) == len(years) and set(years) <= set(query["god"]),
                "Invalid year columns",
            )
            require(
                any(", " + single["tippos"][code] + ", " in section for code in query["tippos"]),
                "CSV lower-level type",
            )
            # The source can place period before activity/financing (e.g. munst61).
            # Reconstruct its exact selected tail from captured grouping positions.
            selected_dimensions = {"okved2", "period"} | finance
            ordered_dimensions = [
                name.removesuffix("_z")
                for name in sorted(groups, key=lambda name: int(groups[name]))
                if name.removesuffix("_z") in selected_dimensions
            ]
            tail = ", " + ", ".join(single[name][query[name][0]] for name in ordered_dimensions)
            require(section.casefold().endswith(tail.casefold()), "CSV selected totals differ")
            continue
        if years is None:
            require(section is not None and len(row) == 1, "Unknown multiline header")
            section = norm(section + " " + row[0])
            continue
        require(row[-1] == "" and len(row) >= len(years) + 2, "Invalid data column count")
        caption = norm(";".join(row[: -len(years) - 1]))
        code = labels.get(caption)
        rule = "exact_full_source_option"
        if code is None:
            require(len(query["oktmo"]) == 1, "Unknown caption in multi-code response")
            code = query["oktmo"][0]
            label = single["oktmo"][code]
            require(
                label.endswith("...") and caption.startswith(label[:-3].rstrip()),
                "Unknown isolated caption",
            )
            rule = "explicit_UI_ellipsis_resolved_by_isolated_exact_code_query"
        unit_match = re.search(
            r", (тысяча рублей|тысяч рублей|тысячи рублей|тыс\.\s*рублей),", section, re.I
        )
        require(unit_match is not None, "Unverified CSV monetary unit")
        normal_scope = "без субъектов малого предпринимательства"
        alias = SCOPE_EXCEPTIONS.get((region, indicator))
        require(
            normal_scope in section.casefold() or (alias and alias in section.casefold()),
            "Unverified exclusion of small businesses",
        )
        scope_rule = (
            "explicit_normal_scope"
            if normal_scope in section.casefold()
            else "reviewed_literal_source_typo:" + alias
        )
        for year, token in zip(years, row[-len(years) - 1 : -1], strict=True):
            value = None
            if re.fullmatch(r"\d+(?:[.,]\d+)?", token.replace(" ", "")):
                value = Decimal(token.replace(" ", "").replace(",", "."))
                status = "observed"
                if sentinel and value == Decimal("999999999"):
                    value, status = None, "suppressed_explicit_source_footnote"
            else:
                require(
                    token in {"", "-", "…", "...", "х", "x", "X", "к", "К"}, "Unknown source marker"
                )
                status = "unresolved_source_marker"
            result.append(
                dict(
                    source_oktmo=code,
                    year=year,
                    indicator_id=indicator,
                    value=value,
                    value_raw=token,
                    value_status=status,
                    source_csv_row=row_index,
                    source_section=section,
                    source_oktmo_name=caption,
                    source_option_label=single["oktmo"][code],
                    caption_evidence_rule=rule,
                    source_unit_literal=unit_match.group(1),
                    scope_evidence_rule=scope_rule,
                    source_footnotes=notes,
                )
            )
    require(result, "No independently parsed cells")
    indexed(result, ("source_oktmo", "year", "indicator_id"))
    return result, metadata


def audit(repo: Path, cache: Path):
    directory = repo / "reports/data_round4/economy"
    old_directory = repo / "reports/data_round3/economy"
    old = indexed(records(old_directory / "annual_candidate_panel.csv"), KEY)
    candidate = indexed(records(directory / "annual_candidate_panel.csv"), KEY)
    source_rows = records(directory / "recovered_source_cells.csv")
    source = indexed(source_rows, SOURCE_KEY)
    delta = indexed(records(directory / "delta_all_cells.csv"), KEY)
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["old_input_hashes"].items():
        require(digest(old_directory / name) == expected, f"Old artifact changed: {name}")
    require(
        digest(repo / "enrichment/src/sber_enrichment/round4_economy.py")
        == manifest["script_sha256"],
        "Collector code changed",
    )
    for name, expected in manifest["cache_files"].items():
        require(digest(cache / name) == expected, f"Cache artifact changed: {name}")
    for name, expected in manifest["output_hashes"].items():
        require(digest(directory / name) == expected, f"Output artifact changed: {name}")
    with duckdb.connect() as con:
        ids = con.execute(
            "SELECT DISTINCT territory_id FROM read_parquet(?)",
            [str(repo / "reports/initial_study/assignments.parquet")],
        ).fetchall()
    expected_grid = set(
        itertools.product([str(row[0]) for row in ids], ["2023", "2024"], INDICATORS)
    )
    require(
        len(ids) == 2016 and set(old) == set(candidate) == set(delta) == expected_grid,
        "Frozen grid differs",
    )

    by_request = defaultdict(list)
    for row in source_rows:
        by_request[row["request_key"]].append(row)
    checked = 0
    parsed_statuses = Counter()
    for request, rows in by_request.items():
        require(
            len({r["indicator_id"] for r in rows}) == 1
            and len({r["form_request_key"] for r in rows}) == 1,
            "Mixed request provenance",
        )
        parsed, metadata = independent_cells(
            cache, request, rows[0]["form_request_key"], rows[0]["indicator_id"]
        )
        actual = indexed(rows, ("source_oktmo", "year", "indicator_id"))
        expected = indexed(parsed, ("source_oktmo", "year", "indicator_id"))
        require(set(actual) == set(expected), "Source export cell grid differs")
        for k, item in actual.items():
            e = expected[k]
            for field in (
                "value_raw",
                "value_status",
                "source_section",
                "source_oktmo_name",
                "source_option_label",
                "caption_evidence_rule",
                "source_unit_literal",
                "scope_evidence_rule",
            ):
                require(item[field] == e[field], f"Source cell field differs: {field}/{k}")
            require(number(item["value"]) == e["value"], f"Source value differs: {k}")
            require(int(item["source_csv_row"]) == e["source_csv_row"], "CSV row locator differs")
            require(
                json.loads(item["source_footnotes"]) == e["source_footnotes"],
                "Source footnotes differ",
            )
            require(
                item["source_sha256"] == metadata["raw_sha256"]
                and item["source_url"] == metadata["url"],
                "Cell raw provenance differs",
            )
            require(
                item["source_region_url"] == metadata["url"].removesuffix("DBInet.cgi"),
                "Region differs",
            )
            require(
                item["unit"] == "thousand_rubles" and item["activity_code"] == "101",
                "Unit/activity differs",
            )
            require(item["period_code"] == INDICATORS[item["indicator_id"]], "Cell period differs")
            parsed_statuses[item["value_status"]] += 1
            checked += 1

    counts = Counter()
    numeric_overlap = 0
    coverage = defaultdict(Counter)
    annual_source_keys = set()
    for k, prior in old.items():
        now, change = candidate[k], delta[k]
        annual_source_keys.add(key(prior, SOURCE_KEY))
        incoming = source.get(key(prior, SOURCE_KEY))
        old_value, new_value = number(prior["value"]), number(now["value"])
        incoming_value = number(incoming["value"]) if incoming else None
        numeric_overlap += old_value is not None and incoming_value is not None
        for field in (
            "source_region_url",
            "source_oktmo",
            "official_oktmo_8",
            "code_reference_date",
            "temporal_review_required",
        ):
            require(now[field] == prior[field], f"Annual code/date link changed: {k}/{field}")
        require(now["geographic_equivalence_proven"] == "", "Unsupported geography approval")
        if old_value is not None:
            require(new_value == old_value, f"Old number changed: {k}")
        if now["round4_selection"] == "round4_source":
            require(incoming is not None and new_value == incoming_value, "Wrong source join")
            for field in (
                "value_raw",
                "value_status",
                "source_sha256",
                "request_key",
                "form_request_key",
                "source_section",
            ):
                require(now[field] == incoming[field], f"Incoming source field differs: {field}")
        else:
            require(now["round4_selection"] == "round3_preserved", "Unexpected selection/conflict")
            require(
                all(now[name] == value for name, value in prior.items()),
                f"Preserved row changed: {k}",
            )
        gain = old_value is None and new_value is not None
        require(
            number(change["old_value"]) == old_value
            and number(change["candidate_value"]) == new_value,
            "Delta value differs",
        )
        require(
            number(change["new_capture_value"]) == incoming_value, "Delta capture value differs"
        )
        require(
            change["numeric_gain"] == str(gain) and change["numeric_revision_conflict"] == "False",
            "Delta flags differ",
        )
        counts["old_numeric"] += old_value is not None
        counts["new_numeric"] += new_value is not None
        counts["gain"] += gain
        cov = coverage[(now["indicator_id"], now["year"])]
        cov["denominator"] += 1
        cov["old_numeric"] += old_value is not None
        cov["new_numeric"] += new_value is not None
        cov["numeric_gain"] += gain
    for row in records(directory / "coverage_delta.csv"):
        expected = coverage[(row["indicator_id"], row["year"])]
        for name, count in expected.items():
            require(int(row[name]) == count, "Coverage summary differs")
        require(row["revision_conflicts"] == "0", "Reported revision conflict")
    outside = indexed(records(directory / "source_only_outside_year_link.csv"), SOURCE_KEY)
    require(set(outside) == set(source) - annual_source_keys, "Outside-year quarantine differs")
    require(
        all(number(row["value"]) is None for row in outside.values()),
        "Unexpected outside-year number",
    )
    refusals = records(directory / "refusals_remaining.csv")
    require(
        {r["key"] for r in refusals} == {"munst07_8401003", "munst07_8401011", "munst63_8401003"},
        "Remaining refusals changed",
    )
    for row in refusals:
        text, metadata = captured(cache, row["request_key"])
        require(metadata["raw_sha256"] == row["raw_sha256"], "Refusal SHA differs")
        titles = [
            line
            for line in text.splitlines()
            if line.casefold().startswith(PREFIXES[row["key"].split("_")[1]].casefold())
        ]
        require(titles, "No source title for refusal")
        if row["key"].startswith("munst07"):
            require(
                any("без субъектов малого предпринимательства" not in t.casefold() for t in titles),
                "Scope refusal contradicted",
            )
        else:
            require(any(", -," in t for t in titles), "Unknown-unit refusal contradicted")
    require(
        counts == Counter(old_numeric=10829, new_numeric=11608, gain=779), "Frozen counts differ"
    )
    require(
        checked == 805 and parsed_statuses == Counter(observed=779, unresolved_source_marker=26),
        "Source counts differ",
    )
    return {
        "status": "PASS",
        "network_requests": 0,
        "production_parser_imported": False,
        "candidate_rows": len(candidate),
        "frozen_municipalities": len(ids),
        "independently_replayed_source_cells": checked,
        "csv_requests": len(by_request),
        "html_forms": len({r["form_request_key"] for r in source_rows}),
        "source_statuses": dict(parsed_statuses),
        "counts": dict(counts),
        "old_numeric_changes": 0,
        "old_and_new_capture_numeric_overlap": numeric_overlap,
        "source_revision_stability": "not_estimable_no_numeric_overlap"
        if numeric_overlap == 0
        else "limited_to_requeried_cells",
        "exact_annual_links_preserved": True,
        "outside_year_link_cells": len(outside),
        "outside_year_link_numeric": 0,
        "isolated_ellipsis_cells": sum(
            r["caption_evidence_rule"] != "exact_full_source_option" for r in source_rows
        ),
        "source_999999999_cells": sum(
            r["value_raw"].replace(",0", "") == "999999999" for r in source_rows
        ),
        "remaining_refusals_verified": len(refusals),
        "geography_or_calendar_mask_changes": 0,
        "collector_sha256": manifest["script_sha256"],
        "verifier_sha256": digest(Path(__file__)),
        "evidence_hashes": {
            name: digest(directory / name)
            for name in (
                "annual_candidate_panel.csv",
                "recovered_source_cells.csv",
                "delta_all_cells.csv",
                "coverage_delta.csv",
                "refusals_remaining.csv",
                "form_inventory.json",
            )
        },
        "limitations": [
            "No independent legal boundary approval",
            "Retrieval vintage does not establish original publication date",
            "Scope excludes small businesses; not whole-economy totals",
            "Confidential sentinel has synthetic tests only; absent in canonical recovered cells",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qa = audit(args.repo, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
