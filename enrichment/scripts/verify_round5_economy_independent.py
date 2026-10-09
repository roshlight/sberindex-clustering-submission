"""Independent read-only reconstruction of accepted round5 economy source cells."""

import argparse
import csv
import hashlib
import html
import io
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def whitespace(value):
    return " ".join(value.split())


def form_options(text, field):
    section = re.search(
        rf'<SELECT\b[^>]*NAME="{field}"[^>]*>(.*?)</SELECT>', text, re.I | re.S
    ).group(1)
    labels = [
        whitespace(html.unescape(re.sub(r"<[^>]+>", "", value)))
        for value in re.findall(r"<OPTION[^>]*>(.*?)(?=<OPTION|$)", section, re.I | re.S)
    ]
    codes = {
        int(index): value for index, value in re.findall(rf'p_{field}\[(\d+)\]="([^"]*)";', text)
    }
    assert set(codes) == set(range(len(labels))), (field, len(codes), len(labels))
    return {codes[index]: label for index, label in enumerate(labels)}


def source_cells(text):
    """Use displayed annual columns and raw captions; no regional production parser."""
    parsed = list(csv.reader(io.StringIO(text), delimiter=";"))
    result, years, section, caption_lines = {}, {}, "", []
    for row in parsed:
        if not row or not any(s.strip() for s in row):
            continue
        if not row[0].strip() and any(s.strip() in {"2023", "2024"} for s in row):
            years = {i: int(s.strip()) for i, s in enumerate(row) if s.strip() in {"2023", "2024"}}
            section = whitespace(" ".join(caption_lines))
            caption_lines = []
        elif years and len(row) > max(years) and row[0].strip():
            for column, year in years.items():
                key = (section, whitespace(row[0]), year)
                if key in result:
                    raise ValueError(f"Ambiguous displayed cell {key}")
                result[key] = row[column].strip()
        elif len(row) == 1:
            caption_lines.append(row[0])
    return result


def number(value):
    return Decimal(value.replace("\xa0", "").replace(" ", "").replace(",", "."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo, raw = args.repo.resolve(), args.raw.resolve()
    report = repo / "reports/data_round5/economy"
    old_path = repo / "reports/data_round4/economy/annual_candidate_panel.csv"
    candidate_path = report / "annual_candidate_panel.csv"
    old, candidate = rows(old_path), rows(candidate_path)
    fields = ("territory_id", "year", "indicator_id")

    def key(row):
        return tuple(row[field] for field in fields)

    prior, current = {key(r): r for r in old}, {key(r): r for r in candidate}
    assert len(prior) == len(old) == len(current) == len(candidate) == 12096
    assert set(prior) == set(current)
    old_observed = [k for k, r in prior.items() if r["value"]]
    gains = {k for k, r in current.items() if r["value"] and not prior[k]["value"]}
    assert len(old_observed) == 11608 and len(gains) == 64
    for k in old_observed:
        for column, value in prior[k].items():
            assert value == current[k][column], (k, column, value, current[k][column])
    missing = {k for k, r in current.items() if not r["value"]}
    assert len(missing) == 424
    assert {key(r) for r in rows(report / "new_numeric_cells.csv")} == gains
    assert {key(r) for r in rows(report / "remaining_missing_cells.csv")} == missing
    evidence = rows(report / "recovered_source_cells.csv")
    assert len(evidence) == 146
    observed = [r for r in evidence if r["value_status"] == "observed"]
    assert len(observed) == 142
    lookups, forms, traces = {}, {}, []
    inputs = {
        old_path,
        candidate_path,
        report / "recovered_source_cells.csv",
        report / "new_numeric_cells.csv",
        report / "remaining_missing_cells.csv",
    }
    for row in observed:
        request_key, form_key = row["request_key"], row["form_request_key"]
        source = raw / "cache" / f"{request_key}.raw"
        form = raw / "cache" / f"{form_key}.raw"
        metadata = source.with_suffix(".json")
        form_metadata = form.with_suffix(".json")
        inputs.update([source, form, metadata, form_metadata])
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        form_meta = json.loads(form_metadata.read_text(encoding="utf-8"))
        assert sha(form) == form_meta["raw_sha256"] and form_meta["status_code"] == 200
        assert sha(source) == meta["raw_sha256"] == row["source_sha256"]
        assert meta["status_code"] == 200 and row["source_url"] == meta["url"]
        qry = dict(pair.split(":", 1) for pair in meta["payload"]["Qry"].split(";") if pair)
        assert row["source_oktmo"] in qry["oktmo"].split(",")
        assert row["year"] in qry["god"].split(",")
        assert row["indicator_id"] in qry["Pokazateli"].split(",")
        assert row["period_code"] in qry["period"].split(",")
        if request_key not in lookups:
            lookups[request_key] = source_cells(source.read_bytes().decode("cp1251"))
        if form_key not in forms:
            text = form.read_bytes().decode("cp1251")
            forms[form_key] = {
                field: form_options(text, field) for field in ["oktmo", "period", "tippos"]
            }
        label = forms[form_key]["oktmo"][row["source_oktmo"]]
        assert label == row["source_option_label"]
        # Accepted captions must literally identify the exact selected option.
        assert label == row["source_oktmo_name"]
        period = forms[form_key]["period"][row["period_code"]]
        assert period in {
            "январь-декабрь",
            "за январь-декабрь",
            "значение показателя за год",
            "значение за год",
        }
        assert period in row["source_section"]
        scope_normal = "без субъектов малого предпринимательства" in row["source_section"]
        # Literal typo is present in Primorye's official heading; restrict the exception.
        scope_typo = (
            row["source_region_url"] == "https://rosstat.gov.ru/dbscripts/munst/munst05/"
            and row["indicator_id"] == "8401003"
            and "без субъектов малого редпринимательства" in row["source_section"]
            and "без субъектов малого редпринимательства" in form.read_bytes().decode("cp1251")
        )
        assert scope_normal or scope_typo
        assert row["source_unit_literal"] in {"тысяч рублей", "тысяча рублей"}
        assert row["source_unit_literal"] in row["source_section"]
        cell_key = (row["source_section"], label, int(row["year"]))
        raw_value = lookups[request_key][cell_key]
        assert number(raw_value) == number(row["value_raw"]) == number(row["value"])
        matches = [
            k
            for k, r in current.items()
            if r["source_region_url"] == row["source_region_url"]
            and r["source_oktmo"] == row["source_oktmo"]
            and r["year"] == row["year"]
            and r["indicator_id"] == row["indicator_id"]
        ]
        assert len(matches) == 1
        k = matches[0]
        assert number(current[k]["value"]) == number(raw_value)
        if k not in gains:
            assert number(prior[k]["value"]) == number(raw_value)
        else:
            assert forms[form_key]["tippos"]["7"] == "Городской округ"
            assert "7" in qry["tippos"].split(",")
        traces.append(
            dict(
                territory_id=k[0],
                year=k[1],
                indicator_id=k[2],
                source_oktmo=row["source_oktmo"],
                source_sha256=row["source_sha256"],
                value=str(number(raw_value)),
                is_new=k in gains,
            )
        )
    assert len({(t["territory_id"], t["year"], t["indicator_id"]) for t in traces}) == 142
    assert sum(t["is_new"] for t in traces) == 64
    gain_sources = Counter((current[k]["source_region_url"], k[2]) for k in gains)
    assert sorted(gain_sources.values()) == [22, 42]
    assert len({k[0] for k in gains}) == 32
    output = args.output or report / "independent_QA.json"
    result = dict(
        status="PASS",
        independent_parser=True,
        production_modules_imported=False,
        prior_numeric_cells_unchanged=11608,
        candidate_numeric_cells=11672,
        numeric_gain=64,
        remaining_missing=424,
        raw_observed_verified=142,
        raw_old_exact=78,
        new_municipalities=32,
        raw_displayed_years_verified=True,
        raw_form_code_labels_verified=True,
        raw_units_scope_period_verified=True,
        geography_equivalence_newly_proven=False,
        model_adoption=False,
        official_primorye_heading_typo_preserved="без субъектов малого редпринимательства",
        gain_sources=[
            dict(region_url=k[0], indicator_id=k[1], count=v)
            for k, v in sorted(gain_sources.items())
        ],
        inputs=[
            dict(
                path=p.relative_to(repo).as_posix()
                if p.is_relative_to(repo)
                else "work/data-round5/economy/" + p.relative_to(raw).as_posix(),
                sha256=sha(p),
            )
            for p in sorted(inputs)
        ],
        source_cells=traces,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"inputs", "source_cells"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
