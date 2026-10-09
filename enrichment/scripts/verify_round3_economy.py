"""Audit every delivered numeric/marker cell against immutable primary CSV bytes."""

import argparse
import csv
import hashlib
import html
import io
import json
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data / "annual_candidate_panel.csv", dtype=str, keep_default_na=False)
    keys = ["territory_id", "year", "indicator_id"]
    assert len(df) == 2016 * 2 * 3 and not df.duplicated(keys).any()
    assert set(df.year) == {"2023", "2024"}
    assert set(df.indicator_id) == {"8109001", "8401011", "8401003"}
    assert not df.geographic_equivalence_proven.ne("").any()
    links = pd.read_csv(args.links, dtype=str, keep_default_na=False)
    accepted = df[df.request_key.ne("")].copy()
    expected_keys = {
        (r.territory_id, r.year, indicator)
        for r in links.itertuples()
        for indicator in ["8109001", "8401011", "8401003"]
    }
    assert set(df[keys].itertuples(index=False, name=None)) == expected_keys
    matched = df.merge(
        links,
        on=["territory_id", "year"],
        suffixes=("", "_ref"),
        validate="many_to_one",
        how="left",
        indicator=True,
    )
    assert len(matched) == len(df) and matched["_merge"].eq("both").all()
    for col in ["source_oktmo", "official_oktmo_8", "source_region_url", "code_reference_date"]:
        assert matched[col].equals(matched[col + "_ref"])
    raw_cache = {}
    form_cache = {}
    request_cache = {}
    for row in accepted.to_dict("records"):
        key = row["request_key"]
        if key not in raw_cache:
            meta = json.loads((args.cache / (key + ".json")).read_text(encoding="utf-8"))
            path = args.cache / (key + ".raw")
            assert sha(path) == meta["raw_sha256"] == row["source_sha256"]
            request = {k: meta[k] for k in ["method", "url", "payload"]}
            assert hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() == key
            assert meta["status_code"] == 200 and meta["url"] == row["source_url"]
            request_cache[key] = dict(
                x.split(":", 1) for x in meta["payload"]["Qry"].split(";") if x
            )
            raw_cache[key] = list(
                csv.reader(io.StringIO(path.read_bytes().decode("cp1251")), delimiter=";")
            )
        records = raw_cache[key]
        query = request_cache[key]
        assert query["Pokazateli"] == row["indicator_id"]
        assert row["source_oktmo"] in query["oktmo"].split(",")
        assert row["year"] in query["god"].split(",")
        assert query["okved2"] == "101" and query["period"] == row["period_code"]
        if row["indicator_id"] == "8109001":
            assert query["istinv"] == "8"
        line_number = Decimal(row["source_csv_row"])
        assert line_number == line_number.to_integral_value()
        line = int(line_number) - 1
        original = records[line]
        header = next(r for r in reversed(records[:line]) if r and not r[0].strip() and any(r[1:]))
        years = [s.strip() for s in header[1:-1]]
        assert row["year"] in years
        cell = original[-len(years) - 1 : -1][years.index(row["year"])].strip()
        caption = " ".join(";".join(original[: -len(years) - 1]).split())
        assert caption == row["source_oktmo_name"]
        assert cell == row["value_raw"]
        if row["value_status"] == "observed":
            assert Decimal(cell.replace(" ", "").replace(",", ".")) == Decimal(row["value"])
            assert Decimal(row["value"]) >= 0
        else:
            assert row["value"] == "" and cell in {"", "-", "…", "...", "х", "x", "X", "к", "К"}
        fkey = row["form_request_key"]
        if fkey not in form_cache:
            fp = args.cache / (fkey + ".raw")
            fm = json.loads((args.cache / (fkey + ".json")).read_text(encoding="utf-8"))
            assert sha(fp) == fm["raw_sha256"]
            assert fm["payload"] == {"pl": row["indicator_id"]}
            assert fm["url"] == row["source_url"]
            form_cache[fkey] = fp.read_bytes().decode("cp1251")
        document = form_cache[fkey]
        coded = re.findall(r'p_oktmo\[(\d+)\]="([^"\r\n]+)"', document)
        indices = [int(index) for index, code in coded if code == row["source_oktmo"]]
        select = re.search(r'<SELECT\s+NAME="oktmo"[^>]*>(.*?)</SELECT>', document, re.I | re.S)
        captions = re.findall(r"<OPTION\b[^>]*>([^<]*)", select.group(1), re.I)
        assert indices and all(
            " ".join(html.unescape(captions[i]).split()) == caption for i in indices
        )
        assert row["unit"] == "thousand_rubles"
        assert row["period_code"] == ("17" if row["indicator_id"] == "8109001" else "79")
        assert row["activity_code"] == "101"
    checks = {
        "status": "PASS",
        "candidate_grid": len(df),
        "accepted_source_cells_verified": len(accepted),
        "numeric_cells": int(df.value_status.eq("observed").sum()),
        "numeric_zero_cells": int(pd.to_numeric(df.value, errors="coerce").eq(0).sum()),
        "responses_rehashed": len(raw_cache),
        "forms_rehashed": len(form_cache),
        "duplicate_keys": 0,
        "fabricated_or_imputed_values": 0,
        "geographic_equivalence_claims": 0,
        "csv_sha256": sha(args.data / "annual_candidate_panel.csv"),
        "verifier_sha256": sha(Path(__file__)),
    }
    args.out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
