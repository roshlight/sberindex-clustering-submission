"""Independently replay every extracted cell from raw XLSX XML, without openpyxl."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import duckdb

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def letter(index):
    text = ""
    while index:
        index, rem = divmod(index - 1, 26)
        text = chr(65 + rem) + text
    return text


def xml_value(cell, shared):
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(NS + "t"))
    value = cell.find(NS + "v")
    if value is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared[int(value.text)]
    return float(value.text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--monthly-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a previous audit")
    source = args.monthly_dir / "city_monthly_price_cells.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW cells AS SELECT * FROM read_parquet('{source.as_posix()}')")
    duplicates = con.sql("""SELECT count(*) FROM (
        SELECT year,month,source_city_code_literal,product_code,count(*) n
        FROM cells GROUP BY ALL HAVING n<>1)""").fetchone()[0]
    if duplicates:
        raise ValueError("Duplicated cell keys")
    checked, monthly = 0, []
    for year in (2023, 2024):
        workbook = args.raw_dir / f"sred_potreb_cen_{year}.xlsx"
        with zipfile.ZipFile(workbook) as archive:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.iter(NS + "t")) for item in shared_root]
            for month in range(1, 13):
                rows = con.execute(
                    """SELECT source_row,source_column,value,value_status,
                    source_city_code_literal,product_code FROM cells WHERE year=? AND month=?""",
                    [year, month],
                ).fetchall()
                expected = {
                    f"{letter(col)}{row}": (value, status) for row, col, value, status, *_ in rows
                }
                city_codes = {f"A{row}": code for row, _, _, _, code, _ in rows}
                product_codes = {f"{letter(col)}3": product for _, col, _, _, _, product in rows}
                seen, seen_cities, seen_products = set(), set(), set()
                # Workbooks have content at sheet1 and months at sheet2..sheet13;
                # validate the workbook relationships, not only ZIP ordering.
                book = ET.fromstring(archive.read("xl/workbook.xml"))
                sheets = book.find(NS + "sheets")
                if sheets[month].attrib["name"] != f"{month:02}({year})":
                    raise ValueError("Unexpected month ordering")
                rid = sheets[month].attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
                target = next(r.attrib["Target"] for r in rels if r.attrib["Id"] == rid)
                xml_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
                with archive.open(xml_path) as stream:
                    for _, element in ET.iterparse(stream, events=("end",)):
                        if element.tag != NS + "c":
                            continue
                        address = element.attrib["r"]
                        if address in expected:
                            value = xml_value(element, shared)
                            stored, status = expected[address]
                            if isinstance(value, float):
                                if stored is None or not math.isfinite(stored) or value != stored:
                                    raise ValueError(f"Numeric mismatch {year}/{month}/{address}")
                                if status != ("observed_zero" if value == 0 else "observed"):
                                    raise ValueError("Wrong numeric status")
                            elif value is None:
                                if stored is not None or status != "source_blank":
                                    raise ValueError("Wrong blank-cell status")
                            elif value in {"...", "…", "-", "—"}:
                                if stored is not None or status != "source_marker":
                                    raise ValueError("Source marker became numeric")
                            else:
                                raise ValueError(f"Unexpected source content {value!r}")
                            seen.add(address)
                        if address in city_codes:
                            if str(int(xml_value(element, shared))) != city_codes[address]:
                                raise ValueError("Wrong source city code at source row")
                            seen_cities.add(address)
                        if address in product_codes:
                            if int(xml_value(element, shared)) != product_codes[address]:
                                raise ValueError("Wrong product code at source column")
                            seen_products.add(address)
                        element.clear()
                if (
                    seen != expected.keys()
                    or seen_cities != city_codes.keys()
                    or seen_products != product_codes.keys()
                ):
                    raise ValueError("Not all source cells/codes were independently replayed")
                checked += len(seen)
                monthly.append(
                    {
                        "year": year,
                        "month": month,
                        "verified_price_cells": len(seen),
                        "verified_city_codes": len(seen_cities),
                        "verified_product_codes": len(seen_products),
                    }
                )
                print(f"Verified raw XML {year}-{month:02}: {len(seen)} cells", flush=True)
    result = {
        "status": "PASS",
        "independent_reader": "stdlib ZIP/XML (not openpyxl)",
        "verified_price_cells": checked,
        "duplicate_keys": duplicates,
        "monthly": monthly,
        "source_parquet_sha256": sha(source),
        "verifier_sha256": sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
