"""Reconcile extracted numeric cells with raw XLSX XML and four labelled source rows."""

import argparse
import hashlib
import json
import math
import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import duckdb
import pandas as pd

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_xml(file):
    values = {}
    with zipfile.ZipFile(file) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        by_id = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        for sheet in wb.findall("x:sheets/x:sheet", NS):
            target = by_id[sheet.attrib[f"{{{RNS}}}id"]]
            path = (
                target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            )
            root = ET.fromstring(z.read(path))
            for cell in root.findall(".//x:sheetData/x:row/x:c", NS):
                value = cell.find("x:v", NS)
                if (
                    cell.attrib.get("t", "n") == "n"
                    and value is not None
                    and value.text is not None
                ):
                    if cell.find("x:f", NS) is None:
                        values[(sheet.attrib["name"], cell.attrib["r"])] = float(value.text)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new output")
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((args.report / "manifest.json").read_text(encoding="utf-8"))
    for file, digest in m["source_hashes"].items():
        assert sha(args.raw / file) == digest
    for file, digest in m["outputs"].items():
        assert sha(args.report / file) == digest
    data = duckdb.read_parquet(str(args.report / "source_cells.parquet")).df()
    numbers = data[data.numeric_value.notna()]
    checked = 0
    for file in sorted(args.raw.glob("*.xlsx")):
        d = numbers[numbers.source_id == file.stem]
        observed = dict(zip(zip(d.sheet, d.cell, strict=True), d.numeric_value, strict=True))
        direct = numeric_xml(file)
        assert direct.keys() == observed.keys(), file.name
        assert all(
            math.isclose(v, observed[k], rel_tol=1e-12, abs_tol=1e-12) for k, v in direct.items()
        )
        checked += len(direct)
    assert checked == m["numeric_cells"]
    pilots = []
    for sid, sheet, label_cell, label, row, columns, candidate_id, region, population_scope in [
        (
            "census_01",
            "Таб. 2",
            "F1019",
            "Верхнеколымский муниципальный район",
            1021,
            "GHIJLNP",
            318,
            "Республика Саха (Якутия)",
            "urban_and_rural",
        ),
        (
            "census_01",
            "Таб. 2",
            "F2907",
            "Нижнеколымский муниципальный район",
            2909,
            "GHIJLNP",
            331,
            "Республика Саха (Якутия)",
            "urban_and_rural",
        ),
        (
            "census_05",
            "Свободненский",
            "A5",
            "Свободненский муниципальный район",
            11,
            "BCDEGIK",
            883,
            "Амурская область",
            "urban_and_rural",
        ),
        ("census_08", "Куркино", "A2", "Куркино", 9, "BCDEGIK", 2527, "Москва", "urban_population"),
    ]:
        subset = data[(data.source_id == sid) & (data.sheet == sheet)].set_index("cell")
        assert subset.loc[label_cell, "raw_value"] == label
        fields = [
            "population_private_households_15plus",
            "reported_labor_status",
            "labor_force",
            "employed",
            "unemployed",
            "outside_labor_force",
            "labor_status_not_reported",
        ]
        values = {
            field: int(subset.loc[f"{col}{row}", "numeric_value"])
            for field, col in zip(fields, columns, strict=True)
        }
        assert values["labor_force"] == values["employed"] + values["unemployed"]
        assert (
            values["reported_labor_status"] == values["labor_force"] + values["outside_labor_force"]
        )
        assert (
            values["population_private_households_15plus"]
            == values["reported_labor_status"] + values["labor_status_not_reported"]
        )
        pilots.append(
            {
                "candidate_territory_id": candidate_id,
                "source_territory_label": label,
                "source_region": region,
                "reference_date": "2021-10-01",
                "labor_reference_week_start": "2021-09-24",
                "labor_reference_week_end": "2021-09-30",
                "labor_period_methodology": "census_13, PDF page 1",
                "population_scope": population_scope,
                "source_id": sid,
                "sheet": sheet,
                "row": row,
                "label_cell": label_cell,
                "source_sha256": m["source_hashes"][sid + ".xlsx"],
                "join_status": "source_row_verified_historical_boundary_link_unresolved",
                **values,
            }
        )
    pd.DataFrame(pilots).to_csv(args.out / "four_case_labor_status_2021.csv", index=False)
    qa = {
        "status": "PASS",
        "numeric_source_cells_reconciled": checked,
        "independent_reader": "Python standard library ZIP+XML, no openpyxl",
        "verified_pilot_rows": len(pilots),
        "pilot_additivity_checks": len(pilots) * 3,
        "primary_panel_joins_accepted": 0,
        "source_manifest_sha256": sha(args.report / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "not_accepted": [
            "Красноярский край substring is not Красноярский район Астраханской области",
            "No municipality-level commuting shares recovered for all six cases",
            "2021 labor status cannot be relabelled as 2023/2024 employment",
        ],
        "outputs": {
            "four_case_labor_status_2021.csv": sha(args.out / "four_case_labor_status_2021.csv")
        },
    }
    (args.out / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
