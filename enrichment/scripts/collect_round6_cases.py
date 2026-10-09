"""Snapshot bounded round-six primary documents and replay reviewed case evidence."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from decimal import Decimal
from pathlib import Path

import openpyxl

SPEC = importlib.util.spec_from_file_location(
    "round6_immutable_helper", Path(__file__).with_name("collect_round5_cases.py")
)
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)


def extract_snapshot(record: dict, raw: Path, *, write_aux: bool = False) -> str:
    content = (raw / record["path"]).read_bytes()
    if content.startswith(b"PK") and ".xlsx" in record["url"]:
        workbook = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        cells = [
            {
                "sheet": sheet.title,
                "cell": cell.coordinate,
                "value": (
                    cell.value.isoformat() if hasattr(cell.value, "isoformat") else cell.value
                ),
            }
            for sheet in workbook
            for row in sheet
            for cell in row
            if cell.value is not None
        ]
        if write_aux:
            prior.save(raw / "parsed" / f"{record['id']}.cells.json", cells)
        return "\n".join(json.dumps(cell, ensure_ascii=False, default=str) for cell in cells)
    return (
        ORIGINAL_EXTRACT(record, raw, write_aux=write_aux).replace("\r\n", "\n").replace("\r", "\n")
    )


ORIGINAL_EXTRACT = prior.extract_snapshot
prior.extract_snapshot = extract_snapshot
ORIGINAL_FETCH = prior.fetch_source


def fetch_source(item: dict, raw: Path, ca: Path | None) -> dict:
    record = ORIGINAL_FETCH(item, raw, ca)
    if ".xlsx" in record["url"] and record.get("path"):
        original = raw / record["path"]
        if original.suffix != ".xlsx":
            target = original.with_suffix(".xlsx")
            if target.exists():
                raise ValueError("Refuse to replace an existing spreadsheet snapshot")
            original.rename(target)
            record["path"] = target.relative_to(raw).as_posix()
            prior.save(raw / "metadata" / f"{record['id']}.json", record)
    return record


prior.fetch_source = fetch_source


def amount_sum(values: list) -> str:
    """Missing components remain missing, and an observed zero remains zero."""
    if not values:
        raise ValueError("A sum requires at least one source cell")
    if any(value is None for value in values):
        return ""
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Only literal numeric source cells can be aggregated")
    return str(sum((Decimal(str(value)) for value in values), Decimal(0)))


def check_financing_layout(sheet) -> None:
    expected = {
        "A4": "(тыс. рублей)",
        "N5": "2022 год",
        "V5": "2023 год",
        "AD5": "2024 год",
        "R6": "кассовое исполнение",
        "Z6": "кассовое исполнение",
        "AH6": "кассовое исполнение",
        "D56": "00000000209970",
        "C56": "Республика Саха (Якутия), пос.Черский",
        "E56": "Реконструкция аэропорта Черский, Республика Саха (Якутия)",
        "B56": "0408 241V7 24100 414",
        "B57": "0408 241V7 2410F 414",
    }
    for cell, value in expected.items():
        if sheet[cell].value != value:
            raise ValueError(f"Financing row/header changed: {cell}")
    if not {"E56:E57", "D56:D57", "AH56:AH57"}.issubset({str(span) for span in sheet.merged_cells}):
        raise ValueError("Project row spans must be checked before aggregation")


def finance_rows(raw: Path, records: dict) -> list[dict]:
    source = records["ach_rtm_financing"]
    workbook = openpyxl.load_workbook(raw / source["path"], data_only=False)
    sheet = workbook["прил 3"]
    check_financing_layout(sheet)
    output = []
    for year, columns in [(2022, ("P", "R")), (2023, ("X", "Z")), (2024, ("AF", "AH"))]:
        for kind, col in zip(
            ["budget_limit_year_end", "annual_cash_execution"], columns, strict=True
        ):
            # In 2024 the two budget-code rows share one merged value cell.
            coordinates = [f"{col}56"] if year == 2024 else [f"{col}56", f"{col}57"]
            values = [sheet[cell].value for cell in coordinates]
            amount = amount_sum(values)
            output.append(
                {
                    "project": "Аэропорт Черский",
                    "year": year,
                    "financial_kind": kind,
                    "value_thousand_rubles": amount,
                    "value_million_rubles": str(Decimal(amount) / 1000) if amount else "",
                    "observation_status": "observed" if amount else "blank_not_zero",
                    "sheet": sheet.title,
                    "source_cells": ";".join(coordinates),
                    "source_values": json.dumps(values),
                    "source_url": source["url"],
                    "source_sha256": source["sha256"],
                    "publication_date": "2025-12-03",
                    "scope": (
                        "Object-level federal customer; not reconciled to municipal statistics"
                    ),
                    "model_join_accepted": False,
                }
            )
    return output


def replay(raw: Path, recipe: Path, out: Path) -> dict:
    reviewed = prior.load(recipe)
    records = {record["id"]: record for record in prior.load(raw / "retrievals.json")}
    verified_sheets = set()
    for row in reviewed["evidence"]:
        if row.get("sheet"):
            cells = prior.load(raw / "parsed" / f"{row['source_id']}.cells.json")
            if row["source_id"] not in verified_sheets:
                raw_cells = [
                    json.loads(line)
                    for line in extract_snapshot(records[row["source_id"]], raw).splitlines()
                ]
                if cells != raw_cells:
                    raise ValueError("Cached spreadsheet cells differ from source bytes")
                verified_sheets.add(row["source_id"])
            source_cells = {(cell["sheet"], cell["cell"]): cell["value"] for cell in cells}
            for cell, expected in row["cell_checks"].items():
                if source_cells.get((row["sheet"], cell)) != expected:
                    raise ValueError(f"Spreadsheet coordinate mismatch: {cell}")
        if row["value_kind"] == "annual_cash_execution":
            if not row.get("year") or row["unit"] != "thousand_rubles":
                raise ValueError("Annual execution requires explicit year and units")
            if not row.get("scope_note"):
                raise ValueError("Cash execution needs a territorial and scope caveat")
        if records[row["source_id"]]["role"] == "nonofficial_mirror_lead":
            raise ValueError("Mirrors cannot be accepted as primary financial evidence")
    summary = prior.replay(raw, recipe, out)
    annual = finance_rows(raw, records)
    prior.csv_save(out / "airport_annual_finance.csv", annual)
    prior.csv_save(out / "financial_semantics.csv", reviewed["financial_semantics"])
    summary.update(
        annual_finance_rows=len(annual),
        annual_finance_observed_rows=sum(r["observation_status"] == "observed" for r in annual),
        annual_cash2024_status="blank_not_zero",
        spreadsheet_auxiliary_replays_verified=len(verified_sheets),
        immutable_helper_script="enrichment/scripts/collect_round5_cases.py",
    )
    prior.save(out / "QA.json", summary)
    prior.save(
        out / "manifest.json",
        {
            "script_sha256": prior.sha(Path(__file__)),
            "helper_script_sha256": prior.sha(Path(prior.__file__)),
            "recipe_sha256": prior.sha(recipe),
            "retrievals_sha256": prior.sha(raw / "retrievals.json"),
            "outputs": {
                path.name: prior.sha(path)
                for path in sorted(out.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--ca-bundle", type=Path)
    args = parser.parse_args()
    if not args.derive_only:
        if args.plan is None:
            parser.error("--plan is required for collection")
        prior.acquire(args.plan, args.raw_root, args.ca_bundle)
    if args.recipe:
        if args.out is None:
            parser.error("--out is required for replay")
        print(json.dumps(replay(args.raw_root, args.recipe, args.out)))


if __name__ == "__main__":
    main()
