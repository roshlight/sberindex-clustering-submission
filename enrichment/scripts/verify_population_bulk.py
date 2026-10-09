"""Reconcile every exported population cell with its saved CSV row and year column."""

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import pandas as pd


def normalized(value):
    return " ".join(html.unescape(value).split())


def verify(directory, cache):
    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in {"complete", "complete_with_gaps"}:
        raise ValueError("Only a completed extraction can be verified")
    for name, expected in manifest["output_hashes"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest["implementation_snapshot_hashes"].items():
        path = directory / "implementation_snapshot" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    frame = pd.read_csv(directory / "population_source.csv", dtype=str, keep_default_na=False)
    keys = ["source_region_url", "source_oktmo", "reference_date", "population_type"]
    assert not frame.duplicated(keys).any(), "Duplicate source observations"
    evidence = {}
    for record in frame.to_dict("records"):
        key = (record["request_key"], record["source_sha256"])
        if key not in evidence:
            meta_path = cache / f"{key[0]}.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["raw_sha256"] != key[1]:
                candidates = [
                    path
                    for path in (cache / "history" / key[0]).glob("*.json")
                    if json.loads(path.read_text(encoding="utf-8"))["raw_sha256"] == key[1]
                ]
                assert len(candidates) >= 1, "Missing historical raw response"
                meta_path = candidates[0]
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = meta_path.with_suffix(".raw").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == key[1], "Raw response hash mismatch"
            assert metadata["status_code"] == 200
            assert metadata["payload"]["Format"] == "CSV"
            dimensions = dict(
                piece.split(":", 1) for piece in metadata["payload"]["Qry"].split(";") if piece
            )
            rows = list(
                csv.reader(io.StringIO(raw.decode("cp1251").replace("\r\r\n", "\n")), delimiter=";")
            )
            evidence[key] = (rows, dimensions)
        rows, dimensions = evidence[key]
        assert record["source_oktmo"] in dimensions["oktmo"].split(",")
        assert record["population_type"] in dimensions["mest"].split(",")
        assert dimensions["Pokazateli"] == "8112027" and dimensions["period"] == "208"
        number = int(record["source_csv_row"]) - 1
        source_row = [normalized(value) for value in rows[number]]
        assert source_row[0] == record["population_type_label"], "Wrong source population row"
        header = None
        for row in reversed(rows[:number]):
            cells = [normalized(value) for value in row]
            if cells and cells[0] == "" and record["reference_date"][:4] in cells[1:]:
                header = cells
                break
        assert header is not None, "No explicit preceding year header"
        column = header.index(record["reference_date"][:4])
        raw_value = source_row[column]
        assert raw_value == record["value_raw"], "Export differs from source row/year column"
        if record["value_status"] == "observed":
            assert raw_value.isdigit() and int(raw_value) == int(record["value"])
        else:
            assert record["value"] == "" and raw_value in {"", "-", "…", "..."}
    return {
        "status": "PASS",
        "verified_cells": len(frame),
        "verified_numeric_cells": int(frame.value_status.eq("observed").sum()),
        "unique_numeric_responses": len(evidence),
        "all_output_and_snapshot_hashes_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.cache)))


if __name__ == "__main__":
    main()
