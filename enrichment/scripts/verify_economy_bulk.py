"""Independently reconcile exported economy cells with saved raw coordinates."""

import argparse
import csv
import hashlib
import html
import io
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd


def normalize(value):
    return " ".join(html.unescape(value).split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory, cache):
    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in {"complete_with_gaps", "checkpointed"}:
        raise ValueError("Extraction is not finished or checkpointed")
    for name, digest in manifest["output_hashes"].items():
        assert sha(directory / name) == digest, f"Output hash: {name}"
    for name, digest in manifest["implementation_snapshot_hashes"].items():
        assert sha(directory / "implementation_snapshot" / name) == digest, name
    frame = pd.read_csv(directory / "economy_source.csv", dtype=str, keep_default_na=False)
    key = ["source_region_url", "source_oktmo", "indicator_id", "year"]
    assert not frame.duplicated(key).any(), "Duplicate source cells"
    evidence = {}
    for record in frame.to_dict("records"):
        request = record["request_key"]
        if request not in evidence:
            metadata = json.loads((cache / f"{request}.json").read_text(encoding="utf-8"))
            raw = (cache / f"{request}.raw").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == metadata["raw_sha256"]
            request_identity = {name: metadata[name] for name in ["method", "url", "payload"]}
            assert (
                hashlib.sha256(json.dumps(request_identity, sort_keys=True).encode()).hexdigest()
                == request
            )
            dimensions = dict(
                part.split(":", 1) for part in metadata["payload"]["Qry"].split(";") if part
            )
            rows = list(
                csv.reader(io.StringIO(raw.decode("cp1251").replace("\r\r\n", "\n")), delimiter=";")
            )
            evidence[request] = metadata, dimensions, rows
        metadata, dimensions, rows = evidence[request]
        assert metadata["status_code"] == 200 and metadata["raw_sha256"] == record["source_sha256"]
        assert dimensions["Pokazateli"] == record["indicator_id"]
        assert dimensions["period"] == "79" and dimensions["okved2"] == "101"
        assert record["source_oktmo"] in dimensions["oktmo"].split(",")
        assert record["year"] in dimensions["god"].split(",")
        number = int(record["source_csv_row"]) - 1
        source = rows[number]
        headers = [
            row
            for row in rows[:number]
            if row
            and not normalize(row[0])
            and any(normalize(v) in {"2023", "2024"} for v in row[1:])
        ]
        assert headers, "Missing preceding year header"
        header = [normalize(v) for v in headers[-1]][1:-1]
        assert len(set(header)) == len(header) and set(header) <= {"2023", "2024"}
        assert source[-1] == "" and len(source) >= len(header) + 2
        label = normalize(";".join(source[: -len(header) - 1]))
        assert label == record["source_oktmo_name"], "Wrong source municipality row"
        raw_value = normalize(source[-len(header) - 1 : -1][header.index(record["year"])])
        assert raw_value == record["value_raw"], "Wrong value or year column"
        if record["value_status"] == "observed":
            assert Decimal(raw_value.replace(" ", "").replace(",", ".")) == Decimal(record["value"])
        else:
            assert record["value"] == ""
        option_key = record["options_request_key"]
        assert sha(cache / f"{option_key}.raw") == record["options_sha256"]
    return {
        "status": "PASS",
        "verified_cells": len(frame),
        "verified_numeric_cells": int(frame.value_status.eq("observed").sum()),
        "unique_responses": len(evidence),
        "all_output_and_snapshot_hashes_verified": True,
        "verifier_sha256": sha(__file__),
        "manifest_sha256": sha(directory / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.cache)))


if __name__ == "__main__":
    main()
