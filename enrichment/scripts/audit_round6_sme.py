"""Offline source audit: SME address lineage and separate FNS worker publication dates."""

import argparse
import ast
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from collect_round6_sme import COMMIT, PACKAGE_SHA, digest, write_json
from sample_round5_sme_archives import RangeReader


def write_csv(path, rows):
    if not rows:
        raise ValueError("An expected evidence table is empty: " + path.name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(raw):
    manifest = json.loads((raw / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    paths = []
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Nonportable raw manifest path")
        path = raw / rel
        data = path.read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError("Raw source SHA/size differs: " + row["path"])
        paths.append(row["path"])
    actual = {
        p.relative_to(raw).as_posix()
        for p in raw.rglob("*")
        if p.is_file() and p.name != "SOURCE_MANIFEST.json"
    }
    if len(paths) != len(set(paths)) or set(paths) != actual:
        raise ValueError("Raw source inventory is incomplete or duplicated")
    return manifest["files"]


def worker_state(value):
    if value is None:
        return "missing_attribute"
    if value == "":
        return "empty_attribute"
    if not re.fullmatch(r"-?\d+", value):
        return "invalid_integer"
    return "negative" if int(value) < 0 else "zero" if int(value) == 0 else "positive"


def inspect_workers(content):
    root = ET.fromstring(content)
    records = root.findall("Документ")
    if int(root.get("КолДок", "-1")) != len(records):
        raise ValueError("XML declared document count differs")
    groups = Counter()
    keys = []
    for record in records:
        values = record.findall("СведССЧР")
        taxpayers = record.findall("СведНП")
        if len(values) != 1 or len(taxpayers) != 1:
            raise ValueError("Expected one worker record and one taxpayer per document")
        data_date = record.get("ДатаСост")
        document_date = record.get("ДатаДок")
        year = datetime.strptime(data_date, "%d.%m.%Y").year
        pubyear = datetime.strptime(document_date, "%d.%m.%Y").year
        state = worker_state(values[0].get("КолРаб"))
        groups[(data_date, document_date, year, pubyear, state)] += 1
        keys.append((taxpayers[0].get("ИННЮЛ"), data_date))
    if any(k[0] is None for k in keys):
        raise ValueError("Expected legal-entity tax key")
    rows = [
        dict(
            data_date=data_date,
            document_date=document_date,
            data_year=year,
            document_year=pubyear,
            workers_state=state,
            records=count,
            denominator_records=len(records),
            representative_sample=False,
        )
        for (data_date, document_date, year, pubyear, state), count in sorted(groups.items())
    ]
    return rows, dict(
        records=len(records),
        unique_taxpayer_data_dates=len(set(keys)),
        repeated_taxpayer_data_date_rows=len(keys) - len(set(keys)),
        records_2023_or_2024=sum(v for k, v in groups.items() if k[2] in (2023, 2024)),
        historical_municipal_join_approved=False,
    )


def code_state(value):
    if value == "":
        return "empty"
    if re.fullmatch(r"\d{8}|\d{11}", value):
        return "exact_8_or_11_digits"
    if re.fullmatch(r"\d+", value):
        return "other_digit_length"
    if re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", value):
        return "scientific_notation"
    return "other_syntax"


def inspect_assets(raw):
    summaries, codes, federal = [], [], []
    folder = raw / "author_source/russian_sme_registry/assets"
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        # foundation_year is a settlement attribute, not validity of this crosswalk.
        temporal = [
            f for f in fields if f.lower() in {"start_date", "end_date", "valid_from", "valid_to"}
        ]
        summaries.append(
            dict(
                asset=path.name,
                rows=len(rows),
                columns="|".join(fields),
                explicit_validity_columns="|".join(temporal),
                explicit_validity_columns_count=len(temporal),
                historical_join_approved=False,
                sha256=digest(path.read_bytes()),
            )
        )
        if "oktmo" not in fields:
            continue
        counts = Counter(code_state(row["oktmo"]) for row in rows)
        for state in [
            "empty",
            "exact_8_or_11_digits",
            "other_digit_length",
            "scientific_notation",
            "other_syntax",
        ]:
            codes.append(
                dict(
                    asset=path.name,
                    code_state=state,
                    rows=counts[state],
                    denominator_rows=len(rows),
                )
            )
        for number, row in enumerate(rows, 2):
            if row.get("region") not in ("Москва", "Санкт-Петербург", "Севастополь"):
                continue
            if path.name.startswith("cities") or row.get("settlement") == row.get("region"):
                federal.append(
                    dict(
                        asset=path.name,
                        csv_line=number,
                        region=row["region"],
                        area=row.get("area", row.get("municipality", "")),
                        city=row.get("city", ""),
                        settlement=row.get("settlement", ""),
                        source_oktmo=row["oktmo"],
                        code_state=code_state(row["oktmo"]),
                        source_text_preserved=True,
                    )
                )
    return summaries, codes, federal


def address_contracts(raw):
    path = raw / "generator_geocode.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    contracts = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in (
            "LocalGeocoder",
            "DaDataGeocoder",
        ):
            continue
        for assignment in node.body:
            if not isinstance(assignment, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "ADDR_COLS" for t in assignment.targets):
                continue
            fields = ast.literal_eval(assignment.value)
            contracts.append(
                dict(
                    route=node.name,
                    line=assignment.lineno,
                    address_fields="|".join(fields),
                    address_field_count=len(fields),
                    includes_year_or_asof=False,
                    code_sha256=digest(path.read_bytes()),
                    upstream_commit=COMMIT,
                    tochno_v20260605_route_proven=False,
                )
            )
            if any("date" in field or "year" in field for field in fields):
                raise ValueError("Address date contract changed")
    if len(contracts) != 2:
        raise ValueError("Expected two distinct author geocoding routes")
    return contracts


def replay_worker(raw):
    head = json.loads((raw / "workers_current_head.http.json").read_text(encoding="utf-8"))
    expected = json.loads((raw / "workers_sample_provenance.json").read_text(encoding="utf-8"))
    stream = RangeReader(None, raw / "workers_current", head, offline=True)
    with zipfile.ZipFile(stream) as archive:
        selected = min(
            (m for m in archive.infolist() if m.filename.endswith(".xml")),
            key=lambda m: (m.compress_size, m.filename),
        )
        if selected.filename != expected["selected_member"]:
            raise ValueError("Worker sample selection differs")
        data = archive.read(selected)
    if data != (raw / "workers_current/sample.xml").read_bytes():
        raise ValueError("Worker member replay differs")
    return expected, data


def retrieval_rows(raw):
    result = []
    for path in sorted(raw.glob("*.http.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            dict(
                metadata=path.name,
                url=r["url"],
                method=r.get("method", "GET"),
                retrieved_at=r.get("retrieved_at", ""),
                status=r.get("status", ""),
                bytes=r.get("bytes", 0),
                body=r.get("path", ""),
                sha256=r.get("sha256", ""),
                error=r.get("error", ""),
            )
        )
    return result


def verify_author_lineage(raw):
    rows = json.loads((raw / "author_source_lineage.json").read_text(encoding="utf-8"))
    tree = json.loads((raw / "generator_tree.html").read_text(encoding="utf-8"))
    blobs = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    for row in rows:
        content = (raw / row["path"]).read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        member = row["path"].removeprefix("author_source/")
        if (
            digest(content) != row["sha256"]
            or row["prior_package_sha256"] != PACKAGE_SHA
            or row["git_commit"] != COMMIT
            or git_sha != blobs[member]
        ):
            raise ValueError("Author package/Git lineage mismatch")
    if (raw / "generator_geocode.py").read_bytes() != (
        raw / "author_source/russian_sme_registry/stages/geocode.py"
    ).read_bytes():
        raise ValueError("Direct Git source differs from extracted package")
    return rows


def code_evidence(raw):
    """Pin substantive code readings to exact source lines without executing source."""
    specs = [
        (
            "fixed_lookup_vintage",
            "author_source/README.md",
            "The belonging to a particular region",
            "Author documents 2021 geography with lookup releases in late 2021/early 2022; "
            "this is not proof of the route used for tochno v20260605.",
        ),
        (
            "workers_legal_entities_only",
            "author_source/README.md",
            "again, no sole traders data",
            "Separate FNS workers enrichment is described for legal entities, not sole traders.",
        ),
        (
            "workers_latest_document_revision",
            "author_source/russian_sme_registry/stages/aggregate.py",
            'Window.partitionBy("tin", "data_date").orderBy(F.desc("doc_date"))',
            "Within taxpayer/data_date the author's window orders document revisions descending; "
            "availability as of a historical prediction date is not established.",
        ),
        (
            "workers_year_from_data_date",
            "author_source/russian_sme_registry/stages/aggregate.py",
            '.select("tin", F.year("data_date").alias("year"), "employees_count")',
            "Worker year comes from data_date, not publication filename or doc_date.",
        ),
        (
            "local_raw_addresses_removed",
            "author_source/russian_sme_registry/stages/geocode.py",
            "data.drop(columns=self.ADDR_COLS, inplace=True)",
            "The local route removes original address fields after constructing the mapping; "
            "its output alone does not preserve the entire raw address record.",
        ),
        (
            "ambiguous_lookup_rows_excluded",
            "author_source/russian_sme_registry/stages/geocode.py",
            "standard.drop_duplicates(subset=right_cols, keep=False, inplace=True)",
            "Duplicate lookup keys are excluded for each matching option in the local route; "
            "raw duplicate keys are not automatically an erroneous accepted join.",
        ),
        (
            "dadata_first_suggestion",
            "author_source/russian_sme_registry/stages/geocode.py",
            'suggestion_data = suggestions[0]["data"]',
            "The alternative route takes the first returned DaData suggestion; "
            "no DaData requests were made in this audit.",
        ),
        (
            "workers_panel_join",
            "author_source/russian_sme_registry/stages/panelize.py",
            'panel = panel.join(empl_data, on=["tax_number", "year"], how="leftouter")',
            "Annual workers join uses tax_number/year; this code does not prove the exact "
            "source archive revisions used by tochno.",
        ),
    ]
    result = []
    for claim, name, anchor, meaning in specs:
        path = raw / name
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = [i + 1 for i, line in enumerate(lines) if anchor in line]
        if claim == "workers_latest_document_revision":
            functions = [
                node
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.FunctionDef) and node.name == "_process_empl_data"
            ]
            if len(functions) != 1:
                raise ValueError("Employee aggregation function is ambiguous")
            scope = functions[0]
            matches = [line for line in matches if scope.lineno <= line <= scope.end_lineno]
        if len(matches) != 1:
            raise ValueError("Evidence anchor missing or ambiguous: " + claim)
        result.append(
            dict(
                evidence_id=claim,
                source_path=name,
                line=matches[0],
                source_sha256=digest(path.read_bytes()),
                claim=meaning,
                scope="inspected author package/Git commit only",
                proves_tochno_exact_build=False,
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    sources = verify_sources(args.raw)
    lineage = verify_author_lineage(args.raw)
    sample, content = replay_worker(args.raw)
    workers, summary = inspect_workers(content)
    assets, codes, federal = inspect_assets(args.raw)
    contracts = address_contracts(args.raw)
    evidence = code_evidence(args.raw)
    retrievals = retrieval_rows(args.raw)
    args.out.mkdir(parents=True)
    tables = {
        "worker_dates.csv": workers,
        "worker_sample_summary.csv": [summary | sample],
        "asset_schema_audit.csv": assets,
        "asset_code_syntax.csv": codes,
        "federal_city_source_rows.csv": federal,
        "geocoder_address_contracts.csv": contracts,
        "code_evidence.csv": evidence,
        "author_source_lineage.csv": lineage,
        "retrieval_attempts.csv": retrievals,
        "raw_inventory.csv": sources,
    }
    for name, rows in tables.items():
        write_csv(args.out / name, rows)
    qa = dict(
        status="PASS",
        raw_files_verified=len(sources),
        author_package_members_verified=len(lineage),
        selected_worker_xml_byte_exact_offline_replay=True,
        selected_worker_xml_crc32_verified=True,
        selected_worker_document_count_verified=True,
        full_xsd_validation="NOT RUN",
        full_archive_hash_verified=False,
        representative_sample=False,
        source_code_executed=False,
        historical_2023_2024_workers_xml_records=summary["records_2023_or_2024"],
        historical_municipal_join_approved=False,
        model_adoption=False,
        tochno_v20260605_exact_build_link_proven=False,
        decision="ready with conditions: source lineage and schema audit only",
    )
    write_json(args.out / "QA.json", qa)
    files = [
        dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p.read_bytes()))
        for p in sorted(args.out.iterdir())
    ]
    repo = Path(__file__).resolve().parents[2]
    dependencies = [
        "enrichment/uv.lock",
        "enrichment/scripts/collect_round6_sme.py",
        "enrichment/scripts/audit_round6_sme.py",
        "enrichment/scripts/collect_round5_sme.py",
        "enrichment/scripts/sample_round5_sme_archives.py",
    ]
    write_json(
        args.out / "manifest.json",
        dict(
            schema_version=1,
            files=files,
            raw_manifest_sha256=digest((args.raw / "SOURCE_MANIFEST.json").read_bytes()),
            raw_source_root="work/data-round6/sme (workspace-relative; override --raw)",
            dependency_hashes={name: digest((repo / name).read_bytes()) for name in dependencies},
        ),
    )
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
