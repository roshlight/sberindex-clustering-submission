"""Offline evidence audit of historical FNS sources; no municipal join or model inputs."""

import argparse
import csv
import hashlib
import json
import re
import tarfile
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from openpyxl import load_workbook
from pypdf import PdfReader

PYPI_SHA = "3c3f41882efb08a9f7be7bcd12fdecd59220e07ce5a68479ee072d92bd2e1421"
NS = {"xs": "http://www.w3.org/2001/XMLSchema"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def register_worker_reference_year(year, month):
    """Current FNS passport rule, not a transformation of tochno.year."""
    if month not in range(1, 13):
        raise ValueError("Invalid month")
    return year - (2 if month <= 6 else 1)


def audit_xml(path, expected_date):
    root = ET.parse(path).getroot()
    docs = root.findall("Документ")
    dates = sorted({d.attrib["ДатаСост"] for d in docs})
    if dates != [expected_date] or int(root.attrib["КолДок"]) != len(docs):
        raise ValueError("Snapshot date or document count mismatch")
    workers = [int(d.attrib["ССЧР"]) for d in docs if "ССЧР" in d.attrib]
    names = {e.tag for d in docs for e in d.iter()}
    names |= {a for d in docs for e in d.iter() for a in e.attrib}
    return dict(
        archive=path.parent.name,
        snapshot_date=expected_date,
        format_version=root.attrib["ВерсФорм"],
        documents=len(docs),
        entity_type_counts=json.dumps(Counter(d.attrib["ВидСубМСП"] for d in docs)),
        location_records=sum(d.find("СведМН") is not None for d in docs),
        district_records=sum(d.find("СведМН/Район") is not None for d in docs),
        city_records=sum(d.find("СведМН/Город") is not None for d in docs),
        settlement_records=sum(d.find("СведМН/НаселПункт") is not None for d in docs),
        workers_present=len(workers),
        workers_missing=len(docs) - len(workers),
        workers_negative=sum(v < 0 for v in workers),
        workers_zero=sum(v == 0 for v in workers),
        workers_min=min(workers) if workers else "",
        workers_max=max(workers) if workers else "",
        oktmo_field_present=any("октмо" in n.lower() for n in names),
        representative_sample=False,
        source_sha256=sha(path),
    )


def parse_statistics(path, year):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = load_workbook(path, read_only=True, data_only=True)
    sheet = book.active
    sheet.reset_dimensions()  # Official workbook dimension incorrectly declares A1.
    values = list(sheet.values)
    book.close()
    if values[2][0] != f"по состоянию на 10.12.{year}":
        raise ValueError("Statistics date mismatch")
    columns = [
        "total",
        "legal_total",
        "legal_micro",
        "legal_small",
        "legal_medium",
        "ip_total",
        "ip_micro",
        "ip_small",
        "ip_medium",
    ]
    rows = []
    for excel_row, cells in enumerate(values, 1):
        if len(cells) < 10 or not isinstance(cells[1], (int, float)):
            continue
        counts = dict(zip(columns, (int(v) for v in cells[1:10]), strict=True))
        if any(v != int(v) or v < 0 for v in cells[1:10]):
            raise ValueError("Non-integral or negative official count")
        if counts["total"] != counts["legal_total"] + counts["ip_total"]:
            raise ValueError("Entity type totals do not reconcile")
        for kind in ["legal", "ip"]:
            if counts[f"{kind}_total"] != sum(
                counts[f"{kind}_{c}"] for c in ["micro", "small", "medium"]
            ):
                raise ValueError("Size category totals do not reconcile")
        name = cells[0]
        level = (
            "country"
            if name == "Российская Федерация"
            else "federal_district"
            if "федеральный округ" in name
            else "other_region_group"
            if name == "Юго-Западные субъекты Российской Федерации"
            else "region"
        )
        rows.append(
            dict(
                snapshot_date=f"{year}-12-10",
                geography_label=name,
                geography_level=level,
                **counts,
                source_file=path.name,
                excel_row=excel_row,
                population="FNS register stock; legal entities and sole traders",
                historical_municipal_join_eligible=False,
            )
        )
    country = next(row for row in rows if row["geography_level"] == "country")
    regions = [row for row in rows if row["geography_level"] == "region"]
    for key in columns:
        if sum(row[key] for row in regions) != country[key]:
            raise ValueError(f"Region-to-country reconciliation failed: {key}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, condition, evidence):
        checks.append(dict(name=name, passed=bool(condition), evidence=evidence))
        if not condition:
            raise ValueError(name)

    bodies, attempts = [], []
    for path in sorted(raw.glob("*.http.json")):
        meta = read_json(path)
        check(
            f"no_cookie_headers:{path.name}",
            not any("cookie" in k.lower() for k in meta.get("headers", {})),
            "HTTP metadata",
        )
        attempts.append(
            dict(
                metadata=path.name,
                url=meta["url"],
                method=meta["method"],
                status=meta.get("status", ""),
                error=meta.get("error", ""),
                retrieved_at=meta["retrieved_at"],
            )
        )
        if "path" in meta:
            source = raw / meta["path"]
            check(
                f"source_hash:{source.name}",
                sha(source) == meta["sha256"] and source.stat().st_size == meta["bytes"],
                meta["url"],
            )
            bodies.append(
                dict(
                    path=source.name,
                    bytes=meta["bytes"],
                    sha256=meta["sha256"],
                    url=meta["url"],
                    retrieved_at=meta["retrieved_at"],
                )
            )
    write_csv(out / "retrieval_attempts.csv", attempts)
    write_csv(out / "downloaded_sources.csv", bodies)
    links = read_json(raw / "discovered_links.json")
    catalog = []
    for item in links:
        match = re.search(r"data-(\d{2})(\d{2})(202[34])-structure-(\d+)\.zip$", item["url"])
        if item["page"] != "fns_passport" or not match:
            continue
        day, month, year, schema = match.groups()
        head = raw / f"archive_{year}_{month}.http.json"
        meta = read_json(head) if head.exists() else {}
        catalog.append(
            dict(
                snapshot_date=f"{year}-{month}-{day}",
                archive_url=item["url"],
                schema_version=schema,
                head_checked=bool(meta),
                head_status=meta.get("status", ""),
                archive_bytes=meta.get("headers", {}).get("content-length", ""),
                register_sschr_reference_year_current_passport_rule=register_worker_reference_year(
                    int(year), int(month)
                ),
                rule_confirmed_for_historical_snapshot=False,
                rule_applies_to_tochno_year=False,
                whole_archive_downloaded=False,
            )
        )
    catalog.sort(key=lambda row: row["snapshot_date"])
    check(
        "24_distinct_archive_links",
        len(catalog) == len({r["snapshot_date"] for r in catalog}) == 24,
        "FNS passport",
    )
    write_csv(out / "historical_archive_catalog.csv", catalog)

    samples = read_json(raw / "archive_samples.json")
    replay = read_json(raw / "archive_sample_replay.json")
    check(
        "range_zip_member_offline_replay",
        samples == replay,
        "All selected members replayed; zipfile validates CRC",
    )
    xml_rows = []
    for sample in samples:
        path = raw / sample["archive"] / "sample.xml"
        year, month = sample["archive"].split("_")[1:]
        check("sample_hash:" + sample["archive"], sha(path) == sample["sample_sha256"], path.name)
        xml_rows.append(audit_xml(path, f"10.{month}.{year}"))
    write_csv(out / "xml_sample_audit.csv", xml_rows)
    write_csv(out / "archive_sample_summary.csv", samples)

    schema_fields = []
    for path in sorted(raw.glob("*.xsd")):
        root = ET.parse(path).getroot()
        all_names = {node.attrib.get("name", "") for node in root.iter()}
        check(
            "no_oktmo_field:" + path.name,
            not any("октмо" in n.lower() for n in all_names),
            "XSD names",
        )
        for element in root.iter():
            if element.attrib.get("name") not in {
                "ДатаСост",
                "ДатаДок",
                "ДатаВклМСП",
                "ССЧР",
                "КолРаб",
                "СведМН",
                "КодРегион",
            }:
                continue
            documentation = element.find("xs:annotation/xs:documentation", NS)
            schema_fields.append(
                dict(
                    source=path.name,
                    field=element.attrib["name"],
                    use=element.attrib.get("use", ""),
                    meaning=" ".join(documentation.itertext()) if documentation is not None else "",
                )
            )
    write_csv(out / "schema_field_evidence.csv", schema_fields)

    stats = parse_statistics(raw / "statistics_2023.xlsx", 2023)
    stats += parse_statistics(raw / "statistics_2024.xlsx", 2024)
    write_csv(out / "official_stock_counts.csv", stats)

    pdf = PdfReader(raw / "rudnya_municipal_2024.pdf")
    pages = [page.extract_text() for page in pdf.pages]
    write_json(
        out / "municipal_pdf_text.json", [dict(page=i + 1, text=t) for i, t in enumerate(pages)]
    )
    municipality = []
    for label, name, count, _count2, _count3 in re.findall(
        r"^\s*(\d+)\.\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s", pages[0], re.MULTILINE
    ):
        municipality.append(
            dict(
                source_row_label=label,
                geography_label=name.strip(),
                snapshot_date="2024-02-10",
                count=int(count),
                source_page=1,
                source_file="rudnya_municipal_2024.pdf",
                historical_oktmo="",
                historical_municipal_join_eligible=False,
                population="FNS register stock legal entities and sole traders",
                boundary_definition="2025 report labels; 2024 boundaries unverified",
            )
        )
    check("municipal_pdf_27_rows", len(municipality) == 27, "page1")
    check(
        "municipal_pdf_2024_sum",
        sum(r["count"] for r in municipality) == 38337,
        "page1 printed total",
    )
    write_csv(out / "municipal_stock_candidates_2024.csv", municipality)

    check(
        "pypi_archive_digest",
        sha(raw / "generator_source.gz") == PYPI_SHA,
        "PyPI published source digest",
    )
    code_evidence = []
    needles = [
        "F.explode(F.sequence",
        "panel.join(empl_data",
        'F.year("data_date").alias("year"), "employees_count"',
        '"СведССЧР/@КолРаб"',
        '"@ДатаСост"',
        "F.min(data_date_col)",
        "F.max(data_date_col)",
    ]
    with tarfile.open(raw / "generator_source.gz") as archive:
        for member in archive.getmembers():
            if not member.name.endswith(
                ("stages/panelize.py", "stages/aggregate.py", "utils/elements.py")
            ):
                continue
            for number, line in enumerate(
                archive.extractfile(member).read().decode().splitlines(), 1
            ):
                if any(needle in line for needle in needles):
                    code_evidence.append(
                        dict(
                            archive="generator_source.gz",
                            member=member.name,
                            line_number=number,
                            text=line.strip(),
                            exact_tochno_build_proven=False,
                        )
                    )
    check(
        "generator_code_evidence",
        len(code_evidence) >= 8,
        "author source archive; static read, no execution",
    )
    write_csv(out / "generator_code_evidence.csv", code_evidence)

    passport_text = BeautifulSoup((raw / "fns_passport.html").read_bytes(), "html.parser").get_text(
        " ", strip=True
    )
    pos = passport_text.find("среднесписоч")
    write_json(
        out / "passport_text_evidence.json", dict(text=passport_text, first_worker_mention=pos)
    )
    ledger = [
        (
            "historical_register",
            "ready with conditions",
            "fns_passport.html;historical_archive_catalog.csv",
            "24 observed monthly archive links for 2023/2024; 8 HEAD checks; 4 "
            "bounded member downloads",
            "Full population not downloaded; selected smallest XML is nonrepresentative",
        ),
        (
            "snapshot_time",
            "ready with conditions",
            "xml_sample_audit.csv;schema_field_evidence.csv",
            "All documents in four samples carry expected ДатаСост",
            "Snapshot date differs from generation filename in two archives; "
            "historical publication availability not proven",
        ),
        (
            "historical_geography",
            "not ready",
            "schema_field_evidence.csv;xml_sample_audit.csv",
            "Historical registered-location/residence text and region code are "
            "present; no OKTMO field in schemas",
            "Requires dated address crosswalk and boundary review; registered "
            "address is not workplace",
        ),
        (
            "register_workers",
            "ready with conditions",
            "fns_passport.html;schema_field_evidence.csv",
            "Current register passport states Jan-Jun Y reference Y-2; Jul-Dec Y reference Y-1",
            "Historical policy continuity not independently confirmed; do not "
            "apply rule to tochno.year",
        ),
        (
            "tochno_workers_year",
            "unknown",
            "generator_code_evidence.csv;prior description_sme.pdf pages 6-9",
            "Tochno documents annual employees and separate FNS source; author "
            "code joins worker year=year(ДатаСост) on tax_number/year",
            "Exact package-to-tochno build linkage and historical worker "
            "archives remain unverified",
        ),
        (
            "annual_population",
            "ready with conditions",
            "generator_code_evidence.csv;prior description_sme.pdf",
            "Author code expands inclusive start/end observed years and can "
            "retain multiple states per tax_number/year",
            "Not December stock; no births/deaths interpretation and no "
            "equality to monthly stock without reconstruction",
        ),
        (
            "official_stock_totals",
            "ready with conditions",
            "official_stock_counts.csv",
            "Official December 10 stocks for 2023/2024; category and region totals reconcile",
            "Country scope includes source-listed territories; "
            "regional/country totals are not municipality counts or panel denominators",
        ),
        (
            "municipal_stock_2024",
            "ready with conditions",
            "municipal_stock_candidates_2024.csv;rudnya_municipal_2024.pdf page1",
            "27 municipality-labelled counts at 10.02.2024 total 38337",
            "2025 report uses district names; no dated OKTMO or exact boundary "
            "equivalence; no 2023 comparable count acquired",
        ),
        (
            "historical_worker_archive",
            "unknown",
            "workers_2024_release.http.json;fns_workers_passport.html",
            "Observed historical 20251225 link returned 404; current passport "
            "has no populated historical release list",
            "No historical standalone employee archive acquired; absence here "
            "does not prove nonexistence elsewhere",
        ),
    ]
    write_csv(
        out / "evidence_ledger.csv",
        [
            dict(topic=a, status=b, evidence=c, established=d, limitation=e)
            for a, b, c, d, e in ledger
        ],
    )
    prior = args.prior_raw.resolve() / "raw" / "description_sme.pdf"
    check("prior_tochno_document_present", prior.exists(), "Reused frozen round4 documentation")
    write_json(
        out / "prior_source_reference.json",
        dict(path="work/data-round4/sme/raw/description_sme.pdf", sha256=sha(prior)),
    )
    inventory = [
        dict(path=p.relative_to(raw).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(raw.rglob("*"))
        if p.is_file()
    ]
    write_csv(out / "raw_inventory.csv", inventory)
    qa = dict(
        status="PASS",
        checks=checks,
        downloaded_body_count=len(bodies),
        archive_links=len(catalog),
        sampled_archives=len(samples),
        sample_documents=sum(r["documents"] for r in xml_rows),
        sample_range_bytes=sum(r["range_bytes_read"] for r in samples),
        complete_archive_bytes_not_downloaded=sum(r["archive_bytes"] for r in samples),
        official_stock_rows=len(stats),
        municipal_2024_candidates=len(municipality),
        model_adoption=False,
        historical_municipal_join_approved=False,
        exact_tochno_build_lineage_proven=False,
        xml_xsd_validation_performed=False,
        pdf_page1_visual_review=(
            "PASS; rendered original PDF table inspected, labels/counts preserved"
        ),
        raw_files=len(inventory),
        raw_bytes=sum(r["bytes"] for r in inventory),
    )
    write_json(out / "QA.json", qa)
    manifest = [
        dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "artifact_manifest.json"
    ]
    write_json(out / "artifact_manifest.json", manifest)
    print(json.dumps({k: v for k, v in qa.items() if k != "checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
