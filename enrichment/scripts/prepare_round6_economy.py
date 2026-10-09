"""Extract separate regional context, preserving scope and publication vintage."""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
UNIT = "thousand_rubles"
LINK_METHOD = "exact_normalized_name_region_year_with_document_review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_name(value):
    value = value.casefold().replace("ё", "е")
    value = re.sub(
        r"\b(?:муниципальный район|муниципальный округ|городской округ|город-курорт|город|г\.)\b",
        "",
        value,
    )
    return re.sub(r"\s+", " ", value).strip(" .")


def number(value):
    literal = value.strip()
    if literal in {"...", "…", "-", "–", "—", "Х", "X"}:
        return None
    if not re.fullmatch(r"\d[\d ]*(?:[.,]\d+)?", literal):
        raise ValueError(f"Unreviewed numeric token: {literal!r}")
    return Decimal(literal.replace(" ", "").replace(",", "."))


def shipment_totals(text, year):
    compact = re.sub(r"\s+", " ", text).strip()
    first = re.search(r"1\.1\.\s*ОТГРУЖЕНО(.*?)1\.2\.", compact, re.S)
    second = re.search(r"1\.2\.\s*ПРОДАНО(.*)", compact, re.S)
    if not first or not second:
        raise ValueError("Expected two economic sections")
    block = first[1]
    for phrase in [
        "БЕЗ СУБЪЕКТОВ МАЛОГО",
        f"{year} ГОДА",
        "(тыс. рублей)",
        "По хозяйственным",
        "По чистым",
    ]:
        if phrase not in block:
            raise ValueError(f"Missing source semantics: {phrase}")
    values = re.search(r"Всего\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", block)
    resale = re.search(r"Всего\s+(\S+)\s+(\S+)", second[1])
    if not values or not resale:
        raise ValueError("Missing total row")
    first_tokens = list(values.groups())
    second_tokens = list(resale.groups())
    for token in [*first_tokens, *second_tokens]:
        number(token)
    return first_tokens, second_tokens


def docx_totals(path):
    with zipfile.ZipFile(path) as z:
        document = ET.fromstring(z.read("word/document.xml"))
    tables = []
    for table in document.findall(".//w:tbl", NS):
        rows = []
        for row in table.findall("w:tr", NS):
            cells = [
                "".join(t.text or "" for t in c.findall(".//w:t", NS)).strip()
                for c in row.findall("w:tc", NS)
            ]
            rows.append(cells)
        tables.append(rows)
    shipment_tables = [
        t
        for t in tables
        if any("По хозяйственным" in " ".join(r) and "По чистым" in " ".join(r) for r in t)
    ]
    if len(shipment_tables) != 1:
        raise ValueError("Ambiguous shipment table")
    table = shipment_tables[0]
    index = tables.index(table)
    totals = [r[1:] for r in table if r[0] == "Всего"]
    resale = [r[1:] for r in tables[index + 1] if r[0] == "Всего"]
    if len(totals) != 1 or len(resale) != 1:
        raise ValueError("Ambiguous total row")
    return totals[0], resale[0]


def food_rows(path, year):
    pages = [p.extract_text() for p in PdfReader(path).pages]
    all_text = "\n".join(pages)
    for phrase in [
        f"за {year} финансовый год",
        "(тыс. рублей)",
        "Без учета объема скрытой деятельности",
    ]:
        if phrase not in all_text:
            raise ValueError("Unverified food sales semantics")
    rows = []
    for page, text in enumerate(pages, 1):
        for line in text.splitlines():
            match = re.fullmatch(r"\s*([А-Яа-яЁё].+?)\s+(\d[\d ]*,\d)\s*", line)
            if match:
                rows.append(
                    {
                        "source_name": match[1].strip(),
                        "source_value_literal": match[2],
                        "value": str(number(match[2])),
                        "source_page": page,
                    }
                )
    total_match = re.search(r"в том числе:\s*(\d[\d ]*,\d)", all_text)
    if not total_match:
        raise ValueError("Regional food total absent")
    return rows, number(total_match[1])


def prepare(repo, raw, out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    source_inputs = {}

    def pin(path):
        source_inputs[path.relative_to(raw).as_posix()] = sha(path)
        return path

    frozen_path = repo / "reports/enriched_analysis/core_v2/annual_panel.csv"
    frozen = pd.read_csv(frozen_path, dtype={"official_oktmo_8": str})
    lookup = {}
    for r in frozen.loc[frozen.region_code.isin([26, 64])].to_dict("records"):
        key = (int(r["region_code"]), int(r["year"]), normalized_name(r["municipal_district_name"]))
        if key in lookup:
            raise ValueError("Ambiguous municipal name")
        lookup[key] = r

    archive_records = json.loads(pin(raw / "extracted/manifest.json").read_text("utf-8"))
    conversion = json.loads(
        pin(raw / "converted2024-final/conversion_manifest.json").read_text("utf-8-sig")
    )
    converted = {x["input"]: x for x in conversion["records"]}
    rows, documents, cross_checks = [], [], []
    for entry in archive_records:
        year = entry["year"]
        original = pin(raw / entry["path"])
        if sha(original) != entry["sha256"]:
            raise ValueError("Original member changed")
        if year == 2023:
            member_name = entry["member"].encode("cp437").decode("cp866")
            pdf = original
        else:
            member_name = Path(entry["member"]).name
            record = converted[original.name]
            if record["input_sha256"] != sha(original):
                raise ValueError("Conversion source mismatch")
            pdf = pin(raw / "converted2024-final" / record["pdf"])
            docx = pin(raw / "converted2024-final" / record["docx"])
            if sha(pdf) != record["pdf_sha256"] or sha(docx) != record["docx_sha256"]:
                raise ValueError("Converted document changed")
        name = re.sub(r"\s+2024$", "", Path(member_name).stem)
        match = lookup[(26, year, normalized_name(name))]
        reader = PdfReader(pdf)
        pages = [p.extract_text() for p in reader.pages]
        cover = re.sub(r"\s+", " ", pages[0])
        if not re.search(rf"январь[-–]декабрь {year} года", cover, re.I):
            raise ValueError("Cover reporting period differs")
        choices = [
            (i, text) for i, text in enumerate(pages, 1) if "1.1." in text and "ОТГРУЖЕНО" in text
        ]
        if len(choices) != 1:
            raise ValueError("Ambiguous source page")
        page, text = choices[0]
        totals, resale = shipment_totals(text, year)
        if year == 2024:
            xml_totals, xml_resale = docx_totals(docx)
            if [number(t) for t in totals] != [number(t) for t in xml_totals] or [
                number(t) for t in resale
            ] != [number(t) for t in xml_resale]:
                raise ValueError("PDF and DOCX table disagree")
            cross_checks.append(
                {
                    "territory_id": int(match["territory_id"]),
                    "year": year,
                    "pdf_docx_values_exact": True,
                }
            )
        archive = pin(
            raw / "stavropol_annual" / ("annual_2023.zip" if year == 2023 else "annual_2024.rar")
        )
        retrieval = json.loads(
            pin(archive.with_name(archive.name + ".retrieval.json")).read_text("utf-8")
        )
        for variable, literal, scope in [
            (
                "shipments_economic_activity",
                totals[0],
                "without_small_business_including_organizations_under_15_workers_excluding_VAT_excise",
            ),
            (
                "shipments_pure_activity",
                totals[2],
                "without_small_business_excluding_organizations_under_15_workers_excluding_VAT_excise",
            ),
            (
                "resale_non_own_goods",
                resale[0],
                "without_small_business_including_organizations_under_15_workers_not_retail_turnover",
            ),
        ]:
            value = number(literal)
            rows.append(
                {
                    "territory_id": int(match["territory_id"]),
                    "year": year,
                    "region_code": 26,
                    "source_name": name,
                    "panel_name": match["municipal_district_name"],
                    "variable": variable,
                    "value": str(value) if value is not None else None,
                    "source_value_literal": literal,
                    "value_status": "numeric" if value is not None else "source_marker_unresolved",
                    "unit": UNIT,
                    "source_scope": scope,
                    "source_status": "operational_statistics",
                    "source_url": retrieval["url"],
                    "archive_sha256": sha(archive),
                    "source_member": member_name,
                    "source_document_sha256": sha(original),
                    "source_page": page,
                    "raw_path": entry["path"],
                    "link_method": LINK_METHOD,
                    "source_has_oktmo": False,
                    "historical_boundary_equivalence_proven": False,
                    "replace_frozen_bdpmo_value": False,
                }
            )
        documents.append(
            {
                "year": year,
                "territory_id": int(match["territory_id"]),
                "source_name": name,
                "source_member": member_name,
                "source_document_sha256": sha(original),
                "pages": len(pages),
                "data_page": page,
                "raw_path": entry["path"],
                "pdf_path": pdf.relative_to(raw).as_posix(),
                "cover_text": cover,
            }
        )

    reconciliations = []
    for year in [2023, 2024]:
        pdf = pin(raw / f"saratov{year}/food_{year}.pdf")
        retrieval = json.loads(pin(pdf.with_name(pdf.name + ".retrieval.json")).read_text("utf-8"))
        food, regional = food_rows(pdf, year)
        for item in food:
            # Source districts omit their legal type; two city rows explicitly use г.
            name = item["source_name"].removeprefix("г. ")
            match = lookup[(64, year, normalized_name(name))]
            rows.append(
                {
                    **item,
                    "territory_id": int(match["territory_id"]),
                    "year": year,
                    "region_code": 64,
                    "panel_name": match["municipal_district_name"],
                    "variable": "food_sales_all_visible_activity",
                    "value_status": "numeric",
                    "unit": UNIT,
                    "source_scope": "food_only_without_hidden_activity",
                    "source_status": "published_annual_food_sales",
                    "source_url": retrieval["url"],
                    "source_document_sha256": sha(pdf),
                    "raw_path": pdf.relative_to(raw).as_posix(),
                    "link_method": LINK_METHOD,
                    "source_has_oktmo": False,
                    "historical_boundary_equivalence_proven": False,
                    "replace_frozen_bdpmo_value": False,
                }
            )
        total = sum(Decimal(r["value"]) for r in food)
        reconciliations.append(
            {
                "year": year,
                "municipal_rows": len(food),
                "municipal_sum_thousand_rubles": str(total),
                "regional_total_thousand_rubles": str(regional),
                "difference_thousand_rubles": str(total - regional),
            }
        )

    catalog_rows = []
    for folder, filename in [
        ("pages", "saratov_retail.html"),
        ("stavropol_catalog", "catalog.html"),
    ]:
        page = pin(raw / folder / filename)
        for item in BeautifulSoup(page.read_bytes(), "html.parser").select(".document-list__item"):
            title = item.select_one(".document-list__item-title")
            info = item.select_one(".document-list__item-info")
            if title and info:
                catalog_rows.append(
                    {
                        "catalog": page.relative_to(raw).as_posix(),
                        "title": title.get_text(" ", strip=True),
                        "publication_label": info.get_text(" ", strip=True),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["territory_id", "year", "variable"]).any():
        raise ValueError("Duplicate municipality-year-variable")
    for filename, data in [
        ("regional_context_candidate.csv", frame),
        ("document_inventory.csv", pd.DataFrame(documents)),
        ("pdf_docx_crosscheck.csv", pd.DataFrame(cross_checks)),
        ("food_reconciliation.csv", pd.DataFrame(reconciliations)),
        ("catalog_publication_labels.csv", pd.DataFrame(catalog_rows)),
    ]:
        data.to_csv(out / filename, index=False)
    qa = {
        "status": "ready_with_conditions",
        "rows": len(frame),
        "numeric_rows": int(frame.value.notna().sum()),
        "by_variable": frame.groupby("variable").size().to_dict(),
        "municipality_years": len(frame[["territory_id", "year"]].drop_duplicates()),
        "stavropol_documents": len(documents),
        "pdf_docx_crosschecks": len(cross_checks),
        "old_panel_modified": False,
        "missing_bdpmo_cells_filled": 0,
        "conditions": [
            "Operational publication differs from current BДПМО vintage; candidate is separate.",
            "Food sales and resale of non-own goods are not total retail turnover.",
            "Name links are candidates, not proof of historical boundary equivalence.",
            "Source markers are retained, never coerced to zero.",
        ],
    }
    save(out / "QA.json", qa)
    save(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "raw_inputs": dict(sorted(source_inputs.items())),
            "repo_inputs": {
                frozen_path.relative_to(repo).as_posix(): sha(frozen_path),
                "enrichment/uv.lock": sha(repo / "enrichment/uv.lock"),
            },
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps(qa, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo.resolve(), args.raw.resolve(), args.out.resolve())
