"""Rebuild the bounded round-three geography evidence package, entirely offline.

This verifies downloaded bytes, official publication cards, exact registry blocks
and the two explicit columns of Rosstat's recoding table. It never changes a
geographical join, analytical mask or earlier result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from bs4 import BeautifulSoup
from pypdf import PdfReader

FIELDS = [
    "TER",
    "KOD1",
    "KOD2",
    "KOD3",
    "KC",
    "RAZDEL",
    "NAME1",
    "Centrum",
    "NomDescr",
    "NomAkt",
    "Status",
    "DateUtv",
    "DateVved",
]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
REGISTRIES = {
    "before": "raw/reused/data-20230112T1301-structure-20260210T1102.csv",
    "after": "raw/reused/data-20240125T1401-structure-20260210T1102.csv",
}
CASES = [
    (208, "Беломорский", "86604000000", "86504000000", 59, "2838"),
    (217, "Питкярантский", "86633000000", "86533000000", 36, "2837"),
    (221, "Сегежский", "86645000000", "86545000000", 36, "2839"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Unexpected empty table: {path.name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def registry(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter=";"), 1):
            if len(row) != len(FIELDS) or [len(x) for x in row[:4]] != [2, 3, 3, 3]:
                raise ValueError(f"Invalid registry row: {line}")
            item = dict(zip(FIELDS, row, strict=True))
            item.update(code="".join(row[:4]), source_csv_line=line)
            result.append(item)
    return result


def settlement_block(rows: list[dict], root_code: str, expected_name: str) -> list[dict]:
    """Read the contiguous official hierarchy under an exact section-1 root.

    Lower-level municipal headings have nonzero KOD2; the next section-1 root
    (KOD2=KOD3=000) terminates this block. Names are guards, never join keys.
    """
    anchors = [i for i, x in enumerate(rows) if x["RAZDEL"] == "1" and x["code"] == root_code]
    if len(anchors) != 1:
        raise ValueError("Registry root must exist exactly once")
    start = anchors[0]
    if rows[start]["NAME1"] != expected_name:
        raise ValueError("Unexpected official municipality name")
    result = []
    for row in rows[start + 1 :]:
        if row["RAZDEL"] == "1" and row["KOD2"] == row["KOD3"] == "000":
            break
        if row["RAZDEL"] == "2" and row["KOD3"] != "000":
            # Guard the already delimited source hierarchy, not a geographic join.
            if (row["TER"], row["KOD1"]) != (rows[start]["TER"], rows[start]["KOD1"]):
                raise ValueError("Unexpected hierarchy escape")
            result.append(row)
    if not result or len({x["code"] for x in result}) != len(result):
        raise ValueError("Empty or duplicate settlement block")
    return result


def recoding(path: Path) -> list[dict]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    pairs = []
    for ti, tbl in enumerate(root.findall(".//w:tbl", NS), 1):
        for ri, row in enumerate(tbl.findall("w:tr", NS), 1):
            cells = [
                "".join(x.text or "" for x in c.findall(".//w:t", NS))
                for c in row.findall("w:tc", NS)
            ]
            if len(cells) != 2:
                continue
            codes = [re.sub(r"\s+", "", c) for c in cells]
            valid = [bool(re.fullmatch(r"\d{11}", c)) for c in codes]
            if any(valid) and not all(valid):
                raise ValueError("Incomplete numeric recoding pair")
            if all(valid):
                pairs.append(
                    {"old_code": codes[0], "new_code": codes[1], "table": ti, "table_row": ri}
                )
    for side in ("old_code", "new_code"):
        if len({r[side] for r in pairs}) != len(pairs):
            raise ValueError("Recoding is not one-to-one")
    return pairs


def publication(path: Path, expected_number: str, expected_date: str) -> str:
    text = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)
    if not re.search(r"Номер опубликования:\s*" + re.escape(expected_number), text):
        raise ValueError("Publication number mismatch")
    if not re.search(r"Дата опубликования:\s*" + re.escape(expected_date), text):
        raise ValueError("Publication date mismatch")
    return text


def build(package: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads((package / "inputs/sources.json").read_text(encoding="utf-8"))
    if len({s["path"] for s in sources}) != len(sources):
        raise ValueError("Duplicate source paths")
    for source in sources:
        path = package / source["path"]
        if sha(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError(f"Source hash/size mismatch: {source['path']}")
    source_map = {s["path"]: s for s in sources}
    review = json.loads((package / "inputs/reviewed_evidence.json").read_text(encoding="utf-8"))
    law_rows = []
    for row in review["laws"]:
        item = dict(row)
        publication(
            package / row["card_path"], row["publication_number"], row["publication_date_verbatim"]
        )
        pages = len(PdfReader(package / row["pdf_path"]).pages)
        if pages != row["pdf_pages"]:
            raise ValueError("PDF page count mismatch")
        item.update(
            pdf_sha256=source_map[row["pdf_path"]]["sha256"],
            card_sha256=source_map[row["card_path"]]["sha256"],
            boundary_equivalence="unknown",
            automatic_mask_change=False,
        )
        law_rows.append(item)
    evidence = []
    for row in review["evidence"]:
        s = source_map[row["source_path"]]
        evidence.append({**row, "source_url": s["source_url"], "source_sha256": s["sha256"]})

    registries = {k: registry(package / v) for k, v in REGISTRIES.items()}
    recode_path = "raw/reused/oktmo_change_634_2023_1.docx"
    pairs = recoding(package / recode_path)
    if len(pairs) != 131:
        raise ValueError("Unexpected recoding table scope")
    matched, summaries, detail = [], [], []
    for tid, name, before, after, expected, _ in CASES:
        old = settlement_block(registries["before"], before, name + " муниципальный район")
        new = settlement_block(registries["after"], after, name + " муниципальный округ")
        old_map, new_map = ({r["code"]: r for r in block} for block in (old, new))
        candidate = [r for r in pairs if r["old_code"] in old_map]
        missing_old = sorted(set(old_map) - {r["old_code"] for r in candidate})
        missing_new = sorted(set(new_map) - {r["new_code"] for r in candidate})
        outside_new = sorted({r["new_code"] for r in candidate} - set(new_map))
        if (
            len(old) != expected
            or len(new) != expected
            or missing_old
            or missing_new
            or outside_new
        ):
            raise ValueError("Recoding does not cover both complete source blocks")
        changed_names = 0
        for pair in candidate:
            a, b = old_map[pair["old_code"]], new_map[pair["new_code"]]
            same_name = a["NAME1"] == b["NAME1"]
            changed_names += int(not same_name)
            matched.append(
                {
                    "territory_id": tid,
                    **pair,
                    "old_name": a["NAME1"],
                    "new_name": b["NAME1"],
                    "exact_name_equal": same_name,
                    "old_registry_line": a["source_csv_line"],
                    "new_registry_line": b["source_csv_line"],
                    "old_registry_sha256": source_map[REGISTRIES["before"]]["sha256"],
                    "new_registry_sha256": source_map[REGISTRIES["after"]]["sha256"],
                    "recoding_sha256": source_map[recode_path]["sha256"],
                    "scope": "settlement_code_inventory_only",
                    "boundary_equivalence": "unknown",
                }
            )
        summaries.append(
            {
                "territory_id": tid,
                "old_municipality_code": before,
                "new_municipality_code": after,
                "old_settlement_count": len(old),
                "new_settlement_count": len(new),
                "exact_recoding_pairs": len(candidate),
                "unmatched_old": len(missing_old),
                "unmatched_new": len(missing_new),
                "name_changes": changed_names,
                "result": "exact_registry_inventory_bijection",
                "law_list_to_registry_code_reconciliation": "not_run_law_lists_have_no_OKTMO",
                "geometric_boundary_equivalence": "unknown",
                "sber_polygon_comparison": "not_run",
                "automatic_mask_change": False,
            }
        )
        for vintage, block in (("before", old), ("after", new)):
            detail.extend(
                {
                    "territory_id": tid,
                    "vintage": vintage,
                    **r,
                    "source_path": REGISTRIES[vintage],
                    "source_sha256": source_map[REGISTRIES[vintage]]["sha256"],
                }
                for r in block
            )

    city = []
    for vintage, rows in registries.items():
        for code, section, role in [
            ("19514000000", "1", "municipal_parent"),
            ("19514000001", "2", "city"),
        ]:
            found = [r for r in rows if r["code"] == code and r["RAZDEL"] == section]
            if len(found) != 1:
                raise ValueError("Expected exact city/parent registry row")
            city.append(
                {
                    "territory_id": 1051,
                    "role": role,
                    "registry_vintage": vintage,
                    **found[0],
                    "source_path": REGISTRIES[vintage],
                    "source_sha256": source_map[REGISTRIES[vintage]]["sha256"],
                    "city_code_allowed_at_2023_01_01": "unknown",
                    "price_join_2023_decision": "retain_temporal_review",
                    "law_OKATO_19410_is_not_OKTMO": True,
                }
            )

    prior_path = package / "inputs/calendar_priority_6.csv"
    with prior_path.open(encoding="utf-8-sig", newline="") as stream:
        prior = list(csv.DictReader(stream))
    if len(prior) != 6:
        raise ValueError("Expected six carried-forward calendar cases")
    for item in prior:
        filename = Path(item["source_capture_path"]).name
        local = package / "raw/reused" / filename
        if sha(local) != item["source_sha256"]:
            raise ValueError("Prior calendar evidence hash mismatch")
        item.update(
            source_capture_path="raw/reused/" + filename,
            round3_status="carried_forward_no_new_boundary_adjudication",
            applied_to_old_results=False,
        )

    outputs = {
        "official_laws.csv": law_rows,
        "evidence_index.csv": evidence,
        "settlement_recoding_131.csv": matched,
        "settlement_inventory_reconciliation.csv": summaries,
        "registry_settlement_rows_262.csv": detail,
        "veliky_ustyug_temporal_review.csv": city,
        "calendar_flags_6_carried_forward.csv": prior,
        "source_catalog.csv": sources,
    }
    for name, data in outputs.items():
        table(out / name, data)
    texts = []
    collection = PdfReader(package / "raw/reused/karelia_laws_2023_04.pdf")
    for page in [47, 48, 49, 50, 51, 52, 53, 54, *range(72, 85)]:
        texts.append(
            {
                "pdf_page": page,
                "source_path": "raw/reused/karelia_laws_2023_04.pdf",
                "text": collection.pages[page - 1].extract_text() or "",
            }
        )
    dump(out / "law_collection_extracts.json", texts)
    queries = []
    search_logs = sorted(p.name for p in package.glob("search_log*.json"))
    for log in search_logs:
        data = json.loads((package / log).read_text(encoding="utf-8"))
        queries.extend(q for b in data["batches"] for q in b["queries"])
    qa = {
        "status": "PASS_WITH_LIMITATIONS",
        "source_files_hash_verified": len(sources),
        "targeted_web_queries": len(queries),
        "full_laws_downloaded": len(law_rows),
        "new_law_pdf_pages": sum(x["pdf_pages"] for x in law_rows),
        "exact_recoding_pairs": len(matched),
        "registry_settlement_rows": len(detail),
        "unmatched_settlement_codes": 0,
        "changed_settlement_names": sum(not x["exact_name_equal"] for x in matched),
        "priority_cases": 4,
        "carried_forward_calendar_cases": len(prior),
        "boundary_equivalence_proven": 0,
        "mask_changes": 0,
        "city_2023_temporal_review_retained": True,
        "legal_lists_to_registry_join": "NOT_RUN_no_automatic_name_join",
        "limits": review["limitations"],
    }
    if len(queries) < 10:
        raise ValueError("Fewer than ten targeted web queries")
    dump(out / "QA.json", qa)
    manifest = {
        "script": Path(__file__).name,
        "script_sha256": sha(Path(__file__)),
        "source_files": sources,
        "package_inputs": [
            {"path": str(p.relative_to(package)).replace("\\", "/"), "sha256": sha(p)}
            for p in sorted((package / "inputs").glob("*"))
        ],
        "search_logs": [{"path": n, "sha256": sha(package / n)} for n in search_logs],
        "outputs": [
            {"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(out.iterdir())
            if p.name in {*outputs, "law_collection_extracts.json", "QA.json"}
        ],
        "unchanged_prior_outputs": True,
    }
    dump(out / "manifest.json", manifest)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.out.resolve() == args.package.resolve()
        or args.package.resolve() in args.out.resolve().parents
        and "raw" in args.out.relative_to(args.package).parts
    ):
        parser.error("Output must not overwrite raw evidence")
    print(json.dumps(build(args.package, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
