"""Rebuild the bounded historical geography register from saved primary documents.

No network access, fuzzy joins, prefix joins or changes to prior outputs. Official
section-1 codes are matched in full. Settlement recoding remains a separate table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from bs4 import BeautifulSoup

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
REGISTRY_FIELDS = [
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
REGISTRY_FILES = {
    "2023": "data-20230112T1301-structure-20260210T1102.csv",
    "2024": "data-20231219T1512-structure-20260210T1102.csv",
    "2024_check": "data-20240125T1401-structure-20260210T1102.csv",
}
BOUNDARY_CAVEAT = (
    "Not independently established: neither a status law nor settlement recoding "
    "proves equivalence of Sber geometry and the statistical source territory."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def registry_rows(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter=";"), 1):
            if len(row) != len(REGISTRY_FIELDS):
                raise ValueError(f"Registry width mismatch at line {line}")
            item = dict(zip(REGISTRY_FIELDS, row, strict=True))
            if item["RAZDEL"] != "1":
                continue
            components = [item[c] for c in ("TER", "KOD1", "KOD2", "KOD3")]
            if [len(c) for c in components] != [2, 3, 3, 3]:
                raise ValueError(f"Registry code width mismatch at line {line}")
            code = "".join(components)
            if not code.isdigit() or code in result:
                raise ValueError(f"Invalid/duplicate section-1 code at line {line}")
            result[code] = {**item, "source_csv_line": line, "official_oktmo_11": code}
    return result


def recoding_rows(path: Path, rejected: list | None = None) -> list[dict]:
    """Accept only complete 11-digit codes in the two explicit Word table columns."""
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    footnotes = {}
    for paragraph in root.findall(".//w:p", NS):
        text = "".join(t.text or "" for t in paragraph.findall(".//w:t", NS))
        match = re.match(r"\s*п\.\s*(\d+)\s*[–—-]\s*(.+)", text)
        if match:
            footnotes[match.group(1)] = text
    result = []
    seen = set()
    for table_no, table in enumerate(root.findall(".//w:tbl", NS), 1):
        for row_no, row in enumerate(table.findall("w:tr", NS), 1):
            cells = [
                "".join(t.text or "" for t in cell.findall(".//w:t", NS))
                for cell in row.findall("w:tc", NS)
            ]
            if len(cells) != 2:
                continue
            codes = [re.sub(r"\s+", "", cell) for cell in cells]
            valid = [bool(re.fullmatch(r"\d{11}", c)) for c in codes]
            if any(valid) and not all(valid):
                if rejected is not None:
                    note = re.fullmatch(r"\s*п\.\s*(\d+)\s*", cells[1])
                    note_text = footnotes.get(note.group(1), "") if note else ""
                    rejected.append(
                        {
                            "source_file": path.name,
                            "source_sha256": sha(path),
                            "source_table": table_no,
                            "source_table_row": row_no,
                            "old_code_verbatim": cells[0],
                            "new_code_verbatim": cells[1],
                            "status": "source_code_requires_review",
                            "accepted_pair": False,
                            "source_note_verbatim": note_text,
                            "reason": (
                                "explicit_law_reference_without_numeric_successor"
                                if note_text
                                else "unresolved_non_numeric_code"
                            ),
                        }
                    )
                    continue
                raise ValueError(f"Partial recoding pair: {path.name}, {table_no}, {row_no}")
            if not all(valid):
                continue
            if codes[0] in seen:
                raise ValueError(f"Duplicate old settlement code: {codes[0]}")
            seen.add(codes[0])
            result.append(
                {
                    "old_settlement_oktmo_11": codes[0],
                    "new_settlement_oktmo_11": codes[1],
                    "old_code_verbatim": cells[0],
                    "new_code_verbatim": cells[1],
                    "source_table": table_no,
                    "source_table_row": row_no,
                    "municipal_territory_id": "",
                    "municipal_join_status": "not_joined",
                }
            )
    if not result:
        raise ValueError(f"No complete recoding pairs: {path}")
    return result


def local_ref(path: Path, repo: Path) -> str:
    return Path(os.path.relpath(path, repo)).as_posix()


def build(
    repo: Path, raw_dir: Path, historical_raw: Path, out: Path, inputs_dir: Path | None = None
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "changes": repo / "reports/enrichment_v2/code_bridge/primary_78_code_type_changes.csv",
        "spatial": repo / "reports/enrichment_v3/spatial/sber_yearly_spatial_attributes.csv",
        "calendar": repo
        / "reports/enrichment_v2/code_bridge_adjudication/calendar_code_adjudication.csv",
        "registry_audit": repo / "reports/enrichment_v2/code_bridge/registry_vintage_audit.csv",
        "reviewed_laws": Path(__file__).with_name("round2_geography_reviewed_laws.json"),
    }
    if inputs_dir is not None:
        input_paths = {key: inputs_dir / path.name for key, path in input_paths.items()}
    changes = read_csv(input_paths["changes"])
    spatial = read_csv(input_paths["spatial"])
    calendar = read_csv(input_paths["calendar"])
    if len(changes) != 78 or changes.territory_id.duplicated().any():
        raise ValueError("Expected 78 unique changed territories")
    if spatial.duplicated(["snapshot_date", "territory_id"]).any():
        raise ValueError("Duplicate yearly spatial key")
    if len(calendar) != 6 or calendar.territory_id.duplicated().any():
        raise ValueError("Expected 6 unique calendar cases")
    if set(changes.territory_id) & set(calendar.territory_id):
        raise ValueError("Calendar appendix unexpectedly overlaps the 78 cases")

    catalog = []
    accepted = {}
    excluded = []
    for line in (raw_dir.parent / "retrievals.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        name = row["filename"]
        if "_pages.json" in name or re.search(r"_page\d+\.jpg$", name):
            excluded.append(
                {
                    "filename": name,
                    "status": row.get("status"),
                    "sha256": row.get("sha256"),
                    "reason": "viewer_metadata_or_authorization_placeholder_not_amendment_text",
                }
            )
        elif row.get("error"):
            excluded.append({"filename": name, "url": row["url"], "reason": row["error"]})
        else:
            accepted[name] = row
    for name, row in sorted(accepted.items()):
        path = raw_dir / name
        if sha(path) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        kind = row["purpose"]
        if name.startswith("oktmo_change_"):
            kind = "official_settlement_recoding_table_not_full_amendment"
        catalog.append(
            {
                "source_id": name,
                "source_kind": kind,
                "source_url": row["url"],
                "source_path": local_ref(path, repo),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "retrieved_at_utc": row["retrieved_at_utc"],
                "http_last_modified": row.get("last_modified", ""),
                "reuse_prior_snapshot": False,
            }
        )

    registry_meta = read_csv(input_paths["registry_audit"]).set_index("filename")
    registries = {}
    for year, name in REGISTRY_FILES.items():
        path = historical_raw / name
        meta = registry_meta.loc[name]
        if sha(path) != meta.sha256:
            raise ValueError(f"Prior registry hash mismatch: {name}")
        registries[year] = registry_rows(path)
        catalog.append(
            {
                "source_id": name,
                "source_kind": "official_historical_oktmo_csv",
                "source_url": meta.url,
                "source_path": local_ref(path, repo),
                "sha256": meta.sha256,
                "bytes": path.stat().st_size,
                "retrieved_at_utc": "",
                "http_last_modified": "",
                "reuse_prior_snapshot": True,
            }
        )

    laws = json.loads(input_paths["reviewed_laws"].read_text(encoding="utf-8"))
    law_by_id = {str(item["territory_id"]): item for item in laws}
    if len(law_by_id) != len(laws):
        raise ValueError("Duplicate reviewed law territory")
    law_source = raw_dir / "karelia_laws_2023_04.pdf"
    if any(item["source_sha256"] != sha(law_source) for item in laws):
        raise ValueError("Reviewed law source hash changed")

    cases = []
    evidence = []
    for row in changes.to_dict("records"):
        tid = row["territory_id"]
        item = {
            "territory_id": tid,
            "region_code_sber": row["region_code_2024"],
            "region_name_sber": row["region_name_2024"],
            "same_sber_territory_id": True,
            "code_changed": row["code_changed"],
            "type_changed": row["type_changed"],
            "boundary_equivalence_status": "unknown",
            "boundary_caveat": BOUNDARY_CAVEAT,
            "primary_law_review_status": "not_reviewed",
            "institutional_succession_status": "not_reviewed",
            "legal_creation_date": "",
            "legal_effectivity_date_status": "not_established",
            "priority": "P2_remaining_boundary_and_legal_review",
        }
        if row["region_name_2024"] == "Ставропольский край" or tid == "2618":
            item["priority"] = "P1_nontrivial_code_or_name_change_boundary_review"
        for year in ("2023", "2024"):
            code = row[f"oktmo_digits_{year}"]
            official = registries[year].get(code)
            if official is None:
                raise ValueError(f"Missing exact full official code: {tid}, {year}, {code}")
            date = f"{year}-01-01"
            sp = spatial[(spatial.territory_id == tid) & (spatial.snapshot_date == date)]
            if len(sp) != 1 or sp.iloc[0].oktmo_digits != code:
                raise ValueError(f"Yearly spatial/code conflict: {tid}, {year}")
            name = REGISTRY_FILES[year]
            meta = registry_meta.loc[name]
            for key, value in {
                "snapshot_date": date,
                "sber_oktmo_11": code,
                "official_oktmo_11": official["official_oktmo_11"],
                "sber_name": row[f"municipal_district_name_{year}"],
                "sber_type": row[f"municipal_district_type_{year}"],
                "official_name": official["NAME1"],
                "registry_NomAkt": official["NomAkt"],
                "registry_DateUtv": official["DateUtv"],
                "registry_DateVved": official["DateVved"],
                "geometry_year_from": sp.iloc[0].geometry_year_from,
                "geometry_year_to_exclusive": sp.iloc[0].geometry_year_to_exclusive,
            }.items():
                item[f"{key}_{year}"] = value
            item[f"official_region_name_{year}"] = registries[year][code[:2] + "000000000"]["NAME1"]
            evidence.append(
                {
                    "territory_id": tid,
                    "snapshot_date": date,
                    **official,
                    "source_file": name,
                    "source_url": meta.url,
                    "source_sha256": meta.sha256,
                    "join_basis": "exact_full_section1_code",
                }
            )
        checked = registries["2024_check"][item["official_oktmo_11_2024"]]
        if any(
            checked[key] != registries["2024"][item["official_oktmo_11_2024"]][key]
            for key in ("NAME1", "NomAkt", "DateUtv", "DateVved")
        ):
            raise ValueError(f"2024 registry confirmation conflict: {tid}")
        item["registry_2024_second_vintage_confirmed"] = True
        item["registry_2024_second_vintage_file"] = REGISTRY_FILES["2024_check"]
        item["recoding_source_file"] = (
            f"oktmo_change_{int(item['registry_NomAkt_2024'])}_2023_1.docx"
        )
        item["recoding_scope"] = "settlement_pairs_only_not_municipal_boundary_equivalence"
        if tid in law_by_id:
            law = law_by_id[tid]
            item.update(
                {
                    "priority": "P1_law_reviewed_publication_and_boundaries_open",
                    "primary_law_review_status": "transformation_status_proven",
                    "institutional_succession_status": (
                        "local_government_organs_explicit_successors"
                    ),
                    "law_number": law["law_number"],
                    "law_date": law["law_date"],
                    "law_source_file": law["source_file"],
                    "law_pdf_pages": law["pdf_pages"],
                    "legal_effectivity_rule": law["entry_into_force_rule"],
                    "transition_period_end": law["transition_period_end"],
                }
            )
        cases.append(item)

    recodings = []
    rejected_recodings = []
    for name, row in sorted(accepted.items()):
        if not name.startswith("oktmo_change_") or not name.endswith(".docx"):
            continue
        for item in recoding_rows(raw_dir / name, rejected_recodings):
            recodings.append(
                {
                    "amendment": re.search(r"change_(\d+)_", name).group(1) + "/2023",
                    **item,
                    "source_file": name,
                    "source_url": row["url"],
                    "source_sha256": row["sha256"],
                }
            )

    calendar_rows = []
    for row in calendar.to_dict("records"):
        filename = Path(row["source_capture_path"]).name
        path = historical_raw / filename
        if sha(path) != row["source_sha256"]:
            raise ValueError(f"Calendar source hash mismatch: {filename}")
        if not any(item["source_id"] == filename for item in catalog):
            catalog.append(
                {
                    "source_id": filename,
                    "source_kind": "prior_calendar_source_" + row["source_capture_kind"],
                    "source_url": row["source_url"],
                    "source_path": local_ref(path, repo),
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                    "retrieved_at_utc": "",
                    "http_last_modified": "",
                    "reuse_prior_snapshot": True,
                }
            )
        sp = spatial[
            (spatial.territory_id == row["territory_id"])
            & (spatial.snapshot_date == row["reference_date"])
        ]
        if len(sp) != 1 or sp.iloc[0].oktmo_digits != row["sber_oktmo_11"]:
            raise ValueError("Calendar full code/year mismatch")
        calendar_rows.append(
            {
                **row,
                "region_name": sp.iloc[0].region_name,
                "municipal_district_name": sp.iloc[0].municipal_district_name,
                "in_78_change_cohort": False,
                "priority": "P0_initial_economy_calendar_flag",
                "source_capture_path": local_ref(path, repo),
                "boundary_equivalence_status": "unknown",
                "applied_to_old_results": False,
            }
        )

    card_dates = []
    for act in (633, 634, 637, 648):
        name = f"gost_card_{act}_2023.html"
        text = BeautifulSoup((raw_dir / name).read_bytes(), "html.parser").get_text(" ", strip=True)
        match = re.search(r"Дата введения\s+(\d{2}\.\d{2}\.\d{4})", text)
        if not match:
            raise ValueError(f"Card date not found: {name}")
        card_dates.append(
            {
                "amendment": f"{act}/2023",
                "card_DateVved_verbatim": match.group(1),
                "matched_new_code_registry_DateVved": "01.01.2024",
                "date_basis_resolved": False,
                "source_file": name,
                "source_url": accepted[name]["url"],
                "source_sha256": accepted[name]["sha256"],
                "decision": "retain_distinct_date_fields_do_not_backdate_2024_codes",
            }
        )

    tables = {
        "case_register_78.csv": pd.DataFrame(cases),
        "exact_registry_evidence_156.csv": pd.DataFrame(evidence),
        "settlement_recoding_pairs.csv": pd.DataFrame(recodings),
        "recoding_source_review.csv": pd.DataFrame(rejected_recodings),
        "calendar_priority_6.csv": pd.DataFrame(calendar_rows),
        "reviewed_laws_3.csv": pd.DataFrame(laws),
        "card_registry_date_differences.csv": pd.DataFrame(card_dates),
        "source_catalog.csv": pd.DataFrame(catalog),
    }
    cases_df = tables["case_register_78.csv"]
    tables["region_coverage.csv"] = cases_df.groupby("region_name_sber", as_index=False).agg(
        case_count=("territory_id", "size"),
        amendment_numbers=("registry_NomAkt_2024", lambda x: ";".join(sorted(set(x)))),
        reviewed_law_cases=(
            "primary_law_review_status",
            lambda x: sum(x == "transformation_status_proven"),
        ),
    )
    for name, table in tables.items():
        table.to_csv(out / name, index=False, encoding="utf-8")
    (out / "excluded_retrieval_attempts.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS_WITH_DOCUMENTED_BOUNDARY_AND_PUBLICATION_GAPS",
        "scope": "78 changed Sber annual territories plus 6 disjoint calendar cases",
        "cases": len(cases),
        "regions": cases_df.region_name_sber.nunique(),
        "exact_code_evidence_rows": len(evidence),
        "new_code_registry_date_counts": cases_df.registry_DateVved_2024.value_counts().to_dict(),
        "recoding_documents": len({r["source_file"] for r in recodings}),
        "settlement_recoding_pairs": len(recodings),
        "recoding_source_rows_requiring_review": len(rejected_recodings),
        "recoding_law_reference_rows": sum(
            bool(r["source_note_verbatim"]) for r in rejected_recodings
        ),
        "reviewed_full_laws": len(laws),
        "calendar_cases_reused": len(calendar_rows),
        "calendar_overlap_with_78": 0,
        "independent_boundary_equivalence_confirmed": 0,
        "municipal_joins_from_settlement_prefixes": 0,
        "source_count": len(catalog),
        "new_downloaded_source_count": len(accepted),
        "network_calls_by_builder": 0,
        "inputs": {
            k: {"path": local_ref(p, repo), "sha256": sha(p)} for k, p in input_paths.items()
        },
        "script_sha256": sha(Path(__file__)),
        "outputs": {
            name: {"rows": len(table), "sha256": sha(out / name)} for name, table in tables.items()
        },
        "excluded_attempts_sha256": sha(out / "excluded_retrieval_attempts.json"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {k: v for k, v in manifest.items() if k not in {"inputs", "outputs"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--historical-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--inputs-dir", type=Path, help="Optional bundled input snapshots directory"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.repo.resolve(),
                args.raw_dir.resolve(),
                args.historical_raw.resolve(),
                args.out.resolve(),
                args.inputs_dir.resolve() if args.inputs_dir else None,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
