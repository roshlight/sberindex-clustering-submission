"""Offline audit of official code transitions and candidate historical municipality links.

No company is assigned a historical municipality. Published code transitions,
classifier structure, legal boundaries and enterprise address history are distinct.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

REGISTRY_COLUMNS = [
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
PRIORITY = {"05": "Приморский край", "57": "Пермский край", "98": "Республика Саха (Якутия)"}
VERSIONS = ["20231220", "20240101", "20241102", "20250108", "20260601", "20260901"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode(data):
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1251"), "cp1251"


def normalized_code(text):
    """Remove published digit grouping only. Never pad or interpret a missing destination."""
    value = re.sub(r"[ \u00a0]", "", text)
    return value if re.fullmatch(r"\d{11}", value, flags=re.ASCII) else None


def read_crosswalk(path, label):
    text, encoding = decode(path.read_bytes())
    source_sha = digest(path)
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    if not rows[0][0].startswith("Перекодировочная таблица"):
        raise ValueError("Unexpected crosswalk title")
    if rows[1][:4] != [
        "Субъект Российской Федерации",
        "Аннулированный код",
        "Действующий код",
        "Номер изменения",
    ]:
        raise ValueError("Unexpected crosswalk columns")
    records = []
    for number, row in enumerate(rows[2:], 3):
        if not any(row):
            continue
        if len(row) < 4 or any(cell.strip() for cell in row[4:]):
            raise ValueError("Nonempty undeclared column")
        old, new = normalized_code(row[1]), normalized_code(row[2])
        records.append(
            dict(
                version=label,
                csv_row=number,
                region=row[0],
                old_as_published=row[1],
                new_as_published=row[2],
                old_code=old or "",
                new_code=new or "",
                amendment=row[3],
                old_status="valid_11_digits" if old else "nonstandard_or_missing",
                new_status="valid_11_digits"
                if new
                else "missing"
                if not row[2].strip()
                else "nonstandard",
                source_sha256=source_sha,
            )
        )
    return pd.DataFrame(records), encoding


def registry(path):
    frame = pd.read_csv(path, sep=";", names=REGISTRY_COLUMNS, dtype=str, keep_default_na=False)
    for name, width in [("TER", 2), ("KOD1", 3), ("KOD2", 3), ("KOD3", 3)]:
        if not frame[name].str.fullmatch(rf"[0-9]{{{width}}}").all():
            raise ValueError(f"Bad classifier code component {name}")
    frame["code"] = frame.TER + frame.KOD1 + frame.KOD2
    settlement = frame.RAZDEL.eq("2") & frame.KOD3.ne("000")
    frame.loc[settlement, "code"] += frame.loc[settlement, "KOD3"]
    return frame.loc[frame.RAZDEL.eq("1") | settlement].copy()


def parent_candidates(entries, panel):
    """A classifier-structural candidate, requiring exact existing parent and panel codes."""
    section1 = entries.loc[entries.RAZDEL.eq("1")]
    if section1.code.duplicated().any() or panel.official_oktmo_8.duplicated().any():
        raise ValueError("Ambiguous parent/panel registry")
    parents = section1.set_index("code")
    territories = panel.set_index("official_oktmo_8")
    records = []
    for row in entries.loc[entries.RAZDEL.eq("2")].itertuples():
        immediate = row.TER + row.KOD1 + row.KOD2
        top = row.TER + row.KOD1 + "000"
        for candidate, relation in [
            (immediate, "explicit_section2_parent_fields"),
            (top, "section1_second_level_fields"),
        ]:
            if relation == "section1_second_level_fields" and top == immediate:
                continue
            if candidate not in parents.index or candidate not in territories.index:
                continue
            territory = territories.loc[candidate]
            records.append(
                dict(
                    settlement_code=row.code,
                    settlement_name=row.NAME1,
                    ter=row.TER,
                    parent_code=candidate,
                    parent_name=parents.loc[candidate, "NAME1"],
                    territory_id=str(territory.territory_id),
                    relation=relation,
                    settlement_date_vved=row.DateVved,
                    settlement_amendment=row.NomAkt,
                    municipality_address_join_approved=False,
                    historical_boundary_equivalence_proven=False,
                )
            )
    return pd.DataFrame(records)


def docx_rows(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = ["".join(p.itertext()) for p in root.findall(".//w:p", ns)]
    tables = []
    for tr in root.findall(".//w:tr", ns):
        tables.append(["".join(cell.itertext()).strip() for cell in tr.findall("w:tc", ns)])
    return paragraphs, tables


def build(repo, raw, prior_raw, out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    input_hashes, downloads, attempts = {}, [], []
    for metadata in sorted(raw.glob("*/retrievals.json")):
        input_hashes["raw/" + metadata.relative_to(raw).as_posix()] = digest(metadata)
        for item in json.loads(metadata.read_text(encoding="utf-8")):
            item = {**item, "phase": metadata.parent.name}
            attempts.append(item)
            if item.get("path"):
                p = metadata.parent / item["path"]
                if digest(p) != item["sha256"] or p.stat().st_size != item["bytes"]:
                    raise ValueError(f"Downloaded bytes changed: {p}")
                relative = p.relative_to(raw).as_posix()
                input_hashes["raw/" + relative] = digest(p)
                downloads.append(
                    {k: item.get(k) for k in ["url", "retrieved_at", "status", "bytes", "sha256"]}
                    | {"raw_path": relative}
                )
    pd.DataFrame(downloads).to_csv(out / "downloaded_sources.csv", index=False)
    write_json(out / "retrieval_attempts.json", attempts)
    soup = BeautifulSoup(
        (raw / "remote_initial/rosstat_classification.html").read_bytes(), "html.parser"
    )
    catalog = []
    for item in soup.select(".document-list__item"):
        title = item.select_one(".document-list__item-title")
        if not title:
            continue
        title = title.get_text(" ", strip=True)
        if re.search(r"Изменение \d+/202\d.*[ОO][КK][ТT][МM][ОO]", title):
            catalog.append(
                dict(
                    title=title,
                    url=urljoin("https://rosstat.gov.ru", item.a["href"]),
                    page_info=item.select_one(".document-list__item-info").get_text(
                        " ", strip=True
                    ),
                )
            )
    pd.DataFrame(catalog).to_csv(out / "official_amendment_document_catalog.csv", index=False)
    frames, version_rows = {}, []
    for label in VERSIONS:
        paths = list((raw / "crosswalk_files").glob(f"data-{label}T*.csv"))
        if len(paths) != 1:
            raise ValueError("Expected one declared version")
        frame, encoding = read_crosswalk(paths[0], label)
        frames[label] = frame
        clean = frame.loc[
            frame.old_status.eq("valid_11_digits") & frame.new_status.eq("valid_11_digits")
        ]
        version_rows.append(
            dict(
                version=label,
                raw_path=paths[0].relative_to(raw).as_posix(),
                sha256=digest(paths[0]),
                encoding=encoding,
                rows=len(frame),
                complete_pair_rows=len(clean),
                missing_destination_rows=int(frame.new_status.eq("missing").sum()),
                nonstandard_destination_rows=int(frame.new_status.eq("nonstandard").sum()),
                duplicate_declared_rows=int(
                    frame.duplicated(
                        ["region", "old_as_published", "new_as_published", "amendment"]
                    ).sum()
                ),
                unique_complete_pairs=len(clean[["old_code", "new_code"]].drop_duplicates()),
                effective_date_column_available=False,
            )
        )
    pd.DataFrame(version_rows).to_csv(out / "crosswalk_versions.csv", index=False)
    changes = []
    for older, newer in zip(VERSIONS[:-1], VERSIONS[1:], strict=True):
        columns = ["region", "old_code", "new_code", "amendment", "new_as_published"]
        a = set(frames[older][columns].itertuples(index=False, name=None))
        b = set(frames[newer][columns].itertuples(index=False, name=None))
        changes.append(
            dict(
                older_version=older,
                newer_version=newer,
                added_declared_rows=len(b - a),
                removed_declared_rows=len(a - b),
                release_dates_are_not_effective_dates=True,
            )
        )
    pd.DataFrame(changes).to_csv(out / "transition_version_changes.csv", index=False)
    latest = frames["20260601"].copy()
    priority_mask = latest.region.isin(PRIORITY.values())
    latest.loc[priority_mask].to_csv(out / "priority_transition_rows.csv", index=False)
    anomalies = latest.loc[
        latest.new_status.ne("valid_11_digits") | latest.old_status.ne("valid_11_digits")
    ].copy()
    anomalies.to_csv(out / "transition_nonlinks.csv", index=False)
    edge_columns = ["region", "old_code", "new_code", "amendment"]
    complete = latest.loc[
        latest.new_status.eq("valid_11_digits") & latest.old_status.eq("valid_11_digits")
    ]
    duplicates = complete.loc[complete.duplicated(edge_columns, keep=False)]
    duplicates.to_csv(out / "transition_duplicate_rows.csv", index=False)
    edges = complete.drop_duplicates(edge_columns).copy()
    reverse_counts = edges.groupby("new_code").old_code.nunique()
    forward_counts = edges.groupby("old_code").new_code.nunique()
    ambiguous = edges.loc[
        edges.new_code.map(reverse_counts).gt(1) | edges.old_code.map(forward_counts).gt(1)
    ].copy()
    ambiguous["predecessor_code_count"] = ambiguous.new_code.map(reverse_counts)
    ambiguous["successor_code_count"] = ambiguous.old_code.map(forward_counts)
    ambiguous.to_csv(out / "ambiguous_transition_edges.csv", index=False)
    inventory = pd.read_csv(
        repo / "reports/data_round5/geography/sme_code_inventory.csv",
        dtype={"source_code": str},
        keep_default_na=False,
    )
    panel_path = repo / "reports/round5_analysis/audit/annual_features.csv"
    panel = pd.read_csv(panel_path, dtype={"official_oktmo_8": str, "territory_id": str})
    input_hashes["repo/" + panel_path.relative_to(repo).as_posix()] = digest(panel_path)
    input_hashes["repo/reports/data_round5/geography/sme_code_inventory.csv"] = digest(
        repo / "reports/data_round5/geography/sme_code_inventory.csv"
    )
    manifest_path = repo / "reports/data_round5/geography/registry_versions.csv"
    prior_manifest = pd.read_csv(manifest_path)
    input_hashes["repo/" + manifest_path.relative_to(repo).as_posix()] = digest(manifest_path)
    all_candidates, controls, transitions = [], [], []
    for year, version in [(2023, "2023-02-08"), (2024, "2024-01-25")]:
        manifest_row = prior_manifest.loc[prior_manifest.label.eq(version)].iloc[0]
        path = prior_raw / manifest_row.path
        if digest(path) != manifest_row.sha256:
            raise ValueError("Historical classifier changed")
        input_hashes["prior_raw/" + manifest_row.path] = digest(path)
        entries = registry(path)
        annual_panel = panel.loc[panel.year.eq(year)]
        if len(annual_panel) != 2016 or annual_panel.territory_id.duplicated().any():
            raise ValueError("Expected fixed 2016-municipality annual panel")
        candidates = parent_candidates(entries, annual_panel)
        candidates.insert(0, "year", year)
        candidates["registry_version_label"] = version
        candidates["registry_sha256"] = digest(path)
        candidates["panel_reference_date"] = f"{year}-01-01"
        dates = pd.to_datetime(candidates.settlement_date_vved, format="%d.%m.%Y", errors="coerce")
        candidates["record_date_after_reference"] = dates.gt(pd.Timestamp(f"{year}-01-01"))
        candidates["record_date_unparsed"] = dates.isna()
        all_candidates.append(candidates.loc[candidates.ter.isin(PRIORITY)])
        candidate_counts = candidates.groupby("parent_code").settlement_code.nunique().to_dict()
        entry_codes = set(entries.code)
        for code, row in annual_panel.set_index("official_oktmo_8").iterrows():
            controls.append(
                dict(
                    year=year,
                    territory_id=row.territory_id,
                    panel_oktmo=code,
                    region_code=int(row.region_code),
                    registry_version_label=version,
                    parent_exactly_in_registry=code in entry_codes,
                    candidate_settlement_codes=int(candidate_counts.get(code, 0)),
                    historical_address_join_approved=False,
                )
            )
        # Direct, declared one-step transition only. No transitive closure, nearest name,
        # guessed date, code padding or assignment to a real enterprise is performed.
        priority_edges = edges.loc[edges.region.isin(PRIORITY.values())]
        linked = priority_edges.merge(
            candidates,
            left_on="old_code",
            right_on="settlement_code",
            how="inner",
            validate="many_to_many",
        )
        source = inventory.loc[inventory.year.eq(year)]
        if source.source_code.duplicated().any():
            raise ValueError("Unexpected null/empty duplicate in SME code inventory")
        linked = linked.merge(
            source[["source_code", "source_rows"]],
            left_on="new_code",
            right_on="source_code",
            how="inner",
            validate="many_to_one",
        )
        linked["new_code_predecessors_all_regions"] = linked.new_code.map(reverse_counts)
        linked["candidate_not_assignment"] = True
        linked["address_date_proven"] = False
        transitions.append(linked)
    candidates = pd.concat(all_candidates, ignore_index=True)
    candidates.to_csv(out / "priority_settlement_parent_candidates.csv", index=False)
    pd.DataFrame(controls).to_csv(out / "panel_2016_classifier_control.csv", index=False)
    links = pd.concat(transitions, ignore_index=True)
    links.to_csv(out / "candidate_transitions_to_panel.csv", index=False)
    summary = []
    for (year, region), group in links.groupby(["year", "region"]):
        codes = group[["new_code", "source_rows"]].drop_duplicates()
        summary.append(
            dict(
                year=int(year),
                region=region,
                candidate_edges=len(group),
                source_code_groups=len(codes),
                source_rows_touched=int(codes.source_rows.sum()),
                candidate_panel_municipalities=group.territory_id.nunique(),
                multi_predecessor_source_codes=int(
                    group.loc[group.new_code_predecessors_all_regions.gt(1), "new_code"].nunique()
                ),
                historical_company_assignments_approved=0,
            )
        )
    pd.DataFrame(summary).to_csv(out / "candidate_summary.csv", index=False)
    # Independent official DOCX versus the CSV transition rows, preserving the source.
    doc = raw / "supplement/643_23 ОКТМО.docx"
    paragraphs, cells = docx_rows(doc)
    doc_pairs = [
        (normalized_code(row[0]), normalized_code(row[1]))
        for row in cells
        if len(row) == 2 and normalized_code(row[0]) and normalized_code(row[1])
    ]
    csv_pairs = list(
        edges.loc[edges.amendment.eq("643/2023 ОКТМО"), ["old_code", "new_code"]].itertuples(
            index=False, name=None
        )
    )
    if set(doc_pairs) != set(csv_pairs) or len(doc_pairs) != len(csv_pairs):
        raise ValueError("Official DOCX and CSV transition pairs disagree")
    write_json(
        out / "primorye_643_document_check.json",
        dict(
            status="PASS",
            docx_pairs=len(doc_pairs),
            csv_pairs=len(csv_pairs),
            docx_sha256=digest(doc),
            independent_primary_document_agreement=True,
            proves_historical_enterprise_location=False,
        ),
    )
    paragraphs, _ = docx_rows(raw / "priority_sources/oktmo_introduction_891.docx")
    write_json(
        out / "classifier_structural_evidence.json",
        dict(
            source="priority_sources/oktmo_introduction_891.docx",
            edition="891 (2026 publication)",
            paragraphs=[
                p
                for p in paragraphs
                if any(t in p for t in ["8-ми разрядному", "3–5", "6–8", "границы"])
            ],
            edition_is_not_historical_boundary_evidence=True,
        ),
    )
    pdf_path = raw / "supplement/perm_20240101.pdf"
    pages = [p.extract_text() for p in PdfReader(pdf_path).pages]
    write_json(out / "perm_reference_text.json", dict(source_sha256=digest(pdf_path), pages=pages))
    pdf_codes = sorted(set(re.findall(r"57 \d{3} \d{3}", "\n".join(pages))))
    perm_panel = panel.loc[panel.year.eq(2024) & panel.region_code.eq(59)]
    matches = set(code.replace(" ", "") for code in pdf_codes) & set(perm_panel.official_oktmo_8)
    write_json(
        out / "perm_reference_control.json",
        dict(
            pdf_codes=len(pdf_codes),
            panel_municipalities=len(perm_panel),
            exact_codes_matched=len(matches),
            effective_date_in_document="2023-11-01",
            filename_reference="2024-01-01",
            historical_address_join_proven=False,
        ),
    )
    schema_rows = []
    for filename in ["gar_schemas.zip", "fias_structure_20210312.zip"]:
        with zipfile.ZipFile(raw / "priority_sources" / filename) as archive:
            for member in archive.namelist():
                if any(
                    token in member for token in ["MUN_HIERARCHY", "CHANGE_HISTORY", "ADDR_OBJ_"]
                ):
                    root = ET.fromstring(archive.read(member))
                    ns = {"x": "http://www.w3.org/2001/XMLSchema"}
                    for field in root.findall(".//x:attribute", ns):
                        schema_rows.append(
                            dict(
                                archive=filename,
                                member=member,
                                attribute=field.get("name"),
                                required=field.get("use"),
                                documentation=" ".join(
                                    e.text or ""
                                    for e in field.findall("x:annotation/x:documentation", ns)
                                ),
                            )
                        )
    pd.DataFrame(schema_rows).to_csv(out / "gar_schema_history_fields.csv", index=False)
    gar = json.loads((raw / "remote_initial/fias_download_catalog.json").read_text())
    pd.DataFrame(gar).to_csv(out / "gar_available_download_catalog.csv", index=False)
    evidence = [
        dict(
            claim="explicit_cancelled_to_valid_settlement_codes",
            status="ready_with_conditions",
            source="crosswalk_files",
            evidence="Six publisher releases with source-row provenance",
            limitation="No effective-date field; release date does not date territorial membership",
        ),
        dict(
            claim="primorye_643_csv_consistent_with_primary_document",
            status="ready",
            source="supplement/643_23 ОКТМО.docx",
            evidence=f"{len(doc_pairs)} exact code pairs",
            limitation=(
                "Agreement confirms published recoding, not enterprise location or boundaries"
            ),
        ),
        dict(
            claim="perm_2024_panel_code_reference",
            status="ready_with_conditions",
            source="supplement/perm_20240101.pdf",
            evidence=f"{len(matches)}/42 exact panel codes",
            limitation="Two-page section 1 list, not settlement composition or address history",
        ),
        dict(
            claim="classifier_structural_parent_candidate",
            status="ready_with_conditions",
            source="priority_sources/oktmo_introduction_891.docx",
            evidence="Declared component construction plus exact parent and panel records",
            limitation="2026 introduction, historical snapshots can include future-effective rows",
        ),
        dict(
            claim="gar_history_fields_available_in_schema",
            status="ready_with_conditions",
            source="priority_sources/gar_schemas.zip",
            evidence="STARTDATE ENDDATE PREVID NEXTID PARENTOBJID OKTMO and UPDATEDATE documented",
            limitation="Schema does not supply historical object records; old archive not acquired",
        ),
        dict(
            claim="historical_enterprise_address_to_municipality",
            status="unknown",
            source="candidate_transitions_to_panel.csv",
            evidence="Only code-level candidates; no dated enterprise-address assignment",
            limitation="Modern geocoding lineage, observation address and date remain unproven",
        ),
        dict(
            claim="sakha_settlement_recoding_coverage",
            status="unknown",
            source="crosswalk_files/data-20260601T1106-structure-20231110T1511.csv",
            evidence="No rows labelled Republic of Sakha in the downloaded June recoding table",
            limitation="Absence of recoding rows does not prove unchanged boundaries or addresses",
        ),
    ]
    pd.DataFrame(evidence).to_csv(out / "evidence_ledger.csv", index=False)
    qa = dict(
        status="ready_with_conditions",
        machine_checks="PASS",
        historical_municipal_join="unknown",
        historical_company_assignments_approved=0,
        model_adoption=False,
        downloaded_bodies=len(downloads),
        failed_attempts=sum(bool(x.get("error")) for x in attempts),
        crosswalk_versions=len(frames),
        june_rows=len(latest),
        june_complete_pairs=len(complete),
        june_unique_declared_edges=len(edges),
        priority_transition_rows=int(priority_mask.sum()),
        panel_control_rows=len(controls),
        priority_settlement_parent_candidates=len(candidates),
        candidate_records_after_jan1=int(candidates.record_date_after_reference.sum()),
        transition_candidate_rows=len(links),
        docx_csv_pair_check=len(doc_pairs),
        perm_2024_exact_panel_codes=len(matches),
        gar_catalog_versions=len(gar),
        gar_min_version=min(x["VersionId"] for x in gar),
        gar_max_version=max(x["VersionId"] for x in gar),
        no_company_identifiers_published=True,
        source_code_group_rows_are_not_unique_firms=True,
        legal_boundary_equivalence_proven=False,
        historical_address_lineage_proven=False,
        duplicate_edges_removed_only_from_candidate_graph=True,
        missing_destinations_preserved_as_nonlinks=True,
        reverse_edges_are_candidates_not_invertible_join=True,
    )
    write_json(out / "QA.json", qa)
    write_json(
        out / "manifest.json",
        dict(
            script_sha256=digest(Path(__file__)),
            inputs=input_hashes,
            outputs={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
            historical_join_approved=False,
        ),
    )
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo, args.raw, args.prior_raw, args.out)
