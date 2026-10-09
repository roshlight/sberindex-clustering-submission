"""Append evidence-based calendar-code decisions without changing the original audit.

Rosstandart's explicit early-application comments and the FNS explicit code/date
notice address code use only. They do not adjudicate population-cell boundaries.
The full six-row initial warning list remains immutable.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CARDS = {
    "598": ("598_2023", "cb34ba18-8543-4d72-9a95-40c55c3cbcd9"),
    "600": ("600_2023", "3c1e0930-031a-4ef5-9217-ab13213e5d12"),
    "594": ("594_2022", "85cf7d7a-cca0-4389-ae8b-ed264f54b608"),
    "593": ("593_2022", "28c5d432-50bf-49c4-a508-7c54d524ecb9"),
}
FNS_URL = "https://www.nalog.gov.ru/rn35/taxation/submission_statements/rekvizit/12986966/"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_card(body, amendment):
    soup = BeautifulSoup(body, "html.parser")
    title = soup.find("h1")
    if title is None or f"Изменение {amendment}/" not in title.get_text():
        raise ValueError("Wrong official amendment card")
    label = soup.find(string=lambda text: text and text.strip() == "Комментарий к дате введения")
    if label is None:
        raise ValueError("Missing official introduction-date comment")
    value = label.parent.parent.find("p")
    text = value.get_text(" ", strip=True) if value else ""
    if "правом досрочного применения" not in text or "с 1 января 2023 г." not in text:
        raise ValueError("No explicit permission from 2023-01-01")
    return text


def prepare(audit_dir, source_dir, output_dir):
    audit_dir, source_dir, output_dir = map(Path, (audit_dir, source_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a new output directory")
    audit_path = audit_dir / "temporal_review_2023.csv"
    audit = pd.read_csv(audit_path, dtype=str)
    if audit.duplicated(["territory_id", "reference_date", "official_oktmo_8"]).any():
        raise ValueError("Duplicate initial warning")
    decisions, sources = [], set()
    for row in audit.itertuples():
        if row.reference_date != "2023-01-01":
            raise ValueError("This documentary review addresses only 2023-01-01")
        amendment = row.registry_nomakt
        if amendment in CARDS:
            name, identifier = CARDS[amendment]
            source = source_dir / f"gost_change_{name}.html"
            validate_card(source.read_bytes(), amendment)
            url = "https://protect.gost.ru/classificators/changesdetails/" + identifier
            kind = "raw_html"
            early_application_allowed = True
            basis = "explicit_early_application_permission_in_official_amendment_card"
            locator = (
                f"h1 amendment {name.replace('_', '/')}; "
                "Основные сведения > Даты и регистрация > Комментарий к дате введения; "
                f"exact official code linked by registry NomAkt={amendment}"
            )
            excerpt = "с правом досрочного применения ... с 1 января 2023 г."
        elif row.official_oktmo_8 == "19514000" and amendment == "586":
            source = source_dir / "fns_19514000_web_tool_text.txt"
            text = source.read_text(encoding="utf-8")
            required = [FNS_URL, "с 01.01.2023 вновь образованы:", "код ОКТМО 19514000"]
            if not all(part in text for part in required):
                raise ValueError("Missing exact FNS code/date evidence")
            url, kind = FNS_URL, "web_tool_text"
            early_application_allowed = None
            basis = "official_FNS_notice_explicitly_associates_exact_code_with_2023_01_01"
            locator = "Main article dated 29.12.2022, opening date clause and 19514000 list entry"
            excerpt = "с 01.01.2023 вновь образованы: ... (код ОКТМО 19514000)"
        else:
            raise ValueError(f"No reviewed documentary decision for {row.official_oktmo_8}")
        sources.add(source)
        decisions.append(
            {
                "territory_id": int(row.territory_id),
                "reference_date": row.reference_date,
                "official_oktmo_8": row.official_oktmo_8,
                "sber_oktmo_11": row.sber_oktmo_11,
                "registry_nomakt": amendment,
                "early_application_allowed": early_application_allowed,
                "code_date_use_confirmed": True,
                "allowed_from": "2023-01-01",
                "source_url": url,
                "source_sha256": digest(source),
                "source_capture_kind": kind,
                "source_capture_path": "evidence/" + source.name,
                "locator": locator,
                "source_excerpt": excerpt,
                "evidence_basis": basis,
                "decision": "clear_calendar_code_conflict_only",
                "remaining_boundary_caveat": (
                    "Population-cell geography and independent equivalence with the Sber annual "
                    "territory remain unconfirmed; this decision only clears "
                    "the code/date conflict."
                ),
                "independent_boundary_equivalence_confirmed": None,
                "initial_DateUtv": row.registry_DateUtv,
                "initial_DateVved": row.registry_DateVved,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir()
    for source in sorted(sources):
        shutil.copyfile(source, evidence_dir / source.name)
    frame = pd.DataFrame(decisions)
    frame.to_csv(output_dir / "calendar_code_adjudication.csv", index=False)
    (output_dir / "calendar_code_adjudication.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "initial_temporal_flags": len(audit),
        "calendar_code_conflicts_cleared": len(frame),
        "explicit_Rosstandart_permission_links": int(
            frame.source_capture_kind.eq("raw_html").sum()
        ),
        "explicit_FNS_code_date_links": int(frame.source_capture_kind.eq("web_tool_text").sum()),
        "independent_boundary_equivalence_confirmed_links": 0,
        "original_audit_mutated": False,
        "population_values_mutated": False,
        "adjudication_scope": "Calendar use of exact codes; not legal or population boundary audit",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "audit_source": str(audit_path),
        "audit_source_sha256": digest(audit_path),
        "processor_sha256": digest(__file__),
        "output_hashes": {
            p.relative_to(output_dir).as_posix(): digest(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-dir", "source-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
