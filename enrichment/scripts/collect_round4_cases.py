"""Fetch a bounded explicit official-source plan and replay reviewed evidence offline.

This records publication versions, exact physical PDF pages and source units. It
does not alter prior results or treat another publication as an independent survey.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import ssl
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("An evidence table cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rounded(published: str, prior: str, published_unit: str, prior_unit: str) -> dict:
    """A one-decimal million-ruble cell has a half-step of 50,000 rubles."""
    if (published_unit, prior_unit) != ("million_rubles", "thousand_rubles"):
        raise ValueError("Unsupported unit comparison")
    a, b = Decimal(published), Decimal(prior) / Decimal(1000)
    if not (a.is_finite() and b.is_finite()) or a < 0 or b < 0:
        raise ValueError("Invalid money value")
    return {
        "prior_value_million_rubles": str(b),
        "difference_million_rubles": str(a - b),
        "rounding_half_step_million_rubles": "0.05",
        "within_rounding_interval": abs(a - b) <= Decimal("0.05"),
    }


def replay_evidence(raw: Path, recipe_path: Path, report: Path) -> dict:
    """Replay explicit human-reviewed transcriptions, never infer a name-based join."""
    bundle = raw.parent
    recipe = read_json(recipe_path)
    if sha((raw / "yakutia_publications.html").read_bytes()) != recipe["source_catalogue_sha256"]:
        raise ValueError("Catalogue hash changed")
    lineage = read_json(bundle / "inputs/lineage.json")
    prior_path = bundle / "inputs/round3_investment_reference.csv"
    if sha(prior_path.read_bytes()) != lineage["reference_sha256"]:
        raise ValueError("Prior reference hash changed")
    prior_rows = list(csv.DictReader(prior_path.open(encoding="utf-8", newline="")))
    prior = {}
    for row in prior_rows:
        key = (int(row["territory_id"]), int(row["year"]))
        if key in prior or row["indicator_id"] != "8109001":
            raise ValueError("Ambiguous prior indicator key")
        if row["value_status"] != "observed" or row["source_oktmo"] != row["official_oktmo_8"]:
            raise ValueError("Prior value is not an accepted exact-code observation")
        prior[key] = row
    evidence, comparisons, seen = [], [], set()
    accepted_files = set()
    for spec in recipe["observations"]:
        source_id = spec["source_id"]
        meta = read_json(raw / (source_id + ".meta.json"))
        source_file = raw / meta["raw_file"]
        if (
            sha(source_file.read_bytes()) != spec["source_sha256"]
            or meta["sha256"] != spec["source_sha256"]
        ):
            raise ValueError("Reviewed source hash changed")
        pages = len(PdfReader(source_file).pages)
        if not spec["visually_reviewed"] or not 1 <= spec["physical_pdf_page"] <= pages:
            raise ValueError("Missing page review")
        for image in spec["images"]:
            image_path = bundle / image["file"]
            if sha(image_path.read_bytes()) != image["sha256"]:
                raise ValueError("Reviewed image hash changed")
            if image["physical_pdf_page"] not in (
                spec["physical_pdf_page"],
                spec["scope_physical_page"],
            ):
                raise ValueError("Wrong reviewed page")
            accepted_files.add(image_path)
        if spec["model_join_accepted"] or spec["unit"] != "million_rubles":
            raise ValueError("Unsupported inference or unit")
        for row in spec["rows"]:
            if len(row["values"]) != len(spec["years"]):
                raise ValueError("Years and cells differ in length")
            for year, value in zip(spec["years"], row["values"], strict=True):
                territory = row["territory_id_context_candidate"]
                key = (source_id, territory, year)
                if key in seen:
                    raise ValueError("Duplicate evidence cell")
                seen.add(key)
                if Decimal(value).as_tuple().exponent != -1:
                    raise ValueError("Expected exactly one published decimal")
                cell = {
                    "evidence_id": f"{source_id}:{territory}:{year}",
                    "territory_id_context_candidate": territory,
                    "source_region": "Республика Саха (Якутия)",
                    "source_municipality_label": row["regional_row_label"],
                    "source_oktmo": "",
                    "indicator": "investment_fixed_capital",
                    "period": year,
                    "period_type": "calendar_year",
                    "value": value,
                    "unit": spec["unit"],
                    "price_basis": spec["price_basis"],
                    "scope": spec["scope"],
                    "physical_pdf_page": spec["physical_pdf_page"],
                    "printed_page": spec["printed_page"],
                    "table": spec["table"],
                    "scope_physical_page": spec["scope_physical_page"],
                    "scope_source_id": spec.get("scope_source_id", source_id),
                    "source_id": source_id,
                    "source_sha256": meta["sha256"],
                    "source_url": meta["url"],
                    "edition_year": spec["edition_year"],
                    "catalogue_display_date": spec["catalogue_display_date"],
                    "http_last_modified": meta["last_modified"],
                    "retrieved_at_utc": meta["retrieved_at_utc"],
                    "independent_measurement": False,
                    "join_status": spec["join_status"],
                    "model_join_accepted": False,
                    "extraction": "visually_reviewed_manual_transcription",
                }
                evidence.append(cell)
                if (territory, year) in prior:
                    old = prior[(territory, year)]
                    comparisons.append(
                        {
                            "evidence_id": cell["evidence_id"],
                            "territory_id_context_candidate": territory,
                            "year": year,
                            "published_value_million_rubles": value,
                            "prior_value_thousand_rubles": old["value"],
                            **compare_rounded(value, old["value"], spec["unit"], old["unit"]),
                            "prior_official_oktmo_8": old["official_oktmo_8"],
                            "prior_source_sha256": old["source_sha256"],
                            "prior_request_key": old["request_key"],
                            "prior_source_csv_row": old["source_csv_row"],
                            "comparison_status": "rounding_comparison_of_named_documentary_context",
                            "model_join_accepted": False,
                        }
                    )
    # All historic versions are compared with the current catalogue version; this
    # tests publication stability only, without counting them as independent data.
    latest = {
        (x["territory_id_context_candidate"], x["period"]): x
        for x in evidence
        if x["source_id"] == "yakutia_yearbook2025_current"
    }
    versions = []
    for row in evidence:
        if row["source_id"] == "yakutia_yearbook2025_current":
            continue
        current = latest[(row["territory_id_context_candidate"], row["period"])]
        versions.append(
            {
                "earlier_evidence_id": row["evidence_id"],
                "current_evidence_id": current["evidence_id"],
                "earlier_value": row["value"],
                "current_value": current["value"],
                "same_displayed_value": Decimal(row["value"]) == Decimal(current["value"]),
                "independent_measurement": False,
            }
        )
    write_csv(report / "evidence_ledger.csv", evidence)
    write_csv(report / "rounding_comparison.csv", comparisons)
    write_csv(report / "publication_version_comparison.csv", versions)
    if recipe.get("case_review_status"):
        if any(
            x["cause_resolved"] or x["model_join_accepted"] for x in recipe["case_review_status"]
        ):
            raise ValueError("The reviewed recipe does not authorize causal or model claims")
        write_csv(report / "case_review_status.csv", recipe["case_review_status"])
    # Include only the raw files and rendered pages actually used by the recipe.
    accepted_files.update(raw.glob("*"))
    accepted_files.update((bundle / "inputs").glob("*"))
    allowlist = [
        {
            "path": p.relative_to(bundle).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha(p.read_bytes()),
        }
        for p in sorted(accepted_files)
        if p.is_file()
    ]
    save(report / "raw_allowlist.json", allowlist)
    checks = {
        "raw_recipe_and_reference_hashes_verified": True,
        "unique_evidence_keys": len(seen) == len(evidence),
        "all_prior_comparisons_within_rounding_interval": all(
            x["within_rounding_interval"] for x in comparisons
        ),
        "all_reviewed_versions_same_displayed_values": all(
            x["same_displayed_value"] for x in versions
        ),
        "all_records_context_only": all(not x["model_join_accepted"] for x in evidence),
    }
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence_cells": len(evidence),
        "unique_territory_years": len(latest),
        "prior_bdpmo_comparisons": len(comparisons),
        "publication_version_comparisons": len(versions),
        "independent_new_measurements": 0,
        "accepted_model_joins": 0,
        "reviewed_pdf_versions": len(recipe["observations"]),
        "distinct_yearbook_editions": len({x["edition_year"] for x in recipe["observations"]}),
        "raw_allowlist_files": len(allowlist),
        "raw_allowlist_bytes": sum(x["bytes"] for x in allowlist),
        "prior_source_lineage": lineage,
        "scope": (
            "Hash/cardinality/arithmetic validation of a manual page review, "
            "not independent measurement or causal validation."
        ),
    }
    save(report / "QA.json", qa)
    return qa


def extract(data: bytes, suffix: str) -> dict:
    if data.startswith(b"%PDF"):
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
        return {"format": "pdf", "pages": pages, "text": "\n\f\n".join(pages)}
    if suffix == ".html":
        soup = BeautifulSoup(data, "html.parser")
        links = [
            {"label": a.get_text(" ", strip=True), "href": a["href"]}
            for a in soup.find_all("a", href=True)
        ]
        for node in soup(["script", "style"]):
            node.decompose()
        return {"format": "html", "text": soup.get_text("\n", strip=True), "links": links}
    return {"format": "unparsed_binary", "text": ""}


def retrieve(spec: dict, raw: Path, client: httpx.Client, max_bytes: int) -> dict:
    meta_path = raw / (spec["source_id"] + ".meta.json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["url"] != spec["url"]:
            raise ValueError("A source ID cannot change URL")
        if old.get("raw_file") and sha((raw / old["raw_file"]).read_bytes()) != old["sha256"]:
            raise ValueError("Saved raw bytes differ")
        return old
    allowed = {urlsplit(spec["url"]).hostname, *spec.get("allowed_redirect_hosts", [])}
    meta = {**spec, "retrieved_at_utc": datetime.now(UTC).isoformat(), "attempts": 1}
    url = spec["url"]
    try:
        chain = []
        for _ in range(6):
            parts = urlsplit(url)
            if (
                parts.scheme != "https"
                or parts.hostname not in allowed
                or parts.username
                or parts.password
            ):
                raise ValueError("Unapproved URL or redirect")
            with client.stream("GET", url) as response:
                chain.append({"url": str(response.url), "status": response.status_code})
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                meta.update(
                    http_status=response.status_code,
                    final_url=str(response.url),
                    redirects=chain,
                    content_type=response.headers.get("content-type", ""),
                    last_modified=response.headers.get("last-modified", ""),
                )
                response.raise_for_status()
                chunks, length = [], 0
                for chunk in response.iter_bytes():
                    length += len(chunk)
                    if length > max_bytes:
                        raise ValueError("Response byte limit exceeded")
                    chunks.append(chunk)
                body = b"".join(chunks)
                break
        else:
            raise ValueError("Redirect limit exceeded")
        filename = spec["source_id"] + spec["suffix"]
        if spec["suffix"] == ".pdf" and not body.startswith(b"%PDF"):
            raise ValueError("Expected a PDF body")
        (raw / filename).write_bytes(body)
        meta.update(status="downloaded", raw_file=filename, bytes=len(body), sha256=sha(body))
    except (httpx.HTTPError, ValueError) as error:
        meta.update(status="unavailable", error_type=type(error).__name__)
    save(meta_path, meta)
    return meta


def run(plan: Path, raw: Path, report: Path, derive_only: bool, max_bytes: int) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    specs = json.loads(plan.read_text(encoding="utf-8"))
    if len({s["source_id"] for s in specs}) != len(specs):
        raise ValueError("Duplicate source IDs")
    records = []
    with httpx.Client(
        verify=ssl.create_default_context(), timeout=50, follow_redirects=False
    ) as client:
        for spec in specs:
            meta_path = raw / (spec["source_id"] + ".meta.json")
            if derive_only and not meta_path.exists():
                raise ValueError("Offline replay lacks a retrieval record")
            meta = retrieve(spec, raw, client, max_bytes)
            records.append(meta)
            if meta["status"] == "downloaded":
                text_path = raw / (spec["source_id"] + ".text.json")
                if not text_path.exists():
                    result = extract((raw / meta["raw_file"]).read_bytes(), spec["suffix"])
                    result["source_sha256"] = meta["sha256"]
                    save(text_path, result)
                result = json.loads(text_path.read_text(encoding="utf-8"))
                if result["source_sha256"] != meta["sha256"]:
                    raise ValueError("Text extraction belongs to another source")
            print(
                json.dumps({"source_id": spec["source_id"], "status": meta["status"]}), flush=True
            )
    save(report / "source_inventory.json", records)
    summary = {
        "sources": len(records),
        "downloaded": sum(r["status"] == "downloaded" for r in records),
        "unavailable": sum(r["status"] != "downloaded" for r in records),
        "prior_sources_modified": False,
    }
    save(report / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=45_000_000)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    run(args.plan, args.raw_dir, args.report_dir, args.derive_only, args.max_bytes)
    if args.recipe:
        replay_evidence(args.raw_dir, args.recipe, args.report_dir)
    outputs = [
        "source_inventory.json",
        "summary.json",
        "evidence_ledger.csv",
        "rounding_comparison.csv",
        "publication_version_comparison.csv",
        "case_review_status.csv",
        "raw_allowlist.json",
        "QA.json",
    ]
    save(
        args.report_dir / "manifest.json",
        {
            "script_sha256": sha(Path(__file__).read_bytes()),
            "plan_sha256": sha(args.plan.read_bytes()),
            "recipe_sha256": sha(args.recipe.read_bytes()) if args.recipe else None,
            "files": [
                {"path": name, "sha256": sha((args.report_dir / name).read_bytes())}
                for name in outputs
                if (args.report_dir / name).exists()
            ],
        },
    )


if __name__ == "__main__":
    main()
