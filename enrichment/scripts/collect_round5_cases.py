"""Bounded source snapshots and offline replay of reviewed round-five case evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_save(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Empty evidence is not a successful result")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_snapshot(record: dict, raw: Path, *, write_aux: bool = False) -> str:
    """Derive reviewed text from saved bytes, including the parliament's public JSON."""
    content = (raw / record["path"]).read_bytes()
    identifier = record["id"]
    if content.startswith(b"%PDF"):
        pages = [
            {"physical_page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(PdfReader(io.BytesIO(content)).pages)
        ]
        if write_aux:
            save(raw / "parsed" / f"{identifier}.pages.json", pages)
        return "\n".join(f"PAGE {page['physical_page']}\n{page['text']}" for page in pages)
    if "application/javascript" in record.get("content_type", ""):
        return content.decode("utf-8")
    if "application/json" in record.get("content_type", ""):
        item = json.loads(content)
        parts = [str(item.get(key) or "") for key in ["name", "publishAt", "updatedAt"]]
        for key in ["contentTop", "content", "contentBottom"]:
            parts.append(
                BeautifulSoup(item.get(key) or "", "html.parser").get_text("\n", strip=True)
            )
        return "\n".join(parts)
    soup = BeautifulSoup(content, "html.parser")
    if write_aux:
        save(
            raw / "parsed" / f"{identifier}.links.json",
            [
                {
                    "text": normalized(a.get_text(" ", strip=True)),
                    "url": urljoin(record.get("final_url", record["url"]), a["href"]),
                }
                for a in soup.select("a[href]")
            ],
        )
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text("\n", strip=True)


def fetch_source(item: dict, raw: Path, ca: Path | None) -> dict:
    identifier = item["id"]
    if not re.fullmatch(r"[a-z0-9_]+", identifier):
        raise ValueError("Unsafe source identifier")
    meta_path = raw / "metadata" / f"{identifier}.json"
    if meta_path.exists():
        return load(meta_path)
    record = {**item, "retrieved_at_utc": datetime.now(UTC).isoformat(), "status": "STARTED"}
    context = ssl.create_default_context()
    if ca:
        context.load_verify_locations(cafile=str(ca))
    try:
        with httpx.Client(
            verify=context,
            timeout=45,
            follow_redirects=True,
            headers={"User-Agent": "Municipal research; public-source audit"},
        ) as client:
            response = client.get(item["url"])
        record.update(
            http_status=response.status_code,
            final_url=str(response.url),
            content_type=response.headers.get("content-type", ""),
            http_last_modified=response.headers.get("last-modified", ""),
            response_date=response.headers.get("date", ""),
            redirects=[{"status": r.status_code, "url": str(r.url)} for r in response.history],
        )
        if len(response.content) > 40_000_000:
            raise ValueError("Response exceeds bounded 40 MB source limit")
        extension = ".pdf" if response.content.startswith(b"%PDF") else ".html"
        target = raw / "raw" / f"{identifier}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        record.update(
            path=target.relative_to(raw).as_posix(),
            bytes=target.stat().st_size,
            sha256=sha(target),
            status="downloaded" if response.is_success else "http_error",
        )
        text = extract_snapshot(record, raw, write_aux=True)
        text_path = raw / "parsed" / f"{identifier}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        record.update(text_path=text_path.relative_to(raw).as_posix(), text_sha256=sha(text_path))
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save(meta_path, record)
    return record


def acquire(plan: Path, raw: Path, ca: Path | None) -> None:
    sources = load(plan)["sources"]
    if len({item["id"] for item in sources}) != len(sources):
        raise ValueError("Duplicate source id")
    raw.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(lambda item: fetch_source(item, raw, ca), sources))
    save(raw / "retrievals.json", records)
    print(
        json.dumps(
            {
                "sources": len(records),
                "downloaded": sum(r["status"] == "downloaded" for r in records),
            }
        )
    )


def validate_evidence(row: dict, sources: dict, raw: Path) -> dict:
    source = sources[row["source_id"]]
    if source["status"] != "downloaded":
        raise ValueError("Cannot accept evidence from a failed response")
    if source["sha256"] != row["source_sha256"]:
        raise ValueError("Evidence recipe belongs to another source snapshot")
    if any(
        row[key] is not False
        for key in [
            "model_join_accepted",
            "causal_attribution_confirmed",
            "annual_bdpm_investment_reconciled",
        ]
    ):
        raise ValueError("Documentary context cannot become a model join or causal attribution")
    text = (raw / source["text_path"]).read_text(encoding="utf-8")
    evidence_text = text
    if row.get("physical_page"):
        pages = dict(re.findall(r"(?:^|\n)PAGE (\d+)\n(.*?)(?=\nPAGE \d+\n|\Z)", text, re.S))
        evidence_text = pages.get(str(row["physical_page"]), "")
        if row.get("unit_needle"):
            unit_page = pages.get(str(row["unit_physical_page"]), "")
            if normalized(row["unit_needle"]) not in normalized(unit_page):
                raise ValueError("Unit header not found on the reviewed PDF page")
    if not row["needle"] or normalized(row["needle"]) not in normalized(evidence_text):
        raise ValueError(f"Reviewed text not found: {row['evidence_id']}")
    if row["value_kind"] in {"planned_funding", "contract_cost_increase", "project_total_cost"}:
        if row["annual_bdpm_investment_reconciled"]:
            raise ValueError("Plans or contract estimates are not annual BDPMO observations")
    return {key: value for key, value in row.items() if key != "needle"} | {
        "source_url": source["url"],
        "retrieved_at_utc": source["retrieved_at_utc"],
        "source_primary_role": source["role"],
        "source_text_sha256": source["text_sha256"],
    }


def replay(raw: Path, recipe_path: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new or empty replay output directory")
    recipe = load(recipe_path)
    records = load(raw / "retrievals.json")
    sources = {record["id"]: record for record in records}
    verified = 0
    for record in records:
        for path_key, sha_key in [("path", "sha256"), ("text_path", "text_sha256")]:
            if path_key in record:
                if sha(raw / record[path_key]) != record[sha_key]:
                    raise ValueError(f"Source hash changed: {record['id']}")
                verified += 1
        if "text_path" in record:
            regenerated = extract_snapshot(record, raw)
            expected = (raw / record["text_path"]).read_text(encoding="utf-8")
            if regenerated != expected:
                raise ValueError(f"Raw-to-text replay differs: {record['id']}")
    for visual in recipe.get("visual_checks", []):
        if sha(raw / visual["path"]) != visual["sha256"]:
            raise ValueError("Visual receipt changed")
        if sources[visual["source_id"]]["sha256"] != visual["source_sha256"]:
            raise ValueError("Visual receipt belongs to another PDF")
    if set(recipe["source_decisions"]) != set(sources):
        raise ValueError("Every attempted source needs exactly one review decision")
    rows = [validate_evidence(row, sources, raw) for row in recipe["evidence"]]
    if len({row["evidence_id"] for row in rows}) != len(rows):
        raise ValueError("Evidence identifiers must be unique")
    out.mkdir(parents=True, exist_ok=True)
    csv_save(out / "evidence_ledger.csv", rows)
    csv_save(
        out / "source_status.csv",
        [
            {
                "source_id": r["id"],
                "url": r["url"],
                "role": r["role"],
                "status": r["status"],
                "http_status": r.get("http_status", ""),
                "bytes": r.get("bytes", ""),
                "sha256": r.get("sha256", ""),
                "retrieved_at_utc": r["retrieved_at_utc"],
                "error": r.get("error", ""),
                "review_status": recipe["source_decisions"][r["id"]],
            }
            for r in records
        ],
    )
    csv_save(out / "hypothesis_status.csv", recipe["hypotheses"])
    csv_save(out / "followup_requests.csv", recipe["followup_requests"])
    save(out / "source_inventory.json", records)
    summary = {
        "status": "PASS_WITH_CONDITIONS",
        "sources_attempted": len(records),
        "sources_downloaded": sum(r["status"] == "downloaded" for r in records),
        "source_and_text_hashes_verified": verified,
        "evidence_rows": len(rows),
        "raw_to_text_replays_verified": sum("text_path" in r for r in records),
        "visual_receipts_verified": len(recipe.get("visual_checks", [])),
        "accepted_municipal_numeric_cells": 0,
        "causal_attributions": 0,
        "models_fit": 0,
        "previous_sources_changed": 0,
        "new_yearbook_transcriptions": 0,
        "network_requests_in_replay": 0,
    }
    save(out / "QA.json", summary)
    save(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "recipe_sha256": sha(recipe_path),
            "retrievals_sha256": sha(raw / "retrievals.json"),
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--derive-only", action="store_true")
    args = parser.parse_args()
    if not args.derive_only:
        if args.plan is None:
            parser.error("--plan is required for acquisition")
        acquire(args.plan, args.raw_root, args.ca_bundle)
    if args.recipe:
        if args.out is None:
            parser.error("--out is required for evidence replay")
        print(json.dumps(replay(args.raw_root, args.recipe, args.out)))


if __name__ == "__main__":
    main()
