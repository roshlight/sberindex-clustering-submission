"""Download an explicit official-source inventory; cache immutable bytes and extract text.

No discovery, fuzzy joins, retries or analytical/model calls are performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import ssl
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from zipfile import ZipFile

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(data: bytes, path: Path, content_type: str) -> dict:
    """Preserve PDF physical page numbers; do not pretend text has visual fidelity."""
    if data.startswith(b"%PDF"):
        pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]
        return {"format": "pdf", "pages": pages, "text": "\n\f\n".join(pages)}
    if data.startswith(b"PK") and path.suffix.lower() == ".docx":
        with ZipFile(io.BytesIO(data)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        texts = [node.text or "" for node in root.iter() if node.tag.endswith("}t")]
        return {"format": "docx", "text": "\n".join(texts)}
    if "html" in content_type or b"<html" in data[:500].lower():
        soup = BeautifulSoup(data, "html.parser")
        links = [
            {"label": a.get_text(" ", strip=True), "href": a.get("href")}
            for a in soup.find_all("a", href=True)
        ]
        for node in soup(["script", "style"]):
            node.decompose()
        return {"format": "html", "text": soup.get_text("\n", strip=True), "links": links}
    return {"format": "unparsed_binary", "text": ""}


def collect(spec_path: Path, raw_dir: Path, ca_bundle: Path | None, interval: float) -> list:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    if ca_bundle:
        context.load_verify_locations(cafile=ca_bundle)
    records = []
    with httpx.Client(verify=context, timeout=60, follow_redirects=False) as client:
        for spec in specs:
            stem = spec["source_id"]
            meta_path = raw_dir / f"{stem}.meta.json"
            if meta_path.exists():
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                if old["url"] != spec["url"]:
                    raise ValueError(f"Immutable source ID reused with another URL: {stem}")
                if (
                    old.get("raw_file")
                    and sha256((raw_dir / old["raw_file"]).read_bytes()) != old["sha256"]
                ):
                    raise ValueError(f"Cached bytes changed: {stem}")
                records.append(old)
                continue
            url = spec["url"]
            allowed = set(spec.get("allowed_redirect_hosts", [])) | {urlsplit(url).hostname}
            record = {**spec, "retrieved_at_utc": datetime.now(UTC).isoformat(), "attempts": 1}
            try:
                chain = []
                for _ in range(6):
                    response = client.get(url)
                    chain.append({"url": str(response.url), "status": response.status_code})
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        break
                    url = urljoin(str(response.url), response.headers["location"])
                    if urlsplit(url).hostname not in allowed:
                        raise ValueError(f"Unapproved redirect hostname: {urlsplit(url).hostname}")
                else:
                    raise ValueError("Redirect limit reached")
                suffix = spec.get("suffix", Path(urlsplit(url).path).suffix or ".html")
                raw_path = raw_dir / f"{stem}{suffix}"
                data = response.content
                raw_path.write_bytes(data)
                record.update(
                    status=response.status_code,
                    final_url=str(response.url),
                    redirect_chain=chain,
                    content_type=response.headers.get("content-type", ""),
                    bytes=len(data),
                    sha256=sha256(data),
                    raw_file=raw_path.name,
                    download_status="downloaded" if response.status_code == 200 else "http_error",
                )
                if response.status_code == 200:
                    try:
                        parsed = extract(data, raw_path, record["content_type"])
                        parsed_path = raw_dir / f"{stem}.extracted.json"
                        parsed_path.write_text(
                            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        (raw_dir / f"{stem}.txt").write_text(parsed["text"], encoding="utf-8")
                        record["extraction_format"] = parsed["format"]
                        record["extraction_sha256"] = sha256(parsed_path.read_bytes())
                    except Exception as error:
                        record["extraction_error"] = f"{type(error).__name__}: {error}"
            except Exception as error:
                record.update(
                    download_status="request_error", error=f"{type(error).__name__}: {error}"
                )
            meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(record)
            print(stem, record["download_status"], record.get("bytes", 0), flush=True)
            time.sleep(interval)
    (raw_dir / "download_inventory.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def finalize(raw_dir: Path, recipe_path: Path, output: Path) -> None:
    """Replay human-reviewed, fixed document locators; fail closed on changed bytes."""
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    metadata = {
        p.stem.removesuffix(".meta"): json.loads(p.read_text(encoding="utf-8"))
        for p in raw_dir.glob("*.meta.json")
    }
    rows = []
    for rule in recipe:
        meta = metadata[rule["source_id"]]
        if meta.get("download_status") != "downloaded":
            raise ValueError(f"Unavailable evidence source {rule['source_id']}")
        source = raw_dir / meta["raw_file"]
        if sha256(source.read_bytes()) != meta["sha256"]:
            raise ValueError(f"Changed raw source {source.name}")
        member_sha = ""
        if rule.get("source_member"):
            member = raw_dir / rule["source_member"]
            member_sha = sha256(member.read_bytes())
            if member_sha != rule["member_sha256"]:
                raise ValueError("Changed archive member")
            with ZipFile(member) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            if rule.get("table"):
                table = root.findall(".//w:tbl", ns)[rule["table"] - 1]
                row = table.findall("./w:tr", ns)[rule["row"] - 1]
                cell = row.findall("./w:tc", ns)[rule["column"] - 1]
                locator_text = "".join(x.text or "" for x in cell.findall(".//w:t", ns))
                if normalized(locator_text) != normalized(rule["value_text"]):
                    raise ValueError(f"Unexpected exact table cell: {rule['evidence_id']}")
            else:
                para = root.findall("./w:body/w:p", ns)[rule["paragraph"] - 1]
                locator_text = "".join(x.text or "" for x in para.findall(".//w:t", ns))
        else:
            parsed_path = raw_dir / f"{rule['source_id']}.extracted.json"
            if sha256(parsed_path.read_bytes()) != meta["extraction_sha256"]:
                raise ValueError(f"Changed extracted text for {rule['source_id']}")
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            locator_text = parsed["pages"][rule["page"] - 1]
        for token in rule.get("must_contain", []):
            if normalized(token) not in normalized(locator_text):
                raise ValueError(f"Evidence token absent for {rule['evidence_id']}: {token}")
        row = {k: v for k, v in rule.items() if k not in {"must_contain", "member_sha256"}}
        row.update(
            source_url=meta["url"],
            source_sha256=meta["sha256"],
            source_member_sha256=member_sha,
            retrieved_at_utc=meta["retrieved_at_utc"],
            locator_check="pass",
        )
        rows.append(row)
    if len({r["evidence_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evidence ID")
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (output / "factual_evidence.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "source_inventory.json").write_text(
        json.dumps(list(metadata.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    file_catalog = [
        {
            "path": p.relative_to(raw_dir).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p.read_bytes()),
        }
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
    ]
    (output / "raw_inventory.json").write_text(
        json.dumps(file_catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "evidence_rows": len(rows),
        "locator_checks_passed": len(rows),
        "unique_cases": len({r["territory_id"] for r in rows}),
        "source_requests": len(metadata),
        "downloaded_sources": sum(
            m.get("download_status") == "downloaded" for m in metadata.values()
        ),
        "raw_files": len(file_catalog),
        "raw_bytes": sum(r["bytes"] for r in file_catalog),
        "automatic_join_or_model_update": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--evidence-recipe", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be nonnegative")
    if args.finalize:
        if not args.evidence_recipe or not args.output:
            parser.error("--finalize requires --evidence-recipe and --output")
        finalize(args.raw_dir, args.evidence_recipe, args.output)
    else:
        if not args.sources:
            parser.error("--sources is required for collection")
        collect(args.sources, args.raw_dir, args.ca_bundle, args.interval)


if __name__ == "__main__":
    main()
