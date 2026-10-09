"""Download observed official regional publications with native TLS verification."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

PAGES = {
    "saratov_retail": "https://64.rosstat.gov.ru/roznica",
    "stavropol_municipal": "https://26.rosstat.gov.ru/main_indicators",
}
HOSTS = {"64.rosstat.gov.ru", "26.rosstat.gov.ru"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in HOSTS:
        raise ValueError("Unreviewed official host")
    if path.exists():
        raise FileExistsError(path)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required")
    # No redirect following: a new destination requires explicit review.
    parts = urlsplit(url)
    request_url = urlunsplit(
        (parts.scheme, parts.netloc, quote(parts.path, safe="/%()"), parts.query, "")
    )
    command = [
        curl,
        "--silent",
        "--show-error",
        "--fail",
        "--max-time",
        "60",
        "--proto",
        "=https",
        "--max-filesize",
        "30000000",
        "--output",
        str(path),
        "--write-out",
        "%{http_code}\t%{content_type}",
        request_url,
    ]
    started = datetime.now(UTC).isoformat()
    result = subprocess.run(command, capture_output=True, timeout=70, check=False)
    fields = result.stdout.decode("utf-8", errors="replace").strip().split("\t", 1)
    record = {
        "url": url,
        "request_url": request_url,
        "retrieved_at": started,
        "curl_returncode": result.returncode,
        "http_status": int(fields[0]) if fields[0].isdigit() else None,
        "content_type": fields[1] if len(fields) > 1 else None,
        "tls_verification": True,
        "redirects_followed": False,
        "file": path.name if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha(path) if path.exists() else None,
        "error": None if result.returncode == 0 else "curl_failed",
    }
    path.with_name(path.name + ".retrieval.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def links(path, url):
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    result = []
    for anchor in soup.find_all("a", href=True):
        target = urljoin(url, anchor["href"])
        if urlsplit(target).hostname not in HOSTS:
            continue
        context = anchor.parent.get_text(" ", strip=True)
        result.append({"url": target, "text": anchor.get_text(" ", strip=True), "context": context})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--name")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    urls = {args.name: args.url} if args.url and args.name else PAGES
    records, anchors = [], []
    for name, url in urls.items():
        filename = name if "." in name else name + ".html"
        record = download(url, args.out / filename)
        records.append(record)
        if record["http_status"] == 200 and "html" in (record["content_type"] or ""):
            anchors.extend(links(args.out / filename, url))
    for name, data in [("retrievals.json", records), ("links.json", anchors)]:
        (args.out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "retrievals": len(records),
                "downloaded": sum(r["http_status"] == 200 for r in records),
                "links": len(anchors),
            }
        )
    )


if __name__ == "__main__":
    main()
