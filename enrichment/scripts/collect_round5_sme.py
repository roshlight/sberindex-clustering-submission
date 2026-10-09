"""Retrieve bounded official FNS SME documentation and historical archive metadata."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

PAGES = {
    "fns_passport": "https://www.nalog.gov.ru/opendata/7707329152-rsmp/",
    "fns_developers": "https://rmsp.nalog.ru/developers.html",
    "fns_statistics": "https://rmsp.nalog.ru/statistics.html",
    "fns_about": "https://rmsp.nalog.ru/about.html",
    "fns_workers_passport": "https://www.nalog.gov.ru/opendata/7707329152-sshr2019/",
    "pypi_generator": "https://pypi.org/project/russian-sme-registry/",
}

# Primary source URLs observed in web search, including a now-unavailable release.
FOLLOWUPS = [
    (
        "workers_2024_release",
        "https://file.nalog.ru/opendata/7707329152-sshr2019/data-20251225-structure-20200408.zip",
        "HEAD",
    ),
    (
        "rudnya_municipal_2024",
        "https://xn--d1ashm6d.xn--p1ai/files/458/analiticheskaya-info2.pdf",
        "GET",
    ),
    (
        "author_article_2024",
        "https://press.psu.ru/index.php/geogr/article/download/9926/6220/22969",
        "GET",
    ),
]


def clean_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in {"set-cookie", "cookie"}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(client, directory, name, url, *, method="GET", cap=25_000_000):
    metadata_path = directory / (name + ".http.json")
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        old["headers"] = clean_headers(old.get("headers", {}))
        metadata_path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
        return old
    record = dict(url=url, method=method, retrieved_at=datetime.now(UTC).isoformat())
    try:
        with client.stream(method, url) as response:
            record.update(
                status=response.status_code,
                final_url=str(response.url),
                headers=clean_headers(dict(response.headers)),
            )
            response.raise_for_status()
            content = b""
            if method == "GET":
                for chunk in response.iter_bytes():
                    content += chunk
                    if len(content) > cap:
                        raise ValueError("Download size cap exceeded; no complete body saved")
                suffix = Path(urlsplit(str(response.url)).path).suffix or ".html"
                path = directory / (name + suffix)
                path.write_bytes(content)
                record.update(path=path.name, bytes=len(content), sha256=digest(content))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    directory = args.raw.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records, links = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name, url in PAGES.items():
            record = fetch(client, directory, name, url)
            records.append(record)
            if "path" not in record:
                continue
            soup = BeautifulSoup((directory / record["path"]).read_bytes(), "html.parser")
            for element in soup.select("a[href], script[src]"):
                href = urljoin(record["final_url"], element.get("href", element.get("src")))
                links.append(dict(page=name, url=href, label=element.get_text(" ", strip=True)))
        for year in [2023, 2024]:
            for month in ["01", "06", "07", "12"]:
                match = [
                    item
                    for item in links
                    if item["page"] == "fns_passport" and f"data-10{month}{year}-" in item["url"]
                ]
                if len(match) != 1:
                    raise ValueError(f"Expected one observed archive URL for {year}-{month}")
                records.append(
                    fetch(
                        client, directory, f"archive_{year}_{month}", match[0]["url"], method="HEAD"
                    )
                )
        schemas = {
            item["url"]
            for item in links
            if item["url"].endswith(("structure-10062023.xsd", "structure-10032022.xsd"))
        }
        docs = {
            item["url"]
            for item in links
            if item["page"] == "fns_passport" and "Описание формата" in item["label"]
        }
        for url in sorted(schemas | docs):
            records.append(fetch(client, directory, Path(urlsplit(url).path).stem, url))
        for item in links:
            if item["page"] == "fns_statistics" and Path(urlsplit(item["url"]).path).name in {
                "statistics.js",
                "main.js",
            }:
                name = Path(urlsplit(item["url"]).path).stem
                records.append(fetch(client, directory, name, item["url"]))
        for item in links:
            if item["page"] == "fns_workers_passport" and item["url"].endswith(".xsd"):
                records.append(fetch(client, directory, "workers_structure", item["url"]))
            if item["page"] == "pypi_generator" and item["url"].endswith(".tar.gz"):
                records.append(fetch(client, directory, "generator_source", item["url"]))
        for year in [2023, 2024]:
            records.append(
                fetch(
                    client,
                    directory,
                    f"statistics_{year}",
                    f"https://rmsp.nalog.ru/statistics.xlsx?statDate=10.12.{year}",
                )
            )
        for name, url, method in FOLLOWUPS:
            records.append(fetch(client, directory, name, url, method=method))
    (directory / "discovered_links.json").write_text(
        json.dumps(links, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "retrievals.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "requests": len(records),
                "downloaded": sum("path" in r for r in records),
                "errors": [{"url": r["url"], "error": r["error"]} for r in records if "error" in r],
            }
        )
    )


if __name__ == "__main__":
    main()
