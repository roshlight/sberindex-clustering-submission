"""Collect a bounded SME lineage bundle; never run the downloaded author's code."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from collect_round5_sme import digest, fetch
from sample_round5_sme_archives import RangeReader

COMMIT = "48a92fa1af335e930f5fb040a7d31a1bf30a44cc"
PACKAGE_SHA = "3c3f41882efb08a9f7be7bcd12fdecd59220e07ce5a68479ee072d92bd2e1421"
BASE = f"https://raw.githubusercontent.com/PavelSyomin/russian-sme-registry/{COMMIT}/"
WORKERS_URL = (
    "https://file.nalog.ru/opendata/7707329152-sshr2019/data-20260825-structure-20200408.zip"
)
SOURCES = [
    ("workers_passport", "https://www.nalog.gov.ru/opendata/7707329152-sshr2019/", "GET"),
    ("workers_current_head", WORKERS_URL, "HEAD"),
    (
        "fns_workers_schedule2023",
        "https://www.nalog.gov.ru/rn77/news/activities_fts/13327467/",
        "GET",
    ),
    ("generator_readme", BASE + "README.md", "GET"),
    ("generator_geocode", BASE + "russian_sme_registry/stages/geocode.py", "GET"),
    ("generator_paper", BASE + "paper/paper.md", "GET"),
    (
        "generator_tree",
        f"https://api.github.com/repos/PavelSyomin/russian-sme-registry/git/trees/{COMMIT}"
        "?recursive=1",
        "GET",
    ),
    (
        "generator_commits",
        "https://api.github.com/repos/PavelSyomin/russian-sme-registry/commits?per_page=10",
        "GET",
    ),
    (
        "author_article_https",
        "https://press.psu.ru/index.php/geogr/article/download/9926/6220/22969",
        "GET",
    ),
]
for _date in ("20240401", "20241225", "20250401"):
    SOURCES.append(
        (
            "workers_candidate" + _date,
            "https://file.nalog.ru/opendata/7707329152-sshr2019/"
            f"data-{_date}-structure-20200408.zip",
            "HEAD",
        )
    )

MEMBERS = [
    "LICENSE",
    "README.md",
    "russian_sme_registry/stages/geocode.py",
    "russian_sme_registry/stages/aggregate.py",
    "russian_sme_registry/stages/panelize.py",
    "russian_sme_registry/utils/elements.py",
    "russian_sme_registry/assets/cities.csv",
    "russian_sme_registry/assets/cities_additional.csv",
    "russian_sme_registry/assets/settlements.csv",
    "russian_sme_registry/assets/regions.csv",
    "russian_sme_registry/assets/abbr.csv",
    "russian_sme_registry/tests/data/empl/empl.html",
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_author_sources(raw, prior):
    """Copy allowlisted bytes; SHA guard the prior package and verify Git blob lineage."""
    package = prior / "generator_source.gz"
    if digest(package.read_bytes()) != PACKAGE_SHA:
        raise ValueError("Prior author package SHA differs")
    package_meta = json.loads((prior / "generator_source.http.json").read_text(encoding="utf-8"))
    if package_meta["sha256"] != PACKAGE_SHA:
        raise ValueError("Prior package HTTP metadata SHA differs")
    prior_meta = raw / "prior_source_metadata"
    prior_meta.mkdir(exist_ok=True)
    for name in ("generator_source.http.json", "workers_structure.http.json"):
        data = (prior / name).read_bytes()
        target = prior_meta / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Existing prior HTTP metadata changed")
        target.write_bytes(data)
    tree = json.loads((raw / "generator_tree.html").read_text(encoding="utf-8"))
    if tree["sha"] != COMMIT or tree.get("truncated"):
        raise ValueError("Incomplete or unexpected Git tree")
    blobs = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    rows = []
    with tarfile.open(package) as archive:
        for member in MEMBERS:
            data = archive.extractfile("russian_sme_registry-1.0.0b10/" + member).read()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != blobs[member]:
                raise ValueError("Package member does not match pinned Git tree: " + member)
            target = raw / "author_source" / member
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Existing extracted source changed")
            if not target.exists():
                target.write_bytes(data)
            rows.append(
                dict(
                    path=target.relative_to(raw).as_posix(),
                    bytes=len(data),
                    sha256=digest(data),
                    git_blob_sha1=blob,
                    git_commit=COMMIT,
                    git_url=BASE + member,
                    transport="extracted from prior verified PyPI sdist; Git tree blob verified",
                    prior_package_sha256=PACKAGE_SHA,
                    prior_package_url=package_meta["url"],
                    prior_package_retrieved_at=package_meta["retrieved_at"],
                    tochno_exact_build_link_proven=False,
                )
            )
    # XSD is included to keep offline auditing self-contained, not counted as a new download.
    schema = prior / "workers_structure.xsd"
    data = schema.read_bytes()
    if digest(data) != "3c63a9447e0c0dbacaf9fe11f1888c5873137f9252f226c148303dd6dfa9d137":
        raise ValueError("Prior workers XSD changed")
    target = raw / "workers_structure.xsd"
    if target.exists() and target.read_bytes() != data:
        raise ValueError("Existing schema changed")
    target.write_bytes(data)
    write_json(raw / "author_source_lineage.json", rows)


class TimedRangeReader(RangeReader):
    """Retain the previous bounded reader and add timing to newly fetched ranges only."""

    def read(self, size=-1):
        start = self.position
        actual = min(self.length - start, size if size >= 0 else self.length)
        metadata = self.directory / f"range_{start}_{start + actual - 1}.json"
        existing = metadata.exists()
        when = datetime.now(UTC).isoformat()
        data = super().read(size)
        if not existing and metadata.exists():
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["retrieved_at"] = when
            write_json(metadata, value)
        return data


def sample_workers(raw, client, offline):
    source = json.loads((raw / "workers_current_head.http.json").read_text(encoding="utf-8"))
    if source["url"] != WORKERS_URL or source.get("status") != 200:
        raise ValueError("Unexpected current workers archive")
    directory = raw / "workers_current"
    reader = TimedRangeReader(client, directory, source, offline)
    with zipfile.ZipFile(reader) as archive:
        members = archive.infolist()
        chosen = min(
            (m for m in members if m.filename.endswith(".xml")),
            key=lambda m: (m.compress_size, m.filename),
        )
        if chosen.file_size > 50_000_000:
            raise ValueError("Uncompressed sample cap exceeded")
        data = archive.read(chosen)  # Full selected member CRC is checked by zipfile.
        target = directory / "sample.xml"
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Selected worker sample changed")
        target.write_bytes(data)
    result = dict(
        archive_url=source["url"],
        archive_bytes=reader.length,
        archive_member_count=len(members),
        selected_member=chosen.filename,
        sample_bytes=len(data),
        sample_sha256=digest(data),
        selected_member_crc32=f"{chosen.CRC:08x}",
        sample_crc32_verified=True,
        range_bytes_read=reader.read_bytes,
        representative_sample=False,
        selection="smallest compressed XML; no inference about entire archive",
        whole_archive_hash_verified=False,
        archive_sha256_server_declared=source["headers"].get("x-amz-meta-sha256"),
    )
    write_json(raw / "workers_sample_provenance.json", result)
    return result


def freeze_inventory(raw):
    entries = []
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name != "SOURCE_MANIFEST.json":
            data = path.read_bytes()
            entries.append(
                dict(path=path.relative_to(raw).as_posix(), bytes=len(data), sha256=digest(data))
            )
    write_json(raw / "SOURCE_MANIFEST.json", {"schema_version": 1, "files": entries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=40, follow_redirects=True) as client:
        for name, url, method in SOURCES:
            cached = args.raw / (name + ".http.json")
            if cached.exists():
                record = json.loads(cached.read_text(encoding="utf-8"))
                if record["url"] != url or record["method"] != method:
                    raise ValueError("Cached metadata identifies another source")
                if "path" in record:
                    if digest((args.raw / record["path"]).read_bytes()) != record["sha256"]:
                        raise ValueError("Cached HTTP body changed")
            elif args.offline:
                raise FileNotFoundError(cached)
            else:
                fetch(client, args.raw, name, url, method=method)
        extract_author_sources(args.raw, args.prior_raw)
        result = sample_workers(args.raw, client, args.offline)
    freeze_inventory(args.raw)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
