"""Extract the two pinned Stavropol archives to generated filenames, never archive paths."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def extract(raw, out):
    raw, out = raw.resolve(), out.resolve()
    if out.exists():
        raise ValueError("Choose a new output directory")
    specs = {
        "annual_2023.zip": "22b1ad11c82352235f3a050729170ad103f30ff58f1bd3f4036484783247008a",
        "annual_2024.rar": "9b02352e7668b8e9c48022f0d63211fcde067ca3e73d6c26187ae3a22fea460d",
    }
    for filename, expected in specs.items():
        path = raw / "stavropol_annual" / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive changed: {filename}")
    out.mkdir(parents=True)
    records = []

    def save(year, index, original, data, extension):
        destination = out / str(year) / f"{index:02d}{extension}"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
        records.append(
            dict(
                year=year,
                member=original,
                path=f"extracted/{year}/{index:02d}{extension}",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    with zipfile.ZipFile(raw / "stavropol_annual/annual_2023.zip") as archive:
        members = archive.infolist()
        if len(members) != 33:
            raise ValueError("Expected 33 PDF members")
        for index, member in enumerate(members):
            if (
                member.is_dir()
                or not member.filename.lower().endswith(".pdf")
                or member.file_size >= 1_000_000
            ):
                raise ValueError("Unexpected ZIP member")
            data = archive.read(member)
            if not data.startswith(b"%PDF"):
                raise ValueError("Not a PDF")
            save(2023, index, member.filename, data, ".pdf")
    rar = raw / "stavropol_annual/annual_2024.rar"
    listing = subprocess.check_output(["tar", "-tf", str(rar)], timeout=60)
    try:
        names = listing.decode("utf-8")
    except UnicodeDecodeError:
        names = listing.decode("cp1251")
    listed = names.splitlines()
    members = [name for name in listed if name.lower().endswith(".doc")]
    parents = {name.rsplit("/", 1)[0] for name in members}
    extras = {name.rstrip("/") for name in listed if name not in members}
    if len(members) != 33 or extras != parents:
        raise ValueError("Expected 33 DOC members")
    for index, member in enumerate(members):
        data = subprocess.check_output(["tar", "-xOf", str(rar), "--", member], timeout=60)
        if not data.startswith(b"\xd0\xcf\x11\xe0") or len(data) >= 3_000_000:
            raise ValueError("Unexpected DOC member")
        save(2024, index, member, data, ".doc")
    (out / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"extracted": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extract(args.raw, args.out)
