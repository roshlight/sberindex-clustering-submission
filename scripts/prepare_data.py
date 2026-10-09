"""Copy only the audited consumption inputs from a local source research folder."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

LANDING_URL = (
    "https://sberindex.ru/ru/research/"
    "data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim"
)
ARCHIVE_URL = (
    "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
)
PORTAL_URL = (
    "https://sberindex.ru/ru/dashboards/"
    "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def prepare_data(source_dir: Path, output_dir: Path) -> dict:
    """Copy unchanged input bytes and write a deterministic provenance manifest.

    Existing identical outputs are retained. Conflicting outputs raise before any
    file is written. The original archive may be expanded or still in ZIP form.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    consumption = source_dir / "hackathon-bundle" / "hackathonlicence" / "consumption.parquet"
    archive = source_dir / "hackathonlicence.zip"
    if consumption.is_file():
        content = consumption.read_bytes()
    else:
        with zipfile.ZipFile(archive) as bundle:
            members = [
                name
                for name in bundle.namelist()
                if not name.endswith("/") and Path(name).name == "consumption.parquet"
            ]
            if len(members) != 1:
                raise ValueError("Expected exactly one consumption.parquet in the archive")
            content = bundle.read(members[0])
    payloads = {
        "consumption.parquet": content,
        "municipality-codebook.csv": (source_dir / "municipality-codebook.csv").read_bytes(),
        "dataset_metadata.json": (
            source_dir / "municipal-spending-metadata-decoded.json"
        ).read_bytes(),
        "source_audit.json": (source_dir / "bundle-data-audit.json").read_bytes(),
    }
    audit = json.loads(payloads["source_audit.json"].decode("utf-8-sig"))
    expected_hashes = {
        record["sha256"]
        for record in audit["file_hashes"]
        if record["file"].replace("\\", "/").endswith("/consumption.parquet")
    }
    if expected_hashes != {_sha256(content)}:
        raise ValueError("consumption.parquet SHA-256 does not match the source audit")
    source_urls = {
        "consumption.parquet": [LANDING_URL, ARCHIVE_URL],
        "municipality-codebook.csv": [PORTAL_URL],
        "dataset_metadata.json": [PORTAL_URL],
        "source_audit.json": [LANDING_URL, ARCHIVE_URL, PORTAL_URL],
    }
    manifest = {
        "manifest_version": 1,
        "source_retrieval_date": audit["retrieval_date"],
        "data_license": "CC BY-SA 4.0",
        "license_scope": "Official historical consumption dataset; see source landing page",
        "codebook_provenance": "Saved display-name mapping from the official portal metadata",
        "audit_provenance": "Local read-only source audit; not an organizer certification",
        "files": [
            {
                "file": name,
                "bytes": len(body),
                "sha256": _sha256(body),
                "source_urls": source_urls[name],
            }
            for name, body in payloads.items()
        ],
    }
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    for name, body in payloads.items():
        target = output_dir / name
        if target.exists() and (not target.is_file() or target.read_bytes() != body):
            raise FileExistsError(f"Refusing to overwrite a different existing file: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, body in payloads.items():
        target = output_dir / name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(body)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    try:
        manifest = prepare_data(args.source_dir, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "copied_input_files": len(manifest["files"])},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
