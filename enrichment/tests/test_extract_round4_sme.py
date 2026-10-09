import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "extract_sme", Path(__file__).parents[1] / "scripts/extract_round4_sme.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root, *, member="entire/example.parquet", extra=False):
    (root / "raw").mkdir()
    archive = root / "raw/snapshot.zip"
    data = b"PAR1 frozen analytical fixture PAR1"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, data)
        if extra:
            z.writestr("unlisted.parquet", b"extra")
        entry = z.infolist()[0]
    record = {
        "year_in_filename": 2023,
        "retrieved_at": "2026-09-16T17:33:46Z",
        "path": "raw/snapshot.zip",
        "bytes": archive.stat().st_size,
        "sha256": MODULE.sha256(archive),
        "entries": [
            {
                "path": "extracted/2023/" + member,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "zip_crc32": entry.CRC,
            }
        ],
    }
    (root / "retrievals.json").write_text(json.dumps([record]), encoding="utf-8")
    return record, data


def test_creates_missing_and_preserves_metadata_and_existing_file(tmp_path):
    record, data = fixture(tmp_path)
    metadata = tmp_path / "retrievals.json"
    before = (metadata.read_bytes(), metadata.stat().st_mtime_ns)
    first = MODULE.extract(tmp_path)
    output = tmp_path / record["entries"][0]["path"]
    assert output.read_bytes() == data
    assert first["created"] == [record["entries"][0]["path"]]
    output_time = output.stat().st_mtime_ns
    second = MODULE.extract(tmp_path)
    assert second["created"] == []
    assert output.stat().st_mtime_ns == output_time
    assert before == (metadata.read_bytes(), metadata.stat().st_mtime_ns)
    assert list((tmp_path / "extracted").rglob("*.partial")) == []


@pytest.mark.parametrize(
    "member", ["../escape.parquet", "/escape.parquet", "C:/evil.parquet", "a\\evil.parquet"]
)
def test_path_traversal_rejected_before_write(tmp_path, member):
    fixture(tmp_path, member=member)
    with pytest.raises(ValueError, match="Unsafe relative path"):
        MODULE.extract(tmp_path)
    assert not (tmp_path / "extracted").exists()


def test_unlisted_and_missing_members_rejected(tmp_path):
    record, _ = fixture(tmp_path, extra=True)
    with pytest.raises(ValueError, match="Unlisted or missing"):
        MODULE.extract(tmp_path)
    record["entries"][0]["path"] = "extracted/2023/missing.parquet"
    (tmp_path / "retrievals.json").write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unlisted or missing"):
        MODULE.extract(tmp_path)
    assert not (tmp_path / "extracted").exists()


def test_unequal_existing_file_never_overwritten(tmp_path):
    record, _ = fixture(tmp_path)
    output = tmp_path / record["entries"][0]["path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"different existing data")
    with pytest.raises(ValueError, match="SHA/size mismatch"):
        MODULE.extract(tmp_path)
    assert output.read_bytes() == b"different existing data"


def test_member_hash_mismatch_leaves_no_final_or_partial(tmp_path):
    record, _ = fixture(tmp_path)
    record["entries"][0]["sha256"] = "0" * 64
    metadata = tmp_path / "retrievals.json"
    metadata.write_text(json.dumps([record]), encoding="utf-8")
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="Streamed member CRC/SHA/size"):
        MODULE.extract(tmp_path)
    assert metadata.read_bytes() == before
    assert not (tmp_path / record["entries"][0]["path"]).exists()
    assert list((tmp_path / "extracted").rglob("*.partial")) == []


def test_archive_hash_mismatch_rejected_before_write(tmp_path):
    record, _ = fixture(tmp_path)
    archive = tmp_path / record["path"]
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA/size mismatch"):
        MODULE.extract(tmp_path)
    assert not (tmp_path / "extracted").exists()
