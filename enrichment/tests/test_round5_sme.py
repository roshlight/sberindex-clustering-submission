"""Verify bounded downloads, tamper detection, and temporal audit guards."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module("audit_round5_sme")
sample = module("sample_round5_sme_archives")


def test_register_year_boundary_does_not_transform_tochno():
    assert audit.register_worker_reference_year(2024, 6) == 2022
    assert audit.register_worker_reference_year(2024, 7) == 2023
    with pytest.raises(ValueError):
        audit.register_worker_reference_year(2024, 0)


def test_ignored_range_is_rejected(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"abcd")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reader = sample.RangeReader(
            client,
            tmp_path,
            dict(url="https://example.test/data", headers={"content-length": "100"}),
        )
        with pytest.raises(ValueError, match="honor exact Range"):
            reader.read(4)
        assert not list(tmp_path.glob("*.bin"))


def test_range_cap_applies_before_network(tmp_path):
    reader = sample.RangeReader(
        None, tmp_path, dict(url="unused", headers={"content-length": "3000000000"})
    )
    with pytest.raises(ValueError, match="Bounded sample"):
        reader.read()


def test_cached_range_tamper_detection(tmp_path):
    path = tmp_path / "range_0_3.bin"
    path.write_bytes(b"evil")
    path.with_suffix(".json").write_text(json.dumps(dict(sha256=sample.digest(b"good"))))
    reader = sample.RangeReader(
        None, tmp_path, dict(url="unused", headers={"content-length": "4"}), offline=True
    )
    with pytest.raises(ValueError, match="Cached range changed"):
        reader.read(4)


def test_xml_snapshot_mismatch_rejected(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text('<Файл КолДок="1"><Документ ДатаСост="10.07.2024"/></Файл>', encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot date"):
        audit.audit_xml(path, "10.06.2024")
