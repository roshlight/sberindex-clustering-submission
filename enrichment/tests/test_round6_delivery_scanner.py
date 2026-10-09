"""Only the reviewed file, path and JPEG-table span may use the new exception."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "round6_packager", Path(__file__).parents[1] / "scripts/package_round6_delivery.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
ALLOWED = "raw6/cases/raw/ach_rtm_report.pdf"


@pytest.fixture
def sample(monkeypatch):
    symbols = b"123456789:" + b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    counts = bytes([0] * 7 + [len(symbols)] + [0] * 8)
    payload = b"\x11" + counts + symbols
    prefix = b"synthetic-fixture\n"
    jpeg = b"\xff\xd8\xff\xc4" + (len(payload) + 2).to_bytes(2, "big") + payload + b"\xff\xd9"
    body = prefix + jpeg
    match = module.SECRETS.search(body)
    exception = {
        "file_bytes": len(body),
        "file_sha256": module.sha_bytes(body),
        "offset": match.start(),
        "length": len(match.group()),
        "match_sha256": module.sha_bytes(match.group()),
        "jpeg_offset": len(prefix),
        "jpeg_bytes": len(jpeg),
        "jpeg_sha256": module.sha_bytes(jpeg),
        "dht": module.containing_dht(body, len(prefix), match.start(), match.end()),
    }
    monkeypatch.setattr(module, "AUDITED_EXCEPTION", exception)
    return body, exception


def test_exact_path_and_source_are_required(sample):
    body, _ = sample
    assert len(module.scan_bytes(body, ALLOWED)) == 1
    for path, content in [("raw6/cases/raw/other.pdf", body), (ALLOWED, b"X" + body[1:])]:
        with pytest.raises(ValueError, match="Credential-like data"):
            module.scan_bytes(content, path)


@pytest.mark.parametrize("field", ["file_sha256", "match_sha256", "offset", "length"])
def test_each_exact_coordinate_is_required(sample, monkeypatch, field):
    body, exception = sample
    value = "0" * 64 if "sha256" in field else 0
    monkeypatch.setattr(module, "AUDITED_EXCEPTION", {**exception, field: value})
    with pytest.raises(ValueError):
        module.scan_bytes(body, ALLOWED)


def test_additional_match_stays_blocked(sample, monkeypatch):
    body, exception = sample
    changed = body + b"\nsk-or-v1-" + b"B" * 30
    monkeypatch.setattr(
        module,
        "AUDITED_EXCEPTION",
        {
            **exception,
            "file_bytes": len(changed),
            "file_sha256": module.sha_bytes(changed),
        },
    )
    with pytest.raises(ValueError):
        module.scan_bytes(changed, ALLOWED)


def test_matching_hashes_cannot_replace_dht_structure(sample, monkeypatch):
    body, exception = sample
    altered = bytearray(body)
    altered[exception["dht"]["segment_offset"] + 1] = 0xE1
    altered = bytes(altered)
    monkeypatch.setattr(
        module,
        "AUDITED_EXCEPTION",
        {
            **exception,
            "file_sha256": module.sha_bytes(altered),
            "jpeg_sha256": module.sha_bytes(altered[exception["jpeg_offset"] :]),
        },
    )
    with pytest.raises(ValueError):
        module.scan_bytes(altered, ALLOWED)


def test_prior_scanner_globals_and_regex_remain_unchanged(sample):
    body, _ = sample
    previous = module.previous.AUDITED_EXCEPTIONS
    assert module.SECRETS.pattern == module.previous.SECRETS.pattern
    module.scan_bytes(body, ALLOWED)
    assert module.previous.AUDITED_EXCEPTIONS is previous
    with pytest.raises(ValueError):
        module.previous.scan_bytes(body)


def test_real_reviewed_pdf_when_saved_raw_is_installed():
    source = Path(__file__).parents[2] / "../../work/data-round6/cases/raw/ach_rtm_report.pdf"
    if not source.exists():
        pytest.skip("Raw PDF is delivered separately")
    body = source.read_bytes()
    assert len(module.scan_bytes(body, ALLOWED)) == 1
    with pytest.raises(ValueError):
        module.scan_bytes(body + b"\n", ALLOWED)
