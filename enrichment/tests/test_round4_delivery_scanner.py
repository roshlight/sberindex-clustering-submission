"""Exact audited false-positive exception; no credential-looking bytes are logged."""

import ast
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_round4_delivery", Path(__file__).parents[1] / "scripts/package_round4_delivery.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_original_regex_is_unchanged():
    original = Path(__file__).parents[1] / "scripts/package_round2_delivery.py"
    parsed = ast.parse(original.read_text("utf-8"))
    assignment = next(
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "secrets" for target in node.targets)
    )
    assert module.SECRETS.pattern == ast.literal_eval(assignment.value.args[0])


@pytest.fixture
def synthetic_dht(monkeypatch):
    # A harmless invented scanner-shaped byte string, used only as JPEG symbols.
    symbols = b"123456789:" + b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    counts = bytes([0] * 7 + [len(symbols)] + [0] * 8)
    payload = b"\x11" + counts + symbols
    prefix = b"synthetic-fixture\n"
    jpeg = b"\xff\xd8\xff\xc4" + (len(payload) + 2).to_bytes(2, "big") + payload + b"\xff\xd9"
    body = prefix + jpeg
    match = module.SECRETS.search(body)
    approved = {
        "exception_id": "synthetic-test-only",
        "file_bytes": len(body),
        "file_sha256": module.sha_bytes(body),
        "offset": match.start(),
        "length": match.end() - match.start(),
        "match_sha256": module.sha_bytes(match.group()),
        "jpeg_offset": len(prefix),
        "jpeg_bytes": len(jpeg),
        "jpeg_sha256": module.sha_bytes(jpeg),
        "dht": module.containing_dht(body, len(prefix), match.start(), match.end()),
    }
    monkeypatch.setattr(module, "AUDITED_EXCEPTION", approved)
    return body, approved


def test_exact_reviewed_match_accepted(synthetic_dht):
    body, _ = synthetic_dht
    hits = module.scan_bytes(body)
    assert len(hits) == 1
    assert "match_sha256" in hits[0]
    assert "match_bytes" not in hits[0]


def test_changed_byte_outside_match_rejected(synthetic_dht):
    body, _ = synthetic_dht
    changed = b"X" + body[1:]
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(changed)


def test_extra_match_not_skipped_even_if_first_is_audited(synthetic_dht, monkeypatch):
    body, approved = synthetic_dht
    changed = body + b"\n" + b"sk-or-v1-" + b"B" * 30
    # Even a newly registered full-file digest must not exempt the whole file.
    approved = {**approved, "file_bytes": len(changed), "file_sha256": module.sha_bytes(changed)}
    monkeypatch.setattr(module, "AUDITED_EXCEPTION", approved)
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(changed)


@pytest.mark.parametrize("field,value", [("offset", 0), ("length", 1), ("match_sha256", "0" * 64)])
def test_each_exact_match_coordinate_required(synthetic_dht, monkeypatch, field, value):
    body, approved = synthetic_dht
    monkeypatch.setattr(module, "AUDITED_EXCEPTION", {**approved, field: value})
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body)


def test_dht_segment_alone_is_insufficient(synthetic_dht):
    body, approved = synthetic_dht
    table_start = approved["dht"]["table_offset"]
    with pytest.raises(ValueError, match="outside DHT symbols"):
        module.containing_dht(body, approved["jpeg_offset"], table_start, table_start + 3)


def test_different_segment_marker_rejected(synthetic_dht):
    body, approved = synthetic_dht
    changed = bytearray(body)
    changed[approved["dht"]["segment_offset"] + 1] = 0xE1
    with pytest.raises(ValueError, match="outside DHT symbols"):
        module.containing_dht(
            changed,
            approved["jpeg_offset"],
            approved["offset"],
            approved["offset"] + approved["length"],
        )


def test_original_scanner_patterns_still_blocked():
    examples = [
        b"sk-or-v1-" + b"C" * 30,
        b"-----BEGIN " + b"PRIVATE KEY-----",
        b"987654321:" + b"D" * 35,
    ]
    for candidate in examples:
        with pytest.raises(ValueError, match="contents are not printed") as result:
            module.scan_bytes(candidate)
        assert str(result.value) == "Credential-like data detected; contents are not printed"


def test_real_pdf_exact_exception_and_modified_pdf():
    repo = Path(__file__).resolve().parents[2]
    source = repo / "../../work/data-round4/geography/raw/pit_planning_materials.pdf"
    if not source.exists():
        pytest.skip("The separately released raw PDF is not installed")
    body = source.read_bytes()
    assert module.sha_bytes(body) == module.AUDITED_EXCEPTION["file_sha256"]
    assert len(module.scan_bytes(body)) == 1
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body + b"\n")
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body + b"\nsk-or-v1-" + b"E" * 30)
