"""Exact JPEG-table exceptions must not weaken credential scanning elsewhere."""

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load_module("package_round5_delivery")
previous = load_module("package_round4_delivery")


def test_original_pattern_prior_exception_and_jpeg_parser_are_preserved():
    assert module.SECRETS.pattern == previous.SECRETS.pattern
    assert module.AUDITED_EXCEPTION == previous.AUDITED_EXCEPTION
    assert module.AUDITED_EXCEPTIONS[0] == previous.AUDITED_EXCEPTION
    functions = []
    for path in ["package_round4_delivery.py", "package_round5_delivery.py"]:
        parsed = ast.parse((SCRIPTS / path).read_text("utf-8"))
        node = next(
            x for x in parsed.body if isinstance(x, ast.FunctionDef) and x.name == "containing_dht"
        )
        functions.append(ast.dump(node, include_attributes=False))
    assert functions[0] == functions[1]


@pytest.fixture
def synthetic_dht(monkeypatch):
    # Invented bytes that exercise the scanner; not a real service credential.
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
    monkeypatch.setattr(module, "AUDITED_EXCEPTIONS", (approved,))
    return body, approved


def test_exact_reviewed_hit_returns_metadata_without_bytes(synthetic_dht):
    body, _ = synthetic_dht
    hits = module.scan_bytes(body)
    assert len(hits) == 1
    assert "match_sha256" in hits[0]
    assert all(not isinstance(value, bytes) for value in hits[0].values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("offset", 0),
        ("length", 1),
        ("match_sha256", "0" * 64),
        ("file_sha256", "0" * 64),
        ("file_bytes", 0),
    ],
)
def test_every_exact_coordinate_is_required(synthetic_dht, monkeypatch, field, value):
    body, approved = synthetic_dht
    monkeypatch.setattr(module, "AUDITED_EXCEPTIONS", ({**approved, field: value},))
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body)


def test_changed_byte_outside_match_is_rejected(synthetic_dht):
    body, _ = synthetic_dht
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(b"X" + body[1:])


def test_extra_match_not_skipped_after_one_approved_match(synthetic_dht, monkeypatch):
    body, approved = synthetic_dht
    changed = body + b"\nsk-or-v1-" + b"B" * 30
    approved = {**approved, "file_bytes": len(changed), "file_sha256": module.sha_bytes(changed)}
    monkeypatch.setattr(module, "AUDITED_EXCEPTIONS", (approved,))
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(changed)


def test_duplicate_exception_does_not_create_ambiguous_permission(synthetic_dht, monkeypatch):
    body, approved = synthetic_dht
    monkeypatch.setattr(module, "AUDITED_EXCEPTIONS", (approved, approved.copy()))
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body)


@pytest.mark.parametrize("mode", ["jpeg_hash", "dht_containment", "not_dht_segment"])
def test_exact_digests_do_not_bypass_structure(synthetic_dht, monkeypatch, mode):
    body, approved = synthetic_dht
    if mode == "jpeg_hash":
        approved = {**approved, "jpeg_sha256": "0" * 64}
    elif mode == "dht_containment":
        approved = {**approved, "dht": {**approved["dht"], "table_id": 3}}
    else:
        changed = bytearray(body)
        changed[approved["dht"]["segment_offset"] + 1] = 0xE1
        body = bytes(changed)
        approved = {
            **approved,
            "file_sha256": module.sha_bytes(body),
            "jpeg_sha256": module.sha_bytes(body[approved["jpeg_offset"] :]),
        }
    monkeypatch.setattr(module, "AUDITED_EXCEPTIONS", (approved,))
    with pytest.raises(ValueError):
        module.scan_bytes(body)


def test_all_original_scanner_patterns_remain_blocked_without_disclosure():
    candidates = [
        b"sk-or-v1-" + b"C" * 30,
        b"987654321:" + b"D" * 35,
        b"-----BEGIN " + b"PRIVATE KEY-----",
    ]
    for body in candidates:
        with pytest.raises(ValueError) as caught:
            module.scan_bytes(body)
        assert str(caught.value) == "Credential-like data detected; contents are not printed"


def test_eight_exact_real_pdf_spans_and_mutation_rejection():
    source = (
        Path(__file__).resolve().parents[2]
        / "../../work/data-round5/cases/raw/mintrans_report2023.pdf"
    )
    if not source.exists():
        pytest.skip("Separately delivered case raw PDF is not installed")
    body = source.read_bytes()
    hits = module.scan_bytes(body)
    assert len(hits) == 8
    assert len({hit["offset"] for hit in hits}) == 8
    assert {hit["pdf_object_number"] for hit in hits} == {76, 208, 209, 210, 391, 403}
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body + b"\n")
    with pytest.raises(ValueError, match="Credential-like data"):
        module.scan_bytes(body + b"\nsk-or-v1-" + b"E" * 30)
