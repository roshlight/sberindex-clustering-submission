"""Fail-closed provenance tests for the fixed round-three evidence collector."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round3_collector", Path(__file__).parents[1] / "scripts/collect_round3_cases.py"
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


@pytest.fixture
def fixture_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "sample.pdf"
    source.write_bytes(b"immutable fixture source")
    parsed = raw / "sample.extracted.json"
    parsed.write_text(json.dumps({"pages": ["workers 17"]}), encoding="utf-8")
    meta = {
        "source_id": "sample",
        "url": "https://example.invalid/source.pdf",
        "download_status": "downloaded",
        "raw_file": source.name,
        "sha256": collector.sha256(source.read_bytes()),
        "extraction_sha256": collector.sha256(parsed.read_bytes()),
        "retrieved_at_utc": "2026-09-16T00:00:00+00:00",
    }
    (raw / "sample.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            [
                {
                    "source_id": "sample",
                    "evidence_id": "e1",
                    "territory_id": 1,
                    "page": 1,
                    "must_contain": ["workers 17"],
                    "value": 17,
                }
            ]
        ),
        encoding="utf-8",
    )
    return raw, recipe, tmp_path / "output"


def test_fixed_locator_replay(fixture_source):
    raw, recipe, output = fixture_source
    collector.finalize(raw, recipe, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["locator_checks_passed"] == 1
    assert summary["automatic_join_or_model_update"] is False


@pytest.mark.parametrize("changed", ["sample.pdf", "sample.extracted.json"])
def test_changed_source_or_text_is_rejected(fixture_source, changed):
    raw, recipe, output = fixture_source
    (raw / changed).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed"):
        collector.finalize(raw, recipe, output)
    assert not (output / "factual_evidence.csv").exists()


def test_wrong_locator_value_is_rejected(fixture_source):
    raw, recipe, output = fixture_source
    rules = json.loads(recipe.read_text())
    rules[0]["must_contain"] = ["workers 18"]
    recipe.write_text(json.dumps(rules))
    with pytest.raises(ValueError, match="token absent"):
        collector.finalize(raw, recipe, output)


def test_forbidden_response_is_cached_without_retry(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            calls.append(url)
            return httpx.Response(403, content=b"Forbidden", request=httpx.Request("GET", url))

    monkeypatch.setattr(collector.httpx, "Client", Client)
    specs = tmp_path / "sources.json"
    specs.write_text(json.dumps([{"source_id": "blocked", "url": "https://example.invalid/"}]))
    first = collector.collect(specs, tmp_path / "raw", None, 0)
    second = collector.collect(specs, tmp_path / "raw", None, 0)
    assert len(calls) == 1
    assert first == second
    assert first[0]["status"] == 403
    assert first[0]["download_status"] == "http_error"
