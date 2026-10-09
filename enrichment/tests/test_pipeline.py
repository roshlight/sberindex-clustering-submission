import argparse
import hashlib
import json

import pytest

from sber_enrichment import pipeline as p
from sber_enrichment.clients import ClientError


def test_query_plan_does_not_research_available_or_out_of_scope_needs():
    needs = [
        {"need_id": "a", "priority": "P0", "search_queries": ["codes"]},
        {
            "need_id": "b",
            "priority": "P1",
            "already_available": True,
            "search_queries": ["distances"],
        },
        {"need_id": "c", "priority": "P2", "search_queries": ["extra"]},
    ]
    assert [row["need_id"] for row in p.search_plan(needs)] == ["a"]


def test_usage_uses_all_http_attempts_without_double_counting(tmp_path):
    p.write_json(
        tmp_path / "model_calls" / "a.json",
        {"status": "complete", "result": {"usage": {"cost": 0.6, "prompt_tokens": 50}}},
    )
    p.write_json(
        tmp_path / "router_attempts.json",
        [
            {"status": 503, "usage": {"cost": 0.1, "prompt_tokens": 10}},
            {"status": 200, "usage": {"cost": 0.6, "prompt_tokens": 50}},
            {"status": None, "usage": None},
        ],
    )
    result = p.usage_summary(tmp_path)
    assert result["reported_cost_usd"] == pytest.approx(0.7)
    assert result["prompt_tokens"] == 60
    assert result["completed_calls"] == 1
    assert result["calls_with_unknown_cost_or_failure"] == 1


def test_model_cache_prevents_repeated_paid_calls_and_ambiguous_retry(tmp_path):
    class Router:
        count = 0

        def complete_json(self, *args, **kwargs):
            self.count += 1
            return {"parsed": {}, "usage": {"cost": 0.1}}

    router = Router()
    for _ in range(2):
        p.model_call(tmp_path, "ok", router, "model", "system", {}, {}, 100)
    assert router.count == 1

    class Broken:
        def complete_json(self, *args, **kwargs):
            raise TimeoutError()

    with pytest.raises(TimeoutError):
        p.model_call(tmp_path, "bad", Broken(), "model", "system", {}, {}, 100)
    with pytest.raises(RuntimeError, match="Previous model attempt failed"):
        p.model_call(tmp_path, "bad", router, "model", "system", {}, {}, 100)
    assert router.count == 1


def test_full_offline_run_and_manifest_integrity(tmp_path, monkeypatch):
    class Search:
        count = 0

        def __init__(self, *args, **kwargs):
            pass

        def config(self):
            return {"engines": [{"name": "mock", "enabled": True}]}

        def search(self, *args, **kwargs):
            Search.count += 1
            return {
                "results": [{"url": "https://rosstat.gov.ru/table", "title": "Data"}],
                "unresponsive_engines": [],
            }

    class Fetch:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, url):
            return {
                "url": url,
                "final_url": url,
                "status": 200,
                "content_type": "text/html",
                "body": b"<html><p>Public data</p></html>",
                "error": None,
            }

    monkeypatch.setattr(p, "SearxClient", Search)
    monkeypatch.setattr(p, "PublicFetcher", Fetch)
    needs, catalog, run = tmp_path / "needs.json", tmp_path / "catalog.json", tmp_path / "run"
    p.write_json(needs, [{"need_id": "a", "priority": "P0", "search_queries": ["codes"]}])
    p.write_json(catalog, [])
    args = argparse.Namespace(
        needs=needs,
        catalog=catalog,
        run_dir=run,
        resume=False,
        engines=["mock"],
        pages=1,
        max_queries=1,
        max_documents=2,
        llm_documents=0,
        expand_queries=False,
        extract_model="none",
        review_model="none",
        search_workers=1,
        fetch_workers=1,
        llm_workers=1,
    )
    p.execute(args)
    manifest = p.read_json(run / "run_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["documents"] == 1
    for name, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="Run exists"):
        p.execute(args)
    args.resume = True
    p.execute(args)
    assert Search.count == 1
    p.write_json(needs, json.loads(needs.read_text()) + [{"need_id": "changed"}])
    with pytest.raises(ValueError, match="changed configuration"):
        p.execute(args)


def test_access_denied_defers_remaining_reviews_without_retry(tmp_path, monkeypatch):
    attempted = []

    def blocked_call(*args, **kwargs):
        attempted.append(args[1])
        raise ClientError("http_error", status=403)

    monkeypatch.setattr(p, "model_call", blocked_call)
    (tmp_path / "text.txt").write_text("Public source text. " * 30)
    records = [
        {
            "source_id": str(i),
            "url": "https://rosstat.gov.ru/" + str(i),
            "text_file": "text.txt",
            "text_chars": 600,
            "sha256": "hash",
            "status": 200,
            "seed": True,
            "authority_hint": "official",
        }
        for i in range(4)
    ]
    cards, failed = p.review_documents(tmp_path, object(), records, [], "model", "model", 4, 1)
    assert cards == [] and failed == 1
    assert len(attempted) == 1
    assert p.read_json(tmp_path / "review_status.json")["deferred"] == 3
