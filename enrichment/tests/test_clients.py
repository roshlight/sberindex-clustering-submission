import gzip
import hashlib
import json
import socket
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs

import certifi
import httpx
import pytest

from sber_enrichment.clients import ClientError, OpenRouterClient, PublicFetcher, SearxClient

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": ["integer", "null"]}},
    "required": ["value"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY_FILE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-offline-tests")
    monkeypatch.setattr("sber_enrichment.clients.time.sleep", lambda _seconds: None)


def completion(content='{"value": 1}', finish="stop", **kwargs):
    return {
        "id": "gen-test",
        "model": "test/model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.01},
        "choices": [{"message": {"content": content, **kwargs}, "finish_reason": finish}],
    }


def test_searx_engine_validation_and_form_contract():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/config":
            return httpx.Response(
                200,
                json={
                    "engines": [
                        {"name": "google cse", "enabled": True},
                        {"name": "off", "enabled": False},
                    ]
                },
            )
        form = parse_qs(request.content.decode())
        assert request.method == "POST"
        assert "categories" not in form
        assert form["engines"] == ["google cse"]
        assert form["pageno"] == ["2"]
        assert form["language"] == ["ru-RU"]
        assert form["format"] == ["json"]
        return httpx.Response(200, json={"results": [], "unresponsive_engines": [["x", "error"]]})

    with SearxClient("http://localhost:8888", transport=httpx.MockTransport(handler)) as client:
        result = client.search("Москва экономика", ["google cse"], page=2)
        assert result["unresponsive_engines"]
        for engine in ("missing", "off"):
            with pytest.raises(ClientError, match="unknown_or_disabled_engine"):
                client.search("q", [engine])
    assert len(seen) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_terminal_http_errors_do_not_retry_or_expose_body(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "SECRET in upstream body"}})

    with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError) as caught:
            client.complete_json("test/model", "system", "user", SCHEMA)
        assert caught.value.status == status
        assert "SECRET" not in str(caught.value)
        assert "SECRET" not in json.dumps(client.attempts)
    assert len(calls) == 1


def test_openrouter_wire_schema_and_file_key(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-test-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(key_file))

    def handler(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer file-test-key"
        body = json.loads(request.content)
        assert body["provider"] == {"require_parameters": True}
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert body["stream"] is False
        assert body["max_tokens"] == 6000
        return httpx.Response(200, json=completion())

    with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        result = client.complete_json("test/model", "system", "user", SCHEMA)
        assert result == {
            "parsed": {"value": 1},
            "response_id": "gen-test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.01},
            "model": "test/model",
            "finish_reason": "stop",
        }
        assert "file-test-key" not in json.dumps(client.attempts)


@pytest.mark.parametrize(
    "response, code",
    [
        (completion(finish="length"), "truncated_completion"),
        (completion(refusal="I cannot"), "model_refusal"),
        (completion(finish="content_filter"), "model_refusal"),
        (completion(content=""), "empty_completion"),
        (completion(content='{"value": "wrong"}'), "schema_validation_failed"),
        (completion(content='{"value": 1, "extra": 2}'), "schema_validation_failed"),
        (completion(content='{"value": 1, "value": 2}'), "invalid_completion_json"),
        (completion(content='{"value": NaN}'), "invalid_completion_json"),
        (completion(finish="error"), "provider_error"),
        ({"id": "gen-test", "error": {"message": "do not expose"}}, "provider_error"),
    ],
)
def test_invalid_completions_preserve_accounting(response, code):
    with OpenRouterClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(ClientError, match=code) as caught:
            client.complete_json("test/model", "system", "user", SCHEMA)
        assert caught.value.response_id == "gen-test"
        assert len(client.attempts) == 1
        if "usage" in response:
            assert caught.value.usage["cost"] == 0.01


def test_retry_records_each_usage_and_honors_retry_after(monkeypatch):
    delays = []
    monkeypatch.setattr("sber_enrichment.clients.time.sleep", delays.append)
    statuses = iter([429, 503, 200])

    def handler(_request):
        status = next(statuses)
        return httpx.Response(status, headers={"Retry-After": "2"}, json=completion())

    with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        client.complete_json("test/model", "system", "user", SCHEMA)
        assert [item["status"] for item in client.attempts] == [429, 503, 200]
        assert len([item for item in client.attempts if item["usage"]["cost"] == 0.01]) == 3
    assert delays == [2.0, 2.0]


def test_retry_limit_and_long_retry_hint():
    with OpenRouterClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503, json={}))
    ) as client:
        with pytest.raises(ClientError, match="http_error"):
            client.models()
        assert len(client.attempts) == 3
    with OpenRouterClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(429, headers={"Retry-After": "3600"}, json={})
        )
    ) as client:
        with pytest.raises(ClientError, match="retry_deferred"):
            client.models()
        assert len(client.attempts) == 1


def test_ambiguous_timeout_never_retries():
    def handler(request):
        raise httpx.ReadTimeout("SECRET timeout body", request=request)

    with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match="timeout_unknown_outcome") as caught:
            client.complete_json("test/model", "system", "user", SCHEMA)
        assert "SECRET" not in str(caught.value)
        assert len(client.attempts) == 1
        assert client.attempts[0]["usage"] is None


def test_missing_usage_is_unknown_and_external_schema_is_blocked():
    response = completion()
    response.pop("usage")
    with OpenRouterClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response))
    ) as client:
        assert client.complete_json("test/model", "sys", "user", SCHEMA)["usage"] is None
        with pytest.raises(ClientError, match="external_schema_reference"):
            client.complete_json("test/model", "sys", "user", {"$ref": "http://localhost/schema"})
        assert len(client.attempts) == 1


def public_fetcher(handler, **kwargs):
    return PublicFetcher(
        transport=httpx.MockTransport(handler), resolver=lambda *_args: ["93.184.216.34"], **kwargs
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/secret",
        "C:\\secret.txt",
        "/etc/passwd",
        "http://localhost/a",
        "http://127.0.0.1",
        "https://[::1]",
        "https://8.8.8.8/a",
        "https://user:SECRET@example.com/a",
        "http://example.com:8888/a",
        "http://example.local/a",
        "http://%31%32%37.0.0.1/a",
        "https://example.com\\@localhost/a",
    ],
)
def test_public_fetcher_rejects_before_network(url):
    def handler(_request):
        pytest.fail("blocked URL reached HTTP transport")

    result = public_fetcher(handler).fetch(url)
    assert result["error"] == "blocked_url"
    assert result["body"] == b""
    assert "SECRET" not in repr(result)


def test_dns_guard_checks_all_addresses(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443)),
        ],
    )
    result = PublicFetcher(
        transport=httpx.MockTransport(lambda _request: pytest.fail("private DNS reached network"))
    ).fetch("https://example.com")
    assert result["error"] == "blocked_dns_address"


def test_public_fetcher_pins_ip_preserves_host_sni_and_relative_redirect():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "authorization" not in request.headers
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/document"})
        return httpx.Response(
            200, stream=httpx.ByteStream(b"evidence"), headers={"content-type": "text/plain"}
        )

    result = public_fetcher(handler).fetch("https://example.com/start")
    assert result["error"] is None
    assert result["final_url"] == "https://example.com/document"
    assert result["body"] == b"evidence"
    assert result["sha256"] == hashlib.sha256(b"evidence").hexdigest()
    assert len(requests) == 2


def test_public_fetcher_checks_redirect_target():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    result = public_fetcher(handler).fetch("https://example.com/start")
    assert result["error"] == "blocked_url"
    assert len(calls) == 1


def test_public_fetcher_bounds_stream_even_without_length():
    result = public_fetcher(
        lambda _request: httpx.Response(200, stream=httpx.ByteStream(b"123456")), max_bytes=5
    ).fetch("https://example.com")
    assert result["error"] == "body_too_large"
    assert result["body"] == b""
    assert result["sha256"] is None


def test_public_fetcher_encoding_and_tls_errors_are_explicit():
    result = public_fetcher(
        lambda _request: httpx.Response(
            200, headers={"content-encoding": "br"}, stream=httpx.ByteStream(b"compressed")
        )
    ).fetch("https://example.com")
    assert result["error"] == "unsupported_content_encoding"

    def fail(request):
        raise httpx.ConnectError("SSL verification failed SECRET", request=request)

    result = public_fetcher(fail).fetch("https://example.com")
    assert result["error"] == "fetch_network_error"
    assert "SECRET" not in repr(result)


def test_read_timeout_preserves_generation_header():
    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"choices":'
            raise httpx.ReadTimeout("opaque transport message")

    with OpenRouterClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, headers={"X-Generation-Id": "gen-interrupted"}, stream=Interrupted()
            )
        )
    ) as client:
        with pytest.raises(ClientError, match="timeout_unknown_outcome") as caught:
            client.complete_json("test/model", "sys", "user", SCHEMA)
        assert caught.value.response_id == "gen-interrupted"
        assert caught.value.status == 200
        assert len(client.attempts) == 1


def test_idn_uses_ascii_dns_host_and_sni():
    def handler(request):
        assert request.headers["Host"] == "xn--e1afmkfd.xn--p1ai"
        assert request.extensions["sni_hostname"] == "xn--e1afmkfd.xn--p1ai"
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    result = public_fetcher(handler).fetch("https://пример.рф/document")
    assert result["error"] is None


def test_html_search_response_is_an_error():
    def handler(request):
        if request.url.path == "/config":
            return httpx.Response(200, json={"engines": [{"name": "one", "enabled": True}]})
        return httpx.Response(200, text="<html>captcha</html>")

    with SearxClient("http://localhost:8888", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ClientError, match="invalid_json_response"):
            client.search("query", ["one"])


def test_models_and_invalid_local_schema_reference():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test/model"}]})
        return httpx.Response(200, json=completion())

    with OpenRouterClient(transport=httpx.MockTransport(handler)) as client:
        assert client.models()["data"][0]["id"] == "test/model"
        with pytest.raises(ClientError, match="schema_validation_failed"):
            client.complete_json("test/model", "sys", "user", {"$ref": "#/$defs/missing"})


def test_nonfinite_exponent_rejected():
    with OpenRouterClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=completion(content='{"value": 1e999}'))
        )
    ) as client:
        with pytest.raises(ClientError, match="invalid_completion_json"):
            client.complete_json("test/model", "sys", "user", SCHEMA)


def test_invalid_secret_is_never_echoed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-with-кириллица")
    with pytest.raises(ClientError) as caught:
        OpenRouterClient()
    assert str(caught.value) == "api_key_missing_or_invalid"


def test_concurrent_completions_keep_their_own_accounting_when_second_finishes_first(tmp_path):
    first_entered = threading.Event()
    second_completed = threading.Event()

    def handler(request):
        model = json.loads(request.content)["model"]
        if model == "test/first":
            first_entered.set()
            assert second_completed.wait(timeout=3)
            number = 1
        else:
            number = 2
        response = completion(content=json.dumps({"value": number}))
        response.update(id=f"gen-{number}", model=model, usage={"cost": number / 100})
        return httpx.Response(200, json=response)

    ledger = tmp_path / "attempts.json"
    with OpenRouterClient(
        transport=httpx.MockTransport(handler), attempt_log_path=ledger
    ) as client:

        def second():
            result = client.complete_json("test/second", "sys", "user", SCHEMA)
            second_completed.set()
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.complete_json, "test/first", "sys", "user", SCHEMA)
            assert first_entered.wait(timeout=3)
            later = pool.submit(second)
            second_result = later.result(timeout=3)
            first_result = first.result(timeout=3)
        assert first_result["parsed"] == {"value": 1}
        assert first_result["response_id"] == "gen-1"
        assert first_result["usage"] == {"cost": 0.01}
        assert second_result["response_id"] == "gen-2"
        assert second_result["usage"] == {"cost": 0.02}
        assert {(row["response_id"], row["model"]) for row in client.attempts} == {
            ("gen-1", "test/first"),
            ("gen-2", "test/second"),
        }
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        assert saved == client.attempts
        assert all(row["state"] == "complete" for row in saved)
        assert len({row["attempt_id"] for row in saved}) == 2
        assert len({row["call_id"] for row in saved}) == 2


def test_durable_ledger_loads_prior_and_keeps_free_models_operation(tmp_path):
    ledger = tmp_path / "attempts.json"
    prior = {
        "attempt_id": "old",
        "state": "headers_received",
        "response_id": "gen-old",
        "usage": None,
        "method": "POST",
        "operation": "/api/v1/chat/completions",
    }
    ledger.write_text(json.dumps([prior]), encoding="utf-8")
    with OpenRouterClient(
        attempt_log_path=ledger,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
    ) as client:
        assert client.attempts == [prior]
        client.models()
    with OpenRouterClient(
        attempt_log_path=ledger,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=completion())),
    ) as recovered:
        assert len(recovered.attempts) == 2
        recovered.complete_json("test/model", "sys", "user", SCHEMA)
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(saved) == 3
    assert saved[0] == prior
    assert saved[1]["method"] == "GET"
    assert saved[1]["operation"] == "/api/v1/models"
    assert saved[2]["method"] == "POST"
    assert saved[2]["usage"]["cost"] == 0.01
    assert not list(tmp_path.glob("*.tmp"))


def test_durable_ledger_survives_interruption_after_headers(tmp_path):
    ledger = tmp_path / "attempts.json"

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            saved = json.loads(ledger.read_text(encoding="utf-8"))
            assert saved[0]["state"] == "headers_received"
            assert saved[0]["response_id"] == "gen-partial"
            raise SystemExit("simulate process interruption")
            yield b""  # pragma: no cover

    def handler(_request):
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        assert saved[0]["state"] == "started"
        return httpx.Response(200, headers={"X-Generation-Id": "gen-partial"}, stream=Interrupted())

    with OpenRouterClient(
        attempt_log_path=ledger, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(SystemExit):
            client.complete_json("test/model", "sys", "user", SCHEMA)
    with OpenRouterClient(
        attempt_log_path=ledger,
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("recovery must not make HTTP requests")
        ),
    ) as recovered:
        assert recovered.attempts[0]["state"] == "headers_received"
        assert recovered.attempts[0]["usage"] is None


def test_durable_retry_and_validation_failure_keep_all_usage(tmp_path):
    ledger = tmp_path / "attempts.json"
    count = 0

    def handler(_request):
        nonlocal count
        count += 1
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        if count == 2:
            assert saved[0]["state"] == "retry_scheduled"
            assert saved[0]["usage"]["cost"] == 0.01
        return httpx.Response(
            503 if count == 1 else 200, json=completion(content='{"value":"bad"}')
        )

    with OpenRouterClient(
        attempt_log_path=ledger, transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ClientError, match="schema_validation_failed"):
            client.complete_json("test/model", "sys", "user", SCHEMA)
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert saved[1]["state"] == "failed"
    assert saved[1]["error"] == "schema_validation_failed"
    assert saved[1]["usage"]["cost"] == 0.01
    assert saved[0]["call_id"] == saved[1]["call_id"]


def test_bad_ledger_fails_before_network_and_is_not_overwritten(tmp_path):
    ledger = tmp_path / "attempts.json"
    ledger.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ClientError, match="attempt_log_unreadable"):
        OpenRouterClient(attempt_log_path=ledger)
    assert ledger.read_text(encoding="utf-8") == "corrupt"


def test_extra_ca_is_exact_host_scoped_and_not_inherited_by_redirects(monkeypatch):
    real_client = httpx.Client
    contexts = []

    def capture_client(**kwargs):
        contexts.append(kwargs["verify"])
        return real_client(**kwargs)

    def handler(request):
        if request.headers["Host"] == "example.com":
            return httpx.Response(302, headers={"Location": "https://sub.example.com/file"})
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    monkeypatch.setattr("sber_enrichment.clients.httpx.Client", capture_client)
    fetcher = public_fetcher(handler, ca_bundles={"example.com": certifi.where()})
    result = fetcher.fetch("https://example.com/file")
    assert result["error"] is None
    assert isinstance(contexts[0], ssl.SSLContext)
    assert contexts[0].check_hostname
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert not contexts[0].verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    assert contexts[1] is True


@pytest.mark.parametrize(
    "hostname", ["*.example.com", "localhost", "example.com/path", "127.0.0.1"]
)
def test_ca_mapping_rejects_nonexact_public_hostnames(hostname):
    with pytest.raises(ClientError, match="invalid_ca_bundle"):
        PublicFetcher(ca_bundles={hostname: certifi.where()})


def test_invalid_ca_file_fails_closed_without_echoing_file_contents(tmp_path):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("SECRET invalid certificate", encoding="ascii")
    with pytest.raises(ClientError) as caught:
        PublicFetcher(ca_bundles={"example.com": bundle})
    assert str(caught.value) == "invalid_ca_bundle"


def gzip_fetch(body, **kwargs):
    return public_fetcher(
        lambda _request: httpx.Response(
            200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(body)
        ),
        **kwargs,
    ).fetch("https://example.com/document")


def test_gzip_decodes_all_members_and_hashes_decoded_document():
    # A member larger than the raw chunk size also exercises boundaries between reads.
    first = bytes(range(256)) * 100
    second = b"second member"
    encoded = gzip.compress(first, compresslevel=0) + gzip.compress(second)
    result = gzip_fetch(encoded)
    assert result["error"] is None
    assert result["body"] == first + second
    assert result["sha256"] == hashlib.sha256(first + second).hexdigest()


def test_gzip_decoded_limit_is_cumulative_across_members():
    result = gzip_fetch(
        gzip.compress(b"123") + gzip.compress(b"456"), max_bytes=5, max_compressed_bytes=100
    )
    assert result["error"] == "body_too_large"
    assert result["body"] == b""
    assert result["sha256"] is None


def test_gzip_expansion_is_bounded_before_full_decompression(monkeypatch):
    from sber_enrichment import clients

    factory = clients.zlib.decompressobj
    returned_sizes = []

    class ObservedDecoder:
        def __init__(self, **kwargs):
            self.inner = factory(**kwargs)

        def decompress(self, data, *, max_length):
            assert 0 < max_length <= 32_001
            output = self.inner.decompress(data, max_length=max_length)
            returned_sizes.append(len(output))
            return output

        def __getattr__(self, attribute):
            return getattr(self.inner, attribute)

    monkeypatch.setattr(clients.zlib, "decompressobj", ObservedDecoder)
    encoded = gzip.compress(b"a" * 2_000_000)
    result = gzip_fetch(encoded, max_bytes=32_000)
    assert len(encoded) < 32_000
    assert result["error"] == "body_too_large"
    assert result["body"] == b""
    assert sum(returned_sizes) == 32_001


def test_gzip_encoded_stream_limit_is_independent_of_decoded_limit():
    encoded = gzip.compress(b"small decoded document", compresslevel=0)
    result = gzip_fetch(encoded, max_bytes=100, max_compressed_bytes=len(encoded) - 1)
    assert result["error"] == "compressed_body_too_large"
    assert result["body"] == b""


@pytest.mark.parametrize(
    ("encoded", "expected_error"),
    [
        (b"", "truncated_gzip_body"),
        (gzip.compress(b"ok")[:-1], "truncated_gzip_body"),
        (gzip.compress(b"ok") + b"junk", "invalid_gzip_body"),
        (gzip.compress(b"ok") + b"\x1f", "truncated_gzip_body"),
        (gzip.compress(b"ok") + gzip.compress(b"second")[:-4], "truncated_gzip_body"),
        (gzip.compress(b"ok")[:-8] + b"\x00" * 8, "invalid_gzip_body"),
    ],
)
def test_gzip_rejects_missing_trailers_crc_errors_and_trailing_garbage(encoded, expected_error):
    result = gzip_fetch(encoded)
    assert result["error"] == expected_error
    assert result["body"] == b""
    assert result["sha256"] is None


def test_gzip_exact_limits_and_empty_member_are_valid():
    encoded = gzip.compress(b"12345") + gzip.compress(b"")
    result = gzip_fetch(encoded, max_bytes=5, max_compressed_bytes=len(encoded))
    assert result["error"] is None
    assert result["body"] == b"12345"


def test_gzip_member_header_can_cross_raw_chunk_boundary():
    first_body = b"x" * 16_360
    first = gzip.compress(first_body, compresslevel=0)
    assert len(first) == 16_383
    result = gzip_fetch(first + gzip.compress(b"second"))
    assert result["error"] is None
    assert result["body"] == first_body + b"second"


def test_gzip_declared_wire_limit_rejects_before_body_iteration():
    class Unreadable(httpx.SyncByteStream):
        def __iter__(self):
            pytest.fail("declared oversized body must not be read")

    result = public_fetcher(
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-length": "1001"},
            stream=Unreadable(),
        ),
        max_bytes=100,
        max_compressed_bytes=1000,
    ).fetch("https://example.com/document")
    assert result["error"] == "compressed_body_too_large"


@pytest.mark.parametrize("hostname", ["www.sberbank.ru", "sber.ru"])
def test_aws_chunked_bare_pdf_accepted_only_with_explicit_provenance_note(hostname):
    body = b"%PDF-1.5\r\nsource document\n%%EOF"
    result = public_fetcher(
        lambda _request: httpx.Response(
            200, headers={"content-encoding": "aws-chunked"}, stream=httpx.ByteStream(body)
        )
    ).fetch(f"https://{hostname}/document.pdf")
    assert result["error"] is None
    assert result["body"] == body
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert result["declared_content_encoding"] == "aws-chunked"
    assert result["encoding_note"] == "declared_aws_chunked_but_received_bare_pdf"


@pytest.mark.parametrize(
    "body",
    [b"5;chunk-signature=abc\r\n%PDF-\r\n0\r\n\r\n", b"<html>error</html>", b"", b" \n%PDF-1.5"],
)
def test_aws_chunked_framed_or_nonpdf_body_is_not_decoded_or_accepted(body):
    result = public_fetcher(
        lambda _request: httpx.Response(
            200, headers={"content-encoding": "aws-chunked"}, stream=httpx.ByteStream(body)
        )
    ).fetch("https://www.sberbank.ru/document.pdf")
    assert result["error"] == "unsupported_aws_chunked_body"
    assert result["body"] == b""
    assert result["sha256"] is None
    assert result["encoding_note"] is None
    assert result["declared_content_encoding"] == "aws-chunked"


@pytest.mark.parametrize(
    "limits",
    [
        {"max_bytes": 10, "max_compressed_bytes": 100},
        {"max_bytes": 100, "max_compressed_bytes": 10},
    ],
)
def test_aws_chunked_bare_pdf_still_obeys_both_stream_limits(limits):
    result = public_fetcher(
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "aws-chunked"},
            stream=httpx.ByteStream(b"%PDF-1.5\r\nlarge source document"),
        ),
        **limits,
    ).fetch("https://sber.ru/document.pdf")
    assert result["error"] == "body_too_large"
    assert result["body"] == b""
    assert result["sha256"] is None
    assert result["encoding_note"] is None


def test_aws_chunked_compatibility_does_not_follow_redirect_to_other_host():
    def handler(request):
        if request.headers["Host"] == "sber.ru":
            return httpx.Response(302, headers={"location": "https://other.sber.ru/file.pdf"})
        return httpx.Response(
            200, headers={"content-encoding": "aws-chunked"}, stream=httpx.ByteStream(b"%PDF-1.5")
        )

    result = public_fetcher(handler).fetch("https://sber.ru/document.pdf")
    assert result["error"] == "unsupported_content_encoding"
    assert result["body"] == b""
    assert result["encoding_note"] is None
