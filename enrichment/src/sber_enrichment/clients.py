"""HTTP connectors; no credentials, prompts, or raw provider errors are logged."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import random
import socket
import ssl
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import certifi
import httpx
from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable


class ClientError(RuntimeError):
    """Safe error code plus accounting metadata, never the original HTTP exception."""

    def __init__(self, code, *, status=None, response_id=None, usage=None):
        super().__init__(code)
        self.code = code
        self.status = status
        self.response_id = response_id
        self.usage = usage


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite_json_number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_json_number")
        return number

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float
    )


def _usage(value):
    """Keep only known numeric accounting fields; null is not zero."""
    if not isinstance(value, dict):
        return None
    fields = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "is_byok",
        "reasoning_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "audio_tokens",
        "upstream_inference_cost",
        "upstream_inference_prompt_cost",
        "upstream_inference_completions_cost",
    }
    groups = {"prompt_tokens_details", "completion_tokens_details", "cost_details"}
    result = {}
    for key, item in value.items():
        if key in groups and isinstance(item, dict):
            result[key] = _usage(item)
        elif key in fields and (
            item is None or isinstance(item, (int, float)) and math.isfinite(item)
        ):
            result[key] = item
    return result


def _response_id(body, response):
    value = body.get("id") if isinstance(body, dict) else None
    value = value or response.headers.get("X-Generation-Id")
    return value if isinstance(value, str) else None


def _retry_delay(response, attempt):
    hint = response.headers.get("Retry-After")
    if hint:
        try:
            delay = float(hint)
        except ValueError:
            try:
                date = parsedate_to_datetime(hint)
                date = date.replace(tzinfo=UTC) if date.tzinfo is None else date
                delay = (date - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = 0
        if math.isfinite(delay) and delay > 0:
            return delay
    return 2 ** (attempt - 1) + random.uniform(0, 0.5)


class _JsonClient:
    def __init__(self, *, timeout=90.0, max_attempts=3, transport=None, attempt_log_path=None):
        if timeout <= 0 or not 1 <= max_attempts <= 5:
            raise ValueError("invalid_client_limits")
        self._attempt_lock = threading.Lock()
        self._attempt_log_path = Path(attempt_log_path) if attempt_log_path is not None else None
        self.attempts = []
        if self._attempt_log_path is not None and self._attempt_log_path.exists():
            try:
                previous = _strict_json(self._attempt_log_path.read_text(encoding="utf-8-sig"))
                if not isinstance(previous, list) or any(
                    not isinstance(row, dict) for row in previous
                ):
                    raise ValueError
                self.attempts = previous
            except (OSError, ValueError, UnicodeError):
                raise ClientError("attempt_log_unreadable") from None
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )
        self._max_attempts = max_attempts

    def _checkpoint_locked(self):
        """Single-writer process: readers see either the previous or complete new snapshot."""
        path = self._attempt_log_path
        if path is None:
            return
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.attempts, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            if os.name == "posix":
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _update_attempt(self, record=None, **changes):
        with self._attempt_lock:
            if record is None:
                record = {}
                self.attempts.append(record)
            record.update(changes, updated_at=datetime.now(UTC).isoformat())
            try:
                self._checkpoint_locked()
            except (OSError, ValueError, TypeError):
                raise ClientError(
                    "attempt_log_write_failed",
                    status=record.get("status"),
                    response_id=record.get("response_id"),
                    usage=record.get("usage"),
                ) from None
            return record

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _request(self, method, url, **kwargs):
        call_id = uuid4().hex
        for attempt in range(1, self._max_attempts + 1):
            record = self._update_attempt(
                attempt_id=uuid4().hex,
                call_id=call_id,
                attempt=attempt,
                method=method.upper(),
                operation=urlsplit(url).path,
                state="started",
                started_at=datetime.now(UTC).isoformat(),
                status=None,
                response_id=None,
                usage=None,
            )
            started = time.monotonic()
            try:
                with self._http.stream(method, url, **kwargs) as response:
                    self._update_attempt(
                        record,
                        state="headers_received",
                        status=response.status_code,
                        response_id=_response_id(None, response),
                    )
                    response.read()
            except httpx.TimeoutException:
                self._update_attempt(
                    record,
                    state="failed",
                    error="timeout_unknown_outcome",
                    seconds=time.monotonic() - started,
                )
                raise ClientError(
                    "timeout_unknown_outcome",
                    status=record["status"],
                    response_id=record["response_id"],
                ) from None
            except httpx.RequestError:
                self._update_attempt(
                    record,
                    state="failed",
                    error="network_error",
                    seconds=time.monotonic() - started,
                )
                raise ClientError(
                    "network_error", status=record["status"], response_id=record["response_id"]
                ) from None
            try:
                body = _strict_json(response.text)
            except (ValueError, UnicodeError):
                body = None
            self._update_attempt(
                record,
                state="response_received",
                status=response.status_code,
                seconds=time.monotonic() - started,
                response_id=_response_id(body, response),
                usage=_usage(body.get("usage")) if isinstance(body, dict) else None,
                model=body.get("model") if isinstance(body, dict) else None,
            )
            retryable = response.status_code == 429 or 500 <= response.status_code <= 599
            if retryable and attempt < self._max_attempts:
                delay = _retry_delay(response, attempt)
                if delay > 60:
                    self._update_attempt(
                        record, state="failed", error="retry_deferred", retry_after_seconds=delay
                    )
                    raise ClientError(
                        "retry_deferred",
                        status=response.status_code,
                        response_id=record["response_id"],
                        usage=record["usage"],
                    )
                self._update_attempt(
                    record,
                    state="retry_scheduled",
                    error="retryable_http_error",
                    retry_after_seconds=delay,
                )
                time.sleep(delay)
                continue
            if not response.is_success:
                self._update_attempt(record, state="failed", error="http_error")
                raise ClientError(
                    "http_error",
                    status=response.status_code,
                    response_id=record["response_id"],
                    usage=record["usage"],
                )
            if not isinstance(body, dict):
                self._update_attempt(record, state="failed", error="invalid_json_response")
                raise ClientError(
                    "invalid_json_response",
                    status=response.status_code,
                    response_id=record["response_id"],
                    usage=record["usage"],
                )
            if "error" in body:
                self._update_attempt(record, state="failed", error="provider_error")
                raise ClientError(
                    "provider_error",
                    status=response.status_code,
                    response_id=record["response_id"],
                    usage=record["usage"],
                )
            self._update_attempt(record, state="http_complete")
            return body, record
        raise AssertionError("unreachable")


class SearxClient(_JsonClient):
    def __init__(self, base_url, *, timeout=45.0, max_attempts=3, transport=None):
        try:
            parsed = httpx.URL(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo:
                raise ValueError
            if parsed.query or parsed.fragment:
                raise ValueError
        except (httpx.InvalidURL, ValueError):
            raise ValueError("invalid_searx_base_url") from None
        super().__init__(timeout=timeout, max_attempts=max_attempts, transport=transport)
        self._base = str(parsed).rstrip("/")
        self._config = None

    def config(self) -> dict:
        body, _record = self._request(
            "GET", f"{self._base}/config", headers={"Accept": "application/json"}
        )
        if not isinstance(body.get("engines"), list):
            raise ClientError("invalid_searx_config")
        self._config = body
        return body

    def search(self, query, engines, page=1, language="ru-RU") -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("empty_search_query")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("invalid_search_page")
        selected = engines.split(",") if isinstance(engines, str) else list(engines)
        if not selected or any(not isinstance(name, str) or not name.strip() for name in selected):
            raise ValueError("empty_search_engines")
        selected = list(dict.fromkeys(name.strip() for name in selected))
        config = self._config or self.config()
        available = {
            item.get("name")
            for item in config["engines"]
            if isinstance(item, dict) and item.get("enabled") is True
        }
        if any(name not in available for name in selected):
            raise ClientError("unknown_or_disabled_engine")
        body, _record = self._request(
            "POST",
            f"{self._base}/search",
            headers={"Accept": "application/json"},
            data={
                "q": query,
                "engines": ",".join(selected),
                "pageno": page,
                "language": language,
                "format": "json",
            },
        )
        if not isinstance(body.get("results"), list):
            raise ClientError("invalid_search_results")
        return body


def _load_key(api_key_file):
    key_file = api_key_file or os.environ.get("OPENROUTER_API_KEY_FILE")
    if key_file:
        try:
            with Path(key_file).open("r", encoding="utf-8") as handle:
                key = handle.read(8193).strip()
        except (OSError, UnicodeError, ValueError):
            raise ClientError("api_key_file_unreadable") from None
    else:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or len(key) > 8192 or any(not "!" <= char <= "~" for char in key):
        raise ClientError("api_key_missing_or_invalid")
    return key


def _schema_validator(schema):
    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise ClientError("external_schema_reference")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)

    check_refs(schema)
    try:
        cls = validator_for(schema)
        cls.check_schema(schema)
        return cls(schema, format_checker=FormatChecker(), registry=Registry())
    except (SchemaError, TypeError, ValueError):
        raise ClientError("invalid_json_schema") from None


class OpenRouterClient(_JsonClient):
    _BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key_file=None,
        *,
        timeout=180.0,
        max_attempts=3,
        transport=None,
        attempt_log_path=None,
    ):
        key = _load_key(api_key_file)
        super().__init__(
            timeout=timeout,
            max_attempts=max_attempts,
            transport=transport,
            attempt_log_path=attempt_log_path,
        )
        self._key = key

    def _headers(self):
        return {"Authorization": f"Bearer {self._key}", "Accept": "application/json"}

    def models(self) -> dict:
        body, record = self._request("GET", f"{self._BASE}/models", headers=self._headers())
        if not isinstance(body.get("data"), list):
            self._update_attempt(record, state="failed", error="invalid_model_catalog")
            raise ClientError("invalid_model_catalog")
        self._update_attempt(record, state="complete")
        return body

    def complete_json(self, model, system, user, schema, max_tokens=6000) -> dict:
        validator = _schema_validator(schema)
        if not all(isinstance(value, str) and value.strip() for value in (model, system, user)):
            raise ValueError("empty_completion_argument")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("invalid_max_tokens")
        body, record = self._request(
            "POST",
            f"{self._BASE}/chat/completions",
            headers=self._headers(),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "max_tokens": max_tokens,
                "provider": {"require_parameters": True},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "municipal_evidence", "strict": True, "schema": schema},
                },
            },
        )

        def fail(code):
            self._update_attempt(record, state="failed", error=code)
            raise ClientError(
                code,
                status=record["status"],
                response_id=record["response_id"],
                usage=record["usage"],
            ) from None

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            fail("missing_completion")
        choice = choices[0]
        finish = choice.get("finish_reason")
        self._update_attempt(record, finish_reason=finish if isinstance(finish, str) else None)
        message = choice.get("message")
        if choice.get("error") is not None or finish == "error":
            fail("provider_error")
        if not isinstance(message, dict):
            fail("missing_completion")
        if message.get("refusal") or finish == "content_filter":
            fail("model_refusal")
        if finish == "length":
            fail("truncated_completion")
        if finish != "stop":
            fail("unexpected_finish_reason")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            fail("empty_completion")
        try:
            parsed = _strict_json(content)
        except (ValueError, UnicodeError):
            fail("invalid_completion_json")
        try:
            validator.validate(parsed)
        except (ValidationError, Unresolvable, RecursionError):
            fail("schema_validation_failed")
        self._update_attempt(record, state="complete", model=body.get("model"))
        return {
            "parsed": parsed,
            "response_id": record["response_id"],
            "usage": record["usage"],
            "model": body.get("model"),
            "finish_reason": finish,
        }


def _public_url(value):
    if not isinstance(value, str) or any(ord(char) <= 32 for char in value) or "\\" in value:
        raise ClientError("blocked_url")
    try:
        split = urlsplit(value)
        url = httpx.URL(value)
        if (
            split.scheme not in {"http", "https"}
            or not split.netloc
            or not url.host
            or url.userinfo
            or "%" in split.netloc
        ):
            raise ValueError
        host = url.host.rstrip(".").lower()
        if "." not in host or host.endswith((".localhost", ".local", ".internal", ".home.arpa")):
            raise ValueError
        if url.port not in {None, 80, 443}:
            raise ValueError
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError
        return url.copy_with(host=host, fragment=None)
    except (ValueError, httpx.InvalidURL):
        raise ClientError("blocked_url") from None


def _resolve_public(host, port):
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ClientError("dns_error") from None
    addresses = sorted({item[4][0] for item in records})
    if not addresses:
        raise ClientError("dns_error")
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise ClientError("blocked_dns_address") from None
        if not address.is_global or address.is_multicast or address.is_reserved:
            raise ClientError("blocked_dns_address")
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            raise ClientError("blocked_dns_address")
    return addresses


class PublicFetcher:
    """Fetch public documents, pinning DNS and checking every redirect before I/O.

    Identity and gzip bodies are streamed with separate wire and decoded limits.
    HTTPX automatic decompression is never used for untrusted source bodies.
    """

    def __init__(
        self,
        timeout=30.0,
        max_bytes=8_000_000,
        *,
        transport=None,
        resolver: Callable[[str, int], Sequence[str]] | None = None,
        max_redirects=5,
        ca_bundles: Mapping[str, str | Path] | None = None,
        max_compressed_bytes=None,
    ):
        if max_compressed_bytes is None:
            max_compressed_bytes = max_bytes
        if (
            timeout <= 0
            or max_bytes < 1
            or max_compressed_bytes < 1
            or not 0 <= max_redirects <= 10
        ):
            raise ValueError("invalid_fetch_limits")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_compressed_bytes = max_compressed_bytes
        self._transport = transport
        self._resolver = resolver or _resolve_public
        self.max_redirects = max_redirects
        self._tls_contexts = {}
        for hostname, bundle in (ca_bundles or {}).items():
            try:
                candidate = _public_url(f"https://{hostname}/")
                canonical = candidate.raw_host.decode("ascii")
                if (
                    not isinstance(hostname, str)
                    or hostname.lower() != canonical
                    or any(
                        character not in "abcdefghijklmnopqrstuvwxyz0123456789.-"
                        for character in canonical
                    )
                ):
                    raise ValueError("invalid_ca_hostname")
                context = ssl.create_default_context(cafile=certifi.where())
                # Supplemental intermediates must chain to a trusted root; do not
                # silently promote an intermediate into an independent trust anchor.
                context.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
                context.load_verify_locations(cafile=str(bundle))
                self._tls_contexts[canonical] = context
            except (ClientError, ValueError, TypeError, OSError):
                raise ClientError("invalid_ca_bundle") from None

    def _read_body(self, response, encoding, started):
        chunks = []
        wire_count = decoded_count = completed_members = 0
        decoder = None

        def check_deadline():
            if time.monotonic() - started > self.timeout:
                raise ClientError("fetch_timeout")

        def append(chunk):
            nonlocal decoded_count
            decoded_count += len(chunk)
            if decoded_count > self.max_bytes:
                raise ClientError("body_too_large")
            if chunk:
                chunks.append(chunk)

        for chunk in response.iter_raw(chunk_size=16384):
            check_deadline()
            wire_count += len(chunk)
            if wire_count > self.max_compressed_bytes:
                code = "compressed_body_too_large" if encoding == "gzip" else "body_too_large"
                raise ClientError(code)
            if encoding != "gzip":
                append(chunk)
                continue
            pending = chunk
            drain = False
            while pending or drain:
                check_deadline()
                if decoder is None:
                    decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
                limit = min(65536, self.max_bytes - decoded_count + 1)
                try:
                    output = decoder.decompress(pending, max_length=limit)
                except zlib.error:
                    raise ClientError("invalid_gzip_body") from None
                append(output)
                if decoder.eof:
                    # RFC 1952 permits concatenated members. Check every trailer;
                    # leftover garbage is never silently ignored after a valid member.
                    completed_members += 1
                    pending = decoder.unused_data
                    decoder = None
                    drain = False
                else:
                    pending = decoder.unconsumed_tail
                    drain = len(output) == limit
        check_deadline()
        if encoding == "gzip" and (decoder is not None or not completed_members):
            raise ClientError("truncated_gzip_body")
        return b"".join(chunks)

    def fetch(self, url) -> dict:
        result = {
            "url": "[rejected-url]",
            "final_url": None,
            "status": None,
            "content_type": None,
            "declared_content_encoding": None,
            "encoding_note": None,
            "body": b"",
            "sha256": None,
            "error": None,
        }
        started = time.monotonic()
        try:
            current = _public_url(url)
            result["url"] = str(current)
            seen = set()
            for _hop in range(self.max_redirects + 1):
                if str(current) in seen:
                    raise ClientError("redirect_loop")
                seen.add(str(current))
                result["final_url"] = str(current)
                remaining = self.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ClientError("fetch_timeout")
                hostname = current.raw_host.decode("ascii")
                addresses = self._resolver(
                    hostname, current.port or (443 if current.scheme == "https" else 80)
                )
                if not addresses:
                    raise ClientError("dns_error")
                # Also validate injected resolver results: tests do not bypass the guard.
                for value in addresses:
                    address = ipaddress.ip_address(value)
                    if not address.is_global or address.is_multicast or address.is_reserved:
                        raise ClientError("blocked_dns_address")
                    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                        raise ClientError("blocked_dns_address")
                addresses = sorted(addresses, key=lambda item: ipaddress.ip_address(item).version)
                pinned = current.copy_with(host=addresses[0])
                remaining = self.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ClientError("fetch_timeout")
                headers = {
                    "Host": current.netloc.decode("ascii"),
                    "Accept-Encoding": "identity",
                    "User-Agent": "municipal-profiles-research/0.1",
                }
                # A fresh pool per hop prevents reuse of TLS connections across hostnames
                # that share an IP. SNI and certificate verification use the original host.
                with httpx.Client(
                    transport=self._transport,
                    trust_env=False,
                    verify=self._tls_contexts.get(hostname, True),
                    follow_redirects=False,
                    timeout=httpx.Timeout(remaining),
                ) as client:
                    with client.stream(
                        "GET", pinned, headers=headers, extensions={"sni_hostname": hostname}
                    ) as response:
                        result["status"] = response.status_code
                        result["content_type"] = response.headers.get("content-type")
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise ClientError("redirect_without_location")
                            current = _public_url(urljoin(str(current), location))
                            continue
                        if not response.is_success:
                            raise ClientError("http_error", status=response.status_code)
                        encoding = (
                            response.headers.get("content-encoding", "identity").strip().lower()
                        )
                        if "content-encoding" in response.headers:
                            result["declared_content_encoding"] = encoding
                        bare_pdf_compatibility = encoding == "aws-chunked" and hostname in {
                            "www.sberbank.ru",
                            "sber.ru",
                        }
                        if encoding not in {"", "identity", "gzip"} and not bare_pdf_compatibility:
                            raise ClientError("unsupported_content_encoding")
                        length = response.headers.get("content-length")
                        if length and length.isdigit():
                            if int(length) > self.max_compressed_bytes:
                                code = (
                                    "compressed_body_too_large"
                                    if encoding == "gzip"
                                    else "body_too_large"
                                )
                                raise ClientError(code)
                            if encoding != "gzip" and int(length) > self.max_bytes:
                                raise ClientError("body_too_large")
                        body = self._read_body(response, encoding, started)
                        if bare_pdf_compatibility:
                            # These two source hosts have returned bare PDF bytes with a
                            # stale S3 encoding header. Do not decode actual AWS framing.
                            if not body.startswith(b"%PDF-"):
                                raise ClientError("unsupported_aws_chunked_body")
                            result["encoding_note"] = "declared_aws_chunked_but_received_bare_pdf"
                        result.update(body=body, sha256=hashlib.sha256(body).hexdigest())
                        return result
            raise ClientError("too_many_redirects")
        except ClientError as error:
            result["error"] = error.code
        except httpx.TimeoutException:
            result["error"] = "fetch_timeout"
        except (httpx.RequestError, OSError):
            result["error"] = "fetch_network_error"
        except (ValueError, httpx.InvalidURL):
            result["error"] = "blocked_url"
        return result
