"""Independent regressions for reviewed code/year attribution and durable fallback."""

import hashlib
import json

import httpx
import pandas as pd
import pytest

from sber_enrichment.bdpmo import BdpmoClient, parse_population_csv
from sber_enrichment.population_bulk import (
    AttributionReview,
    fetch_group,
    parse_batch_csv,
    run_bulk,
)

URL = "https://rosstat.gov.ru/dbscripts/munst/munst79/"
TITLE = "Оценка численности населения на 1 января текущего года, человек"
PERIOD = "на 1 января"
CODES = ["79701000", "79703000"]
NAMES = ["Город Майкоп", "Город Адыгейск"]
OPTIONS = {
    "Pokazateli": [{"code": "8112027", "label": TITLE}],
    "oktmo": [{"code": code, "label": name} for code, name in zip(CODES, NAMES, strict=True)],
    "munr": [{"code": "79700000", "label": "Городские округа"}],
    "tippos": [{"code": "7", "label": "Городской округ"}],
    "period": [{"code": "208", "label": PERIOD}],
}


def singleton(name, *, year="2024", total="100", urban="-"):
    return (
        f"{TITLE}, {name}, {PERIOD}\n;{year};\n"
        f"Все население;{total};\nГородское население;{urban};\n"
    )


def metadata(raw):
    return {
        "url": URL + "DBInet.cgi",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "request_key": "test-request",
        "retrieved_at": "2026-09-15T00:00:00+00:00",
    }


@pytest.mark.parametrize("parser", ["single", "batch"])
@pytest.mark.parametrize("header", [";2023;;2024;", ";;2024;"])
def test_blank_year_columns_cannot_relabel_an_unheaded_value(parser, header):
    body = "Все население;100;200;;\nГородское население;50;60;;\n"
    if parser == "single":
        document = f"{TITLE}\n{header}\n{body}"
        with pytest.raises(ValueError, match="[Bb]lank.*year"):
            parse_population_csv(document, require_complete=False)
    else:
        document = f"{TITLE}\n{header}\n" + "".join(f"{name};;;;\n{body}" for name in NAMES)
        with pytest.raises(ValueError, match="[Bb]lank.*year"):
            parse_batch_csv(document, OPTIONS, CODES)


def test_reordered_explicit_years_preserve_values_and_nulls():
    document = f"{TITLE}\n;2024;2023;;;\nВсе население;100;200;;;\nГородское население;-;60;;;\n"
    rows = parse_population_csv(document)
    cells = {(r["reference_date"], r["population_type"]): r["value"] for r in rows}
    assert cells == {
        ("2024-01-01", "11"): 100,
        ("2023-01-01", "11"): 200,
        ("2024-01-01", "12"): None,
        ("2023-01-01", "12"): 60,
    }


@pytest.mark.parametrize("prefix", ["", singleton(NAMES[0], year="2023")])
def test_foreign_singleton_caption_is_rejected_even_after_a_correct_section(prefix):
    class FakeClient:
        def request(self, *args, **kwargs):
            raw = (prefix + singleton(NAMES[1], total="15661")).encode("cp1251")
            return raw, metadata(raw)

    with pytest.raises(AttributionReview, match="municipality caption"):
        fetch_group(FakeClient(), URL, OPTIONS, {"request_key": "options"}, [CODES[0]])


def test_singleton_exact_caption_keeps_missing_urban_and_does_not_invent_year():
    class FakeClient:
        def request(self, *args, **kwargs):
            raw = singleton(NAMES[0]).encode("cp1251")
            return raw, metadata(raw)

    rows = fetch_group(FakeClient(), URL, OPTIONS, {"request_key": "options"}, [CODES[0]])[CODES[0]]
    assert {r["source_oktmo"] for r in rows} == {CODES[0]}
    assert {r["reference_date"] for r in rows} == {"2024-01-01"}
    assert [r["value"] for r in rows] == [100, None]
    assert all(r["territory_id"] is None for r in rows)


def test_unknown_nonempty_section_cannot_supply_population_cells():
    document = singleton(NAMES[0]).replace(
        "Все население;100;", "Чужой муниципалитет;\nВсе население;100;"
    )
    with pytest.raises(ValueError, match="Unknown nonempty CSV row"):
        parse_population_csv(document, require_complete=False)


def write_targets(path, codes=CODES):
    path.write_text(
        json.dumps([{"region_url": URL, "oktmo": code} for code in codes]), encoding="utf-8"
    )


def test_successful_fallback_is_checkpointed_before_later_403(tmp_path):
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def options(self, *args, **kwargs):
            return OPTIONS, {"request_key": "options"}

        def request(self, method, url, payload, **kwargs):
            calls.append(payload["Qry"])
            if f"oktmo:{','.join(CODES)};" in payload["Qry"]:
                raw = b"not a CSV batch"
            elif f"oktmo:{CODES[0]};" in payload["Qry"]:
                raw = singleton(NAMES[0]).encode("cp1251")
            else:
                httpx.Response(403, request=httpx.Request(method, url)).raise_for_status()
            return raw, metadata(raw)

    target, output = tmp_path / "targets.json", tmp_path / "out"
    write_targets(target)
    result = run_bulk(target, output, tmp_path / "cache", client_factory=FakeClient)
    assert result["status"] == "paused_network"
    assert len(calls) == 3
    assert result["record_count"] == 2
    items = list((output / "items").glob("*.json"))
    assert len(items) == 1
    item = json.loads(items[0].read_text(encoding="utf-8"))
    assert item["target"]["oktmo"] == CODES[0]
    assert item["records"][1]["value"] is None
    exported = pd.read_csv(output / "population_source.csv", dtype={"source_oktmo": str})
    assert set(exported.source_oktmo) == {CODES[0]}
    assert len(exported) == 2


def test_retry_failed_requests_fresh_csv_and_keeps_previously_failed_target(tmp_path):
    refresh_flags = []
    option_refresh_flags = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def options(self, *args, refresh=False):
            option_refresh_flags.append(refresh)
            return OPTIONS, {"request_key": "options"}

        def request(self, method, url, payload, *, refresh=False):
            refresh_flags.append(refresh)
            raw = singleton(NAMES[0]).encode("cp1251") if refresh else b"<html>temporary</html>"
            return raw, metadata(raw)

    target, output = tmp_path / "targets.json", tmp_path / "out"
    write_targets(target, CODES[:1])
    first = run_bulk(target, output, tmp_path / "cache", client_factory=FakeClient)
    assert first["record_count"] == 0
    second = run_bulk(
        target,
        output,
        tmp_path / "cache",
        resume=True,
        retry_failed=True,
        client_factory=FakeClient,
    )
    assert refresh_flags == [False, True]
    assert option_refresh_flags == [False, True]
    assert second["record_count"] == 2
    assert second["status_counts"] == {"partial_source": 1}


def test_explicit_cache_refresh_preserves_old_http200_body_and_metadata(tmp_path):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, content=b"old invalid CSV" if len(seen) == 1 else b"new CSV")

    client = BdpmoClient(tmp_path / "cache", interval=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=False)
    try:
        old_raw, old_metadata = client.request("POST", URL + "DBInet.cgi", {"Qry": "test"})
        cached_raw, _ = client.request("POST", URL + "DBInet.cgi", {"Qry": "test"})
        assert cached_raw == old_raw
        new_raw, _ = client.request("POST", URL + "DBInet.cgi", {"Qry": "test"}, refresh=True)
        assert new_raw == b"new CSV"
        assert len(seen) == 2
        history = tmp_path / "cache" / "history" / old_metadata["request_key"]
        assert [p.read_bytes() for p in history.glob("*.raw")] == [old_raw]
        saved = [json.loads(p.read_text(encoding="utf-8")) for p in history.glob("*.json")]
        assert saved == [old_metadata]
    finally:
        client.close()


def test_regional_options_refresh_bypasses_memory_and_post_cache(tmp_path, monkeypatch):
    requests = []
    client = BdpmoClient(tmp_path / "cache", interval=0)

    def fake_request(method, url, payload=None, *, refresh=False):
        requests.append((method, url, payload, refresh))
        return b"/dbscripts/munst/munst79/DBInet.cgi", {"request_key": "fresh-options"}

    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr("sber_enrichment.bdpmo.parse_options", lambda document: OPTIONS)
    client._options[URL] = ({"stale": True}, {"request_key": "old-options"})
    try:
        result = client.options(URL, refresh=True)
        assert result == (OPTIONS, {"request_key": "fresh-options"})
        assert requests == [
            ("POST", URL + "DBInet.cgi", {"pl": "8112027"}, True),
        ]
        assert client.options(URL) == result
        assert len(requests) == 1
    finally:
        client.close()
