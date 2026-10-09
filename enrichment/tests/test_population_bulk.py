import json

import httpx
import pytest

from sber_enrichment.bdpmo import parse_population_csv
from sber_enrichment.population_bulk import (
    NetworkPause,
    batch_payload,
    parse_batch_csv,
    run_bulk,
    safe_groups,
    with_backoff,
)

OPTIONS = {
    "oktmo": [
        {"code": "79701000", "label": "Город Майкоп"},
        {"code": "79703000", "label": "Город Адыгейск"},
    ],
    "munr": [{"code": "79700000", "label": "Городские округа"}],
    "tippos": [{"code": "7", "label": "Городской округ"}],
}
TITLE = "Оценка численности населения на 1 января текущего года, человек"
BATCH = f"""{TITLE}
;2023;2024;
Город Майкоп;;;
Все население;163766;161898;
Городское население;139687;137965;
Город Адыгейск;;;
Все население;15614;15661;
Городское население;13191;13224;
"""
URL = "https://rosstat.gov.ru/dbscripts/munst/munst79/"


def test_real_layout_batch_is_bound_to_exact_source_options():
    rows = parse_batch_csv(BATCH, OPTIONS, ["79701000", "79703000"])
    assert len(rows["79701000"]) == 4
    assert rows["79703000"][0]["value"] == 15614
    assert rows["79703000"][0]["source_csv_row"] == 7
    payload = batch_payload(OPTIONS, ["79701000", "79703000"])
    assert "oktmo:79701000,79703000;" in payload["Qry"]


def test_duplicate_label_anywhere_in_regional_form_requires_single_code():
    options = {
        **OPTIONS,
        "oktmo": OPTIONS["oktmo"] + [{"code": "99999999", "label": "Город Майкоп"}],
    }
    assert safe_groups(options, ["79701000", "79703000"], 20) == [["79703000"], ["79701000"]]
    with pytest.raises(ValueError, match="unique"):
        parse_batch_csv(BATCH, options, ["79701000", "79703000"])


@pytest.mark.parametrize(
    "document",
    [
        BATCH.replace("Город Майкоп;;;", "Майкоп;;;"),
        BATCH.replace("Город Адыгейск;;;", "Город Майкоп;;;"),
        BATCH + "Все население;1;2;\n",
        BATCH.split("Город Адыгейск")[0],
    ],
)
def test_uncertain_or_duplicate_batch_is_rejected(document):
    with pytest.raises(ValueError):
        parse_batch_csv(document, OPTIONS, ["79701000", "79703000"])


def test_partial_year_does_not_discard_valid_population_or_invent_other_year():
    document = f"{TITLE}\n;2024;\nВсе население;100;\nГородское население;-;\n"
    rows = parse_population_csv(document, require_complete=False)
    assert len(rows) == 2
    assert {row["reference_date"] for row in rows} == {"2024-01-01"}
    assert rows[1]["value"] is None
    with pytest.raises(ValueError):
        parse_population_csv(document)


def test_batch_can_keep_one_year_when_source_splits_territorial_context():
    document = f"""{TITLE}
;2024;
Город Майкоп;;
Все население;100;
Городское население;50;
{TITLE}
;2023;
Город Адыгейск;;
Все население;90;
Городское население;40;
"""
    rows = parse_batch_csv(document, OPTIONS, ["79701000", "79703000"])
    assert len(rows["79701000"]) == 2
    assert rows["79701000"][0]["reference_date"] == "2024-01-01"
    assert rows["79703000"][0]["reference_date"] == "2023-01-01"


def test_429_backoff_respects_retry_after_and_403_pauses_immediately():
    delays, calls = [], []
    request = httpx.Request("GET", URL)

    def transient():
        calls.append(1)
        if len(calls) == 1:
            response = httpx.Response(429, headers={"retry-after": "12"}, request=request)
            response.raise_for_status()
        return "accepted"

    assert with_backoff(transient, sleep=delays.append) == "accepted"
    assert delays == [12]

    def forbidden():
        httpx.Response(403, request=request).raise_for_status()

    with pytest.raises(NetworkPause, match="403"):
        with_backoff(forbidden, sleep=delays.append)
    assert delays == [12]


def test_durable_resume_keeps_completed_targets_and_reports_missing_cells(tmp_path):
    requested = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def options(self, url):
            return OPTIONS, {"request_key": "official-options-hash"}

        def request(self, method, url, payload):
            code = "79701000" if "oktmo:79701000;" in payload["Qry"] else "79703000"
            requested.append(code)
            label = next(row["label"] for row in OPTIONS["oktmo"] if row["code"] == code)
            raw = (
                f"{TITLE}, {label}, на 1 января\n;2024;\n"
                "Все население;100;\nГородское население;0;\n"
            ).encode("cp1251")
            return raw, {
                "url": url,
                "raw_sha256": "raw-hash",
                "request_key": code,
                "retrieved_at": "2026-09-15T00:00:00+00:00",
            }

    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            [
                {
                    "region_url": URL,
                    "oktmo": code,
                    "reference_links": [{"reference_date": "2024-01-01", "join_eligible": False}],
                }
                for code in ["79701000", "79703000"]
            ]
        )
    )
    output = tmp_path / "output"
    first = run_bulk(
        targets, output, tmp_path / "cache", limit=1, batch_size=1, client_factory=FakeClient
    )
    assert first["status"] == "checkpointed"
    assert first["record_count"] == 2
    second = run_bulk(
        targets, output, tmp_path / "cache", resume=True, batch_size=1, client_factory=FakeClient
    )
    assert second["status"] == "complete_with_gaps"
    assert second["status_counts"] == {"partial_source": 2}
    assert requested == ["79701000", "79703000"]
    assert second["record_count"] == 4
    items = [json.loads(path.read_text()) for path in (output / "items").glob("*.json")]
    assert all(len(item["missing_cells"]) == 2 for item in items)
    assert all(row["territory_id"] is None for item in items for row in item["records"])
    with pytest.raises(ValueError, match="changed"):
        run_bulk(
            targets,
            output,
            tmp_path / "cache",
            resume=True,
            batch_size=2,
            client_factory=FakeClient,
        )
