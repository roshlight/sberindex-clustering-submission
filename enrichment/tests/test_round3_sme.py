"""Historical-date, source-scope and bounded-download guards for SME collection."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from openpyxl import Workbook

SPEC = importlib.util.spec_from_file_location(
    "round3_sme", Path(__file__).resolve().parents[1] / "scripts" / "collect_round3_sme.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_official_hosts_are_exact_domains_not_similar_suffixes():
    assert MOD.valid_official_url("https://rmsp.nalog.ru/statistics.html")
    assert not MOD.valid_official_url("https://nalog.ru.evil.example/test")
    assert not MOD.valid_official_url("http://rosstat.gov.ru/file")
    assert not MOD.valid_official_url("https://user:pass@rosstat.gov.ru/file")


def test_oversized_response_is_not_saved(tmp_path):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-length": "2000000000"})
    )
    item = {"source_id": "large", "url": "https://file.nalog.ru/archive.zip"}
    with httpx.Client(transport=transport) as client:
        got = MOD.fetch_one(client, item, tmp_path, 100)
    assert got["access_status"] == "not_downloaded_exceeds_byte_limit"
    assert not (tmp_path / "large.raw").exists()


def test_cached_response_is_verified_and_changed_plan_rejected(tmp_path):
    item = {"source_id": "sample", "url": "https://rosstat.gov.ru/file"}
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"abc"))
    ) as client:
        MOD.fetch_one(client, item, tmp_path, 100)
        changed = {**item, "url": "https://rosstat.gov.ru/other"}
        with pytest.raises(ValueError, match="Changed existing source plan"):
            MOD.fetch_one(client, changed, tmp_path, 100)
        (tmp_path / "sample.raw").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            MOD.fetch_one(client, item, tmp_path, 100)


@pytest.mark.parametrize("date,expected", [("10.06.2024", 2022), ("10.07.2024", 2023)])
def test_fns_employee_year_is_distinct_from_snapshot_year(tmp_path, date, expected):
    path = tmp_path / "data.json"
    row = {
        "stat_date": date + " 00:00:00",
        "cnt_total": 5,
        "cnt_ul_total": 3,
        "cnt_ip_total": 2,
        "cnt_worker_total": 12,
        "cnt_worker_ul_total": 10,
        "cnt_worker_ip_total": 2,
        "cnt_name": "Region",
        "cnt_level": 0,
        "ssrf": "77",
    }
    path.write_text(json.dumps([row]), "utf-8")
    source = {
        "request": {
            "source_id": "stat",
            "url": "https://rmsp.nalog.ru/statistics-proc.json",
            "data": {"statDate": date},
        },
        "sha256": "a" * 64,
    }
    got = MOD.fns_statistics(path, source)
    assert got[0]["employee_reference_year"] == expected
    assert got[0]["territory_id"] is None
    row["stat_date"] = "10.09.2026 00:00:00"
    path.write_text(json.dumps([row]), "utf-8")
    with pytest.raises(ValueError, match="historical snapshot"):
        MOD.fns_statistics(path, source)


def test_municipal_counts_keep_source_labels_markers_and_year(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "T25"
    sheet.append(["Группировка малых предприятий в 2024 году"])
    sheet.append([])
    sheet.append([None, "Количество, единиц"])
    sheet.append([None, None, None, "количество"])
    sheet.append(["город Example 3"])
    sheet.append(["ВСЕГО", 5, 100, 5, 100, 100, "-"])
    sheet.append(["Manufacturing", 0, 0, "-", "-", "-", "-"])
    path = tmp_path / "table.xlsx"
    book.save(path)
    source = {
        "request": {"source_id": "table", "url": "https://35.rosstat.gov.ru/file.xlsx"},
        "sha256": "a" * 64,
    }
    got = MOD.vologda_counts(path, 2024, source)
    assert got[0]["source_municipality_label"] == "город Example 3"
    assert got[0]["loss_organizations"] is None
    assert got[0]["loss_raw"] == "-"
    assert got[1]["organizations"] == 0
    assert got[0]["official_oktmo_8"] is None
    with pytest.raises(ValueError, match="title/year/scope"):
        MOD.vologda_counts(path, 2023, source)
