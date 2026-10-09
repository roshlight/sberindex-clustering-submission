import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round2_sector", Path(__file__).parents[1] / "scripts/run_round2_sector.py"
)
sector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sector)
sys.modules["run_round2_sector"] = sector
BULK_SPEC = importlib.util.spec_from_file_location(
    "round2_sector_bulk", Path(__file__).parents[1] / "scripts/run_round2_sector_bulk.py"
)
bulk = importlib.util.module_from_spec(BULK_SPEC)
BULK_SPEC.loader.exec_module(bulk)
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "round2_sector_audit", Path(__file__).parents[1] / "scripts/audit_round2_sector.py"
)
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)
CAPTION_SPEC = importlib.util.spec_from_file_location(
    "round2_sector_caption", Path(__file__).parents[1] / "scripts/review_round2_sector_caption.py"
)
caption = importlib.util.module_from_spec(CAPTION_SPEC)
CAPTION_SPEC.loader.exec_module(caption)


@pytest.fixture
def options():
    return {
        "Pokazateli": {"8423005": "Среднесписочная численность, человек"},
        "munr": {"1": "Район"},
        "tippos": {"10": "Муниципальный район"},
        "oktmo": {"12345678": "Точный МО", "22222222": "Другой МО"},
        "okved2": {"101": "Всего", "45000": "Торговля; ремонт", "85000": "Образование"},
        "god": {"2023": "2023", "2024": "2024"},
        "period": {"79": "январь-декабрь"},
    }


def document(rows, municipality="Точный МО"):
    return (
        "Среднесписочная численность, человек, Район, Муниципальный район, "
        + municipality
        + ", январь-декабрь\r\r\n;2023;2024;\r\r\n"
        + rows
    )


def test_embedded_semicolon_blank_and_absent_are_distinct(options):
    rows = sector.parse_sector_csv(
        document("Всего;100;110;\nТорговля; ремонт;0;;\n"), options, "12345678"
    )
    lookup = {(r["source_activity_code"], r["year"]): r for r in rows}
    assert lookup["45000", "2023"]["value"] == "0"
    assert lookup["45000", "2023"]["status"] == "observed"
    assert lookup["45000", "2024"]["status"] == "source_marker_unresolved"
    assert lookup["85000", "2024"]["status"] == "not_returned"


def test_exact_municipality_context_required(options):
    with pytest.raises(ValueError, match="context"):
        sector.parse_sector_csv(document("Всего;1;2;", "Другой МО"), options, "12345678")


def test_duplicate_and_unknown_activity_rejected(options):
    with pytest.raises(ValueError, match="Duplicate"):
        sector.parse_sector_csv(document("Всего;1;2;\nВсего;3;4;"), options, "12345678")
    with pytest.raises(ValueError, match="Unknown complete"):
        sector.parse_sector_csv(document("Торг;1;2;"), options, "12345678")


def test_unknown_markers_and_years_rejected(options):
    with pytest.raises(ValueError, match="Unknown numerical"):
        sector.parse_sector_csv(document("Всего;тайна;2;"), options, "12345678")
    with pytest.raises(ValueError, match="year column"):
        sector.parse_sector_csv(document("Всего;1;2;").replace("2024", "2025"), options, "12345678")


def test_duplicate_source_activity_labels_rejected(options):
    bad = copy.deepcopy(options)
    bad["okved2"]["999"] = "Образование"
    with pytest.raises(ValueError, match="unique"):
        sector.sector_payload(bad, "12345678")


def test_payload_preserves_internal_keys_and_annual_period(options):
    payload = sector.sector_payload(options, "12345678")
    assert "okved2:101,45000,85000;" in payload["Qry"]
    assert "period:79;" in payload["Qry"]
    assert "oktmo:12345678;" in payload["Qry"]


def batch_document(activity, rows):
    return (
        "Среднесписочная численность, человек, Район, Муниципальный район, "
        + activity
        + ", январь-декабрь\r\r\n;2023;2024;\r\r\n"
        + rows
        + "\n"
    )


def test_batch_full_activity_and_municipality_keys(options):
    doc = batch_document("Всего", "Точный МО;100;120;\nДругой МО;200;210;")
    doc += batch_document("Торговля; ремонт", "Точный МО;3;4;\nДругой МО;5;;")
    records = bulk.parse_batch(doc, options, ["12345678", "22222222"])
    lookup = {(r["source_oktmo"], r["okved2_code"], r["year"]): r for r in records}
    assert lookup["12345678", "45000", "2023"]["value"] == "3"
    assert lookup["22222222", "45000", "2024"]["value_status"] == "unresolved_source_marker"
    assert lookup["22222222", "85000", "2024"]["value_status"] == "not_returned"
    assert len(records) == 12


def test_batch_rejects_repeated_sections_or_truncated_labels(options):
    doc = batch_document("Всего", "Точный МО;100;120;")
    with pytest.raises(ValueError, match="Duplicate"):
        bulk.parse_batch(doc + doc, options, ["12345678"])
    with pytest.raises(ValueError, match="caption"):
        bulk.parse_batch(doc.replace("Точный МО;", "Точный;"), options, ["12345678"])


def test_batch_rejects_ambiguous_entire_form_even_one_selected(options):
    bad = copy.deepcopy(options)
    bad["oktmo"]["99999999"] = "Точный МО"
    with pytest.raises(ValueError, match="unique"):
        bulk.payload_for(bad, ["12345678"])


def test_batch_rejects_wrong_parent_and_wrong_activity(options):
    doc = batch_document("Всего", "Точный МО;100;120;")
    with pytest.raises(ValueError, match="context"):
        bulk.parse_batch(doc.replace(", Район,", ", Чужой район,"), options, ["12345678"])
    with pytest.raises(ValueError, match="context"):
        bulk.parse_batch(doc.replace("Всего, январь", "Другое, январь"), options, ["12345678"])


def test_global_request_pacing_and_backoff_without_real_sleep():
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    pacer = bulk.StartPacer(clock=lambda: clock[0], sleeper=sleep)
    starts = []
    for _ in range(3):
        pacer.start()
        starts.append(clock[0])
    assert starts == [0.0, 2.5, 5.0]
    pacer.defer(30)
    pacer.start()
    assert clock[0] == 35


def test_wrapped_context_with_semicolon_is_verified_and_locator_preserved(options):
    doc = batch_document("Торговля; ремонт", "Точный МО;4;5;")
    doc = doc.replace("Торговля; ремонт", "Торговля;\nремонт")
    records = bulk.parse_batch(doc, options, ["12345678"])
    row = next(r for r in records if r["okved2_code"] == "45000" and r["year"] == "2023")
    assert row["value"] == "4"
    assert (
        row["source_csv_row"]
        == doc.replace("\r\r\n", "\n").splitlines().index("Точный МО;4;5;") + 1
    )


def test_null_without_a_reference_row_is_not_a_reconciliation_match():
    frame = pd.DataFrame(
        [
            {
                "territory_id": "1",
                "year": "2024",
                "indicator_id": "8423005",
                "source_activity_code": "101",
                "value": float("nan"),
            }
        ]
    )
    old = pd.DataFrame(columns=["territory_id", "year", "indicator_id", "value"])
    result = audit.reconcile(frame, old)
    assert result["unmatched_reference_rows"] == 1
    assert result["exact_value_or_null_matches"] == 0
    assert len(result["mismatches"]) == 1


def test_caption_exception_requires_exact_capture_hash_and_complete_name(options):
    options = copy.deepcopy(options)
    options["oktmo"] = {caption.CODE: "Полный..."}
    body = document("Всего;4;5;", "Полный МО").encode("cp1251")
    decision = {
        "source_oktmo": caption.CODE,
        "indicator_id": "8423005",
        "source_csv_sha256": hashlib.sha256(body).hexdigest(),
        "source_form_sha256": "form-sha",
        "source_option_caption": "Полный...",
        "approved_csv_municipality_caption": "Полный МО",
        "decision": "accept_exact_single_code_and_reviewed_full_caption",
    }
    rows = caption.apply_caption_decision(options, body, decision, "form-sha", "8423005")
    assert (
        next(r for r in rows if r["year"] == "2023" and r["source_activity_code"] == "101")["value"]
        == "4"
    )
    with pytest.raises(ValueError, match="exact CSV"):
        caption.apply_caption_decision(options, body + b" ", decision, "form-sha", "8423005")
    wrong = {**decision, "approved_csv_municipality_caption": "Другой МО"}
    with pytest.raises(ValueError, match="context"):
        caption.apply_caption_decision(options, body, wrong, "form-sha", "8423005")


def test_canonical_reader_preserves_padded_official_code_and_raw_decimals(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text(
        "territory_id,year,indicator_id,source_oktmo,official_oktmo_8,"
        "okved2_code,value,value_raw\n1,2024,8423005,3701000,03701000,101,1.20,1.20\n",
        encoding="utf-8",
    )
    row = audit.canonical_frame(path, bulk=True).iloc[0]
    assert row.official_oktmo_8 == "03701000"
    assert row.source_oktmo == "3701000"
    assert row.value_raw == "1.20"
