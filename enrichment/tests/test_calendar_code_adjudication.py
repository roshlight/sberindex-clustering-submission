"""A document number or a date alone is not permission for early code application."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/adjudicate_calendar_code_dates.py"
SPEC = importlib.util.spec_from_file_location("calendar_adjudication", SCRIPT)
adjudication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adjudication)


def card(comment, number="598"):
    return (
        f"<h1>Изменение {number}/2023 ОКТМО</h1><div>"
        "<span>Комментарий к дате введения</span><p>" + comment + "</p></div>"
    ).encode()


def test_requires_explicit_early_permission_and_correct_amendment():
    valid = "с правом досрочного применения в правоотношениях, возникших с 1 января 2023 г."
    assert adjudication.validate_card(card(valid), "598") == valid
    with pytest.raises(ValueError, match="Wrong official"):
        adjudication.validate_card(card(valid, number="600"), "598")


@pytest.mark.parametrize("comment", ["-", "01.01.2023", "с правом досрочного применения"])
def test_a_date_or_incomplete_comment_does_not_clear_conflict(comment):
    with pytest.raises(ValueError, match="No explicit permission"):
        adjudication.validate_card(card(comment), "598")


def test_fns_code_date_notice_does_not_become_literal_early_application_permission(tmp_path):
    audit, sources, output = tmp_path / "audit", tmp_path / "sources", tmp_path / "output"
    audit.mkdir()
    sources.mkdir()
    pd.DataFrame(
        [
            {
                "territory_id": "1051",
                "reference_date": "2023-01-01",
                "official_oktmo_8": "19514000",
                "sber_oktmo_11": "19514000000",
                "registry_nomakt": "586",
                "registry_DateUtv": "01.12.2022",
                "registry_DateVved": "16.01.2023",
            }
        ]
    ).to_csv(audit / "temporal_review_2023.csv", index=False)
    (sources / "fns_19514000_web_tool_text.txt").write_text(
        adjudication.FNS_URL + "\nс 01.01.2023 вновь образованы:\nкод ОКТМО 19514000",
        encoding="utf-8",
    )
    adjudication.prepare(audit, sources, output)
    decision = json.loads((output / "calendar_code_adjudication.json").read_text(encoding="utf-8"))[
        0
    ]
    assert decision["early_application_allowed"] is None
    assert decision["code_date_use_confirmed"] is True
    assert decision["independent_boundary_equivalence_confirmed"] is None
    assert decision["initial_DateVved"] == "16.01.2023"
