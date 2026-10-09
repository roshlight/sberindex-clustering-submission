"""Guard missing destinations, code widths, duplicates and explicit parent existence."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round6_geography", Path(__file__).resolve().parents[1] / "scripts/audit_round6_geography.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_grouping_normalization_never_pads_or_fills():
    assert V.normalized_code("05 614 154 051") == "05614154051"
    assert V.normalized_code("05\u00a0614 154 051") == "05614154051"
    for text in ["", "-", "0", "5614154051", "05614154", "056141540510"]:
        assert V.normalized_code(text) is None


def test_missing_destination_and_duplicate_rows_survive_raw_parser(tmp_path):
    p = tmp_path / "source.csv"
    p.write_text(
        "Перекодировочная таблица;\n"
        "Субъект Российской Федерации;Аннулированный код;Действующий код;Номер изменения\n"
        "Пермский край;57 528 000 382;;547/2022 ОКТМО\n"
        "Приморский край;05 614 154 051;05 514 000 052;643/2023 ОКТМО\n"
        "Приморский край;05 614 154 051;05 514 000 052;643/2023 ОКТМО\n",
        encoding="cp1251",
    )
    data, encoding = V.read_crosswalk(p, "test")
    assert encoding == "cp1251" and len(data) == 3
    assert data.iloc[0].new_status == "missing" and data.iloc[0].new_code == ""
    assert data.iloc[1].old_code == "05614154051"
    assert data.iloc[1].new_code == "05514000052"
    assert data.csv_row.tolist() == [3, 4, 5]


def test_unknown_extra_data_column_is_not_silently_discarded(tmp_path):
    p = tmp_path / "source.csv"
    p.write_text(
        "Перекодировочная таблица;\n"
        "Субъект Российской Федерации;Аннулированный код;Действующий код;Номер изменения\n"
        "Приморский край;05 614 154 051;05 514 000 052;643/2023 ОКТМО;meaningful\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="undeclared"):
        V.read_crosswalk(p, "test")


def test_no_parent_candidate_without_exact_section1_record():
    settlement = dict(
        code="05614154051",
        TER="05",
        KOD1="614",
        KOD2="154",
        RAZDEL="2",
        NAME1="Населённый пункт",
        DateVved="01.01.2014",
        NomAkt="1",
    )
    parent = dict(
        code="05614000",
        TER="05",
        KOD1="614",
        KOD2="000",
        RAZDEL="1",
        NAME1="Муниципальный район",
        DateVved="01.01.2014",
        NomAkt="1",
    )
    panel = pd.DataFrame([dict(official_oktmo_8="05614000", territory_id="test")])
    assert V.parent_candidates(pd.DataFrame([settlement]), panel).empty
    result = V.parent_candidates(pd.DataFrame([parent, settlement]), panel)
    assert len(result) == 1 and result.iloc[0].parent_code == "05614000"
    assert not result.iloc[0].historical_boundary_equivalence_proven
    assert not result.iloc[0].municipality_address_join_approved
