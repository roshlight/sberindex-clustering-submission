"""Preserve code length, ambiguity and section meaning before comparing versions."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_round5_geography.py"
SPEC = importlib.util.spec_from_file_location("round5_geography", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(**updates):
    value = dict.fromkeys(module.COLS, "")
    value.update(TER="01", KOD1="601", KOD2="101", KOD3="000", RAZDEL="1", NAME1="Town")
    return value | updates


def test_settlement_headings_are_not_identifiers():
    frame = pd.DataFrame([row(), row(RAZDEL="2"), row(RAZDEL="2", KOD3="001")])
    result = module.registry_entries(frame)
    assert result.code.tolist() == ["01601101", "01601101001"]
    assert result.code_kind.tolist() == ["section1_submunicipal_or_group", "section2_settlement"]


def test_no_zero_padding_or_prefix_join():
    entries = module.registry_entries(pd.DataFrame([row(), row(RAZDEL="2", KOD3="001")]))
    codes = pd.DataFrame({"source_code": ["01601101", "1601101", "01601101999", ""]})
    result = module.attach_presence(codes, entries, "v")
    assert result.v_present.tolist() == [True, False, False, False]


def test_ambiguous_codes_never_select_a_name():
    entries = module.registry_entries(pd.DataFrame([row(), row(NAME1="Other")]))
    result = module.attach_presence(pd.DataFrame({"source_code": ["01601101"]}), entries, "v")
    assert result.v_present.item() and result.v_ambiguous.item()
    assert pd.isna(result.v_name.item())


@pytest.mark.parametrize("changes", [{"TER": "1"}, {"KOD3": "001"}, {"RAZDEL": "3"}])
def test_reject_changed_registry_schema(changes):
    with pytest.raises(ValueError):
        module.registry_entries(pd.DataFrame([row(**changes)]))
