import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/summarize_economy_bulk.py"
SPEC = importlib.util.spec_from_file_location("economy_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_annual_changed_code_uses_its_year_and_keeps_zero_null_and_absence_separate():
    links = pd.DataFrame(
        [
            {
                "source_region_url": "region",
                "source_oktmo": code,
                "territory_id": "1",
                "reference_date": f"{year}-01-01",
                "temporal_review_required": flag,
            }
            for code, year, flag in [("old", "2023", "True"), ("new", "2024", "False")]
        ]
    )
    source = pd.DataFrame(
        [
            {
                "source_region_url": "region",
                "source_oktmo": code,
                "indicator_id": indicator,
                "year": year,
                "value": value,
                "value_status": status,
            }
            for code, indicator, year, value, status in [
                ("old", "8423005", "2023", "0", "observed"),
                ("new", "8423005", "2023", "999", "observed"),
                ("new", "8423007", "2024", "", "unresolved_source_marker"),
            ]
        ]
    )
    statuses = links[["source_region_url", "source_oktmo"]].merge(
        pd.DataFrame({"indicator_id": ["8423005", "8423007"]}), how="cross"
    )
    statuses["status"], statuses["error"] = "partial_source", ""
    result = MODULE.candidate_panel(source, links, statuses)
    assert len(result) == 4
    assert "999" not in result.value.to_list()
    zero = result.loc[result.cell_status.eq("observed")].iloc[0]
    assert zero.value == "0" and zero.temporal_review_required == "True"
    assert result.cell_status.value_counts().to_dict() == {
        "not_returned": 2,
        "observed": 1,
        "returned_source_marker": 1,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.candidate_panel(pd.concat([source, source]), links, statuses)
