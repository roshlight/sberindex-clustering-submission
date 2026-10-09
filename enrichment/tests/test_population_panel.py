import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "population_panel", Path(__file__).parents[1] / "scripts/prepare_population_panel.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_frames():
    links, source, targets = [], [], []
    region = "https://rosstat.gov.ru/dbscripts/munst/munst01/"
    for territory_id in (1, 2):
        code = str(10000000 + territory_id)
        targets.append(
            {"region_url": region, "source_oktmo": code, "status": "complete", "error": None}
        )
        for year in (2023, 2024):
            date = f"{year}-01-01"
            links.append(
                {
                    "territory_id": str(territory_id),
                    "reference_date": date,
                    "source_region_url": region,
                    "source_oktmo": code,
                    "official_oktmo_8": code,
                    "official_code_present": "true",
                    "sber_annual_code_present": "true",
                    "temporal_review_required": "false",
                }
            )
            for kind, value in [("11", 100 * territory_id), ("12", 30 * territory_id)]:
                source.append(
                    {
                        "reference_date": date,
                        "source_region_url": region,
                        "source_oktmo": code,
                        "population_type": kind,
                        "value": str(value),
                        "value_status": "observed",
                        "unit": "persons",
                        "indicator_id": "8112027",
                        "territory_id": None,
                    }
                )
    return pd.DataFrame(source), pd.DataFrame(links), pd.DataFrame(targets)


def test_missing_and_temporal_review_are_not_usable_zero_values():
    source, links, targets = fixture_frames()
    source = source.drop(source.index[-1])
    links.loc[0, "temporal_review_required"] = "true"
    panel, wide, coverage = module.assemble(source, links, targets, [1, 2])
    assert len(panel) == 8 and len(wide) == 4
    reviewed = panel.loc[panel.territory_id.eq("1") & panel.reference_date.eq("2023-01-01")]
    assert reviewed.value.tolist() == [100, 30]
    assert reviewed.analysis_value.isna().all()
    absent = panel.loc[~panel.source_row_present]
    assert len(absent) == 1 and absent.value.isna().all()
    assert absent.analysis_status.iloc[0] == "not_returned"
    assert wide.urban_share.notna().sum() == 2
    assert sum(r["source_values_observed"] for r in coverage) == 7
    assert sum(r["analysis_values_available"] for r in coverage) == 5


def test_source_missing_marker_remains_a_returned_but_unknown_value():
    source, links, targets = fixture_frames()
    source.loc[1, ["value", "value_status"]] = [None, "missing_symbol"]
    panel, wide, _ = module.assemble(source, links, targets, [1, 2])
    missing = panel.loc[panel.availability_status.eq("missing_symbol")]
    assert missing.source_row_present.all() and missing.value.isna().all()
    assert wide.urban_share.notna().sum() == 3


def test_numeric_html_option_preserves_official_leading_zero_in_reference():
    source, links, targets = fixture_frames()
    for frame in (source, links, targets):
        frame.loc[frame.source_oktmo.eq("10000001"), "source_oktmo"] = "1000001"
    links.loc[links.territory_id.eq("1"), "official_oktmo_8"] = "01000001"
    panel, _, _ = module.assemble(source, links, targets, [1, 2])
    matched = panel.loc[panel.territory_id.eq("1")]
    assert matched.official_oktmo_8.eq("01000001").all()
    assert matched.analysis_value.notna().all()


@pytest.mark.parametrize("status", ["attribution_review", "error", "pending"])
def test_stale_values_cannot_override_rejected_acquisition_status(status):
    source, links, targets = fixture_frames()
    targets.loc[0, "status"] = status
    with pytest.raises(ValueError, match="acquisition status"):
        module.assemble(source, links, targets, [1, 2])


def calendar_decision():
    return pd.DataFrame(
        [
            {
                "territory_id": "1",
                "reference_date": "2023-01-01",
                "official_oktmo_8": "10000001",
                "allowed_from": "2023-01-01",
                "code_date_use_confirmed": "true",
                "decision": "clear_calendar_code_conflict_only",
                "source_sha256": "a" * 64,
                "source_url": "https://protect.gost.ru/example",
                "source_capture_kind": "raw_html",
                "locator": "date comment",
                "independent_boundary_equivalence_confirmed": None,
            }
        ]
    )


def test_calendar_decision_retains_initial_warning_and_boundary_unknown():
    source, links, targets = fixture_frames()
    links.loc[0, "temporal_review_required"] = "true"
    panel, _, coverage = module.assemble(source, links, targets, [1, 2], calendar_decision())
    resolved = panel.loc[panel.calendar_review_cleared]
    assert len(resolved) == 2 and resolved.initial_temporal_review.all()
    assert not resolved.temporal_review.any()
    assert resolved.analysis_value.tolist() == [100, 30]
    assert resolved.calendar_independent_boundary_equivalence_confirmed.isna().all()
    assert sum(r["calendar_review_cleared_links"] for r in coverage) == 2


@pytest.mark.parametrize("defect", ["wrong_code", "wrong_date", "late_permission", "unflagged"])
def test_calendar_decision_cannot_clear_an_unrelated_or_unproven_link(defect):
    source, links, targets = fixture_frames()
    links.loc[0, "temporal_review_required"] = "true"
    decision = calendar_decision()
    if defect == "wrong_code":
        decision.loc[0, "official_oktmo_8"] = "10000002"
    elif defect == "wrong_date":
        decision.loc[0, "reference_date"] = "2024-01-01"
    elif defect == "late_permission":
        decision.loc[0, "allowed_from"] = "2023-01-02"
    else:
        links.loc[0, "temporal_review_required"] = "false"
    with pytest.raises(ValueError):
        module.assemble(source, links, targets, [1, 2], decision)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_source",
        "duplicate_link",
        "missing_cohort",
        "unknown_timing",
        "urban_exceeds_total",
    ],
)
def test_rejects_silent_join_and_value_corruption(defect):
    source, links, targets = fixture_frames()
    if defect == "duplicate_source":
        source = pd.concat([source, source.iloc[:1]], ignore_index=True)
    elif defect == "duplicate_link":
        links = pd.concat([links, links.iloc[:1]], ignore_index=True)
    elif defect == "missing_cohort":
        links = links.iloc[:-1]
    elif defect == "unknown_timing":
        links.loc[0, "temporal_review_required"] = "unknown"
    else:
        source.loc[1, "value"] = "999"
    with pytest.raises(ValueError):
        module.assemble(source, links, targets, [1, 2])
