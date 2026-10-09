"""Semantic guards for incomplete published organizational worker sections."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "round2_sector_analysis.py"
SPEC = importlib.util.spec_from_file_location("round2_sector_analysis", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_official_total_denominator_does_not_renormalize_disclosures():
    got = MOD.derive_shares(
        pd.Series([10.0, 20.0]), pd.Series(["observed", "observed"]), pd.Series([100.0, 100.0])
    )
    assert got.share.tolist() == [0.1, 0.2]
    assert got.share.sum() == pytest.approx(0.3)


def test_exceeds_total_excluded_not_clipped_and_zero_is_retained():
    got = MOD.derive_shares(
        pd.Series([120.0, 0.0, np.nan, np.nan]),
        pd.Series(["observed", "observed", "unresolved_source_marker", None]),
        pd.Series([100.0] * 4),
    )
    assert pd.isna(got.share.iloc[0])
    assert got.share.iloc[1] == 0
    assert got.share.iloc[2:].isna().all()
    assert got.share_status.tolist() == [
        "exceeds_total",
        "valid",
        "unresolved_source_marker",
        "not_offered",
    ]


@pytest.mark.parametrize("total", [np.nan, 0.0, -10.0, np.inf])
def test_unusable_total_cannot_create_a_share(total):
    got = MOD.derive_shares(pd.Series([5.0]), pd.Series(["observed"]), pd.Series([total]))
    assert got.share.isna().all()
    assert got.share_status.iloc[0] == "invalid_total"


def sample_frames():
    core = pd.DataFrame(
        [
            {
                "territory_id": 2710,
                "year": 2024,
                "official_oktmo_8": "28714000",
                "workers": np.nan,
                "core_eligible": False,
                "conservative_eligible": False,
                "calendar_review": True,
                "code_type_changed": True,
                "geography_status": "unknown",
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "territory_id": 2710,
                "year": 2024,
                "official_oktmo_8": "28714000",
                "indicator_id": MOD.WORKERS,
                "source_activity_code": code,
                "value": value,
                "status": "observed",
                "source_sha256": "a" * 64,
                "source_csv_row": row,
                "source_url": "https://example.org/source.csv",
                "temporal_review_required": True,
            }
            for code, value, row in [("101", 100.0, 1), ("10000", 80.0, 2), ("20000", 30.0, 3)]
        ]
    )
    return source, core


def test_new_total_preserves_frozen_eligibility_and_diagnostic_sum_over_total():
    source, core = sample_frames()
    features, long = MOD.build_features(source, core)
    row = features.iloc[0]
    assert row.sector_A_share == 0.8
    assert row.sector_C_share == 0.3
    assert pd.isna(row.sector_B_share)
    assert row.observed_sum_over_official_total_diagnostic == 1.1
    assert row.observed_sum_exceeds_official_total
    assert not row.core_eligible
    assert row.calendar_review and row.code_type_changed
    assert row.newly_available_workers_total
    assert pd.isna(row.workers_total_frozen_core)
    assert pd.isna(row.geographic_equivalence_confirmed)
    assert set(long.loc[long.canonical_section.eq("B"), "share_status"]) == {"not_offered"}
    # Source-valid ratios do not revise or replace the frozen temporal eligibility mask.
    assert row.n_selected_shares_available == 2


def test_duplicate_cell_and_changed_code_are_rejected():
    source, core = sample_frames()
    with pytest.raises(ValueError, match="Duplicate worker"):
        MOD.build_features(pd.concat([source, source.iloc[[0]]]), core)
    core.loc[0, "official_oktmo_8"] = "28714001"
    with pytest.raises(ValueError, match="annual codes differ"):
        MOD.build_features(source, core)


def test_exact_form_label_and_section_prefix_are_independent_checks():
    row = {
        "source_region_url": "https://example.org/",
        "indicator_id": MOD.WORKERS,
        "source_activity_code": "10000",
        "source_activity_label": "Раздел A: сельское хозяйство",
        "options_sha256": "a" * 64,
    }
    source = pd.DataFrame([row])
    options = source.copy()
    dictionary = MOD.label_dictionary(source, options)
    assert dictionary.canonical_section.tolist() == ["A"]
    source.loc[0, "source_activity_label"] = "Раздел B: сельское хозяйство"
    with pytest.raises(ValueError, match="absent from exact"):
        MOD.label_dictionary(source, options)
    with pytest.raises(ValueError, match="Conflicting section prefix"):
        MOD.label_dictionary(source, source.copy())


def test_comparison_needs_both_published_values_and_retains_unknown_boundaries():
    source, core = sample_frames()
    source["territory_id"] = 331
    core["territory_id"] = 331
    source["temporal_review_required"] = False
    core[["calendar_review", "code_type_changed"]] = False
    core["municipal_district_name"] = "Synthetic municipality"
    source["source_activity_label"] = "Reviewed section"
    source_2023 = source.copy().assign(year=2023)
    core_2023 = core.copy().assign(year=2023)
    source_2023.loc[source_2023.source_activity_code.eq("10000"), "value"] = 70.0
    source_2023.loc[source_2023.source_activity_code.eq("20000"), ["value", "status"]] = [
        np.nan,
        "not_returned",
    ]
    features, long = MOD.build_features(
        pd.concat([source_2023, source]), pd.concat([core_2023, core])
    )
    _, pairs = MOD.example_tables(features, long)
    a = pairs.loc[pairs.canonical_section.eq("A")].iloc[0]
    c = pairs.loc[pairs.canonical_section.eq("C")].iloc[0]
    assert a.conditional_same_code_share_change_pp == pytest.approx(10)
    assert pd.isna(a.independent_boundary_equivalence_confirmed)
    assert pd.isna(c.conditional_same_code_share_change_pp)
    assert c.comparison_status == "one_or_both_sector_values_unavailable"


def test_raw_example_check_handles_wrapped_explicit_sections_but_rejects_wrong_year(tmp_path):
    raw = tmp_path / "source.raw"
    raw.write_bytes(
        b"\nWorker indicator,\nexact section total\n;2023;2024;\nOther;1;2;\nTarget;100;110;\n\n"
        b"Worker indicator, exact section A\n;2023;2024;\nOther;3;4;\nTarget;70;80;\n"
    )
    digest = MOD.sha256(raw)
    source = pd.DataFrame(
        [
            {
                "territory_id": 331,
                "year": 2024,
                "indicator_id": MOD.WORKERS,
                "source_activity_code": code,
                "source_sha256": digest,
                "source_csv_row": line,
                "source_section": section,
                "actual_source_oktmo_label": "Target",
                "value": value,
                "official_oktmo_8": "98637000",
                "source_url": "https://example.org/source.csv",
            }
            for code, line, section, value in [
                ("101", 6, "Worker indicator, exact section total", 110.0),
                ("10000", 11, "Worker indicator, exact section A", 80.0),
            ]
        ]
    )
    examples = pd.DataFrame(
        [{"territory_id": 331, "year": 2024, "max_among_observed_sections": "A"}]
    )
    inventory = [{"sha256": digest, "path": "source.raw"}]
    trace = MOD.verify_example_raw(source, examples, tmp_path, inventory)
    assert trace.value.tolist() == [80.0, 110.0]
    assert trace.source_year_header_row.tolist() == [9, 4]
    source.loc[0, "value"] = 100.0  # The 2023 cell must not validate a 2024 observation.
    with pytest.raises(ValueError, match="Raw source value differs"):
        MOD.verify_example_raw(source, examples, tmp_path, inventory)
