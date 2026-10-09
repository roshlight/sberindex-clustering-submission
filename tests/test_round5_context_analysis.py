"""Protect row grain, missingness and the boundary between budgets and investment."""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round5_context", Path(__file__).parents[1] / "scripts/round5_context_analysis.py"
)
context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context)


@pytest.fixture
def fixture_data(tmp_path):
    records = [
        ("a", "45000000", "77", 0, 5),
        ("a", "45000000", "77", 0, 5),
        ("b", "01234567890", "01", 1, None),
        ("c", None, None, 1, None),
        ("d", "", "77", 0, -1),
        ("e", "01000000", "01", 0, 0),
        ("f", "12345678", "23", 0, None),
    ]
    raw = pd.DataFrame(
        records,
        columns=[
            "tax_number",
            "municipality_code",
            "region_code",
            "is_sole_trader",
            "employees_count",
        ],
    )
    raw["year"] = 2023
    raw["region_iso_code"] = raw.region_code
    raw["region"] = raw.region_code
    path = tmp_path / "source.parquet"
    with duckdb.connect() as con:
        con.from_df(raw).write_parquet(str(path))
    dimension = pd.DataFrame(
        [
            (2023, "45000000", False, 2, True, "region_group", False),
            (2023, "01234567890", False, 1, True, "section2_settlement", False),
            (2023, "", True, 1, False, None, False),
            (2023, "", False, 1, False, None, False),
            (2023, "01000000", False, 1, True, "region_group", False),
            (2023, "12345678", False, 1, False, None, False),
        ],
        columns=[
            "year",
            "source_code",
            "source_code_is_null",
            "source_rows",
            "v20260601_present",
            "v20260601_kind",
            "v20260601_ambiguous",
        ],
    )
    return path, dimension


def test_exact_annotation_preserves_repeated_period_rows_and_null_vs_empty(fixture_data):
    path, dimension = fixture_data
    with duckdb.connect() as con:
        n = context.create_base(con, path, context.code_dimension(dimension), 2023)
        result = context.annual_tables(con, 2023, n)
    annual = result["annual_population"].set_index("legal_kind").loc["ALL"]
    assert n == 7
    assert annual.distinct_tax_ids == 6
    assert annual.positive_employee_rows == 2
    assert annual.nonnegative_employee_rows == 3
    assert annual.negative_employee_rows == 1
    assert annual.missing_employee_rows == 3
    unmatched = result["unmatched_audit"].set_index("unmatched_reason")
    assert set(unmatched.index) == {"source_null", "source_empty", "valid_syntax_not_in_june2026"}
    assert unmatched.source_rows.sum() == 3
    assert result["geography_strata"].source_rows.sum() == 7
    assert (
        result["geography_strata"]
        .set_index("geo_stratum")
        .loc["federal_city_region_group", "source_rows"]
        == 2
    )
    assert all("sum_workers" not in name for name in result)


def test_dimension_duplicates_are_rejected_before_join(fixture_data):
    _, dimension = fixture_data
    with pytest.raises(ValueError, match="many-to-one"):
        context.code_dimension(pd.concat([dimension, dimension.iloc[:1]], ignore_index=True))


def test_changed_raw_group_counts_do_not_pass_national_total(fixture_data):
    path, dimension = fixture_data
    dimension.loc[0, "source_rows"] = 1
    dimension.loc[1, "source_rows"] = 2
    with duckdb.connect() as con, pytest.raises(ValueError, match="code-group counts differ"):
        context.create_base(con, path, context.code_dimension(dimension), 2023)


def test_unmatched_code_is_not_promoted_by_nonnull_kind(fixture_data):
    _, dimension = fixture_data
    dimension.loc[5, "v20260601_kind"] = "section1_top_level_or_group"
    with pytest.raises(ValueError, match="absent exact code"):
        context.code_dimension(dimension)


@pytest.mark.parametrize("value", ["section2_settlement", "section1_top_level_or_group"])
def test_federal_group_requires_exact_reviewed_kind(fixture_data, value):
    _, dimension = fixture_data
    dimension.loc[0, "v20260601_kind"] = value
    with pytest.raises(ValueError, match="Federal-city grouping"):
        context.code_dimension(dimension)


def test_budget_revision_is_exact_decimal_and_dash_stays_unknown():
    rows = []
    for sid, date, values in [
        ("gov_contract_uplift_b_http", "2022-12-16", ["370152,2", "134642,6", "-", "504794,8"]),
        ("gov_contract_uplift_a_http", "2023-04-21", ["370152,2", "126838", "-", "496990,2"]),
    ]:
        for year, value in zip(["2022", "2023", "2024", "2022–2024"], values, strict=True):
            rows.append(
                {
                    "source_id": sid,
                    "reported_year": year,
                    "unit": "thousand_RUB",
                    "value_as_published": value,
                    "source_publication_date": date,
                    "source_sha256": "hash",
                    "source_url": "https://example.test/report",
                    "physical_page": "28",
                    "model_join_accepted": "False",
                    "causal_attribution_confirmed": "False",
                    "annual_bdpm_investment_reconciled": "False",
                }
            )
    frame = pd.DataFrame(rows)
    result = context.budget_revision(frame).set_index("year_column")
    assert result.loc["2023", "revision_delta_thousand_rub"] == "-7804.6"
    assert result.loc["2023", "revision_delta_million_rub"] == "-7.8046"
    assert result.loc["2024", "revision_delta_thousand_rub"] == ""
    assert not result.share_of_bdpm_peak_computed.any()
    frame.loc[0, "annual_bdpm_investment_reconciled"] = "True"
    with pytest.raises(ValueError, match="promoted"):
        context.budget_revision(frame)


def test_stock_contrast_never_labels_population_gap_an_error():
    stock = pd.DataFrame(
        [
            {
                "snapshot_date": f"{year}-12-10",
                "geography_level": "country",
                "total": 5,
                "legal_total": 2,
                "ip_total": 3,
                "source_file": "official.xlsx",
                "excel_row": 9,
            }
            for year in [2023, 2024]
        ]
    )
    annual = pd.DataFrame(
        [
            {"year": year, "legal_kind": kind, "source_rows": n, "distinct_tax_ids": ids}
            for year in [2023, 2024]
            for kind, n, ids in [("ALL", 10, 8), ("legal_entity", 4, 3), ("IP", 6, 5)]
        ]
    )
    result = context.fns_contrast(stock, annual)
    assert len(result) == 6
    assert not result.same_population_and_moment.any()
    assert not result.coverage_error_or_growth_estimate.any()


def test_old_outputs_cannot_be_overwritten(tmp_path):
    existing = tmp_path / "out"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        context.run(tmp_path, tmp_path, existing)
