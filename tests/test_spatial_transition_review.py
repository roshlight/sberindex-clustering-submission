"""Regression checks for version identity and municipality-level permutation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "spatial_review", Path(__file__).resolve().parents[1] / "scripts/review_spatial_transitions.py"
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def test_half_open_versions_do_not_duplicate_a_boundary_year():
    versions = pd.DataFrame(
        {
            "territory_id": [1, 1],
            "year_from": [2018, 2024],
            "year_to": [2024, 9999],
            "oktmo": ["001", "002"],
        }
    )
    assert review.snapshot(versions, 2023).oktmo.tolist() == ["001"]
    assert review.snapshot(versions, 2024).oktmo.tolist() == ["002"]
    versions.loc[0, "year_to"] = 2025
    with pytest.raises(ValueError, match="duplicate"):
        review.snapshot(versions, 2024)


def test_unknown_interval_end_is_not_invented():
    versions = pd.DataFrame({"territory_id": [1], "year_from": [2023], "year_to": [np.nan]})
    with pytest.raises(ValueError, match="Unknown"):
        review.snapshot(versions, 2024)


def test_permutations_preserve_changed_counts_within_regions():
    flags = np.array([True, False, False, True, True, False])
    strata = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    rng = np.random.default_rng(13)
    for _ in range(20):
        shuffled = review.permuted_flags(flags, strata, rng)
        assert [shuffled[s].sum() for s in strata] == [1, 2]


def test_region_confounding_is_not_evidence_of_excess_switching():
    # All high-rate territories belong to one region: unrestricted shuffling
    # would manufacture evidence; the conditional null equals the observation.
    result, draws = review.regional_permutation(
        [1, 1, 0, 0], [True, True, False, False], [1, 1, 2, 2], permutations=99, seed=13
    )
    assert result["difference"] == 1
    assert result["excess_over_null"] == 0
    assert result["status"] == "degenerate_null"
    assert result["p_upper"] == 1
    np.testing.assert_array_equal(draws, np.ones(99))


def test_one_sided_direction_and_repeatability():
    args = ([0, 0, 1, 1], [True, True, False, False], [1, 1, 1, 1])
    result, draws = review.regional_permutation(*args, permutations=99, seed=13)
    again, repeated = review.regional_permutation(*args, permutations=99, seed=13)
    assert result == again
    np.testing.assert_array_equal(draws, repeated)
    assert result["difference"] == -1
    assert result["p_upper"] == 1


def test_holm_known_example():
    np.testing.assert_allclose(review.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def transition_fixture():
    return pd.DataFrame(
        {
            "territory_id": [1, 2],
            "method": ["leiden_hellinger"] * 2,
            "from_month": ["2023-12"] * 2,
            "to_month": ["2024-01"] * 2,
            "lag": [1] * 2,
            "from_label": [0, 1],
            "to_label": [8, 9],
            "aligned_to_label": [0, 0],
            "switched": [False, True],
            "robust_switch": pd.Series([False, False], dtype="boolean"),
            "hellinger_profile_change": [0.01, 0.02],
        }
    )


WINDOW = {"from_year": "2023", "to_year": "2024", "lag": 1, "pairs_per_territory": 1}


def test_pair_alignment_not_raw_labels_controls_switches():
    result = review.municipality_rates(transition_fixture(), [1, 2], WINDOW, "leiden_hellinger")
    assert result.switches.tolist() == [0, 1]
    assert result.robust_switches.tolist() == [0, 0]


def test_missing_robust_support_is_not_false():
    t = transition_fixture()
    t.loc[0, "robust_switch"] = pd.NA
    with pytest.raises(ValueError, match="expected observed Booleans"):
        review.municipality_rates(t, [1, 2], WINDOW, "leiden_hellinger")


def test_missing_municipality_cannot_silently_change_denominator():
    with pytest.raises(ValueError, match="cohort"):
        review.municipality_rates(transition_fixture().iloc[:1], [1, 2], WINDOW, "leiden_hellinger")


def test_bad_saved_alignment_is_rejected():
    t = transition_fixture()
    t.loc[0, "switched"] = True
    with pytest.raises(ValueError, match="aligned labels"):
        review.municipality_rates(t, [1, 2], WINDOW, "leiden_hellinger")


def test_existing_output_is_preserved(tmp_path):
    marker = tmp_path / "important.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        review.run(tmp_path, tmp_path / "absent.json", tmp_path)
    assert marker.read_text() == "keep"


def test_delivered_summary_reconciles_with_source_rows_via_sql():
    """A separate SQL path checks counts/rates against the pre-existing ID audit."""
    import duckdb

    root = Path(__file__).resolve().parents[1]
    output = root / "data/derived/reviews/spatial_transitions_v1"
    checks = pd.read_csv(output / "spatial_checks.csv")
    transitions = duckdb.read_parquet(str(root / "reports/initial_study/transitions.parquet")).df()
    changed = pd.read_csv(
        root / "reports/enrichment_v2/market_access_followup/sber_attribute_changes_78.csv"
    )[["territory_id"]]
    predicates = {
        "boundary_2023_2024": "from_month='2023-12' AND to_month='2024-01' AND lag=1",
        "within_2024": "from_month LIKE '2024-%' AND to_month LIKE '2024-%' AND lag=1",
        "year_over_year": "from_month LIKE '2023-%' AND to_month LIKE '2024-%' AND lag=12",
    }
    with duckdb.connect() as con:
        con.register("source_transitions", transitions)
        con.register("changed_ids", changed)
        for row in checks.itertuples():
            # Predicates above are fixed test code, not user-supplied SQL.
            actual = con.sql(f"""
                SELECT c.territory_id IS NOT NULL AS changed,
                       count(DISTINCT t.territory_id) AS n,
                       count(*) AS pairs,
                       avg(CAST(t.switched AS INTEGER)) AS rate,
                       sum(CAST(t.robust_switch AS INTEGER)) AS robust_count
                FROM source_transitions t
                LEFT JOIN changed_ids c USING (territory_id)
                WHERE t.method='leiden_hellinger' AND {predicates[row.window]}
                GROUP BY 1
            """).df()
            for record in actual.itertuples():
                group = "changed" if record.changed else "no_flag"
                assert record.n == getattr(row, f"n_{group}")
                assert record.pairs == getattr(row, f"pairs_{group}")
                assert record.rate == pytest.approx(getattr(row, f"switch_rate_{group}"))
                assert record.robust_count == getattr(row, f"robust_count_{group}")
            assert row.n_all == row.n_changed + row.n_no_flag
            assert row.pairs_all == row.pairs_changed + row.pairs_no_flag
