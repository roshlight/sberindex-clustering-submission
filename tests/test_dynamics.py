import numpy as np
import pandas as pd
import pytest

from municipal_profiles.dynamics import PROFILE_COLUMNS, temporal_tables


def _month(month, labels, ids=None, method="leiden_hellinger", profiles=None):
    if ids is None:
        ids = np.arange(1, len(labels) + 1)
    if profiles is None:
        profiles = np.full((len(labels), 5), 0.2)
    frame = pd.DataFrame(profiles, columns=PROFILE_COLUMNS)
    frame["month"] = month
    frame["territory_id"] = ids
    frame["method"] = method
    frame["label"] = labels
    return frame


def _agreement(assignments, score=0.9):
    result = assignments[["month", "territory_id"]].drop_duplicates().copy()
    result["agreement"] = score
    result["n_comparisons"] = 3
    return result


def _variants(assignments):
    result = {}
    primary = assignments.loc[assignments["method"].eq("leiden_hellinger")]
    for month, frame in primary.groupby("month"):
        frame = frame.sort_values("territory_id")
        result[month] = {
            "ids": frame["territory_id"].to_numpy(),
            "labels": np.tile(frame["label"].to_numpy(), (21, 1)),
            "support": np.ones((21, len(frame)), dtype=bool),
        }
    return result


def test_label_permutation_is_not_a_transition():
    assignments = pd.concat([_month("2023-01", [0, 0, 1, 1]), _month("2023-02", [70, 70, 4, 4])])
    transitions, flows, summary = temporal_tables(
        assignments, _agreement(assignments), variant_evidence=_variants(assignments)
    )
    assert len(transitions) == 4
    assert not transitions["switched"].any()
    np.testing.assert_array_equal(transitions["aligned_to_label"], transitions["from_label"])
    assert transitions["robust_endpoints"].all()
    assert not transitions["robust_switch"].any()
    np.testing.assert_allclose(transitions["transition_reproduction_fraction"], 1)
    assert summary.iloc[0]["ari"] == 1
    assert summary.iloc[0]["switch_share"] == 0
    assert summary.iloc[0]["robust_switch_share_of_all"] == 0
    assert summary.iloc[0]["median_profile_change"] == 0
    assert set(zip(flows["from_label"], flows["to_label"], strict=True)) == {(0, 70), (1, 4)}
    assert flows["count"].sum() == 4


def test_split_retains_every_raw_flow_and_unmatched_group():
    assignments = pd.concat(
        [_month("2023-01", [0, 0, 0, 0, 1, 1]), _month("2023-02", [10, 10, 10, 20, 30, 30])]
    )
    transitions, flows, summary = temporal_tables(
        assignments, _agreement(assignments), variant_evidence=_variants(assignments)
    )
    assert transitions["switched"].sum() == 1
    assert transitions.loc[transitions["to_label"].eq(20), "aligned_to_label"].iloc[0] < 0
    assert summary.iloc[0]["ncommon"] == 6
    assert summary.iloc[0]["switch_share"] == pytest.approx(1 / 6)
    assert summary.iloc[0]["robust_switch_share_of_all"] == pytest.approx(1 / 6)
    split_flows = flows.set_index(["from_label", "to_label"])
    assert split_flows.loc[(0, 10), "count"] == 3
    assert split_flows.loc[(0, 10), "row_share"] == 0.75
    assert split_flows.loc[(0, 20), "row_share"] == 0.25
    np.testing.assert_allclose(flows.groupby("from_label")["row_share"].sum(), 1)
    np.testing.assert_allclose(flows.groupby("to_label")["col_share"].sum(), 1)


def test_merge_column_shares_use_common_destination_group():
    assignments = pd.concat([_month("2023-01", [0, 0, 0, 1]), _month("2023-02", [10, 10, 10, 10])])
    transitions, flows, _ = temporal_tables(assignments, pd.DataFrame())
    assert transitions["switched"].sum() == 1
    assert flows.set_index("from_label").loc[0, "col_share"] == 0.75
    assert flows.set_index("from_label").loc[1, "col_share"] == 0.25
    np.testing.assert_allclose(flows["row_share"], 1)


def test_inner_join_excludes_entering_and_leaving_entities():
    assignments = pd.concat(
        [
            _month("2023-01", [0, 0, 1], ids=[1, 2, 3]),
            _month("2023-02", [8, 7, 8], ids=[2, 3, 4]),
        ]
    )
    transitions, flows, summary = temporal_tables(assignments, _agreement(assignments))
    assert transitions["territory_id"].tolist() == [2, 3]
    assert not transitions["switched"].any()
    assert summary["ncommon"].tolist() == [2]
    assert flows["count"].sum() == 2


def test_calendar_pairs_skip_gaps_but_include_year_and_december_boundary():
    assignments = pd.concat(
        [_month(month, [0, 1]) for month in ("2023-01", "2023-03", "2023-12", "2024-01")]
    )
    transitions, _, summary = temporal_tables(assignments, pd.DataFrame())
    pairs = set(summary[["from_month", "to_month", "lag"]].itertuples(index=False, name=None))
    assert pairs == {("2023-01", "2024-01", 12), ("2023-12", "2024-01", 1)}
    assert len(transitions) == 4


def test_agreement_is_assessed_only_with_both_endpoints_and_primary_method():
    assignments = pd.concat(
        [
            _month(month, labels, method=method)
            for method in ("leiden_hellinger", "leiden_cosine", "kmeans")
            for month, labels in (("2023-01", [0, 0, 0]), ("2023-02", [1, 1, 2]))
        ]
    )
    agreement = _agreement(assignments)
    agreement.loc[agreement["territory_id"].eq(2), "agreement"] = 0.7
    agreement = agreement.loc[~(agreement["month"].eq("2023-02") & agreement["territory_id"].eq(3))]
    transitions, _, summary = temporal_tables(assignments, agreement)
    primary = transitions.loc[transitions["method"].eq("leiden_hellinger")].set_index(
        "territory_id"
    )
    assert bool(primary.loc[1, "robust_endpoints"])
    assert not bool(primary.loc[2, "robust_endpoints"])
    assert pd.isna(primary.loc[3, "robust_endpoints"])
    assert pd.isna(primary.loc[3, "robust_switch"])
    alternatives = transitions.loc[~transitions["method"].eq("leiden_hellinger")]
    assert alternatives[["from_agreement", "to_agreement"]].isna().all().all()
    assert alternatives[["robust_endpoints", "robust_switch"]].isna().all().all()
    assert summary["robust_switch_share_of_all"].isna().all()
    assert str(transitions["robust_switch"].dtype) == "boolean"


@pytest.mark.parametrize("agreement_kind", ["empty", "zero_comparisons", "null_comparisons"])
def test_lack_of_agreement_is_not_assessed_as_zero(agreement_kind):
    assignments = pd.concat([_month("2023-01", [0, 0]), _month("2023-02", [3, 3])])
    agreement = _agreement(assignments, score=1)
    if agreement_kind == "empty":
        agreement = pd.DataFrame()
    else:
        agreement["n_comparisons"] = 0 if agreement_kind == "zero_comparisons" else np.nan
    transitions, _, summary = temporal_tables(assignments, agreement)
    assert transitions[["robust_endpoints", "robust_switch"]].isna().all().all()
    assert summary["robust_switch_share_of_all"].isna().all()
    assert summary["switch_share"].iloc[0] == 0


def test_hellinger_change_and_threshold_inclusive_boundary():
    assignments = pd.concat(
        [
            _month("2023-01", [0, 0], profiles=[[1, 0, 0, 0, 0], [0.2] * 5]),
            _month("2023-02", [0, 1], profiles=[[0, 1, 0, 0, 0], [0.2] * 5]),
        ]
    )
    transitions, _, summary = temporal_tables(
        assignments,
        _agreement(assignments, score=0.8),
        variant_evidence=_variants(assignments),
    )
    np.testing.assert_allclose(transitions["hellinger_profile_change"], [1, 0])
    assert transitions["robust_endpoints"].all()
    assert summary["median_profile_change"].iloc[0] == pytest.approx(0.5)
    _, _, higher_threshold = temporal_tables(
        assignments, _agreement(assignments, score=0.8), 0.81, _variants(assignments)
    )
    assert higher_threshold["robust_switch_share_of_all"].iloc[0] == 0.5


def test_high_marginal_agreement_without_variant_evidence_is_not_robustness():
    assignments = pd.concat([_month("2023-01", [0, 0]), _month("2023-02", [3, 4])])
    transitions, _, summary = temporal_tables(assignments, _agreement(assignments, score=1))
    assert transitions["robust_endpoints"].all()
    assert transitions["transition_reproduction_fraction"].isna().all()
    assert transitions["robust_switch"].isna().all()
    assert summary["robust_switch_share_of_all"].isna().all()


def test_two_17_of_21_supported_endpoints_with_13_overlap_are_not_a_robust_pair():
    assignments = pd.concat([_month("2023-01", [0, 0, 0]), _month("2023-02", [1, 1, 2])])
    evidence = _variants(assignments)
    evidence["2023-01"]["support"][17:, 2] = False
    evidence["2023-02"]["support"][:4, 2] = False
    transitions, _, summary = temporal_tables(
        assignments, _agreement(assignments, score=17 / 21), variant_evidence=evidence
    )
    switched = transitions.loc[transitions["territory_id"].eq(3)].iloc[0]
    assert bool(switched["switched"]) and bool(switched["robust_endpoints"])
    assert switched["transition_reproduction_fraction"] == pytest.approx(13 / 21)
    assert not bool(switched["robust_switch"])
    assert summary["robust_switch_share_of_all"].iloc[0] == 0


def test_variant_switch_disagreement_reduces_reproduction_even_with_full_support():
    assignments = pd.concat([_month("2023-01", [0, 0, 0]), _month("2023-02", [1, 1, 2])])
    evidence = _variants(assignments)
    evidence["2023-02"]["labels"][:5, :] = 9
    transitions, _, _ = temporal_tables(assignments, pd.DataFrame(), variant_evidence=evidence)
    switched = transitions.loc[transitions["territory_id"].eq(3)].iloc[0]
    assert switched["transition_reproduction_fraction"] == pytest.approx(16 / 21)
    assert not bool(switched["robust_switch"])
    # Independent pair evidence can be assessed without the marginal score table.
    assert transitions["robust_endpoints"].isna().all()
    assert transitions["robust_switch"].notna().all()


def test_variant_comparison_uses_same_common_ids_and_preserves_method_scope():
    assignments = pd.concat(
        [
            _month("2023-01", [0, 0, 1], ids=[1, 2, 3], method=method)
            for method in ("leiden_hellinger", "leiden_cosine")
        ]
        + [
            _month("2023-02", [8, 7, 8], ids=[2, 3, 4], method=method)
            for method in ("leiden_hellinger", "leiden_cosine")
        ]
    )
    transitions, _, summary = temporal_tables(
        assignments, _agreement(assignments), variant_evidence=_variants(assignments)
    )
    primary = transitions.loc[transitions["method"].eq("leiden_hellinger")]
    alternatives = transitions.loc[transitions["method"].eq("leiden_cosine")]
    assert primary["territory_id"].tolist() == [2, 3]
    np.testing.assert_allclose(primary["transition_reproduction_fraction"], 1)
    assert not primary["switched"].any()
    assert alternatives["transition_reproduction_fraction"].isna().all()
    assert alternatives["robust_switch"].isna().all()
    assert pd.isna(
        summary.loc[summary["method"].eq("leiden_cosine"), "robust_switch_share_of_all"].iloc[0]
    )


def test_missing_variant_month_is_unassessed_and_missing_common_id_is_rejected():
    assignments = pd.concat([_month("2023-01", [0, 0]), _month("2023-02", [3, 4])])
    evidence = _variants(assignments)
    partial = {"2023-01": evidence["2023-01"]}
    transitions, _, summary = temporal_tables(assignments, pd.DataFrame(), variant_evidence=partial)
    assert transitions["robust_switch"].isna().all()
    assert summary["robust_switch_share_of_all"].isna().all()
    evidence["2023-02"] = {
        "ids": np.array([1]),
        "labels": np.ones((21, 1), dtype=int),
        "support": np.ones((21, 1), dtype=bool),
    }
    with pytest.raises(ValueError, match="common primary IDs"):
        temporal_tables(assignments, pd.DataFrame(), variant_evidence=evidence)


def test_no_common_ids_yields_undefined_summary_and_no_flows():
    assignments = pd.concat([_month("2023-01", [0], ids=[1]), _month("2023-02", [0], ids=[2])])
    transitions, flows, summary = temporal_tables(assignments, pd.DataFrame())
    assert transitions.empty and flows.empty
    assert summary["ncommon"].iloc[0] == 0
    assert summary[["ari", "switch_share", "robust_switch_share_of_all"]].isna().all().all()
    assert str(transitions["robust_switch"].dtype) == "boolean"


def test_empty_assignments_and_nonadjacent_singletons_have_stable_output_schema():
    assignments = _month("2023-01", [0])
    for frame in (assignments, assignments.iloc[:0]):
        transitions, flows, summary = temporal_tables(frame, pd.DataFrame())
        assert transitions.empty and flows.empty and summary.empty
        assert "territory_id" in transitions
        assert "row_share" in flows
        assert "ncommon" in summary


def test_duplicate_assignment_and_agreement_keys_are_rejected():
    assignments = pd.concat([_month("2023-01", [0]), _month("2023-02", [1])])
    with pytest.raises(ValueError, match="Duplicate.*assignment"):
        temporal_tables(pd.concat([assignments, assignments.iloc[:1]]), pd.DataFrame())
    agreement = _agreement(assignments)
    with pytest.raises(ValueError, match="Duplicate.*agreement"):
        temporal_tables(assignments, pd.concat([agreement, agreement.iloc[:1]]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("month", "2023-13", "YYYY-MM"),
        ("territory_id", 1.5, "integers"),
        ("label", np.nan, "nulls"),
        ("method", "unknown", "method"),
        ("p_food", -0.1, "Profiles"),
        ("p_food", 1, "Profiles"),
    ],
)
def test_invalid_assignments_are_rejected(column, value, message):
    assignments = _month("2023-01", [0])
    assignments[column] = value
    with pytest.raises(ValueError, match=message):
        temporal_tables(assignments, pd.DataFrame())


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        temporal_tables(_month("2023-01", [0]), pd.DataFrame(), threshold)


def test_shuffled_input_is_deterministic_and_does_not_mutate_inputs():
    assignments = pd.concat(
        [_month("2023-01", [0, 0, 1]), _month("2023-02", [7, 7, 9])], ignore_index=True
    )
    agreement = _agreement(assignments)
    original_assignments, original_agreement = assignments.copy(), agreement.copy()
    expected = temporal_tables(assignments, agreement)
    actual = temporal_tables(assignments.sample(frac=1, random_state=13), agreement.sample(frac=1))
    for expected_table, actual_table in zip(expected, actual, strict=True):
        pd.testing.assert_frame_equal(expected_table, actual_table)
    pd.testing.assert_frame_equal(assignments, original_assignments)
    pd.testing.assert_frame_equal(agreement, original_agreement)
