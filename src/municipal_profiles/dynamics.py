"""Calendar-aware changes in municipality profiles and monthly partitions.

Raw cluster numbers belong to a method and month, not a global cluster lineage.
The maximum-overlap alignment is recomputed for each pair on common territories;
its unmatched negative labels are local bookkeeping, not persistent new clusters.
All raw pair flows are retained so that a one-to-one alignment does not conceal
splits or merges. Agreement is a sensitivity assessment, not a correctness score.
"""

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from municipal_profiles.networks import align_labels

PROFILE_COLUMNS = (
    "p_food",
    "p_health",
    "p_marketplaces",
    "p_foodservice",
    "p_transport",
)
_METHODS = {"leiden_hellinger", "leiden_cosine", "kmeans", "kmeans_matched_k"}
_PRIMARY_METHOD = "leiden_hellinger"
_KEY = ["month", "territory_id"]
_PAIR_COLUMNS = ["from_month", "to_month", "lag", "method"]
_TRANSITION_COLUMNS = [
    *_PAIR_COLUMNS,
    "territory_id",
    "from_label",
    "to_label",
    "aligned_to_label",
    "switched",
    "hellinger_profile_change",
    "from_agreement",
    "to_agreement",
    "robust_endpoints",
    "transition_reproduction_fraction",
    "robust_switch",
]
_FLOW_COLUMNS = [*_PAIR_COLUMNS, "from_label", "to_label", "count", "row_share", "col_share"]
_SUMMARY_COLUMNS = [
    *_PAIR_COLUMNS,
    "ncommon",
    "ari",
    "switch_share",
    "robust_switch_share_of_all",
    "median_profile_change",
]
_MONTH_PATTERN = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")


def _check_months(frame: pd.DataFrame) -> None:
    if (
        not frame["month"]
        .map(lambda value: isinstance(value, str) and _MONTH_PATTERN.fullmatch(value) is not None)
        .all()
    ):
        raise ValueError("month must contain strings in YYYY-MM format")


def _integer_column(frame: pd.DataFrame, column: str, *, positive: bool = False) -> None:
    values = pd.to_numeric(frame[column], errors="raise")
    if (
        not np.isfinite(values.to_numpy(dtype=float)).all()
        or not values.eq(np.floor(values)).all()
        or (positive and not values.gt(0).all())
    ):
        raise ValueError(f"{column} must contain finite {'positive ' if positive else ''}integers")
    frame[column] = values.astype("int64")


def _assignments(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*_KEY, "method", "label", *PROFILE_COLUMNS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")
    frame = frame.loc[:, required].copy()
    if frame.isna().any().any():
        raise ValueError("Assignment columns must not contain nulls")
    _check_months(frame)
    _integer_column(frame, "territory_id", positive=True)
    _integer_column(frame, "label")
    if not frame["method"].isin(_METHODS).all():
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if frame.duplicated([*_KEY, "method"]).any():
        raise ValueError("Duplicate (month, territory_id, method) assignment keys")
    profiles = frame.loc[:, list(PROFILE_COLUMNS)].to_numpy(dtype=float)
    if (
        not np.isfinite(profiles).all()
        or (profiles < 0).any()
        or not np.allclose(profiles.sum(axis=1), 1, rtol=1e-6, atol=1e-9)
    ):
        raise ValueError("Profiles must be finite, nonnegative, and sum to one")
    frame[list(PROFILE_COLUMNS)] = profiles
    return frame.sort_values(["month", "method", "territory_id"]).reset_index(drop=True)


def _agreement(frame: pd.DataFrame) -> pd.DataFrame:
    # A completely empty table is valid even before a producer defines its schema.
    if frame.empty:
        return pd.DataFrame(columns=[*_KEY, "agreement"])
    required = [*_KEY, "agreement", "n_comparisons"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing agreement columns: {sorted(missing)}")
    frame = frame.loc[:, required].copy()
    _check_months(frame)
    _integer_column(frame, "territory_id", positive=True)
    if frame.duplicated(_KEY).any():
        raise ValueError("Duplicate (month, territory_id) agreement keys")
    scores = pd.to_numeric(frame["agreement"], errors="raise").astype(float)
    counts = pd.to_numeric(frame["n_comparisons"], errors="raise").astype(float)
    present_scores = scores.dropna()
    present_counts = counts.dropna()
    if not np.isfinite(present_scores).all() or not present_scores.between(0, 1).all():
        raise ValueError("Non-null agreement must be finite and between zero and one")
    if (
        not np.isfinite(present_counts).all()
        or not present_counts.ge(0).all()
        or not present_counts.eq(np.floor(present_counts)).all()
    ):
        raise ValueError("Non-null n_comparisons must be finite nonnegative integers")
    frame["agreement"] = scores.where(counts.gt(0))
    return frame[[*_KEY, "agreement"]]


def _month_ordinal(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:]) - 1


def _variants(evidence: Mapping | None) -> dict:
    if evidence is None:
        return {}
    result = {}
    for month, values in evidence.items():
        if not isinstance(month, str) or _MONTH_PATTERN.fullmatch(month) is None:
            raise ValueError("variant_evidence keys must be YYYY-MM month strings")
        if not {"ids", "labels", "support"}.issubset(values):
            raise ValueError("Each variant month needs ids, labels, and support")
        ids = np.asarray(values["ids"])
        labels = np.asarray(values["labels"])
        support = np.asarray(values["support"])
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or (ids <= 0).any()
            or (np.diff(ids) <= 0).any()
        ):
            raise ValueError("Variant ids must be strictly increasing positive integer codes")
        if labels.shape != (21, len(ids)) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Variant labels must be an integer array of shape (21, n_ids)")
        if support.shape != labels.shape or support.dtype != np.dtype(bool):
            raise ValueError("Variant support must be a Boolean array of shape (21, n_ids)")
        result[month] = {"ids": ids, "labels": labels, "support": support}
    return result


def _reproduction(
    transitions: pd.DataFrame, evidence: dict, from_month: str, to_month: str
) -> np.ndarray:
    if from_month not in evidence or to_month not in evidence:
        return np.full(len(transitions), np.nan)
    ids = transitions["territory_id"].to_numpy()
    selected = []
    for month in (from_month, to_month):
        values = evidence[month]
        if not np.isin(ids, values["ids"]).all():
            raise ValueError(f"Variant evidence for {month} does not cover all common primary IDs")
        indices = np.searchsorted(values["ids"], ids)
        selected.append((values["labels"][:, indices], values["support"][:, indices]))
    (before, support_before), (after, support_after) = selected
    alternative_switched = np.vstack(
        [left != align_labels(left, right) for left, right in zip(before, after, strict=True)]
    )
    reproduced = (
        support_before
        & support_after
        & (alternative_switched == transitions["switched"].to_numpy()[None, :])
    )
    return reproduced.mean(axis=0)


def _pair_transitions(
    common: pd.DataFrame,
    scores: pd.DataFrame,
    evidence: dict,
    metadata: dict[str, str | int],
    threshold: float,
) -> pd.DataFrame:
    records = pd.DataFrame({"territory_id": common["territory_id"]})
    for name, value in metadata.items():
        records[name] = value
    before = common["label_from"].to_numpy()
    after = common["label_to"].to_numpy()
    records["from_label"] = before
    records["to_label"] = after
    records["aligned_to_label"] = align_labels(before, after)
    records["switched"] = before != records["aligned_to_label"].to_numpy()
    p = common[[f"{name}_from" for name in PROFILE_COLUMNS]].to_numpy(dtype=float)
    q = common[[f"{name}_to" for name in PROFILE_COLUMNS]].to_numpy(dtype=float)
    records["hellinger_profile_change"] = np.clip(
        np.linalg.norm(np.sqrt(p) - np.sqrt(q), axis=1) / np.sqrt(2), 0, 1
    )
    for endpoint in ("from", "to"):
        endpoint_scores = scores.loc[
            scores["month"].eq(metadata[f"{endpoint}_month"]), ["territory_id", "agreement"]
        ].set_index("territory_id")["agreement"]
        records[f"{endpoint}_agreement"] = (
            records["territory_id"].map(endpoint_scores).astype(float)
            if metadata["method"] == _PRIMARY_METHOD
            else np.nan
        )
    assessed = records[["from_agreement", "to_agreement"]].notna().all(axis=1)
    endpoints = records["from_agreement"].ge(threshold) & records["to_agreement"].ge(threshold)
    records["robust_endpoints"] = endpoints.astype("boolean").where(assessed, pd.NA)
    records["transition_reproduction_fraction"] = (
        _reproduction(records, evidence, metadata["from_month"], metadata["to_month"])
        if metadata["method"] == _PRIMARY_METHOD
        else np.nan
    )
    reproduced = records["transition_reproduction_fraction"]
    records["robust_switch"] = (
        (reproduced.ge(threshold) & records["switched"])
        .astype("boolean")
        .where(reproduced.notna(), pd.NA)
    )
    return records[_TRANSITION_COLUMNS]


def temporal_tables(
    assignments: pd.DataFrame,
    agreement: pd.DataFrame,
    threshold: float = 0.8,
    variant_evidence: Mapping | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return territory transitions, raw cluster flows, and pair-level summaries.

    Inputs must contain only the primary settings for each method; this function
    does not select seeds or resolutions. Pairs are exactly 1 or 12 calendar
    months apart within a method. Only common territory IDs enter comparison;
    appearance/disappearance is never a switch. Empty intersections have only a
    summary row with ``ncommon=0`` and undefined comparison statistics.

    ``row_share`` divides a raw flow by its source-group size among common IDs;
    ``col_share`` divides it by its destination-group size among common IDs.
    ``robust_endpoints`` is a nullable marginal comparison: both endpoint
    agreement scores >= threshold and both have positive comparison counts.
    It does not establish transition robustness. ``variant_evidence`` maps each
    month to sorted integer ``ids``, integer ``labels`` of shape (21, n_ids), and
    Boolean ``support`` of the same shape. Variant ordering must be identical
    across months; support means co-membership Jaccard to primary >= 0.8.

    Each variant pair is independently aligned on the same common IDs as the
    primary pair. ``transition_reproduction_fraction`` is the fraction of all 21
    variants with support at both ends AND the same switch status as primary.
    ``robust_switch`` requires a primary switch and this fraction >= threshold.
    Missing month evidence leaves it NA, even with high marginal agreement.
    Robustness is assessed only for leiden_hellinger. The summary share divides
    by all common IDs and is NaN if transition reproduction is unassessed.
    Reproduction cannot establish that a transition is economically meaningful.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    frame = _assignments(assignments)
    scores = _agreement(agreement)
    evidence = _variants(variant_evidence)
    transition_parts = []
    flow_parts = []
    summaries = []
    for method, method_frame in frame.groupby("method", sort=True):
        months = sorted(method_frame["month"].unique())
        ordinals = {_month_ordinal(month): month for month in months}
        monthly = {month: part for month, part in method_frame.groupby("month", sort=True)}
        for from_month in months:
            for lag in (1, 12):
                to_month = ordinals.get(_month_ordinal(from_month) + lag)
                if to_month is None:
                    continue
                metadata = {
                    "from_month": from_month,
                    "to_month": to_month,
                    "lag": lag,
                    "method": method,
                }
                common = monthly[from_month].merge(
                    monthly[to_month],
                    on="territory_id",
                    how="inner",
                    suffixes=("_from", "_to"),
                    validate="one_to_one",
                    sort=True,
                )
                if common.empty:
                    summaries.append(
                        {
                            **metadata,
                            "ncommon": 0,
                            "ari": np.nan,
                            "switch_share": np.nan,
                            "robust_switch_share_of_all": np.nan,
                            "median_profile_change": np.nan,
                        }
                    )
                    continue
                transitions = _pair_transitions(common, scores, evidence, metadata, threshold)
                transition_parts.append(transitions)
                flows = (
                    transitions.groupby(["from_label", "to_label"], sort=True)
                    .size()
                    .reset_index(name="count")
                )
                flows["row_share"] = flows["count"] / flows.groupby("from_label")[
                    "count"
                ].transform("sum")
                flows["col_share"] = flows["count"] / flows.groupby("to_label")["count"].transform(
                    "sum"
                )
                for name, value in metadata.items():
                    flows[name] = value
                flow_parts.append(flows[_FLOW_COLUMNS])
                all_assessed = transitions["robust_switch"].notna().all()
                summaries.append(
                    {
                        **metadata,
                        "ncommon": len(common),
                        "ari": float(
                            adjusted_rand_score(transitions["from_label"], transitions["to_label"])
                        ),
                        "switch_share": float(transitions["switched"].mean()),
                        "robust_switch_share_of_all": (
                            float(transitions["robust_switch"].mean()) if all_assessed else np.nan
                        ),
                        "median_profile_change": float(
                            transitions["hellinger_profile_change"].median()
                        ),
                    }
                )
    transitions = (
        pd.concat(transition_parts, ignore_index=True)
        if transition_parts
        else pd.DataFrame(columns=_TRANSITION_COLUMNS)
    )
    for column in ("robust_endpoints", "robust_switch"):
        transitions[column] = transitions[column].astype("boolean")
    flows = (
        pd.concat(flow_parts, ignore_index=True)
        if flow_parts
        else pd.DataFrame(columns=_FLOW_COLUMNS)
    )
    summary = pd.DataFrame(summaries, columns=_SUMMARY_COLUMNS)
    return (
        transitions.sort_values([*_PAIR_COLUMNS, "territory_id"]).reset_index(drop=True),
        flows.sort_values([*_PAIR_COLUMNS, "from_label", "to_label"]).reset_index(drop=True),
        summary.sort_values(_PAIR_COLUMNS).reset_index(drop=True),
    )
