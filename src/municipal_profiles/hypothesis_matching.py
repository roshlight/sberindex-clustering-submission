"""Outcome-blind matching and fixed randomization checks for hypotheses H3/H4.

The geometry and eligibility rules are fixed before examining profile outcomes.
These observational checks do not identify an effect of manufacturing employment.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

from municipal_profiles.dynamics import PROFILE_COLUMNS
from municipal_profiles.enriched_analysis import CORE, clr

SECTOR_COLUMNS = [f"sector_{section}_share" for section in "ABCDHOPQ"]
MODEL_PANEL = "reports/round2_analysis/models/model_panel.csv"
SOURCE_PATHS = [
    MODEL_PANEL,
    "reports/enriched_analysis/core_v2/annual_panel.csv",
    "reports/round2_analysis/models/selected_sector_features.csv",
]
CORE_CALIPER = 0.5
SECTOR_RMS_CALIPER = 1.0
PAIR_COLUMNS = [
    "left_index",
    "right_index",
    "left_territory_id",
    "right_territory_id",
    "left_region_code",
    "right_region_code",
    "left_municipal_district_type",
    "right_municipal_district_type",
    "cost",
    "core_max_abs_z",
    "core_rms_z",
    "sector_rms_z",
    "share_C_difference",
]
BALANCE_COLUMNS = [
    "hypothesis",
    "feature",
    "contrast",
    "n_pairs",
    "eligible_sd",
    "left_mean",
    "right_mean",
    "mean_difference_z",
    "mean_abs_difference_z",
    "rms_difference_z",
    "max_abs_difference_z",
]


def _standardize(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    center = values.mean(axis=0) if len(values) else np.zeros(len(columns))
    scale = values.std(axis=0, ddof=0) if len(values) else np.ones(len(columns))
    scale = np.where(scale > 0, scale, 1.0)
    return (values - center) / scale, center, scale


def _minimum_cost_matching(n_nodes, edges, costs, seed=13):
    """Maximum cardinality, then minimum total cost; each node is used at most once.

    Seeded edge ordering resolves solver tie/order behavior without adding jitter
    to the distance objective. HiGHS is deterministic for fixed inputs.
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    costs = np.asarray(costs, dtype=float)
    if len(costs) != len(edges) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Matching requires finite nonnegative costs for every edge")
    if not len(edges):
        return np.empty(0, dtype=int)
    if (
        (edges < 0).any()
        or (edges >= n_nodes).any()
        or (edges[:, 0] == edges[:, 1]).any()
        or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)
    ):
        raise ValueError("Invalid or duplicate matching edge")
    order = np.random.default_rng(seed).permutation(len(edges))
    ordered_edges = edges[order]
    incidence = csc_matrix(
        (
            np.ones(2 * len(edges)),
            (ordered_edges.ravel(), np.repeat(np.arange(len(edges)), 2)),
        ),
        shape=(n_nodes, len(edges)),
    )
    bounds = Bounds(np.zeros(len(edges)), np.ones(len(edges)))
    integrality = np.ones(len(edges), dtype=int)
    options = {"mip_rel_gap": 0.0}
    first = milp(
        -np.ones(len(edges)),
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(incidence, 0, 1),
        options=options,
    )
    if not first.success:
        raise RuntimeError(f"Maximum-cardinality matching failed: {first.message}")
    cardinality = int(np.rint(first.x.sum()))
    constraints = vstack([incidence, csc_matrix(np.ones((1, len(edges))))], format="csc")
    second = milp(
        costs[order],
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            constraints,
            np.r_[np.zeros(n_nodes), cardinality],
            np.r_[np.ones(n_nodes), cardinality],
        ),
        options=options,
    )
    if not second.success:
        raise RuntimeError(f"Minimum-cost matching failed: {second.message}")
    selected = np.sort(order[np.flatnonzero(second.x > 0.5)])
    if len(selected) != cardinality or len(np.unique(edges[selected])) != 2 * cardinality:
        raise RuntimeError("Solver returned an invalid matching")
    return selected


def _pair_frame(cohort, edges, selected, core_z, sector_z=None):
    rows = []
    for edge_index in selected:
        left, right = edges[edge_index]
        a, b = cohort.iloc[left], cohort.iloc[right]
        core_delta = core_z[left] - core_z[right]
        core_rms = float(np.sqrt(np.mean(core_delta**2)))
        sector_rms = (
            float(np.sqrt(np.mean((sector_z[left] - sector_z[right]) ** 2)))
            if sector_z is not None
            else np.nan
        )
        cost = (
            float(np.sqrt(0.5 * core_rms**2 + 0.5 * sector_rms**2))
            if sector_z is not None
            else float(np.linalg.norm(core_delta))
        )
        rows.append(
            [
                left,
                right,
                a.territory_id,
                b.territory_id,
                a.region_code,
                b.region_code,
                a.municipal_district_type,
                b.municipal_district_type,
                cost,
                float(np.max(np.abs(core_delta))),
                core_rms,
                sector_rms,
                float(a.sector_C_share - b.sector_C_share),
            ]
        )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _h3_pairs(cohort, seed=13):
    """Pair high/low manufacturing share within region and municipal type."""
    cohort = cohort.copy().reset_index(drop=True)
    core_z, center, scale = _standardize(cohort, CORE)
    cohort["share_C_median"] = cohort.groupby(
        ["region_code", "municipal_district_type"], sort=True
    ).sector_C_share.transform("median")
    cohort["high_manufacturing"] = cohort.sector_C_share.gt(cohort.share_C_median)
    edges, costs = [], []
    for _, stratum in cohort.groupby(["region_code", "municipal_district_type"], sort=True):
        high = stratum.index[stratum.high_manufacturing].to_numpy()
        low = stratum.index[~stratum.high_manufacturing].to_numpy()
        for left in high:
            delta = core_z[left] - core_z[low]
            allowed = np.all(np.abs(delta) <= CORE_CALIPER, axis=1)
            for right, difference in zip(low[allowed], delta[allowed], strict=True):
                edges.append((left, right))
                costs.append(float(np.linalg.norm(difference)))
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    selected = _minimum_cost_matching(len(cohort), edges, costs, seed)
    pairs = _pair_frame(cohort, edges, selected, core_z)
    return cohort, pairs, {"core_center": center.tolist(), "core_scale": scale.tolist()}, edges


def _h4_pairs(cohort, seed=13):
    """Pair different regions using economic and sector data, never profiles."""
    cohort = cohort.copy().reset_index(drop=True)
    core_z, core_center, core_scale = _standardize(cohort, CORE)
    sector_z, sector_center, sector_scale = _standardize(cohort, SECTOR_COLUMNS)
    regions = cohort.region_code.to_numpy()
    edges, costs = [], []
    for left in range(len(cohort)):
        rights = np.arange(left + 1, len(cohort))
        core_delta = core_z[left] - core_z[rights]
        sector_delta = sector_z[left] - sector_z[rights]
        sector_rms = np.sqrt(np.mean(sector_delta**2, axis=1))
        allowed = (
            (regions[left] != regions[rights])
            & np.all(np.abs(core_delta) <= CORE_CALIPER, axis=1)
            & (sector_rms <= SECTOR_RMS_CALIPER)
        )
        for right, c_delta, s_rms in zip(
            rights[allowed], core_delta[allowed], sector_rms[allowed], strict=True
        ):
            edges.append((left, right))
            costs.append(float(np.sqrt(0.5 * np.mean(c_delta**2) + 0.5 * s_rms**2)))
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    selected = _minimum_cost_matching(len(cohort), edges, costs, seed)
    pairs = _pair_frame(cohort, edges, selected, core_z, sector_z)
    scales = {
        "core_center": core_center.tolist(),
        "core_scale": core_scale.tolist(),
        "sector_center": sector_center.tolist(),
        "sector_scale": sector_scale.tolist(),
    }
    return cohort, pairs, scales, edges


def _sign_flip_null(differences, regions, n_resamples, seed):
    """One shared sign per region for both H3 outcomes and all its pairs."""
    differences = np.asarray(differences, dtype=float)
    regions = np.asarray(regions)
    unique = np.unique(regions)
    sums = np.asarray([differences[regions == region].sum(axis=0) for region in unique])
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], (n_resamples, len(unique)))
    return signs @ sums / len(differences)


def _hellinger_pair_mean(profiles, left, right):
    roots = np.sqrt(np.asarray(profiles, dtype=float))
    return float(np.sqrt(np.sum((roots[left] - roots[right]) ** 2, axis=1) / 2).mean())


def _profile_permutation_null(profiles, regions, left, right, n_resamples, seed):
    """Permute whole profile vectors among every eligible row within its region."""
    regions = np.asarray(regions)
    groups = [np.flatnonzero(regions == region) for region in np.unique(regions)]
    rng = np.random.default_rng(seed)
    roots = np.sqrt(np.asarray(profiles, dtype=float))
    null = np.empty(n_resamples)
    for repeat in range(n_resamples):
        order = np.arange(len(profiles))
        for indices in groups:
            order[indices] = rng.permutation(indices)
        delta = roots[order[left]] - roots[order[right]]
        null[repeat] = np.sqrt(np.sum(delta**2, axis=1) / 2).mean()
    return null


def _tail_p(observed, null, direction):
    if direction not in {"positive", "negative"}:
        raise ValueError("Unknown test direction")
    # Count numerical ties conservatively, including profiles fixed by singletons.
    tolerance = 1e-12 * max(1.0, abs(observed))
    extreme = (
        null >= observed - tolerance if direction == "positive" else null <= observed + tolerance
    )
    return float((1 + extreme.sum()) / (len(null) + 1))


def _balance(cohort, pairs, hypothesis):
    rows = []
    columns = [*CORE, "sector_C_share"] if hypothesis == "H3" else [*CORE, *SECTOR_COLUMNS]
    _, _, scale = _standardize(cohort, columns)
    if pairs.empty:
        return rows
    left, right = pairs.left_index.to_numpy(), pairs.right_index.to_numpy()
    a = cohort.iloc[left][columns].to_numpy(dtype=float)
    b = cohort.iloc[right][columns].to_numpy(dtype=float)
    for index, column in enumerate(columns):
        delta = (a[:, index] - b[:, index]) / scale[index]
        rows.append(
            {
                "hypothesis": hypothesis,
                "feature": column,
                "contrast": "high_minus_low" if hypothesis == "H3" else "cross_region_pair",
                "n_pairs": len(pairs),
                "eligible_sd": scale[index],
                "left_mean": float(a[:, index].mean()),
                "right_mean": float(b[:, index].mean()),
                "mean_difference_z": float(delta.mean()),
                "mean_abs_difference_z": float(np.abs(delta).mean()),
                "rms_difference_z": float(np.sqrt(np.mean(delta**2))),
                "max_abs_difference_z": float(np.abs(delta).max()),
            }
        )
    return rows


def _read_panel(root, year):
    panel = pd.read_csv(root / MODEL_PANEL)
    required = [
        "territory_id",
        "year",
        "region_code",
        "municipal_district_type",
        "core_eligible",
        *CORE,
        *SECTOR_COLUMNS,
        *PROFILE_COLUMNS,
    ]
    missing = set(required).difference(panel.columns)
    if missing:
        raise ValueError(f"Missing model-panel columns: {sorted(missing)}")
    if panel.duplicated(["territory_id", "year"]).any():
        raise ValueError("Nonunique model-panel territory/year key")
    if panel[["territory_id", "year"]].isna().any().any():
        raise ValueError("Missing model-panel territory/year key")
    panel = (
        panel.loc[panel.year.eq(year), required].sort_values("territory_id").reset_index(drop=True)
    )
    flag = panel.core_eligible.astype(str).str.lower().map({"true": True, "false": False})
    if flag.isna().any():
        raise ValueError("Invalid core_eligible flag")
    panel["core_eligible"] = flag.astype(bool)
    for column in [*CORE, *SECTOR_COLUMNS, *PROFILE_COLUMNS]:
        panel[column] = pd.to_numeric(panel[column], errors="raise")
    sectors = panel[SECTOR_COLUMNS].to_numpy(dtype=float)
    invalid = ~np.isnan(sectors) & (~np.isfinite(sectors) | (sectors < 0) | (sectors > 1))
    if invalid.any():
        raise ValueError("Sector shares must be observed fractions or missing")
    profiles = panel[list(PROFILE_COLUMNS)].to_numpy(dtype=float)
    if len(panel) and (
        not np.isfinite(profiles).all()
        or (profiles <= 0).any()
        or not np.allclose(profiles.sum(axis=1), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("Expected finite strictly positive normalized annual profiles")
    valid_core = (
        panel.core_eligible
        & np.isfinite(panel[CORE]).all(axis=1)
        & panel.region_code.notna()
        & panel.municipal_district_type.notna()
    )
    panel["eligible_H3"] = valid_core & panel.sector_C_share.notna()
    panel["eligible_H4"] = valid_core & panel[SECTOR_COLUMNS].notna().all(axis=1)
    return panel


def _result(test_id, hypothesis, year, cohort, pairs, direction, method, unit):
    matched = pd.unique(pairs[["left_index", "right_index"]].to_numpy().ravel())
    regions = pd.unique(pairs[["left_region_code", "right_region_code"]].to_numpy().ravel())
    return {
        "test_id": test_id,
        "hypothesis": hypothesis,
        "year": year,
        "estimate": np.nan,
        "expected_direction": direction,
        "p_raw": 1.0,
        "n_entities": len(matched),
        "n_regions": len(regions),
        "n_pairs": len(pairs),
        "n_eligible": len(cohort),
        "method": method,
        "unit": unit,
        "available": bool(len(pairs)),
        "observed": np.nan,
        "null_mean": np.nan,
        "improvement": np.nan,
    }


def run_matching_tests(
    root: Path, output: Path, *, year=2024, seed=13, n_resamples=9999
) -> pd.DataFrame:
    """Return three raw directional p-values and save the complete matching audit.

    ``n_entities`` counts matched municipalities; ``n_eligible`` and coverage.json
    retain the full cohorts. No correction for multiple testing is applied here.
    H4's estimate is observed distance minus the regional-permutation null mean.
    Unavailable checks retain p=1 so they remain in the declared Holm family.
    """
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    root, output = Path(root), Path(output)
    panel = _read_panel(root, year)
    h3, pairs3, scales3, edges3 = _h3_pairs(panel.loc[panel.eligible_H3], seed)
    h4, pairs4, scales4, edges4 = _h4_pairs(panel.loc[panel.eligible_H4], seed)
    rows = [
        _result(
            f"H3_manufacturing_{target}",
            "H3",
            year,
            h3,
            pairs3,
            direction,
            "maximum-cardinality minimum-cost matching; regional-block sign flips",
            "mean paired CLR difference (high minus low)",
        )
        for target, direction in [("marketplaces", "positive"), ("transport", "negative")]
    ]
    rows.append(
        _result(
            "H4_cross_region_similarity",
            "H4",
            year,
            h4,
            pairs4,
            "negative",
            "maximum-cardinality minimum-cost matching; within-region profile permutation",
            "mean Hellinger distance minus permutation mean",
        )
    )
    null3 = pd.DataFrame(columns=["resample", "marketplaces", "transport"])
    null4 = pd.DataFrame(columns=["resample", "mean_distance"])
    if len(pairs3):
        profiles = clr(h3[list(PROFILE_COLUMNS)].to_numpy(dtype=float))
        left, right = pairs3.left_index.to_numpy(), pairs3.right_index.to_numpy()
        targets = [
            list(PROFILE_COLUMNS).index(f"p_{name}") for name in ["marketplaces", "transport"]
        ]
        differences = profiles[left][:, targets] - profiles[right][:, targets]
        observed = differences.mean(axis=0)
        null = _sign_flip_null(differences, pairs3.left_region_code.to_numpy(), n_resamples, seed)
        for index, row in enumerate(rows[:2]):
            row.update(
                estimate=float(observed[index]),
                observed=float(observed[index]),
                null_mean=float(null[:, index].mean()),
                p_raw=_tail_p(observed[index], null[:, index], row["expected_direction"]),
            )
        pairs3["marketplaces_CLR_difference"] = differences[:, 0]
        pairs3["transport_CLR_difference"] = differences[:, 1]
        null3 = pd.DataFrame(null, columns=["marketplaces", "transport"])
        null3.insert(0, "resample", np.arange(1, n_resamples + 1))
    if len(pairs4):
        profiles = h4[list(PROFILE_COLUMNS)].to_numpy(dtype=float)
        left, right = pairs4.left_index.to_numpy(), pairs4.right_index.to_numpy()
        observed = _hellinger_pair_mean(profiles, left, right)
        null = _profile_permutation_null(
            profiles, h4.region_code.to_numpy(), left, right, n_resamples, seed
        )
        mean_null = float(null.mean())
        rows[2].update(
            estimate=observed - mean_null,
            observed=observed,
            null_mean=mean_null,
            improvement=mean_null - observed,
            p_raw=_tail_p(observed, null, "negative"),
        )
        roots = np.sqrt(profiles)
        pairs4["hellinger_distance"] = np.sqrt(
            np.sum((roots[left] - roots[right]) ** 2, axis=1) / 2
        )
        null4 = pd.DataFrame({"resample": np.arange(1, n_resamples + 1), "mean_distance": null})
    result = pd.DataFrame(rows)
    panel["share_C_median_H3"] = panel.territory_id.map(h3.set_index("territory_id").share_C_median)
    panel["high_manufacturing_H3"] = panel.territory_id.map(
        h3.set_index("territory_id").high_manufacturing
    )
    for name, cohort, pairs in [("H3", h3, pairs3), ("H4", h4, pairs4)]:
        matched = cohort.iloc[
            np.unique(pairs[["left_index", "right_index"]].to_numpy().ravel()).astype(int)
        ].territory_id
        panel[f"matched_{name}"] = panel.territory_id.isin(matched)
    coverage = {
        "year": year,
        "annual_entities": len(panel),
        "core_eligible_entities": int(panel.core_eligible.sum()),
        "H3": {
            "eligible_entities": len(h3),
            "eligible_regions": int(h3.region_code.nunique()),
            "high_entities": int(h3.high_manufacturing.sum()),
            "low_entities": int((~h3.high_manufacturing).sum()),
            "candidate_edges": len(edges3),
            "candidate_entities": len(np.unique(edges3)),
            "matched_entities": 2 * len(pairs3),
            "pairs": len(pairs3),
            "matched_regions": int(pairs3.left_region_code.nunique()),
        },
        "H4": {
            "eligible_entities": len(h4),
            "eligible_regions": int(h4.region_code.nunique()),
            "candidate_edges": len(edges4),
            "candidate_entities": len(np.unique(edges4)),
            "matched_entities": 2 * len(pairs4),
            "pairs": len(pairs4),
            "matched_regions": len(
                pd.unique(pairs4[["left_region_code", "right_region_code"]].to_numpy().ravel())
            ),
            "singleton_regions": int(h4.groupby("region_code").size().eq(1).sum()),
            "permutable_entities": int(h4.groupby("region_code").size().loc[lambda s: s > 1].sum()),
        },
    }
    protocol = {
        "year": year,
        "seed": seed,
        "n_resamples": n_resamples,
        "core_columns": CORE,
        "sector_columns": SECTOR_COLUMNS,
        "scaling": "eligible cohort population SD (ddof=0); zero SD uses scale 1",
        "core_caliper_each": CORE_CALIPER,
        "sector_RMS_caliper_H4": SECTOR_RMS_CALIPER,
        "matching": "maximum cardinality then minimum total distance; no entity reuse",
        "tie_order": "numpy default_rng(seed) edge ordering; deterministic HiGHS; no cost jitter",
        "H3_split": "share_C > stratum median vs <= median; same region and municipal type",
        "H3_cost": "Euclidean distance of three standardized economic logs; no share_C gap",
        "H3_null": "shared regional signs for all matched pairs and both CLR targets",
        "H4_cost": "sqrt(0.5*mean(core_z_difference^2)+0.5*mean(sector_z_difference^2))",
        "H4_null": "whole five-category profiles permuted within all eligible regional cohorts",
        "H4_singletons": "fixed profile vectors; included in the conditional null",
        "p_value": "one-sided Monte Carlo (1 + extreme count)/(n_resamples + 1); ties included",
        "profile": "arithmetic mean of twelve monthly normalized five-category profiles",
        "H3_scaling": scales3,
        "H4_scaling": scales4,
    }
    hashes = [
        {"source": source, "sha256": hashlib.sha256((root / source).read_bytes()).hexdigest()}
        for source in SOURCE_PATHS
        if (root / source).is_file()
    ]
    hashes.append(
        {
            "source": "src/municipal_profiles/hypothesis_matching.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [
        ("results.csv", result),
        ("pairs_H3.csv", pairs3),
        ("pairs_H4.csv", pairs4),
        ("sample.csv", panel),
        ("nulls_H3.csv", null3),
        ("nulls_H4.csv", null4),
        (
            "balance.csv",
            pd.DataFrame(
                [*_balance(h3, pairs3, "H3"), *_balance(h4, pairs4, "H4")], columns=BALANCE_COLUMNS
            ),
        ),
    ]:
        frame.to_csv(output / filename, index=False)
    for filename, value in [
        ("coverage.json", coverage),
        ("matching_protocol.json", protocol),
        ("source_hashes.json", hashes),
    ]:
        (output / filename).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result
