"""Describe frozen December partitions using external context and regional permutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

METHODS = ("leiden_hellinger", "kmeans")
MONTHS = ("2023-12", "2024-12")
PROFILES = ("p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport")
INPUTS = {
    "assignments": "reports/initial_study/assignments.parquet",
    "population": "reports/enrichment_v2/population/municipal_population_wide.parquet",
    "economy": "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.parquet",
    "yearly_spatial": "reports/enrichment_v3/spatial/sber_yearly_spatial_attributes.csv",
    "static_spatial": "reports/enrichment_v3/spatial/sber_static_context_2024.csv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frame(path):
    if path.suffix == ".parquet":
        with duckdb.connect() as db:
            return db.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require_unique(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate join keys: {keys}")


def ss_decomposition(values, labels):
    values, labels = np.asarray(values, float), np.asarray(labels)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite complete observations are required")
    _, encoded = np.unique(labels, return_inverse=True)
    counts = np.bincount(encoded)
    means = np.bincount(encoded, weights=values) / counts
    total = np.sum((values - values.mean()) ** 2)
    between = np.sum(counts * (means - values.mean()) ** 2)
    within = np.sum((values - means[encoded]) ** 2)
    if not np.isclose(total, between + within, rtol=1e-10, atol=1e-10):
        raise AssertionError("SS decomposition failed")
    return float(total), float(between), float(within)


def regional_null(values, labels, regions, repeats=499, seed=13):
    values, labels, regions = np.asarray(values), np.asarray(labels), np.asarray(regions)
    positions = [np.flatnonzero(regions == code) for code in np.unique(regions)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        shuffled = labels.copy()
        for selected in positions:
            shuffled[selected] = rng.permutation(labels[selected])
        draws.append(ss_decomposition(values, shuffled)[1])
    movable = sum(len(p) for p in positions if len(np.unique(labels[p])) > 1)
    return np.asarray(draws), movable


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ordered = np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1))
    result = np.empty_like(pvalues)
    result[order] = np.minimum(1, ordered)
    return result


def self_checks():
    x, labels = np.array([0.0, 0.0, 2.0, 2.0]), np.array([0, 0, 1, 1])
    assert ss_decomposition(x, labels) == (4.0, 4.0, 0.0)
    assert ss_decomposition(x, 3 - labels) == (4.0, 4.0, 0.0)
    null, movable = regional_null(x, labels, labels, 19)
    assert (null == 4).all() and movable == 0
    assert np.isclose((1 + np.sum(null >= 4)) / 20, 1)
    null, movable = regional_null(x, labels, np.zeros(4), 99)
    assert 0 in null and 4 in null and movable == 4
    assert ss_decomposition(x, np.zeros(4)) == (4.0, 0.0, 4.0)
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    return {
        "status": "PASS",
        "checks": 7,
        "scope": "hand-calculated SS, blocked null, label invariance, Holm",
    }


def prepare_cohort(repo):
    frames = {name: read_frame(repo / path) for name, path in INPUTS.items()}
    for frame in frames.values():
        frame["territory_id"] = pd.to_numeric(frame.territory_id, errors="raise").astype(int)
    assignments = frames["assignments"]
    assignments = assignments.loc[
        assignments.month.isin(MONTHS) & assignments.method.isin(METHODS)
    ].copy()
    require_unique(assignments, ["month", "method", "territory_id"])
    if not np.allclose(assignments[list(PROFILES)].sum(axis=1), 1):
        raise ValueError("Expected normalized five-category profiles")
    assignments["year"] = assignments.month.str[:4].astype(int)
    populations = frames["population"].copy()
    populations["year"] = populations.reference_date.str[:4].astype(int)
    require_unique(populations, ["territory_id", "year"])
    wages = frames["economy"].loc[frames["economy"].indicator_id.eq("8423007")].copy()
    wages["year"] = wages.year.astype(int)
    require_unique(wages, ["territory_id", "year"])
    valid = (
        wages.cell_status.eq("observed")
        & wages.unit.eq("рубль")
        & wages.period_code.eq("79")
        & wages.okved2_code.eq("101")
        & wages.official_code_present.eq("True")
        & wages.sber_annual_code_present.eq("True")
    )
    wages["wage"] = pd.to_numeric(wages.value.where(valid), errors="coerce")
    wages["wage_temporal_review"] = wages.temporal_review_required.eq("True")
    spatial = frames["yearly_spatial"].copy()
    spatial["year"] = spatial.snapshot_date.str[:4].astype(int)
    require_unique(spatial, ["territory_id", "year"])
    static = frames["static_spatial"]
    require_unique(static, ["territory_id"])
    cohort = (
        assignments.merge(
            populations[["territory_id", "year", "reference_date", "population_total"]],
            on=["territory_id", "year"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            wages[["territory_id", "year", "wage", "wage_temporal_review", "cell_status"]],
            on=["territory_id", "year"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            spatial[
                ["territory_id", "year", "region_code", "region_name", "municipal_district_type"]
            ],
            on=["territory_id", "year"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            static[["territory_id", "market_access", "road_distance_to_sber_flagged_center_km"]],
            on="territory_id",
            how="left",
            validate="many_to_one",
        )
    )
    for field in [
        "wage",
        "population_total",
        "market_access",
        "road_distance_to_sber_flagged_center_km",
    ]:
        cohort[field] = pd.to_numeric(cohort[field], errors="coerce")
    for source, target in [("wage", "log_wage"), ("population_total", "log_population")]:
        cohort[target] = np.log(cohort[source].where(cohort[source] > 0))
    cohort["log1p_road_distance"] = np.log1p(
        cohort.road_distance_to_sber_flagged_center_km.where(
            cohort.road_distance_to_sber_flagged_center_km >= 0
        )
    )
    cohort.loc[
        cohort.year.ne(2024),
        ["market_access", "log1p_road_distance", "road_distance_to_sber_flagged_center_km"],
    ] = np.nan
    counts = cohort.groupby(["month", "method"]).territory_id.nunique()
    if len(counts) != 4 or not counts.eq(2016).all():
        raise ValueError("Expected all four frozen partitions on 2016 municipalities")
    return cohort.sort_values(["month", "method", "territory_id"])


def run(repo, output, repeats=499, seed=13):
    if output.exists():
        raise FileExistsError("Choose a new analysis directory; frozen results are not overwritten")
    checks = self_checks()
    inputs = {name: {"path": path, "sha256": sha(repo / path)} for name, path in INPUTS.items()}
    cohort = prepare_cohort(repo)
    dispersions, permutations, concentration, compositions, groups, agreement = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for (month, method), frame in cohort.groupby(["month", "method"], sort=True):
        variables = ["log_wage", "log_population"]
        if month == "2024-12":
            variables += ["market_access", "log1p_road_distance"]
        for variable in variables:
            selected = frame.loc[
                np.isfinite(frame[variable]) & frame.region_code.notna() & frame.region_code.ne("")
            ]
            x, labels = selected[variable].to_numpy(), selected.label.to_numpy()
            total, between, within = ss_decomposition(x, labels)
            draws, movable = regional_null(
                x, labels, selected.region_code.to_numpy(), repeats, seed
            )
            if total <= 0:
                raise ValueError("Constant feature does not define a variance share")
            p = (1 + np.sum(draws >= between - 1e-12)) / (repeats + 1)
            dispersions.append(
                {
                    "month": month,
                    "method": method,
                    "variable": variable,
                    "n_available": len(selected),
                    "n_partition": len(frame),
                    "n_clusters_complete_cases": selected.label.nunique(),
                    "ss_total": total,
                    "ss_between": between,
                    "ss_within": within,
                    "between_share": between / total,
                    "within_share": within / total,
                    "null_between_share_median": np.median(draws) / total,
                    "null_between_share_q025": np.quantile(draws, 0.025) / total,
                    "null_between_share_q975": np.quantile(draws, 0.975) / total,
                    "excess_above_null_mean_share": (between - draws.mean()) / total,
                    "p_within_region_one_sided": p,
                    "permutations": repeats,
                    "seed": seed,
                    "n_regions": selected.region_code.nunique(),
                    "n_permutable_units": movable,
                    "wage_flagged_annual_links": int(selected.wage_temporal_review.sum())
                    if variable == "log_wage"
                    else 0,
                }
            )
            permutations.extend(
                {
                    "month": month,
                    "method": method,
                    "variable": variable,
                    "replicate": i + 1,
                    "ss_between": b,
                    "between_share": b / total,
                }
                for i, b in enumerate(draws)
            )
        for attribute in ["region_code", "municipal_district_type"]:
            selected = frame.loc[frame[attribute].notna() & frame[attribute].ne("")]
            table = pd.crosstab(selected.label, selected[attribute])
            concentration.append(
                {
                    "month": month,
                    "method": method,
                    "attribute": attribute,
                    "n_available": len(selected),
                    "n_partition": len(frame),
                    "n_attribute_classes": selected[attribute].nunique(),
                    "ami": adjusted_mutual_info_score(selected.label, selected[attribute]),
                    "ari": adjusted_rand_score(selected.label, selected[attribute]),
                    "weighted_dominant_attribute_share": table.max(axis=1).sum() / len(selected),
                }
            )
            for label, row in table.iterrows():
                compositions.extend(
                    {
                        "month": month,
                        "method": method,
                        "label": label,
                        "attribute": attribute,
                        "attribute_value": value,
                        "count": int(count),
                        "group_share": count / row.sum(),
                    }
                    for value, count in row.items()
                    if count
                )
        for label, members in frame.groupby("label"):
            row = {"month": month, "method": method, "label": label, "n": len(members)}
            for feature in PROFILES:
                row[feature + "_mean"] = members[feature].mean()
            for feature in ["wage", "population_total", "market_access", "log1p_road_distance"]:
                row[feature + "_n"] = int(members[feature].notna().sum())
                row[feature + "_median"] = members[feature].median()
            for attr in ["region_code", "municipal_district_type"]:
                shares = members[attr].value_counts(normalize=True)
                row[attr + "_dominant"] = shares.index[0]
                row[attr + "_dominant_share"] = shares.iloc[0]
                row[attr + "_hhi"] = np.square(shares).sum()
            groups.append(row)
    for month, frame in cohort.groupby("month"):
        table = frame.pivot(index="territory_id", columns="method", values="label")
        agreement.append(
            {
                "month": month,
                "n": len(table),
                "ari_leiden_kmeans": adjusted_rand_score(table[METHODS[0]], table[METHODS[1]]),
                "ami_leiden_kmeans": adjusted_mutual_info_score(
                    table[METHODS[0]], table[METHODS[1]]
                ),
            }
        )
    dispersion = pd.DataFrame(dispersions)
    dispersion["p_holm_all_12_descriptive_tests"] = holm_adjust(
        dispersion.p_within_region_one_sided.to_numpy()
    )
    output.mkdir(parents=True)
    tables = {
        "feature_dispersion.csv": dispersion,
        "permutation_draws.csv": pd.DataFrame(permutations),
        "partition_concentration.csv": pd.DataFrame(concentration),
        "group_attribute_composition.csv": pd.DataFrame(compositions),
        "group_context.csv": pd.DataFrame(groups),
        "partition_method_agreement.csv": pd.DataFrame(agreement),
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    with duckdb.connect() as db:
        db.register("cohort", cohort)
        db.execute(
            "COPY cohort TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(output / "analytical_cohort.parquet")],
        )
    manifest = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "exploratory frozen-partition description; no fitting or economic true labels",
        "months": MONTHS,
        "methods": METHODS,
        "seed": seed,
        "permutations": repeats,
        "population_reference": "January 1 of same year, not December-end population",
        "wage_reference": "calendar-year average monthly nominal wage, excluding small businesses",
        "geometry": "conditional annual code links; geographic equivalence unproven",
        "permutation_contract": "labels permuted within regions on feature-complete samples; "
        "region-by-label counts preserved",
        "aggregation": "equal municipality weights; natural logs of positive wage and population",
        "input_hashes": inputs,
        "script_sha256": sha(__file__),
        "self_checks": checks,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "duckdb"]},
        "python": platform.python_version(),
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "results": dispersion.to_dict("records"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_checks()))
    else:
        directory = args.output or args.repo_root / "reports/enriched_analysis/cluster_context"
        result = run(args.repo_root, directory, args.permutations, args.seed)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(directory),
                    "tests": len(result["results"]),
                }
            )
        )
