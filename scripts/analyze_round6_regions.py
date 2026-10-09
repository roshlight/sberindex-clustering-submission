"""Conditional regional interpretation using frozen annual profiles and source snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROTOCOL = "configs/round6_analysis_v1.json"
INPUTS = [
    PROTOCOL,
    "reports/data_round6/economy/regional_context_candidate.csv",
    "reports/round5_analysis/audit/annual_features.csv",
    "reports/enriched_analysis/annual_networks/assignments.parquet",
    "reports/enriched_analysis/annual_networks/annual_profiles.parquet",
    "reports/enriched_analysis/annual_networks/yearly_support.csv",
    "uv.lock",
]
OUTCOMES = ["log_total", "p_food"]
KEYS = ["territory_id", "year"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def correlation(x, y):
    x = x - np.mean(x, axis=-1, keepdims=True)
    y = y - np.mean(y, axis=-1, keepdims=True)
    den = np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-12, np.sum(x * y, axis=-1) / den, np.nan)


def rank_statistics(data):
    """Rows are municipalities, columns x/y/controls; accepts a batch."""
    data = np.asarray(data, dtype=float)
    ranks = rankdata(data, axis=-2)
    x, y = ranks[..., 0], ranks[..., 1]
    controls = np.concatenate([np.ones_like(x)[..., None], ranks[..., 2:]], axis=-1)
    xy = ranks[..., :2]
    residual = xy - controls @ (np.linalg.pinv(controls) @ xy)
    return correlation(x, y), correlation(residual[..., 0], residual[..., 1])


def uncertainty(data, indices):
    rho, partial = rank_statistics(data)
    boot_rho, boot_partial = rank_statistics(data[indices])
    loo = np.asarray([rank_statistics(np.delete(data, i, axis=0)) for i in range(len(data))])
    result = {"n": len(data), "rho": float(rho), "partial_rho": float(partial)}
    for name, values, omitted in [
        ("rho", boot_rho, loo[:, 0]),
        ("partial_rho", boot_partial, loo[:, 1]),
    ]:
        valid = values[np.isfinite(values)]
        require(len(valid) >= 0.99 * len(values), "Too many undefined bootstrap draws")
        lo, hi = np.quantile(valid, [0.025, 0.975])
        result.update(
            {
                name + "_ci_low": float(lo),
                name + "_ci_high": float(hi),
                name + "_bootstrap_valid": len(valid),
                name + "_loo_min": float(np.nanmin(omitted)),
                name + "_loo_max": float(np.nanmax(omitted)),
                name + "_loo_valid": int(np.isfinite(omitted).sum()),
            }
        )
    return result


def rank_eta_squared(values, labels):
    ranked = rankdata(values)
    total = np.sum((ranked - ranked.mean()) ** 2)
    require(total > 0, "No attribute variation")
    between = sum(
        np.sum(labels == label) * (ranked[labels == label].mean() - ranked.mean()) ** 2
        for label in np.unique(labels)
    )
    return float(between / total)


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(q, 1)
    return result


def paired_ids(frame):
    require(not frame.duplicated(KEYS).any(), "Duplicate municipality-year key")
    return sorted(
        set(frame.loc[frame.year.eq(2023), "territory_id"])
        & set(frame.loc[frame.year.eq(2024), "territory_id"])
    )


def build(repo, out):
    require(not out.exists(), "Choose a new output directory")
    inputs = {p: sha(repo / p) for p in INPUTS}
    config = json.loads((repo / PROTOCOL).read_text("utf-8"))
    raw = pd.read_csv(repo / INPUTS[1])
    annual = pd.read_csv(repo / INPUTS[2])
    require(len(raw) == 276 and not raw.duplicated([*KEYS, "variable"]).any(), "Raw keys")
    require(not annual.duplicated(KEYS).any(), "Frozen annual keys")
    require(not raw.historical_boundary_equivalence_proven.any(), "Changed geographic contract")
    require(not raw.replace_frozen_bdpmo_value.any(), "Changed admission contract")
    c = duckdb.connect()
    assignments = c.execute("select * from read_parquet(?)", [str(repo / INPUTS[3])]).df()
    profiles = c.execute("select * from read_parquet(?)", [str(repo / INPUTS[4])]).df()
    c.close()
    labels = assignments.loc[assignments.method.eq("leiden_hellinger")].copy()
    labels["year"] = labels.month.str[:4].astype(int)
    require(not labels.duplicated(KEYS).any(), "Frozen label keys")
    check = annual.merge(profiles, on=KEYS, suffixes=("", "_frozen"), validate="one_to_one")
    require(len(check) == len(annual), "Frozen profiles coverage")
    for name in ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]:
        require(np.allclose(check[name], check[name + "_frozen"], atol=1e-12, rtol=0), name)
    primary = raw.loc[
        [
            config["regional_primary_attributes"].get(str(r.region_code)) == r.variable
            for r in raw.itertuples()
        ]
    ].copy()
    require(len(primary) == 144, "Primary rows")
    keep = [
        *KEYS,
        "population_total",
        "urban_share",
        "log_total",
        "total_mean",
        "p_food",
        "core_eligible",
        "conservative_eligible",
        "code_type_changed",
        "calendar_review",
    ]
    panel = primary.merge(annual[keep], on=KEYS, validate="one_to_one")
    panel = panel.rename(columns={"value": "value_thousand_rubles"})
    panel = panel.merge(labels[[*KEYS, "label"]], on=KEYS, validate="one_to_one")
    panel["rubles_per_person"] = panel.value_thousand_rubles * 1000 / panel.population_total
    panel["log_intensity"] = np.log(panel.rubles_per_person.where(panel.rubles_per_person > 0))
    valid = (
        panel.value_thousand_rubles.gt(0)
        & panel.population_total.gt(0)
        & np.isfinite(panel[["log_intensity", "log_total", "p_food"]]).all(axis=1)
    )
    panel["row_eligible"] = valid
    panel["paired_available"] = False
    panel["paired_stable_code"] = False
    panel["paired_without_largest"] = False
    panel["paired_urban_complete"] = False
    panel["paired_stable_urban_complete"] = False
    coverage, estimates, alignments, cluster_profiles, summaries = [], [], [], [], []
    for region in sorted(panel.region_code.unique()):
        regional = panel.loc[panel.region_code.eq(region)].copy()
        ids = paired_ids(regional.loc[regional.row_eligible])
        require(len(ids) >= config["minimum_n"], "Insufficient paired sample")
        available = regional.loc[regional.territory_id.isin(ids)].copy()
        changed = set(available.loc[available.code_type_changed, "territory_id"])
        stable = sorted(set(ids) - changed)
        urban = paired_ids(available.loc[np.isfinite(available.urban_share)])
        first = available.loc[available.year.eq(2023)].sort_values(
            ["population_total", "territory_id"], ascending=[False, True]
        )
        largest = int(first.iloc[0].territory_id)
        samples = {
            "paired_available": ids,
            "paired_stable_code": stable,
            "paired_without_largest": sorted(set(ids) - {largest}),
            "paired_urban_complete": urban,
            "paired_stable_urban_complete": sorted(set(stable) & set(urban)),
        }
        for sample, members in samples.items():
            panel.loc[panel.region_code.eq(region) & panel.territory_id.isin(members), sample] = (
                True
            )
            rng = np.random.default_rng(config["seed"] + int(region) + len(members))
            draws = rng.integers(0, len(members), size=(config["bootstrap_draws"], len(members)))
            for year in config["years"]:
                part = available.loc[available.year.eq(year) & available.territory_id.isin(members)]
                part = part.sort_values("territory_id")
                coverage.append(
                    dict(
                        region_code=int(region),
                        year=year,
                        sample=sample,
                        n=len(part),
                        candidates=len(regional.loc[regional.year.eq(year)]),
                        largest_excluded_id=largest if sample == "paired_without_largest" else "",
                        conditional_name_join=True,
                    )
                )
                require(len(part) == len(members), "Unpaired year population")
                for outcome in OUTCOMES:
                    record = dict(
                        region_code=int(region),
                        year=year,
                        sample=sample,
                        variable=part.variable.iloc[0],
                        outcome=outcome,
                        n=len(part),
                    )
                    if len(part) < config["minimum_n"]:
                        record.update(status="NOT_ESTIMATED_BELOW_MINIMUM_N")
                    else:
                        data = part[["log_intensity", outcome, "population_total"]].to_numpy()
                        record.update(status="ESTIMATED", **uncertainty(data, draws))
                        if sample in {"paired_urban_complete", "paired_stable_urban_complete"}:
                            require(
                                np.isfinite(part.urban_share).all(), "Urban subset has missing data"
                            )
                            urban_data = part[
                                ["log_intensity", outcome, "population_total", "urban_share"]
                            ].to_numpy()
                            urban_stats = uncertainty(urban_data, draws)
                            record.update(
                                {
                                    "urban_" + k: v
                                    for k, v in urban_stats.items()
                                    if k.startswith("partial_rho")
                                }
                            )
                    estimates.append(record)
                if sample != "paired_available":
                    continue
                values, labs = part.log_intensity.to_numpy(), part.label.to_numpy()
                observed = rank_eta_squared(values, labs)
                prng = np.random.default_rng(config["seed"] + int(region) * 100 + year)
                null = np.array(
                    [
                        rank_eta_squared(values, prng.permutation(labs))
                        for _ in range(config["permutation_draws"])
                    ]
                )
                alignments.append(
                    dict(
                        region_code=int(region),
                        year=year,
                        n=len(part),
                        clusters=part.label.nunique(),
                        rank_eta_squared=observed,
                        null_mean=float(null.mean()),
                        permutation_p=(1 + int((null >= observed - 1e-15).sum())) / (1 + len(null)),
                        permutation_draws=len(null),
                    )
                )
                for label, group in part.groupby("label", sort=True):
                    cluster_profiles.append(
                        dict(
                            region_code=int(region),
                            year=year,
                            label=int(label),
                            n=len(group),
                            small_group=len(group) < 5,
                            intensity_median=float(group.rubles_per_person.median()),
                            food_share_median=float(group.p_food.median()),
                            total_median=float(group.total_mean.median()),
                        )
                    )
                summaries.append(
                    dict(
                        region_code=int(region),
                        year=year,
                        n=len(part),
                        variable=part.variable.iloc[0],
                        intensity_median=float(part.rubles_per_person.median()),
                        intensity_q25=float(part.rubles_per_person.quantile(0.25)),
                        intensity_q75=float(part.rubles_per_person.quantile(0.75)),
                        largest_id=largest,
                        largest_value=float(
                            part.loc[part.territory_id.eq(largest), "value_thousand_rubles"].iloc[0]
                        ),
                    )
                )
    aligned = pd.DataFrame(alignments)
    aligned["permutation_q_bh"] = bh_adjust(aligned.permutation_p)
    out.mkdir(parents=True)
    outputs = {
        "regional_panel.csv": panel.sort_values(["region_code", *KEYS]),
        "coverage.csv": pd.DataFrame(coverage),
        "associations.csv": pd.DataFrame(estimates),
        "cluster_alignment.csv": aligned,
        "cluster_profiles.csv": pd.DataFrame(cluster_profiles),
        "regional_summary.csv": pd.DataFrame(summaries),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    qa = dict(
        status="PASS_WITH_INTERPRETATION_LIMITS",
        candidate_rows=len(panel),
        association_rows=len(estimates),
        cluster_checks=len(alignments),
        frozen_profiles_matched=True,
        models_trained=0,
        new_clusters=0,
        accepted_historical_joins=0,
        old_panel_changed=False,
        same_municipalities_in_both_years=True,
        confidence_scope=(
            "Conditional paired-ID resampling; no spatial-dependence or multiplicity correction"
        ),
        cluster_null=(
            "Conditional exchangeability, four-test BH; no spatial or cluster-stability validation"
        ),
    )
    save(out / "QA.json", qa)
    require(inputs == {p: sha(repo / p) for p in INPUTS}, "Inputs changed during analysis")
    save(
        out / "manifest.json",
        dict(
            inputs=inputs,
            script_sha256=sha(Path(__file__)),
            outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    print(json.dumps(qa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo.resolve(), args.out.resolve())
