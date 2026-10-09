"""Exploratory regional-CPI adjustment of municipality total spending, 2023–2024.

Real values use each region's own December 2022 price base, not a common spatial
price level. The five normalized category profiles are only checked for scalar
invariance; incompatible category-specific CPI groups are never applied.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from municipal_profiles.data import (
    EXPECTED_PERIODS,
    PUBLISHED_CATEGORIES,
    TOTAL_CATEGORY,
    load_panel,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: pd.Series) -> dict:
    x = values.to_numpy(dtype=float)
    require(len(x) > 0 and np.isfinite(x).all(), "Summary requires finite observations")
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "sd_population": float(np.std(x)),
        **{
            key: float(np.quantile(x, q))
            for key, q in {
                "minimum": 0,
                "p05": 0.05,
                "p10": 0.10,
                "p25": 0.25,
                "median": 0.5,
                "p75": 0.75,
                "p90": 0.90,
                "p95": 0.95,
                "maximum": 1,
            }.items()
        },
        "positive_n": int(np.sum(x > 0)),
        "positive_share": float(np.mean(x > 0)),
        "zero_n": int(np.sum(x == 0)),
        "negative_n": int(np.sum(x < 0)),
    }


def formula_checks() -> dict:
    chain = np.cumprod(np.array([101.0, 102.0]) / 100)
    require(np.allclose(chain, [1.01, 1.0302], rtol=0, atol=1e-14), "January base check")
    require(np.allclose(100 * chain / chain, 100), "Constant real spending check")
    monthly_average = np.mean(np.array([10.0, 10.0]) / np.array([1.0, 2.0]))
    shortcut = np.mean([10.0, 10.0]) / np.mean([1.0, 2.0])
    require(
        np.isclose(monthly_average, 7.5) and not np.isclose(monthly_average, shortcut),
        "Monthly deflation must precede annual averaging",
    )
    return {
        "status": "PASS",
        "jan2023_index_included": True,
        "constant_real_spending_recovered": True,
        "annual_mean_shortcut_counterexample": {"correct": monthly_average, "shortcut": shortcut},
    }


def variance_decomposition(frame: pd.DataFrame, field: str) -> dict:
    values = frame[field]
    grand = values.mean()
    group = frame.groupby("source_territory_code")[field]
    group_mean = group.transform("mean")
    ss_total = float(((values - grand) ** 2).sum())
    ss_within = float(((values - group_mean) ** 2).sum())
    ss_between = float(((group_mean - grand) ** 2).sum())
    require(
        np.isclose(ss_total, ss_within + ss_between, rtol=1e-12, atol=1e-10),
        "Within/between variance identity failed",
    )
    return {
        "field": field,
        "n_municipalities": len(frame),
        "n_regions": int(frame.source_territory_code.nunique()),
        "weighting": "equal municipality; regional means weighted by cohort municipality count",
        "ss_total": ss_total,
        "ss_within": ss_within,
        "ss_between": ss_between,
        "within_fraction": ss_within / ss_total if ss_total else None,
        "between_fraction": ss_between / ss_total if ss_total else None,
        "interpretation": (
            "descriptive arithmetic decomposition; not causal explained variance or inference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir or root / "reports/enriched_analysis/prices"
    out.mkdir(parents=True, exist_ok=True)
    price_root = root / "reports/enrichment_v3/prices"
    paths = [
        root / "data/raw/consumption.parquet",
        price_root / "cpi_monthly_2023_2024.csv",
        price_root / "municipal_region_price_links.csv",
    ]
    synthetic = formula_checks()
    panel = load_panel(paths[0], complete_only=True)
    ids = panel.territory_id.unique()
    require(len(ids) == 2016 and len(panel) == 2016 * 24 * 6, "Unexpected complete cohort")
    cpi = pd.read_csv(paths[1], dtype={"source_territory_code": str})
    cpi = cpi[cpi.category.eq("all") & cpi.territorial_level.eq("region")].copy()
    require(not cpi.duplicated(["source_territory_code", "month"]).any(), "Duplicate CPI key")
    require(
        cpi.index_base.eq("previous_month_100").all() and cpi.unit.eq("percent").all(),
        "Unsupported CPI units or index base",
    )
    require(np.isfinite(cpi.value).all() and cpi.value.gt(0).all(), "Invalid CPI values")
    links = pd.read_csv(paths[2], dtype={"source_territory_code": str, "region_code": str})
    links = links[links.territory_id.isin(ids)].copy()
    require(
        len(links) == 4032 and not links.duplicated(["territory_id", "reference_date"]).any(),
        "Expected unique cohort-year price links",
    )
    require(set(links.reference_date) == {"2023-01-01", "2024-01-01"}, "Unexpected link dates")
    require(links.reference_date.eq(links.snapshot_date).all(), "Snapshot/reference mismatch")
    require(
        links.groupby("territory_id").source_territory_code.nunique().eq(1).all(),
        "Municipality price region changes; require explicit temporal adjudication",
    )
    require(links.groupby("territory_id").region_code.nunique().eq(1).all(), "Region code changes")
    require(
        links[["source_territory_code", "region_code", "region_name"]].notna().all().all(),
        "Missing regional mapping",
    )
    selected_regions = set(links.source_territory_code)
    require(selected_regions <= set(cpi.source_territory_code), "Linked CPI region absent")
    cpi = cpi[cpi.source_territory_code.isin(selected_regions)].sort_values(
        ["source_territory_code", "month"]
    )
    require(
        all(tuple(g.month) == EXPECTED_PERIODS for _, g in cpi.groupby("source_territory_code")),
        "Every region must have all 24 ordered CPI months",
    )
    cpi["cpi_month_factor"] = cpi.value / 100
    cpi["price_level_dec2022_1"] = cpi.groupby("source_territory_code").cpi_month_factor.cumprod()
    cpi["year"] = cpi.month.str[:4].astype(int)
    jan = cpi[cpi.month.eq("2023-01")]
    require(
        np.array_equal(jan.price_level_dec2022_1, jan.cpi_month_factor),
        "January2023 CPI must enter the chain",
    )
    dec = cpi[cpi.month.isin(["2023-12", "2024-12"])]
    for code, part in cpi.groupby("source_territory_code"):
        require(
            np.isclose(
                part.price_level_dec2022_1.iloc[-1], part.cpi_month_factor.prod(), rtol=1e-14
            ),
            f"Chain reconciliation failed for {code}",
        )
    cpi.to_csv(out / "regional_cpi_chain.csv", index=False)
    one_link = links[links.reference_date.eq("2024-01-01")][
        ["territory_id", "region_code", "region_name", "source_territory_code"]
    ]
    monthly = panel[panel.category.eq(TOTAL_CATEGORY)][["territory_id", "date", "value"]].copy()
    monthly = monthly.rename(columns={"date": "month", "value": "nominal_total"})
    monthly = monthly.merge(one_link, on="territory_id", validate="many_to_one")
    monthly = monthly.merge(
        cpi[["source_territory_code", "month", "price_level_dec2022_1"]],
        on=["source_territory_code", "month"],
        validate="many_to_one",
        how="left",
    )
    require(
        len(monthly) == 48384 and monthly.price_level_dec2022_1.notna().all(),
        "Monthly CPI join must be complete without duplication",
    )
    monthly["real_total_dec2022"] = monthly.nominal_total / monthly.price_level_dec2022_1
    monthly["year"] = monthly.month.str[:4].astype(int)
    monthly = monthly.sort_values(["territory_id", "month"])
    monthly.to_csv(out / "municipal_monthly_price_context.csv", index=False)
    annual = (
        monthly.groupby(["territory_id", "year"])
        .agg(
            nominal_mean=("nominal_total", "mean"),
            real_mean_dec2022=("real_total_dec2022", "mean"),
            n_months=("month", "nunique"),
            annual_mean_price_level=("price_level_dec2022_1", "mean"),
        )
        .reset_index()
    )
    require(annual.n_months.eq(12).all(), "Annual means require 12 months")
    annual["annual_mean_deflator_shortcut"] = annual.nominal_mean / annual.annual_mean_price_level
    annual["shortcut_level_error_pct"] = 100 * (
        annual.annual_mean_deflator_shortcut / annual.real_mean_dec2022 - 1
    )
    annual.to_csv(out / "municipal_annual_price_context.csv", index=False)
    growth = one_link.sort_values("territory_id").copy()
    for year in (2023, 2024):
        selected = (
            annual[annual.year.eq(year)]
            .drop(columns="year")
            .rename(
                columns={
                    col: f"{col}_{year}"
                    for col in annual.columns
                    if col not in {"territory_id", "year"}
                }
            )
        )
        growth = growth.merge(selected, on="territory_id", validate="one_to_one")
    # Keep the public wide schema explicit and stable for the parent analysis.
    growth = growth.rename(
        columns={f"real_mean_dec2022_{y}": f"real_mean_{y}_dec2022" for y in (2023, 2024)}
    )
    growth["nominal_growth_pct"] = 100 * (growth.nominal_mean_2024 / growth.nominal_mean_2023 - 1)
    growth["real_growth_pct"] = 100 * (
        growth.real_mean_2024_dec2022 / growth.real_mean_2023_dec2022 - 1
    )
    growth["nominal_log_growth"] = np.log(growth.nominal_mean_2024 / growth.nominal_mean_2023)
    growth["real_log_growth"] = np.log(
        growth.real_mean_2024_dec2022 / growth.real_mean_2023_dec2022
    )
    growth["nominal_positive"] = growth.nominal_growth_pct.gt(0)
    growth["real_positive"] = growth.real_growth_pct.gt(0)
    growth["sign_changed"] = np.sign(growth.nominal_growth_pct).ne(np.sign(growth.real_growth_pct))
    growth["shortcut_real_growth_pct"] = 100 * (
        growth.annual_mean_deflator_shortcut_2024 / growth.annual_mean_deflator_shortcut_2023 - 1
    )
    growth["shortcut_growth_error_pp"] = growth.shortcut_real_growth_pct - growth.real_growth_pct
    growth["deflator_use"] = "regional_temporal_price_adjustment_not_spatial_PPP"
    require(len(growth) == 2016 and not growth.territory_id.duplicated().any(), "Growth output key")
    growth.to_csv(out / "municipal_price_growth.csv", index=False)

    # This is a numerical check of a scalar identity, not category-specific deflation.
    five = panel[panel.category.isin(PUBLISHED_CATEGORIES)].pivot(
        index=["territory_id", "date"], columns="category", values="value"
    )
    five = five.loc[:, list(PUBLISHED_CATEGORIES)]
    factors = monthly.set_index(["territory_id", "month"]).price_level_dec2022_1
    factors.index.names = five.index.names
    factors = factors.reindex(five.index).to_numpy()
    x = five.to_numpy()
    before = x / x.sum(axis=1, keepdims=True)
    scalar_scaled = x / factors[:, None]
    after = scalar_scaled / scalar_scaled.sum(axis=1, keepdims=True)
    difference = np.abs(before - after)
    require(
        np.allclose(before, after, atol=1e-14, rtol=0), "Common CPI changes normalized profiles"
    )

    regional_records = []
    for source_code, part in growth.groupby("source_territory_code", sort=True):
        row = {
            "source_territory_code": source_code,
            "region_code": part.region_code.iloc[0],
            "region_name": part.region_name.iloc[0],
            "n_municipalities": len(part),
        }
        for field in ("nominal_growth_pct", "real_growth_pct"):
            row.update({f"{field}_{k}": v for k, v in stats(part[field]).items()})
        regional_records.append(row)
    regional = pd.DataFrame(regional_records)
    regional.to_csv(out / "regional_growth_summary.csv", index=False)
    annual_cpi = cpi.groupby(["source_territory_code", "year"]).agg(
        annual_mean_price_level=("price_level_dec2022_1", "mean")
    )
    inflation = annual_cpi.unstack("year")["annual_mean_price_level"]
    inflation["mean_price_level_growth_pct_2024_vs_2023"] = 100 * (
        inflation[2024] / inflation[2023] - 1
    )
    inflation = inflation.rename(
        columns={2023: "mean_price_level_2023", 2024: "mean_price_level_2024"}
    )
    inflation = inflation.join(
        dec.pivot(
            index="source_territory_code", columns="month", values="price_level_dec2022_1"
        ).rename(columns={"2023-12": "dec2023_price_level", "2024-12": "dec2024_price_level"})
    )
    inflation["dec2024_vs_dec2023_pct"] = 100 * (
        inflation.dec2024_price_level / inflation.dec2023_price_level - 1
    )
    inflation.reset_index().to_csv(out / "regional_price_summary.csv", index=False)
    decomposition = [
        variance_decomposition(growth, field)
        for field in (
            "nominal_growth_pct",
            "real_growth_pct",
            "nominal_log_growth",
            "real_log_growth",
        )
    ]
    summary = {
        "status": "exploratory_descriptive",
        "n_municipalities": len(growth),
        "n_regions": len(regional),
        "weighting": "each municipality equal; no population/client weights",
        "municipal_growth_pct": {
            field: stats(growth[field]) for field in ("nominal_growth_pct", "real_growth_pct")
        },
        "municipal_annual_levels": {
            field: stats(growth[field])
            for field in (
                "nominal_mean_2023",
                "nominal_mean_2024",
                "real_mean_2023_dec2022",
                "real_mean_2024_dec2022",
            )
        },
        "sign_transitions": growth.groupby(["nominal_positive", "real_positive"])
        .size()
        .reset_index(name="n")
        .to_dict("records"),
        "between_regions_equal_region_weight": {
            field: stats(regional[field])
            for field in (
                "nominal_growth_pct_median",
                "real_growth_pct_median",
                "nominal_growth_pct_mean",
                "real_growth_pct_mean",
            )
        },
        "regional_cohort_size": stats(regional.n_municipalities),
        "descriptive_variance_decomposition": decomposition,
        "annual_mean_deflator_shortcut_error": {
            "growth_error_pp": stats(growth.shortcut_growth_error_pp),
            "absolute_growth_error_pp": stats(growth.shortcut_growth_error_pp.abs()),
            "sign_differs_from_correct_real_growth_n": int(
                np.sign(growth.shortcut_real_growth_pct).ne(np.sign(growth.real_growth_pct)).sum()
            ),
            "annual_level_error_pct": stats(annual.shortcut_level_error_pct),
        },
        "region_price_growth_equal_region_weight": stats(
            inflation.mean_price_level_growth_pct_2024_vs_2023
        ),
        "limitations": [
            "2024 previously inspected: all analysis exploratory, no confirmatory holdout.",
            "Sber values are estimated monthly means, "
            "not municipal expenditure totals or resident weights.",
            "Regional CPI is repeated across municipalities, "
            "not independent municipal price evidence.",
            "Real levels retain region-specific Dec2022 price bases, "
            "not spatially comparable purchasing power.",
            "Common deflator cancels in each monthly five-category profile; "
            "no category-specific adjustment attempted.",
            "Changes do not establish welfare, physical consumption, "
            "causality or forecast performance.",
            "Source revision/publication vintage not reconstructed; "
            "actual input files acquired later.",
        ],
    }
    qa = {
        "status": "PASS",
        "formula_checks": synthetic,
        "cohort_rows": len(panel),
        "monthly_total_rows": len(monthly),
        "annual_rows": len(annual),
        "growth_rows": len(growth),
        "region_months": len(cpi),
        "n_regions": len(regional),
        "all_municipality_years_have_12_months": True,
        "all_cpi_chains_have_24_months": True,
        "regional_mapping_stable_across_years": True,
        "january2023_enters_chain": True,
        "chain_endpoint_product_reconciled": True,
        "regional_aggregate_rows_excluded": True,
        "monthly_profile_invariance": {
            "rows": len(five),
            "components": difference.size,
            "max_absolute_error": float(difference.max()),
            "atol": 1e-14,
            "identity": "(x_j / D) / sum_k(x_k / D) = x_j / sum_k(x_k) for D>0",
        },
        "variance_decomposition_identity": "PASS",
        "no_external_requests": True,
    }
    for filename, document in (("summary.json", summary), ("qa.json", qa)):
        (out / filename).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_with_interpretation_limits",
        "script_sha256": sha(Path(__file__)),
        "inputs": [
            {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in paths
        ],
        "outputs": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(out.iterdir())
            if p.suffix in {".csv", ".json"} and p.name != "manifest.json"
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "qa": qa,
                "growth_pct": summary["municipal_growth_pct"],
                "sign_transitions": summary["sign_transitions"],
                "decomposition": decomposition[:2],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
