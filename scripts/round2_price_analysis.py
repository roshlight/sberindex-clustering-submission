"""Descriptive city-price context and strictly matched product-price comparisons.

Uses frozen approved links only; no price imputation, city-to-district broadcasting,
clustering refit or official inflation claims. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

PROFILES = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
CONTROLS = ["log_population", "log_wage", "log_worker_concentration"]
TARGETS = ["log_total"] + ["clr_" + field for field in PROFILES]
PRICE_ROOT = "reports/data_round2/prices"
CORE_ROOT = "reports/enriched_analysis/core_v2"
PROTOCOL = {
    "status": "exploratory_observational_city_context_only",
    "unit": "approved city inside a primary municipality, annual reference geography conditional",
    "iszh_predictor": "log annual city ISZh, Russia=1; not local inflation",
    "outcomes": TARGETS,
    "controls": CONTROLS + ["municipality type", "region fixed effects"],
    "raw_and_partial_sample": "same complete core_eligible city-year sample, separate years",
    "sensitivity": "conservative_eligible stable-code sample, same controls",
    "bootstrap": "999 draws of whole regions with replacement; nuisance regression refit each draw",
    "seed": 13,
    "intervals": "percentile 95%; descriptive, marginal, not multiplicity adjusted or causal",
    "product_pair_key": (
        "same municipality, city OKTMO, source city code, month, product and full label/unit"
    ),
    "price_filter": "strictly positive in both years; no missing/zero/source-code repair",
    "annual_product_estimand": "mean2024/mean2023 over exactly 12 common positive months",
    "city_estimand": "unweighted median of eligible annual product price ratios, not CPI",
    "primary_min_products": 100,
    "sensitivity_min_products": [50, 100, 200, 300],
    "sensitivity_min_month_pairs": [10, 12],
    "common_basket_sensitivity": (
        "products available in >=90% of primary cities, cities with ALL these products"
    ),
    "no_cross_validation_or_prediction_claim": True,
    "no_sber_category_deflator": True,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8"
    )


def unique(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate keys: {keys}")


def verify_inputs(repo):
    prices = repo / PRICE_ROOT
    manifest = json.loads((prices / "manifest.json").read_text(encoding="utf8"))
    for item in manifest["outputs"]:
        if sha(prices / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed frozen price artifact: {item['path']}")
    core = repo / CORE_ROOT
    cm = json.loads((core / "manifest.json").read_text(encoding="utf8"))
    if sha(core / "annual_panel.parquet") != cm["outputs"]["annual_panel.parquet"]:
        raise ValueError("Changed core annual panel")
    verification = json.loads(
        (prices / "source_cell_verification_v3.json").read_text(encoding="utf8")
    )
    if verification["status"] != "PASS" or verification["verified_price_cells"] != 3836682:
        raise ValueError("Frozen full source-cell audit is missing")
    paths = [core / "annual_panel.parquet", core / "manifest.json", prices / "manifest.json"]
    paths += [
        prices / p
        for p in [
            "bridge_v2/city_context_observations.csv",
            "bridge_v2/primary_cohort_city_context.csv",
            "monthly_v3/primary_city_price_context.parquet",
            "monthly_v3/product_dictionary.csv",
            "monthly_v3/city_monthly_dimensions.csv",
            "source_cell_verification_v3.json",
        ]
    ]
    return {p.relative_to(repo).as_posix(): sha(p) for p in paths}


def prepare_panel(repo, out):
    db = duckdb.connect()
    core = db.from_parquet(str(repo / CORE_ROOT / "annual_panel.parquet")).df()
    price = pd.read_csv(
        repo / PRICE_ROOT / "bridge_v2/city_context_observations.csv",
        dtype={"source_city_oktmo_11": str},
    )
    unique(core, ["year", "territory_id"])
    unique(price, ["year", "territory_id"])
    if (
        price.temporal_review_required.any()
        or not price.bridge_status.eq("approved_evidence").all()
    ):
        raise ValueError("Non-approved price context")
    if not price.context_scope.eq("city_context_only").all() or not (price.value > 0).all():
        raise ValueError("Invalid ISZh scope/value")
    selected = [
        "year",
        "territory_id",
        "value",
        "source_city_name",
        "source_city_oktmo_11",
        "city_only_listed_settlement_in_parent",
    ]
    panel = core.merge(
        price[selected].rename(columns={"value": "iszh"}),
        on=["year", "territory_id"],
        how="left",
        validate="one_to_one",
    )
    if len(panel) != 4032 or panel.groupby("year").size().to_dict() != {2023: 2016, 2024: 2016}:
        raise ValueError("Changed full cohort")
    if panel.groupby("year").iszh.count().to_dict() != {2023: 215, 2024: 214}:
        raise ValueError("Changed price coverage")
    p = panel[PROFILES].to_numpy(float)
    if (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Positive normalized annual profiles required")
    clr = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
    panel[["clr_" + field for field in PROFILES]] = clr
    panel["log_iszh"] = np.log(panel.iszh)
    panel["price_available"] = panel.iszh.notna()
    panel.to_csv(out / "annual_context_panel.csv", index=False)
    return panel


def describe_selection(panel, out):
    rows, categories, counts, levels = [], [], [], []
    fields = ["population_total", "wage", "workers", "total_mean", "urban_share"] + PROFILES
    for year, data in panel.groupby("year"):
        for group, mask in [
            ("full_cohort", np.ones(len(data), bool)),
            ("price_selected", data.price_available),
            ("not_selected", ~data.price_available),
        ]:
            d = data.loc[mask]
            counts.append(
                {
                    "year": year,
                    "group": group,
                    "n": len(d),
                    "regions": d.region_code.nunique(),
                    "core_eligible": int(d.core_eligible.sum()),
                    "conservative_eligible": int(d.conservative_eligible.sum()),
                }
            )
            for field in fields:
                v = d[field].dropna()
                rows.append(
                    {
                        "year": year,
                        "group": group,
                        "field": field,
                        "n_numeric": len(v),
                        "median": v.median(),
                        "mean": v.mean(),
                        "q25": v.quantile(0.25),
                        "q75": v.quantile(0.75),
                    }
                )
        for field in ["municipal_district_type", "region_name"]:
            for value, d in data.groupby(field, dropna=False):
                categories.append(
                    {
                        "year": year,
                        "dimension": field,
                        "category": value,
                        "full_n": len(d),
                        "selected_n": int(d.price_available.sum()),
                        "within_category_coverage": d.price_available.mean(),
                        "share_of_full": len(d) / len(data),
                        "share_of_selected": d.price_available.sum() / data.price_available.sum(),
                    }
                )
        for field in [
            "log_population",
            "log_wage",
            "log_worker_concentration",
            "log_total",
        ] + PROFILES:
            a = data.loc[data.price_available, field].dropna()
            b = data.loc[~data.price_available, field].dropna()
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            levels.append(
                {
                    "year": year,
                    "field": field,
                    "selected_n": len(a),
                    "unselected_n": len(b),
                    "standardized_mean_difference": (a.mean() - b.mean()) / pooled,
                }
            )
    for name, data in [
        ("selection_numeric", rows),
        ("selection_categories", categories),
        ("coverage", counts),
        ("selection_smd", levels),
    ]:
        pd.DataFrame(data).to_csv(out / (name + ".csv"), index=False)
    level_rows = []
    for year, d in panel.loc[panel.price_available].groupby("year"):
        level_rows.append(
            {
                "year": year,
                "n": len(d),
                "median_iszh": d.iszh.median(),
                "q10": d.iszh.quantile(0.1),
                "q90": d.iszh.quantile(0.9),
                "min": d.iszh.min(),
                "max": d.iszh.max(),
                "below_russia_1": int((d.iszh < 1).sum()),
            }
        )
    pd.DataFrame(level_rows).to_csv(out / "iszh_levels.csv", index=False)


def within_region(matrix, groups):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    means = np.column_stack([np.bincount(inverse, weights=col) / sizes for col in matrix.T])
    return matrix - means[inverse]


def residualize(values, controls):
    beta, _, rank, _ = linalg.lstsq(controls, values, lapack_driver="gelsy")
    return values - controls @ beta, rank


def correlation_and_slope(x, y):
    x = x - x.mean()
    y = y - y.mean(axis=0)
    xx, yy = x @ x, np.sum(y * y, axis=0)
    if xx <= 1e-15 or np.any(yy <= 1e-15):
        return np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
    return (x @ y) / np.sqrt(xx * yy), (x @ y) / xx


def adjusted_associations(data, repeats, seed):
    type_columns = pd.get_dummies(data.municipal_district_type, drop_first=True, dtype=float)
    controls = np.column_stack([data[CONTROLS].to_numpy(float), type_columns.to_numpy(float)])
    xy = data[["log_iszh"] + TARGETS].to_numpy(float)
    groups = data.region_code.to_numpy()
    all_within = within_region(np.column_stack([controls, xy]), groups)
    cw, xyw = all_within[:, : controls.shape[1]], all_within[:, controls.shape[1] :]
    residual, rank = residualize(xyw, cw)
    raw, raw_slope = correlation_and_slope(xy[:, 0], xy[:, 1:])
    adjusted, adjusted_slope = correlation_and_slope(residual[:, 0], residual[:, 1:])
    region_indexes = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    boot_r, boot_b = [], []
    for _ in range(repeats):
        positions = np.concatenate(
            [region_indexes[i] for i in rng.integers(len(region_indexes), size=len(region_indexes))]
        )
        rb, _ = residualize(xyw[positions], cw[positions])
        r, b = correlation_and_slope(rb[:, 0], rb[:, 1:])
        boot_r.append(r)
        boot_b.append(b)
    boot_r, boot_b = np.asarray(boot_r), np.asarray(boot_b)
    records = []
    for j, target in enumerate(TARGETS):
        records.append(
            {
                "target": target,
                "n": len(data),
                "regions": len(region_indexes),
                "regions_with_2plus": sum(len(x) > 1 for x in region_indexes),
                "residual_dof": len(data) - len(region_indexes) - rank - 1,
                "raw_pearson": raw[j],
                "raw_slope": raw_slope[j],
                "partial_pearson": adjusted[j],
                "partial_r_ci_low": np.nanquantile(boot_r[:, j], 0.025),
                "partial_r_ci_high": np.nanquantile(boot_r[:, j], 0.975),
                "adjusted_slope": adjusted_slope[j],
                "slope_ci_low": np.nanquantile(boot_b[:, j], 0.025),
                "slope_ci_high": np.nanquantile(boot_b[:, j], 0.975),
                "bootstrap_finite": int(np.isfinite(boot_r[:, j]).sum()),
                "residual_price_variance_fraction": float(
                    np.sum(residual[:, 0] ** 2) / np.sum((xy[:, 0] - xy[:, 0].mean()) ** 2)
                ),
            }
        )
    evidence = data[["territory_id", "year", "source_city_name", "region_code"]].copy()
    evidence[["residual_log_iszh"] + ["residual_" + x for x in TARGETS]] = residual
    return pd.DataFrame(records), evidence


def associations(panel, out, repeats):
    rows, residuals = [], []
    for year in [2023, 2024]:
        for sample, eligibility in [
            ("core_complete", "core_eligible"),
            ("stable_code", "conservative_eligible"),
        ]:
            d = panel.loc[(panel.year == year) & panel.price_available & panel[eligibility]].copy()
            if not np.isfinite(d[["log_iszh"] + CONTROLS + TARGETS].to_numpy()).all():
                raise ValueError("Eligibility does not guarantee complete cases")
            result, res = adjusted_associations(d, repeats, 13)
            result.insert(0, "sample", sample)
            result.insert(0, "year", year)
            res["sample"] = sample
            rows.append(result)
            residuals.append(res)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out / "iszh_associations.csv", index=False)
    pd.concat(residuals, ignore_index=True).to_csv(out / "iszh_residual_evidence.csv", index=False)
    fields = [
        "territory_id",
        "source_city_name",
        "source_city_oktmo_11",
        "iszh",
        "log_total",
        "code_type_changed",
        "region_name",
    ]
    a = panel.loc[(panel.year == 2023) & panel.price_available, fields]
    b = panel.loc[(panel.year == 2024) & panel.price_available, fields]
    paired = a.merge(b, on="territory_id", suffixes=("_2023", "_2024"), validate="one_to_one")
    paired["same_city_code"] = paired.source_city_oktmo_11_2023.eq(paired.source_city_oktmo_11_2024)
    paired["relative_iszh_ratio_not_inflation"] = paired.iszh_2024 / paired.iszh_2023
    paired.to_csv(out / "paired_iszh.csv", index=False)
    same_city = paired.loc[paired.same_city_code]
    write_json(
        out / "paired_iszh_summary.json",
        {
            "both_approved_years": len(paired),
            "same_city_oktmo": len(same_city),
            "spearman_same_city_levels": same_city.iszh_2023.corr(
                same_city.iszh_2024, method="spearman"
            ),
            "median_same_city_relative_ratio_not_inflation": (
                same_city.relative_iszh_ratio_not_inflation.median()
            ),
            "not_inflation": True,
        },
    )
    return result


def build_matched(db):
    """Single audited query used by both real extraction and synthetic checks."""
    db.execute("""CREATE TEMP TABLE matched AS
        SELECT CAST(a.territory_id AS INTEGER) territory_id,a.month,a.product_code,
        a.source_city_code_literal,a.source_city_oktmo_11,
        a.value price2023,b.value price2024,b.value/a.value monthly_ratio,
        a.source_row source_row2023,a.source_column source_column2023,
        b.source_row source_row2024,b.source_column source_column2024,
        d.source_product_label
        FROM prices a JOIN prices b ON a.territory_id=b.territory_id
          AND a.month=b.month AND a.product_code=b.product_code
          AND a.source_city_code_literal=b.source_city_code_literal
          AND a.source_city_oktmo_11=b.source_city_oktmo_11
        JOIN dictionary d ON d.year=a.year AND d.month=a.month AND d.product_code=a.product_code
        JOIN dictionary e ON e.year=b.year AND e.month=b.month AND e.product_code=b.product_code
        WHERE a.year=2023 AND b.year=2024 AND a.value>0 AND b.value>0
          AND d.source_product_label=e.source_product_label""")


def monthly_prices(repo, out):
    db = duckdb.connect()
    p = repo / PRICE_ROOT / "monthly_v3"
    db.from_parquet(str(p / "primary_city_price_context.parquet")).create_view("prices")
    dictionary = pd.read_csv(p / "product_dictionary.csv")
    unique(dictionary, ["year", "month", "product_code"])
    if dictionary.groupby("product_code").source_product_label.nunique().max() != 1:
        raise ValueError("Changing product labels require new unit/composition review")
    db.register("dictionary", dictionary)
    build_matched(db)
    if db.sql("SELECT count(*) FROM matched WHERE territory_id=1051").fetchone()[0]:
        raise ValueError("Temporal quarantine was bypassed")
    if db.sql(
        "SELECT count(*) FROM matched WHERE month=1 AND source_city_code_literal='28405000000'"
    ).fetchone()[0]:
        raise ValueError("Source code quarantine was bypassed")
    product = db.sql("""SELECT territory_id,source_city_code_literal,source_city_oktmo_11,
        product_code,source_product_label,count(*) positive_month_pairs,
        avg(price2023) mean2023,avg(price2024) mean2024,
        avg(price2024)/avg(price2023) annual_ratio,
        median(monthly_ratio) median_monthly_ratio,min(monthly_ratio) min_monthly_ratio,
        max(monthly_ratio) max_monthly_ratio
        FROM matched GROUP BY 1,2,3,4,5 ORDER BY 1,4""").df()
    unique(product, ["territory_id", "product_code"])
    if (product.positive_month_pairs > 12).any():
        raise ValueError("Duplicate month pairs")
    db.register("product_pairs", product)
    db.sql("SELECT * FROM product_pairs").write_parquet(
        str(out / "city_product_pairs.parquet"), compression="zstd"
    )
    source_coverage = db.sql("""SELECT year,count(*) cells,count(value) numeric_cells,
        count(*) FILTER(WHERE value>0) positive,count(*) FILTER(WHERE value=0) zeros,
        count(distinct territory_id) municipalities FROM prices GROUP BY year ORDER BY year""").df()
    source_coverage.to_csv(out / "monthly_input_coverage.csv", index=False)
    db.sql("""SELECT month,count(*) positive_pair_cells,
        count(distinct territory_id) municipalities,count(distinct product_code) products
        FROM matched GROUP BY month ORDER BY month""").df().to_csv(
        out / "matched_month_coverage.csv", index=False
    )
    db.sql("""SELECT territory_id,count(*) positive_pair_cells,
        count(distinct product_code) products,count(distinct month) month_count
        FROM matched GROUP BY territory_id ORDER BY territory_id""").df().to_csv(
        out / "matched_city_coverage.csv", index=False
    )
    summaries, city_tables = [], []
    for months in [10, 12]:
        eligible = product.loc[product.positive_month_pairs >= months]
        grouped = (
            eligible.groupby("territory_id")
            .agg(
                n_products=("product_code", "size"),
                city_median_ratio=("annual_ratio", "median"),
                product_q10=("annual_ratio", lambda x: x.quantile(0.1)),
                product_q90=("annual_ratio", lambda x: x.quantile(0.9)),
                product_share_gt1=("annual_ratio", lambda x: (x > 1).mean()),
                median_of_monthly_medians=("median_monthly_ratio", "median"),
            )
            .reset_index()
        )
        for threshold in [50, 100, 200, 300]:
            cities = grouped.loc[grouped.n_products >= threshold].copy()
            cities["min_month_pairs"] = months
            cities["min_products"] = threshold
            city_tables.append(cities)
            summaries.append(
                {
                    "min_month_pairs": months,
                    "min_products": threshold,
                    "cities": len(cities),
                    "products_min": cities.n_products.min(),
                    "products_median": cities.n_products.median(),
                    "products_max": cities.n_products.max(),
                    "median_city_ratio": cities.city_median_ratio.median(),
                    "q10_city_ratio": cities.city_median_ratio.quantile(0.1),
                    "q90_city_ratio": cities.city_median_ratio.quantile(0.9),
                    "min_city_ratio": cities.city_median_ratio.min(),
                    "max_city_ratio": cities.city_median_ratio.max(),
                }
            )
    sensitivity = pd.DataFrame(summaries)
    city_ratios = pd.concat(city_tables, ignore_index=True)
    city_ratios.to_csv(out / "city_ratio_sensitivities.csv", index=False)
    sensitivity.to_csv(out / "ratio_threshold_sensitivity.csv", index=False)
    primary = city_ratios.loc[
        (city_ratios.min_month_pairs == 12) & (city_ratios.min_products == 100)
    ].copy()
    eligible = product.loc[
        (product.positive_month_pairs == 12) & product.territory_id.isin(primary.territory_id)
    ]
    frequencies = eligible.groupby("product_code").territory_id.nunique()
    basket = frequencies.loc[frequencies >= np.ceil(0.9 * len(primary))].index
    basket_products = eligible.loc[eligible.product_code.isin(basket)]
    complete_ids = basket_products.groupby("territory_id").size()
    complete_ids = complete_ids.loc[complete_ids == len(basket)].index
    common = (
        basket_products.loc[basket_products.territory_id.isin(complete_ids)]
        .groupby("territory_id")
        .agg(n_products=("product_code", "size"), common_basket_ratio=("annual_ratio", "median"))
        .reset_index()
    )
    common = common.merge(
        primary[["territory_id", "city_median_ratio"]], on="territory_id", validate="one_to_one"
    )
    common.to_csv(out / "common_basket_city_ratios.csv", index=False)
    dictionary.drop_duplicates("product_code").loc[
        lambda x: x.product_code.isin(basket), ["product_code", "source_product_label"]
    ].to_csv(out / "common_basket_products.csv", index=False)
    product.groupby("positive_month_pairs").size().rename("city_products").to_csv(
        out / "pair_completeness.csv"
    )
    products_by_code = (
        eligible.groupby(["product_code", "source_product_label"])
        .agg(
            cities=("territory_id", "nunique"),
            median_ratio=("annual_ratio", "median"),
            q10_ratio=("annual_ratio", lambda x: x.quantile(0.1)),
            q90_ratio=("annual_ratio", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    products_by_code.to_csv(out / "product_ratio_summary.csv", index=False)
    # Small directly replayable evidence set, explicitly selected as diagnostic tails.
    examples = product.loc[product.positive_month_pairs == 12].sort_values("annual_ratio")
    examples = pd.concat([examples.head(5), examples.tail(5)]).drop_duplicates(
        ["territory_id", "product_code"]
    )
    db.register("example_keys", examples[["territory_id", "product_code"]])
    db.sql("""SELECT m.* FROM matched m JOIN example_keys e USING(territory_id,product_code)
        ORDER BY territory_id,product_code,month""").df().to_csv(
        out / "source_cell_examples.csv", index=False
    )
    return (
        primary,
        sensitivity,
        common,
        {
            "matched_positive_month_pairs": db.sql("SELECT count(*) FROM matched").fetchone()[0],
            "city_product_pairs_any": len(product),
            "city_product_pairs_12": int(product.positive_month_pairs.eq(12).sum()),
            "common_basket_products": len(basket),
            "common_basket_complete_cities": len(common),
            "ratio_lt_half_or_gt_double_12month": int(
                ((product.annual_ratio < 0.5) | (product.annual_ratio > 2))
                .where(product.positive_month_pairs.eq(12), False)
                .sum()
            ),
        },
    )


def self_checks():
    groups = np.repeat([1, 2, 3], 4)
    controls = np.tile(np.arange(4), 3).astype(float)
    x = np.tile([1.0, -1.0, -1.0, 1.0], 3)
    y = 2 * x + 10 * groups + 3 * controls
    data = within_region(np.column_stack([controls, x, y]), groups)
    residual, _ = residualize(data[:, 1:], data[:, :1])
    r, b = correlation_and_slope(residual[:, 0], residual[:, 1:])
    assert np.allclose(r, [1]) and np.allclose(b, [2])
    assert np.allclose(within_region(data, groups), data)
    assert np.allclose((np.array([20, 60]).mean() / np.array([10, 20]).mean()), 8 / 3)
    assert not np.isclose(8 / 3, np.median([2, 3]))
    price_rows, labels = [], []
    for year in [2023, 2024]:
        for month in range(1, 13):
            labels.extend(
                [
                    {
                        "year": year,
                        "month": month,
                        "product_code": 111,
                        "source_product_label": "A, kg",
                    },
                    {
                        "year": year,
                        "month": month,
                        "product_code": 222,
                        "source_product_label": "B, kg" if year == 2023 else "B, litre",
                    },
                ]
            )
            for entity in range(1, 6):
                price_rows.append(
                    {
                        "territory_id": str(entity),
                        "year": year,
                        "month": month,
                        "product_code": 222 if entity == 4 else 111,
                        "source_city_code_literal": "changed"
                        if entity == 5 and year == 2024
                        else str(entity),
                        "source_city_oktmo_11": "changed"
                        if entity == 2 and year == 2024
                        else str(entity),
                        "value": 0
                        if entity == 3 and year == 2024 and month == 12
                        else 10 * (year - 2022),
                        "source_row": entity,
                        "source_column": 3,
                    }
                )
    with duckdb.connect() as db:
        db.register("prices", pd.DataFrame(price_rows))
        db.register("dictionary", pd.DataFrame(labels))
        build_matched(db)
        assert db.sql(
            "SELECT territory_id,count(*) FROM matched GROUP BY 1 ORDER BY 1"
        ).fetchall() == [(1, 12), (3, 11)]
        assert db.sql("SELECT min(monthly_ratio),max(monthly_ratio) FROM matched").fetchone() == (
            2.0,
            2.0,
        )
    return {
        "status": "PASS",
        "checks": [
            "region absorption",
            "nuisance partial slope",
            "idempotent within transform",
            "ratio of means differs from median monthly ratios",
            "actual SQL rejects changed city code, changed source code, changed unit and zero",
            "missing positive month stays 11/12; no annual imputation",
        ],
    }


def city_pair_eligibility(panel, primary, out):
    fields = ["territory_id", "source_city_name", "iszh", "source_city_oktmo_11"]
    frame = panel.loc[panel.year.eq(2023), fields].merge(
        panel.loc[panel.year.eq(2024), fields],
        on="territory_id",
        suffixes=("_2023", "_2024"),
        validate="one_to_one",
    )
    frame["approved_both_years"] = frame.iszh_2023.notna() & frame.iszh_2024.notna()
    frame["same_approved_city_code"] = (
        frame.approved_both_years
        & frame.source_city_oktmo_11_2023.eq(frame.source_city_oktmo_11_2024)
    )
    coverage = pd.read_csv(out / "matched_city_coverage.csv")
    frame = frame.merge(coverage, on="territory_id", how="left", validate="one_to_one")
    frame = frame.merge(
        primary[["territory_id", "n_products"]],
        on="territory_id",
        how="left",
        validate="one_to_one",
    )
    frame["pair_status"] = "no_approved_city_in_both_years"
    frame.loc[frame.approved_both_years, "pair_status"] = "city_code_changed_no_temporal_bridge"
    frame.loc[frame.same_approved_city_code, "pair_status"] = "no_complete12_product_set_ge100"
    frame.loc[frame.n_products.notna(), "pair_status"] = "accepted12months_ge100products"
    frame.to_csv(out / "city_pair_eligibility.csv", index=False)
    frame.groupby("pair_status").size().rename("municipalities").to_csv(
        out / "city_pair_exclusion_counts.csv"
    )


def figures(panel, result, primary, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = panel.loc[panel.year == 2024]
    axes[0].boxplot(
        [
            d.loc[~d.price_available, "log_population"].dropna(),
            d.loc[d.price_available, "log_population"].dropna(),
        ],
        tick_labels=["Нет городского ИСЖ", "Есть городской ИСЖ"],
        showfliers=False,
    )
    axes[0].set_ylabel("Логарифм населения МО, 2024")
    axes[0].set_title("Выборка городов отличается от панели")
    d = result.loc[(result.year == 2024) & result["sample"].eq("core_complete")]
    positions = np.arange(len(d))
    axes[1].scatter(d.raw_pearson, positions, marker="x", color="#ad6a22", label="Сырая связь")
    axes[1].errorbar(
        d.partial_pearson,
        positions,
        xerr=np.vstack(
            [d.partial_pearson - d.partial_r_ci_low, d.partial_r_ci_high - d.partial_pearson]
        ),
        fmt="o",
        color="#206a70",
        capsize=3,
        label="После поправок, 95% CI",
    )
    axes[1].axvline(0, color="grey", linewidth=0.8)
    axes[1].set_yticks(
        positions,
        [
            "Log расходов",
            "CLR продукты",
            "CLR здоровье",
            "CLR маркетплейсы",
            "CLR общепит",
            "CLR транспорт",
        ],
    )
    axes[1].set_xlabel("Корреляция с log городского ИСЖ")
    axes[1].set_xlim(-1, 1)
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"2024: {d.n.iloc[0]} МО, {d.regions.iloc[0]} регионов; одинаковая выборка")
    fig.tight_layout()
    fig.savefig(out / "iszh_selection_and_associations.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(100 * (primary.city_median_ratio - 1), bins=20, color="#206a70", edgecolor="white")
    ax.set_xlabel("100 × (медиана отношений среднегодовых цен − 1), %")
    ax.set_ylabel("Наблюдаемые города")
    ax.set_title("12/12 положительных месячных пар, ≥100 товаров; это не ИПЦ")
    fig.tight_layout()
    fig.savefig(out / "matched_price_ratios.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=999)
    args = parser.parse_args()
    if args.bootstrap < 99:
        raise ValueError("At least 99 regional bootstrap draws required")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("Use an empty new output directory")
    protocol = {**PROTOCOL, "bootstrap_draws": args.bootstrap}
    write_json(args.out / "protocol.json", protocol)
    manifest = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha(__file__),
        "protocol_sha256": sha(args.out / "protocol.json"),
    }
    write_json(args.out / "manifest.json", manifest)
    try:
        inputs = verify_inputs(args.repo)
        checks = self_checks()
        panel = prepare_panel(args.repo, args.out)
        describe_selection(panel, args.out)
        with threadpool_limits(limits=1):
            result = associations(panel, args.out, args.bootstrap)
        primary, sensitivity, common, price_qa = monthly_prices(args.repo, args.out)
        city_pair_eligibility(panel, primary, args.out)
        labels = panel.loc[
            panel.year == 2024,
            [
                "territory_id",
                "source_city_name",
                "region_name",
                "municipal_district_name",
                "code_type_changed",
            ],
        ]
        primary = primary.merge(labels, on="territory_id", validate="one_to_one")
        primary.to_csv(args.out / "primary_city_price_ratios.csv", index=False)
        top = primary.sort_values("city_median_ratio")
        pd.concat(
            [
                top.head(5).assign(case="lowest_observed"),
                top.tail(5).assign(case="highest_observed"),
            ]
        ).to_csv(args.out / "city_ratio_cases.csv", index=False)
        figures(panel, result, primary, args.out)
        summary = {
            "analysis_status": "descriptive_conditional_city_context_not_causal",
            "cohort": 2016,
            "iszh_city_years": int(panel.price_available.sum()),
            "primary_ratio_cities": len(primary),
            "median_city_price_ratio": float(primary.city_median_ratio.median()),
            "common_basket_median_ratio": float(common.common_basket_ratio.median()),
            "common_basket_same_city_original_median": float(common.city_median_ratio.median()),
            **price_qa,
        }
        write_json(args.out / "summary.json", summary)
        write_json(
            args.out / "qa.json",
            {
                **checks,
                "frozen_price_manifest_outputs_verified": True,
                "raw_XML_prior_audit": "3836682 cells PASS; no new extraction or source revisions",
                "no_forbidden_temporal_or_january_pairs": True,
                "unique_annual_and_product_keys": True,
                "strict_positive_pairs_only": True,
                "exact_full_product_labels_unchanged": True,
            },
        )
        manifest.update(
            status="complete",
            finished_at=datetime.now(UTC).isoformat(),
            input_hashes=inputs,
            runtime={
                "python": platform.python_version(),
                **{x: version(x) for x in ["numpy", "pandas", "scipy", "duckdb", "matplotlib"]},
            },
            root_lock_sha256=sha(args.repo / "uv.lock"),
            outputs={
                p.name: sha(p)
                for p in sorted(args.out.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(args.out / "manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), finished_at=datetime.now(UTC).isoformat())
        write_json(args.out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
