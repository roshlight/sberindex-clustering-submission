"""Prespecified H1/H2 municipal regressions with regional wild-cluster inference.

CPI adjustment removes price change relative to each region's December 2022
base. It does not put different regions at a common spatial price level.
The primary outcome is the mean of monthly deflated published total spending.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import t
from threadpoolctl import threadpool_limits

from municipal_profiles.dynamics import PROFILE_COLUMNS

ECONOMIC = ["log_population", "log_wage", "log_worker_concentration"]
SOURCES = {
    "annual_panel": "reports/enriched_analysis/core_v2/annual_panel.csv",
    "annual_prices": "reports/enriched_analysis/prices/municipal_annual_price_context.csv",
    "monthly_prices": "reports/enriched_analysis/prices/municipal_monthly_price_context.csv",
    "regional_chain": "reports/enriched_analysis/prices/regional_cpi_chain.csv",
    "raw_cpi": "reports/enrichment_v3/prices/cpi_monthly_2023_2024.csv",
    "price_links": "reports/enrichment_v3/prices/municipal_region_price_links.csv",
}
METHOD = "OLS; region/type FE; CR1; restricted-null Rademacher wild-cluster bootstrap-t"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _unique(frame: pd.DataFrame, keys: list[str], source: str) -> None:
    _require(not frame[keys].isna().any().any(), f"Missing key: {source}")
    _require(not frame.duplicated(keys).any(), f"Nonunique key: {source}")


def _boolean(values: pd.Series, source: str) -> pd.Series:
    normalized = values.astype(str).str.lower()
    _require(normalized.isin(["true", "false"]).all(), f"Invalid boolean: {source}")
    return normalized.eq("true")


def _close(actual, expected, message: str, *, atol: float = 1e-8) -> None:
    _require(
        np.isfinite(np.asarray(actual, dtype=float)).all()
        and np.isfinite(np.asarray(expected, dtype=float)).all()
        and np.allclose(actual, expected, rtol=1e-12, atol=atol),
        message,
    )


def _load_sample(root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reconcile saved annual values to monthly prices and independent CPI chaining."""
    panel = pd.read_csv(root / SOURCES["annual_panel"], dtype={"region_code": str})
    _unique(panel, ["territory_id", "year"], "annual_panel")
    annual = panel.loc[panel.year.eq(year)].copy()
    _require(len(annual) > 0, f"No annual observations for {year}")
    eligible = _boolean(annual.core_eligible, "core_eligible")
    calendar = _boolean(annual.calendar_review, "calendar_review")
    finite_logs = np.isfinite(annual[ECONOMIC].to_numpy(dtype=float)).all(axis=1)
    _require(np.array_equal(eligible, finite_logs & ~calendar), "Core eligibility mismatch")
    sampling = annual[["territory_id", "year", "region_code"]].copy()
    sampling["selected"] = eligible
    sampling["calendar_review"] = calendar
    sampling["finite_economic_logs"] = finite_logs
    sample = annual.loc[eligible].sort_values("territory_id").reset_index(drop=True)
    _require(len(sample) > 0, "Empty core cohort")
    _require(sample[["region_code", "municipal_district_type"]].notna().all().all(), "Missing FE")
    _require(sample.observed_months.eq(12).all(), "Core profiles require 12 observed months")
    profiles = sample[list(PROFILE_COLUMNS)].to_numpy(dtype=float)
    _require(np.isfinite(profiles).all() and (profiles > 0).all(), "Invalid CLR profile")
    _close(profiles.sum(axis=1), np.ones(len(sample)), "Profile must sum to one", atol=1e-12)
    for field, raw in [("log_population", "population_total"), ("log_wage", "wage")]:
        _require(sample[raw].gt(0).all(), f"Nonpositive {raw}")
        _close(sample[field], np.log(sample[raw]), f"Derived {field} mismatch", atol=1e-12)
    _require(sample.workers.gt(0).all(), "Nonpositive workers")
    _close(
        sample.log_worker_concentration,
        np.log(sample.workers / sample.population_total),
        "Worker concentration must be workers / population",
        atol=1e-12,
    )
    sample["clr_foodservice"] = np.log(
        profiles[:, list(PROFILE_COLUMNS).index("p_foodservice")]
    ) - np.log(profiles).mean(axis=1)

    prices = pd.read_csv(root / SOURCES["annual_prices"])
    _unique(prices, ["territory_id", "year"], "annual_prices")
    price_fields = [
        "territory_id",
        "year",
        "nominal_mean",
        "real_mean_dec2022",
        "n_months",
        "annual_mean_price_level",
    ]
    sample = sample.merge(
        prices[price_fields], on=["territory_id", "year"], how="left", validate="one_to_one"
    )
    _require(sample.n_months.eq(12).all(), "Incomplete annual price coverage")
    _require(sample.real_mean_dec2022.gt(0).all(), "Nonpositive real spending")
    _close(sample.nominal_mean, sample.total_mean, "Annual nominal source mismatch")

    monthly = pd.read_csv(
        root / SOURCES["monthly_prices"],
        dtype={"region_code": str, "source_territory_code": str},
    )
    _unique(monthly, ["territory_id", "month"], "monthly_prices")
    _require(monthly.month.str.fullmatch(r"\d{4}-\d{2}").all(), "Invalid price month")
    _require(monthly.year.eq(monthly.month.str[:4].astype(int)).all(), "Price month/year mismatch")
    monthly = monthly.loc[monthly.territory_id.isin(sample.territory_id) & monthly.year.eq(year)]
    monthly = monthly.sort_values(["territory_id", "month"]).reset_index(drop=True)
    months = [f"{year}-{m:02d}" for m in range(1, 13)]
    _require(
        len(monthly) == len(sample) * 12
        and all(list(g.month) == months for _, g in monthly.groupby("territory_id")),
        "Every selected municipality requires 12 ordered price months",
    )
    monthly = monthly.merge(
        sample[["territory_id", "region_code"]].rename(columns={"region_code": "core_region"}),
        on="territory_id",
        validate="many_to_one",
    )
    _require(monthly.region_code.eq(monthly.core_region).all(), "Price/core region mismatch")

    links = pd.read_csv(
        root / SOURCES["price_links"],
        dtype={"region_code": str, "source_territory_code": str},
    )
    _unique(links, ["territory_id", "reference_date"], "price_links")
    links = links.loc[links.territory_id.isin(sample.territory_id)]
    _require(links.reference_date.eq(links.snapshot_date).all(), "Price link date mismatch")
    _require(
        links.groupby("territory_id")[["region_code", "source_territory_code"]]
        .nunique()
        .eq(1)
        .all()
        .all(),
        "Changing municipality price region",
    )
    # The existing CPI pipeline explicitly uses its 2024 mapping for both years.
    selected_links = links.loc[links.reference_date.eq("2024-01-01")]
    mapped = monthly.merge(
        selected_links[["territory_id", "region_code", "source_territory_code"]],
        on="territory_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_link"),
    )
    _require(
        mapped.region_code.eq(mapped.region_code_link).all()
        and mapped.source_territory_code.eq(mapped.source_territory_code_link).all(),
        "Price link coverage/mapping mismatch",
    )

    codes = set(monthly.source_territory_code)
    raw = pd.read_csv(root / SOURCES["raw_cpi"], dtype={"source_territory_code": str})
    raw = raw.loc[
        raw.category.eq("all")
        & raw.territorial_level.eq("region")
        & raw.source_territory_code.isin(codes)
        & raw.month.le(f"{year}-12")
    ].sort_values(["source_territory_code", "month"])
    _unique(raw, ["source_territory_code", "month"], "raw_cpi")
    _require(
        raw.index_base.eq("previous_month_100").all() and raw.unit.eq("percent").all(),
        "Unsupported CPI convention",
    )
    _require(np.isfinite(raw.value).all() and raw.value.gt(0).all(), "Invalid CPI factor")
    chain_months = list(pd.period_range("2023-01", f"{year}-12", freq="M").astype(str))
    _require(
        set(raw.source_territory_code) == codes
        and all(list(g.month) == chain_months for _, g in raw.groupby("source_territory_code")),
        "CPI chain must start in January 2023 and cover every region/month",
    )
    raw["independent_price_level"] = (raw.value / 100).groupby(raw.source_territory_code).cumprod()
    saved_chain = pd.read_csv(
        root / SOURCES["regional_chain"], dtype={"source_territory_code": str}
    )
    _unique(saved_chain, ["source_territory_code", "month"], "regional_chain")
    check = raw.merge(
        saved_chain[["source_territory_code", "month", "price_level_dec2022_1"]],
        on=["source_territory_code", "month"],
        how="left",
        validate="one_to_one",
    )
    _close(check.price_level_dec2022_1, check.independent_price_level, "Saved CPI chain mismatch")
    monthly = monthly.merge(
        raw[["source_territory_code", "month", "independent_price_level"]],
        on=["source_territory_code", "month"],
        how="left",
        validate="many_to_one",
    )
    _close(monthly.price_level_dec2022_1, monthly.independent_price_level, "Monthly CPI mismatch")
    _require(monthly.nominal_total.gt(0).all(), "Nonpositive monthly nominal spending")
    _close(
        monthly.real_total_dec2022,
        monthly.nominal_total / monthly.independent_price_level,
        "Monthly real spending must equal nominal / chained regional CPI",
    )
    reconstructed = monthly.groupby("territory_id", as_index=False).agg(
        checked_real_mean=("real_total_dec2022", "mean"),
        checked_nominal_mean=("nominal_total", "mean"),
        checked_price_mean=("independent_price_level", "mean"),
    )
    sample = sample.merge(reconstructed, on="territory_id", validate="one_to_one")
    _close(sample.real_mean_dec2022, sample.checked_real_mean, "Deflate before annual averaging")
    _close(sample.nominal_mean, sample.checked_nominal_mean, "Monthly/annual nominal mismatch")
    _close(sample.annual_mean_price_level, sample.checked_price_mean, "Annual CPI mean mismatch")
    sample["log_real_spending"] = np.log(sample.real_mean_dec2022)
    sample["log_nominal_spending"] = np.log(sample.nominal_mean)
    sample["log_scalar_deflated_spending"] = np.log(
        sample.nominal_mean / sample.annual_mean_price_level
    )
    audit = {
        "source_year_rows": len(annual),
        "selected_core_rows": len(sample),
        "excluded_rows": len(annual) - len(sample),
        "n_regions": int(sample.region_code.nunique()),
        "monthly_selected_rows": len(monthly),
        "all_source_arithmetic_checks": "PASS",
        "profile_outcome": "CLR of the mean of 12 normalized monthly profiles",
        "spending_outcome": "log(mean over 12 months of nominal_total / regional CPI chain))",
        "price_base": "each region's own December 2022 = 1; temporal, not spatial price parity",
        "workers": "accounted organization workers / municipal population; excludes small firms",
        "price_mapping": "existing 2024 municipality-region links; unchanged across 2023/2024",
    }
    return sample.sort_values("territory_id").reset_index(drop=True), sampling, audit


def _design(sample: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    dummies = pd.get_dummies(
        sample[["region_code", "municipal_district_type"]].astype(str),
        drop_first=True,
        dtype=float,
    )
    names = ["intercept", *dummies.columns]
    nuisance = np.column_stack([np.ones(len(sample)), dummies.to_numpy()])
    _, r, pivot = qr(nuisance, mode="economic", pivoting=True)
    tolerance = max(nuisance.shape) * np.finfo(float).eps * abs(r[0, 0])
    rank = int(np.sum(np.abs(np.diag(r)) > tolerance))
    chosen = sorted(pivot[:rank])
    dropped = [name for i, name in enumerate(names) if i not in chosen]
    return (
        np.column_stack([sample[ECONOMIC].to_numpy(dtype=float), nuisance[:, chosen]]),
        [*ECONOMIC, *[names[i] for i in chosen]],
        dropped,
    )


@dataclass
class _ClusterFit:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    names: list[str]
    beta: np.ndarray
    residual: np.ndarray
    bread: np.ndarray
    covariance: np.ndarray
    identifiable: np.ndarray
    rank: int
    correction: float


def _cluster_fit(x, y, groups, names) -> _ClusterFit:
    x, y, groups = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
    _require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite regression design/outcome")
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    tol = max(x.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tol))
    v = vt[:rank].T
    beta = v @ ((u[:, :rank].T @ y) / singular[:rank])
    bread = (v / singular[:rank] ** 2) @ v.T
    residual = y - x @ beta
    scores = np.array([x[groups == g].T @ residual[groups == g] for g in np.unique(groups)])
    n, g = len(y), len(scores)
    correction = g / (g - 1) * (n - 1) / (n - rank) if g > 1 and n > rank else np.nan
    covariance = correction * bread @ scores.T @ scores @ bread
    # A coefficient is estimable only when its coordinate is in the row space of X.
    identifiable = np.linalg.norm(np.eye(x.shape[1]) - v @ v.T, axis=0) < 1e-8
    return _ClusterFit(
        x,
        y,
        groups,
        names,
        beta,
        residual,
        bread,
        covariance,
        identifiable,
        rank,
        float(correction),
    )


def _wild_cluster_statistics(fit: _ClusterFit, tested: int, weights: np.ndarray) -> np.ndarray:
    """Exactly reproduce null-imposed refits and CR1 studentization, without B OLS solves."""
    x, y, groups = fit.x, fit.y, fit.groups
    restricted_x = np.delete(x, tested, axis=1)
    null_beta = np.linalg.lstsq(restricted_x, y, rcond=None)[0]
    null_residual = y - restricted_x @ null_beta
    levels = np.unique(groups)
    scores0 = np.array([x[groups == g].T @ null_residual[groups == g] for g in levels])
    q = fit.bread[:, tested]
    c = scores0 @ q
    beta_shifts = fit.bread @ scores0.T
    projected_grams = np.array([q @ x[groups == g].T @ x[groups == g] for g in levels])
    score_map = np.diag(c) - projected_grams @ beta_shifts
    weights = np.asarray(weights, dtype=float)
    _require(weights.ndim == 2 and weights.shape[1] == len(levels), "Invalid wild weights")
    _require(np.isin(weights, [-1, 1]).all(), "Wild weights must be regional Rademacher signs")
    numerators = weights @ c
    variances = fit.correction * np.square(weights @ score_map.T).sum(axis=1)
    result = np.full(len(weights), np.nan)
    valid = np.isfinite(variances) & (variances > 0)
    result[valid] = numerators[valid] / np.sqrt(variances[valid])
    return result


def _test_contrast(fit: _ClusterFit, field: str, *, seed: int, n_resamples: int) -> dict:
    j = fit.names.index(field)
    se2 = fit.covariance[j, j]
    se = float(np.sqrt(se2)) if se2 >= 0 and np.isfinite(se2) else np.nan
    estimate = float(fit.beta[j]) if fit.identifiable[j] else np.nan
    n_regions = len(np.unique(fit.groups))
    available = bool(fit.identifiable[j] and n_regions > 1 and np.isfinite(se) and se > 0)
    result = {
        "estimate": estimate,
        "standard_error_cr1": se,
        "statistic_t": estimate / se if available else np.nan,
        "ci95_low_cr1": np.nan,
        "ci95_high_cr1": np.nan,
        "p_raw": 1.0,
        "available": available,
        "reason": "" if available else "contrast_unidentifiable_or_degenerate_cluster_SE",
        "bootstrap_seed": seed,
        "n_resamples": n_resamples,
        "bootstrap_invalid_draws": 0,
        "bootstrap_exceedances": 0,
    }
    if not available:
        return result
    critical = float(t.ppf(0.975, n_regions - 1))
    result["ci95_low_cr1"] = estimate - critical * se
    result["ci95_high_cr1"] = estimate + critical * se
    rng = np.random.default_rng(seed)
    weights = 2 * rng.integers(0, 2, size=(n_resamples, n_regions)) - 1
    statistics = _wild_cluster_statistics(fit, j, weights)
    invalid = int(np.sum(~np.isfinite(statistics)))
    result["bootstrap_invalid_draws"] = invalid
    if invalid:
        result.update(available=False, reason="undefined_bootstrap_t_draws")
        return result
    exceedances = int(np.sum(statistics >= result["statistic_t"]))
    result["bootstrap_exceedances"] = exceedances
    result["p_raw"] = (1 + exceedances) / (n_resamples + 1)
    return result


def _coefficients(fit: _ClusterFit, hypothesis: str) -> list[dict]:
    rows = []
    for j, name in enumerate(fit.names):
        se2 = fit.covariance[j, j]
        rows.append(
            {
                "hypothesis": hypothesis,
                "term": name,
                "estimate": float(fit.beta[j]) if fit.identifiable[j] else np.nan,
                "standard_error_cr1": float(np.sqrt(se2)) if se2 >= 0 else np.nan,
                "identifiable": bool(fit.identifiable[j]),
            }
        )
    return rows


def run_regression_tests(
    root: Path,
    output: Path,
    *,
    year: int = 2024,
    seed: int = 13,
    n_resamples: int = 9999,
) -> pd.DataFrame:
    """Run two fixed primary tests; multiple-testing adjustment belongs to the caller."""
    _require(year in (2023, 2024), "Only existing 2023/2024 annual sources are supported")
    _require(
        isinstance(n_resamples, int) and n_resamples >= 1, "Positive integer resamples required"
    )
    _require(isinstance(seed, int) and seed >= 0, "Nonnegative integer seed required")
    root, output = Path(root).resolve(), Path(output).resolve()
    _require(not output.exists() or not any(output.iterdir()), "Output must be new or empty")
    hashes_before = {name: _sha(root / path) for name, path in SOURCES.items()}
    sample, sampling, audit = _load_sample(root, year)
    x, names, dropped = _design(sample)
    groups = sample.region_code.to_numpy()
    rows, coefficients = [], []
    with threadpool_limits(limits=1):
        for hypothesis, field, outcome, suffix, unit in [
            ("H1", "log_population", "clr_foodservice", 101, "CLR per log population"),
            ("H2", "log_wage", "log_real_spending", 102, "log real spending per log wage"),
        ]:
            fit = _cluster_fit(x, sample[outcome], groups, names)
            result = _test_contrast(fit, field, seed=seed + suffix, n_resamples=n_resamples)
            rows.append(
                {
                    "test_id": "H1_population_foodservice"
                    if hypothesis == "H1"
                    else "H2_wage_real_spending",
                    "hypothesis": hypothesis,
                    "year": year,
                    **result,
                    "expected_direction": "positive",
                    "n_entities": len(sample),
                    "n_regions": len(np.unique(groups)),
                    "n_pairs": 0,
                    "method": METHOD,
                    "unit": unit,
                    "rank": fit.rank,
                    "design_columns": len(names),
                    "cr1_correction": fit.correction,
                }
            )
            coefficients.extend(_coefficients(fit, hypothesis))
        nominal = _cluster_fit(x, sample.log_nominal_spending, groups, names)
        scalar = _cluster_fit(x, sample.log_scalar_deflated_spending, groups, names)
    differences = scalar.beta[: len(ECONOMIC)] - nominal.beta[: len(ECONOMIC)]
    estimable = nominal.identifiable[: len(ECONOMIC)] & scalar.identifiable[: len(ECONOMIC)]
    maximum_difference = float(np.max(np.abs(differences[estimable]))) if estimable.any() else 0.0
    _require(maximum_difference < 1e-9, "Scalar regional deflator FE invariance failed")
    diagnostics = {
        "not_additional_hypothesis_tests": True,
        "nominal_H2": _coefficients(nominal, "H2_nominal_diagnostic")[: len(ECONOMIC)],
        "scalar_regional_deflator": {
            "interpretation": "log of a regional scalar is absorbed by region FE",
            "max_abs_identifiable_economic_coefficient_difference": maximum_difference,
            "coefficient_differences": {
                term: float(differences[j]) if estimable[j] else None
                for j, term in enumerate(ECONOMIC)
            },
        },
    }
    hashes_after = {name: _sha(root / path) for name, path in SOURCES.items()}
    _require(hashes_before == hashes_after, "Source changed while regressions were running")
    results = pd.DataFrame(rows)
    _require(
        len(results) == 2
        and np.isfinite(results.p_raw).all()
        and results.p_raw.between(0, 1).all(),
        "Both prespecified family members must retain finite probabilities",
    )
    audit.update(
        {
            "redundant_FE_dropped": dropped,
            "design_terms": names,
            "equal_municipality_weighting": True,
            "inference": "regional CR1; one-sided restricted-null wild-cluster bootstrap-t",
            "ci95": "two-sided t(G-1) interval with CR1 SE; not bootstrap confidence interval",
            "undefined_test_rule": "available=False, p_raw=1; retain in Holm family",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "regression_tests.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
    sample.to_csv(output / "sample.csv", index=False)
    sampling.sort_values("territory_id").to_csv(output / "sampling_audit.csv", index=False)
    for filename, payload in [("audit.json", audit), ("diagnostics.json", diagnostics)]:
        _write_json(output / filename, payload)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "year": year,
        "seed": seed,
        "n_resamples": n_resamples,
        "test_family": "H1/H2 are two members of caller's Holm family",
        "inputs": {
            name: {"path": path, "sha256": hashes_before[name]} for name, path in SOURCES.items()
        },
        "implementation_sha256": _sha(Path(__file__)),
        "outputs": {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    _write_json(output / "manifest.json", manifest)
    return results
