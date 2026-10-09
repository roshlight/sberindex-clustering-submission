import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from municipal_profiles.dynamics import PROFILE_COLUMNS
from municipal_profiles.hypothesis_regressions import (
    ECONOMIC,
    SOURCES,
    _cluster_fit,
    _design,
    _test_contrast,
    _wild_cluster_statistics,
    run_regression_tests,
)


def _toy_design(seed=30):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(6), 9)
    x = np.column_stack([rng.normal(size=(len(groups), 2)), np.ones(len(groups))])
    y = x @ np.array([0.8, -0.3, 1.4]) + rng.normal(size=len(groups))
    y += np.repeat(rng.normal(size=6), 9)
    return x, y, groups


def test_cluster_covariance_matches_independent_score_formula():
    x, y, groups = _toy_design()
    fit = _cluster_fit(x, y, groups, ["a", "b", "intercept"])
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    meat = np.zeros((3, 3))
    for g in np.unique(groups):
        score = sum(x[i] * residual[i] for i in np.flatnonzero(groups == g))
        meat += np.outer(score, score)
    correction = 6 / 5 * (len(y) - 1) / (len(y) - 3)
    np.testing.assert_allclose(fit.beta, beta, atol=1e-12)
    np.testing.assert_allclose(fit.covariance, correction * bread @ meat @ bread, atol=1e-12)
    assert fit.identifiable.all()


def test_null_imposed_fast_bootstrap_matches_every_brute_refit():
    x, y, groups = _toy_design()
    fit = _cluster_fit(x, y, groups, ["a", "b", "intercept"])
    signs = np.array(list(itertools.product([-1, 1], repeat=6)))
    restricted = x[:, 1:]
    fitted0 = restricted @ np.linalg.lstsq(restricted, y, rcond=None)[0]
    residual0 = y - fitted0
    expected = []
    for draw in signs:
        y_star = fitted0 + residual0 * draw[groups]
        refit = _cluster_fit(x, y_star, groups, fit.names)
        expected.append(refit.beta[0] / np.sqrt(refit.covariance[0, 0]))
    actual = _wild_cluster_statistics(fit, 0, signs)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    result = _test_contrast(fit, "a", seed=77, n_resamples=64)
    generated = 2 * np.random.default_rng(77).integers(0, 2, size=(64, 6)) - 1
    generated_t = _wild_cluster_statistics(fit, 0, generated)
    exceed = np.sum(generated_t >= result["statistic_t"])
    assert result["p_raw"] == (1 + exceed) / 65
    assert result["bootstrap_exceedances"] == exceed


def test_nonestimable_contrast_and_one_region_remain_in_family():
    x, y, groups = _toy_design()
    collinear = np.column_stack([x[:, 0], x[:, 0], x[:, 1:]])
    fit = _cluster_fit(collinear, y, groups, ["a", "duplicate", "b", "intercept"])
    result = _test_contrast(fit, "a", seed=1, n_resamples=9)
    assert not result["available"] and result["p_raw"] == 1
    assert np.isnan(result["estimate"])
    one_region = _cluster_fit(x, y, np.zeros(len(y)), ["a", "b", "intercept"])
    result = _test_contrast(one_region, "a", seed=1, n_resamples=9)
    assert not result["available"] and result["p_raw"] == 1


def test_nested_type_fixed_effects_keep_economic_contrasts_identifiable():
    x, y, groups = _toy_design()
    frame = pd.DataFrame(x[:, :2], columns=ECONOMIC[:2])
    frame[ECONOMIC[2]] = np.random.default_rng(15).normal(size=len(frame))
    frame["region_code"] = groups.astype(str)
    frame["municipal_district_type"] = np.where(groups < 3, "city", "district")
    design, names, dropped = _design(frame)
    assert len(dropped) == 1
    fit = _cluster_fit(design, y, groups, names)
    assert fit.identifiable.all()
    assert fit.rank == design.shape[1]
    relabeled = frame.assign(region_code=frame.region_code.map(lambda value: "r" + value))
    alternative, alternative_names, _ = _design(relabeled)
    other = _cluster_fit(alternative, y, relabeled.region_code, alternative_names)
    np.testing.assert_allclose(fit.beta[:3], other.beta[:3], atol=1e-12)
    np.testing.assert_allclose(fit.covariance[:3, :3], other.covariance[:3, :3], atol=1e-12)


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "source"
    rng = np.random.default_rng(8)
    regions = np.repeat(np.arange(6), 8)
    n = len(regions)
    logs = rng.normal(size=(n, 3)) * [0.7, 0.2, 0.4] + [10, 10.5, -1]
    log_levels = 0.4 * logs[:, 0] + 0.7 * logs[:, 1] + 0.05 * logs[:, 2]
    log_levels += 0.07 * regions + rng.normal(scale=0.15, size=n)
    profile_logs = rng.normal(scale=0.08, size=(n, 5))
    profile_logs[:, 3] += 0.35 * (logs[:, 0] - 10)
    profile_values = np.exp(profile_logs)
    profile_values /= profile_values.sum(axis=1, keepdims=True)
    panel = pd.DataFrame(logs, columns=ECONOMIC)
    for j, col in enumerate(PROFILE_COLUMNS):
        panel[col] = profile_values[:, j]
    panel["territory_id"] = np.arange(n)
    panel["year"] = 2024
    panel["region_code"] = [str(g) for g in regions]
    panel["municipal_district_type"] = np.where(np.arange(n) % 2, "city", "district")
    panel["population_total"] = np.exp(logs[:, 0])
    panel["wage"] = np.exp(logs[:, 1])
    panel["workers"] = panel.population_total * np.exp(logs[:, 2])
    panel["observed_months"] = 12
    panel["calendar_review"] = False
    panel.loc[0, "calendar_review"] = True
    panel["core_eligible"] = True
    panel.loc[0, "core_eligible"] = False

    raw_rows, chain_rows, link_rows, monthly_rows = [], [], [], []
    months = list(pd.period_range("2023-01", "2024-12", freq="M").astype(str))
    chains = {}
    for g in range(6):
        price = 1.0
        for m, month in enumerate(months):
            factor = 1.003 + g * 0.0008 + m * 0.00007
            price *= factor
            row = {
                "source_territory_code": f"source_{g}",
                "territorial_level": "region",
                "month": month,
                "category": "all",
                "value": factor * 100,
                "unit": "percent",
                "index_base": "previous_month_100",
            }
            raw_rows.append(row)
            chain_rows.append({**row, "price_level_dec2022_1": price})
            chains[g, month] = price
    for i, g in enumerate(regions):
        for year in [2023, 2024]:
            link_rows.append(
                {
                    "territory_id": i,
                    "reference_date": f"{year}-01-01",
                    "snapshot_date": f"{year}-01-01",
                    "region_code": str(g),
                    "source_territory_code": f"source_{g}",
                }
            )
        for m, month in enumerate(months):
            # Municipality-specific seasonality makes monthly deflation differ from its shortcut.
            nominal = np.exp(log_levels[i]) * (
                1 + 0.12 * np.sin((m % 12) * np.pi / 6) + 0.01 * logs[i, 0] * (m % 12)
            )
            monthly_rows.append(
                {
                    "territory_id": i,
                    "month": month,
                    "nominal_total": nominal,
                    "region_code": str(g),
                    "source_territory_code": f"source_{g}",
                    "price_level_dec2022_1": chains[g, month],
                    "real_total_dec2022": nominal / chains[g, month],
                    "year": int(month[:4]),
                }
            )
    monthly = pd.DataFrame(monthly_rows)
    annual = monthly.groupby(["territory_id", "year"], as_index=False).agg(
        nominal_mean=("nominal_total", "mean"),
        real_mean_dec2022=("real_total_dec2022", "mean"),
        n_months=("month", "nunique"),
        annual_mean_price_level=("price_level_dec2022_1", "mean"),
    )
    panel["total_mean"] = annual.loc[annual.year.eq(2024)].nominal_mean.to_numpy()
    frames = {
        "annual_panel": panel,
        "annual_prices": annual,
        "monthly_prices": monthly,
        "regional_chain": pd.DataFrame(chain_rows),
        "raw_cpi": pd.DataFrame(raw_rows),
        "price_links": pd.DataFrame(link_rows),
    }
    for name, frame in frames.items():
        path = root / SOURCES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return root


def test_pipeline_has_two_prespecified_rows_monthly_real_outcome_and_provenance(
    source_root, tmp_path
):
    output = tmp_path / "results"
    results = run_regression_tests(source_root, output, n_resamples=99)
    assert results.test_id.tolist() == ["H1_population_foodservice", "H2_wage_real_spending"]
    assert results.n_entities.tolist() == [47, 47]
    assert results.n_regions.tolist() == [6, 6]
    assert results.n_pairs.tolist() == [0, 0]
    assert results.available.all() and results.p_raw.between(0, 1).all()
    assert results.bootstrap_seed.tolist() == [114, 115]
    sample = pd.read_csv(output / "sample.csv")
    assert not np.allclose(
        sample.real_mean_dec2022,
        sample.nominal_mean / sample.annual_mean_price_level,
    )
    np.testing.assert_allclose(sample.log_real_spending, np.log(sample.checked_real_mean))
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["not_additional_hypothesis_tests"]
    assert (
        diagnostics["scalar_regional_deflator"][
            "max_abs_identifiable_economic_coefficient_difference"
        ]
        < 1e-9
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, source in manifest["inputs"].items():
        assert (
            source["sha256"]
            == hashlib.sha256((source_root / SOURCES[name]).read_bytes()).hexdigest()
        )
    for name, digest in manifest["outputs"].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="new or empty"):
        run_regression_tests(source_root, output, n_resamples=99)


@pytest.mark.parametrize("mutation", ["duplicate", "shortcut", "chain", "log_wage"])
def test_invalid_sources_fail_before_producing_a_report(source_root, tmp_path, mutation):
    source = {
        "duplicate": "annual_prices",
        "shortcut": "annual_prices",
        "chain": "regional_chain",
        "log_wage": "annual_panel",
    }[mutation]
    path = source_root / SOURCES[source]
    data = pd.read_csv(path)
    if mutation == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif mutation == "shortcut":
        data["real_mean_dec2022"] = data.nominal_mean / data.annual_mean_price_level
    elif mutation == "chain":
        data["price_level_dec2022_1"] *= 1.001
    else:
        data["log_wage"] += 0.1
    data.to_csv(path, index=False)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_regression_tests(source_root, output, n_resamples=9)
    assert not output.exists()


def test_source_row_order_is_irrelevant_and_unidentified_tests_are_retained(source_root, tmp_path):
    first = run_regression_tests(source_root, tmp_path / "first", n_resamples=49)
    for source in SOURCES.values():
        path = source_root / source
        frame = pd.read_csv(path)
        frame.sample(frac=1, random_state=11).to_csv(path, index=False)
    second = run_regression_tests(source_root, tmp_path / "second", n_resamples=49)
    pd.testing.assert_frame_equal(first, second, check_exact=False, atol=1e-10, rtol=1e-10)
    path = source_root / SOURCES["annual_panel"]
    panel = pd.read_csv(path)
    panel["log_wage"] = panel.log_population
    panel["wage"] = panel.population_total
    panel.to_csv(path, index=False)
    unavailable = run_regression_tests(source_root, tmp_path / "unidentified", n_resamples=49)
    assert unavailable.hypothesis.tolist() == ["H1", "H2"]
    assert unavailable.p_raw.eq(1).all()
    assert not unavailable.available.any()
