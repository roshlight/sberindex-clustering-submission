"""Apply the declared five-test family and joint H3 decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

TEST_IDS = (
    "H1_population_foodservice",
    "H2_wage_real_spending",
    "H3_manufacturing_marketplaces",
    "H3_manufacturing_transport",
    "H4_cross_region_similarity",
)
HYPOTHESES = ("H1", "H2", "H3", "H3", "H4")
DIRECTIONS = ("positive", "positive", "positive", "negative", "negative")


def holm_adjust(pvalues) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm requires a finite one-dimensional p-value array")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return adjusted


def summarize_tests(raw: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Keep unavailable contrasts in the family; H3 requires both outcomes."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if len(raw) != len(TEST_IDS) or set(raw.test_id) != set(TEST_IDS):
        raise ValueError("Expected exactly the five declared contrasts")
    if raw.test_id.duplicated().any() or raw.year.nunique() != 1:
        raise ValueError("Contrasts must be unique and use one declared year")
    result = raw.set_index("test_id").loc[list(TEST_IDS)].reset_index().copy()
    if result.hypothesis.tolist() != list(HYPOTHESES):
        raise ValueError("Declared hypothesis groups cannot change")
    if result.expected_direction.tolist() != list(DIRECTIONS):
        raise ValueError("Declared directions cannot change")
    if "available" not in result:
        result["available"] = True
    if result.available.isna().any() or not result.available.isin([True, False]).all():
        raise ValueError("available must contain explicit boolean values")
    result.loc[~result.available, "p_raw"] = 1.0
    if not np.isfinite(result.loc[result.available, "estimate"]).all():
        raise ValueError("Available effects must be finite")
    if not result.expected_direction.isin(["positive", "negative"]).all():
        raise ValueError("Every contrast needs a declared direction")
    result["p_holm"] = holm_adjust(result.p_raw.to_numpy())
    result["direction_ok"] = np.where(
        result.expected_direction.eq("positive"), result.estimate > 0, result.estimate < 0
    )
    result["supported"] = result.available & result.direction_ok & (result.p_holm < alpha)
    result["hypothesis_supported"] = result.groupby("hypothesis").supported.transform("all")
    result["decision"] = np.where(
        ~result.available, "unavailable", np.where(result.supported, "supported", "not_supported")
    )
    return result
