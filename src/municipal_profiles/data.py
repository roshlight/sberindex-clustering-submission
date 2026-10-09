"""Load coded municipality data and construct normalized expenditure profiles.

The main profile normalizes the five published category means by their own sum.
It is not an estimate of household budget shares: estimator denominators are not
fully documented. ``include_residual`` enables a separately labelled arithmetic
sensitivity calculation using the published overall mean.
"""

import re
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

TOTAL_CATEGORY = "Все категории"
PUBLISHED_CATEGORIES = (
    "Продовольствие",
    "Здоровье",
    "Маркетплейсы",
    "Общественное питание",
    "Транспорт",
)
ARITHMETIC_RESIDUAL = "Прочее (арифметический остаток)"
EXPECTED_PERIODS = tuple(f"{year}-{month:02d}" for year in (2023, 2024) for month in range(1, 13))
_COLUMNS = ("date", "territory_id", "category", "value")
_KEY = ("territory_id", "date", "category")
_MONTH_PATTERN = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return typed columns, rejecting ambiguity rather than repairing rows."""
    missing_columns = sorted(set(_COLUMNS) - set(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    frame = panel.loc[:, list(_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("The consumption panel is empty")
    if frame.isna().any().any():
        raise ValueError("Null values are not allowed in the consumption panel")
    if (
        not frame["date"]
        .map(lambda value: isinstance(value, str) and _MONTH_PATTERN.fullmatch(value) is not None)
        .all()
    ):
        raise ValueError("date must contain month strings in YYYY-MM format")
    ids = pd.to_numeric(frame["territory_id"], errors="raise")
    if (
        not np.isfinite(ids.to_numpy(dtype=float)).all()
        or not ids.gt(0).all()
        or not ids.eq(np.floor(ids)).all()
        or not ids.le(np.iinfo(np.int64).max).all()
    ):
        raise ValueError("territory_id must contain positive integer codes")
    frame["territory_id"] = ids.astype("int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
    if not np.isfinite(frame["value"]).all() or not frame["value"].gt(0).all():
        raise ValueError("value must contain finite, strictly positive numbers")
    if frame.duplicated(list(_KEY)).any():
        raise ValueError("Duplicate (territory_id, date, category) keys are not allowed")
    expected_categories = set(PUBLISHED_CATEGORIES) | {TOTAL_CATEGORY}
    if set(frame["category"]) != expected_categories:
        raise ValueError("Expected exactly the five published categories and Все категории")
    group_counts = frame.groupby(["territory_id", "date"], sort=False).size()
    if not group_counts.eq(len(expected_categories)).all():
        raise ValueError("Every observed municipality-month must contain all 6 categories")
    return frame.sort_values(["date", "territory_id", "category"]).reset_index(drop=True)


def load_panel(
    path: str | Path,
    complete_only: bool = True,
    *,
    expected_periods: Sequence[str] = EXPECTED_PERIODS,
) -> pd.DataFrame:
    """Load an original coded Parquet file and validate its observation grain.

    The observed calendar must equal ``expected_periods`` (2023–2024 by default).
    ``complete_only=True`` retains territories observed in every expected month.
    With ``False``, incomplete territories remain; missing months are never filled.
    A shorter explicit calendar is useful for a deliberately scoped source or test.
    Returned columns are date, territory_id, category and value, sorted by month/ID.
    """
    periods = tuple(expected_periods)
    if (
        not periods
        or len(set(periods)) != len(periods)
        or any(not isinstance(p, str) or _MONTH_PATTERN.fullmatch(p) is None for p in periods)
    ):
        raise ValueError("expected_periods must contain unique YYYY-MM strings")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = _validate_panel(duckdb.read_parquet(str(source)).df())
    observed_periods = set(frame["date"])
    expected = set(periods)
    if observed_periods != expected:
        raise ValueError(
            "Source calendar differs from expected_periods: "
            f"missing={sorted(expected - observed_periods)}, "
            f"unexpected={sorted(observed_periods - expected)}"
        )
    if complete_only:
        month_counts = frame.groupby("territory_id")["date"].nunique()
        complete_ids = month_counts.index[month_counts.eq(len(periods))]
        frame = frame.loc[frame["territory_id"].isin(complete_ids)].reset_index(drop=True)
        if frame.empty:
            raise ValueError("No territory has a complete history for expected_periods")
    return frame


def make_profiles(
    panel: pd.DataFrame,
    include_residual: bool = False,
    drop_category: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Return month → (sorted IDs, normalized means, ordered category names).

    Main definition: P[i,j] = value[i,j] / sum(five published category means).
    With ``include_residual``, append the positive arithmetic difference between
    the reported overall mean and all five component means. This is a sensitivity
    proxy, not a verified sixth expenditure category or household budget share.
    ``drop_category`` removes one selected feature before renormalization; the
    residual, when requested, is always calculated from all five original means.
    Each month's ID array describes that month; an unbalanced input is preserved.
    """
    frame = _validate_panel(panel)
    selected = list(PUBLISHED_CATEGORIES)
    if include_residual:
        selected.append(ARITHMETIC_RESIDUAL)
    if drop_category is not None:
        if drop_category not in selected:
            raise ValueError(f"drop_category must be one of {selected}")
        selected.remove(drop_category)
    result: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for month, monthly in frame.groupby("date", sort=True):
        wide = monthly.pivot(index="territory_id", columns="category", values="value")
        wide = wide.sort_index()
        if include_residual:
            residual = wide[TOTAL_CATEGORY] - wide.loc[:, list(PUBLISHED_CATEGORIES)].sum(axis=1)
            if not np.isfinite(residual).all() or not residual.gt(0).all():
                raise ValueError("Arithmetic residual must be finite and strictly positive")
            wide[ARITHMETIC_RESIDUAL] = residual
        values = wide.loc[:, selected].to_numpy(dtype=np.float64)
        # Scaling by the row maximum avoids overflowing sums of finite inputs.
        scaled = values / values.max(axis=1, keepdims=True)
        normalized = scaled / scaled.sum(axis=1, keepdims=True)
        result[str(month)] = (
            wide.index.to_numpy(dtype=np.int64),
            normalized,
            selected.copy(),
        )
    return result
