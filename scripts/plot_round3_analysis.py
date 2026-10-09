"""Create three source-backed round-three research figures; never refit models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

BLUE, TEAL, INK = "#4056A1", "#397A79", "#273442"
BASELINE = "economic_sectors_newflags"
FULL = "economic_sectors_newvalues"
SAMPLES = {
    "core": "Основная",
    "stable_code": "Без смены кода/типа",
    "complete_new3": "Все 3 новых значения",
}
REGIMES = {
    "within_region": "Муниципалитеты внутри регионов",
    "heldout_regions": "Целые регионы вне обучения",
}
LABELS = {"investment": "Инвестиции", "shipments": "Отгрузка", "retail": "Розница"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(
        out / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None}
    )
    plt.close(fig)


def checked_comparisons(frame: pd.DataFrame, target: str, alternatives: list[str]) -> pd.DataFrame:
    selected = frame.loc[
        frame.seed.eq(13)
        & frame.baseline.eq(BASELINE)
        & frame.target.eq(target)
        & frame.alternative.isin(alternatives)
        & frame["sample"].isin(SAMPLES if target == "profile_clr" else ["core"])
        & frame.regime.isin(REGIMES)
        & frame.year.isin([2023, 2024])
    ].copy()
    key = ["year", "sample", "regime", "seed", "baseline", "alternative", "target"]
    if selected.duplicated(key).any():
        raise ValueError("Duplicate comparison cells")
    expected = 12 if target == "profile_clr" else 8
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} comparisons, found {len(selected)}")
    for column in ["relative_mse_reduction", "ci_low", "ci_high"]:
        if not np.isfinite(selected[column]).all():
            raise ValueError("Missing comparison or interval")
        selected[column + "_pct"] = 100 * selected[column]
    if not (
        selected.ci_low.le(selected.relative_mse_reduction)
        & selected.relative_mse_reduction.le(selected.ci_high)
    ).all():
        raise ValueError("Point estimate lies outside supplied interval")
    return selected


def draw_interval(ax, row, position: int, *, color: str, marker: str, filled: bool) -> None:
    estimate, low, high = (
        100 * getattr(row, c) for c in ["relative_mse_reduction", "ci_low", "ci_high"]
    )
    ax.errorbar(
        estimate,
        position,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt=marker,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgewidth=1.5,
        capsize=4,
        markersize=6,
        elinewidth=1.5,
    )
    ax.annotate(
        f"{estimate:+.2f}",
        (estimate, position),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
    )


def style_forest(ax, labels: list[str], title: str, limits: tuple[float, float], xlabel: str):
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_ylim(len(labels) - 0.5, -0.65)
    ax.axvline(0, color=INK, linewidth=1, linestyle="--")
    ax.set_xlim(*limits)
    ax.set_title(title, loc="left", fontsize=12, pad=18)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.grid(axis="x", color="#E3E5E9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)


def coverage_plot(coverage: pd.DataFrame, out: Path) -> pd.DataFrame:
    data = coverage.loc[coverage["sample"].eq("core")].copy()
    if len(data) != 6 or data.duplicated(["year", "prefix"]).any():
        raise ValueError("Expected six core coverage cells")
    if not data.numeric.eq(data.feature_available).all():
        raise ValueError("Core raw/feature coverage differs; update chart scope explicitly")
    data["coverage_pct"] = 100 * data.numeric / data.n_municipality_years
    data["label"] = data.prefix.map(LABELS)
    data.to_csv(out / "coverage_chart.csv", index=False)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    for year, offset, color, hatch in [(2023, -0.18, TEAL, "//"), (2024, 0.18, BLUE, None)]:
        rows = data.loc[data.year.eq(year)].set_index("prefix").reindex(LABELS)
        positions = np.arange(len(rows)) + offset
        ax.barh(positions, 100, height=0.3, color="#F0F1F4", zorder=0)
        ax.barh(
            positions,
            rows.coverage_pct,
            height=0.3,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
        for position, row in zip(positions, rows.itertuples(), strict=True):
            ax.text(
                row.coverage_pct + 1.2,
                position,
                f"{row.numeric} / {row.n_municipality_years} ({row.coverage_pct:.1f}%)",
                va="center",
                fontsize=10,
            )
    ax.set_yticks(range(3), LABELS.values(), fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Доступные числовые значения, % от frozen core")
    ax.grid(axis="x", color="#E3E5E9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.suptitle("Покрытие новых показателей в основной выборке", x=0.015, ha="left", fontsize=18)
    fig.text(
        0.015,
        0.91,
        "Подписи: число значений / число МО в выборке соответствующего года.",
        fontsize=11,
    )
    fig.legend(
        handles=[
            Patch(facecolor=TEAL, hatch="//", edgecolor="white", label="2023"),
            Patch(facecolor=BLUE, label="2024"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.938),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.015,
        0.035,
        "Пропуски не равны нулю. Состав core сохранён; данные получены в 2026 году.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.87])
    save(fig, out, "activity_coverage")
    return data


def profile_plot(comparisons: pd.DataFrame, out: Path) -> pd.DataFrame:
    data = checked_comparisons(comparisons, "profile_clr", [FULL])
    data["sample_label"] = data["sample"].map(SAMPLES)
    data.to_csv(out / "profile_increment_chart.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), sharex=True)
    span = 100 * (data.ci_high.max() - data.ci_low.min())
    limits = (
        100 * min(0, data.ci_low.min()) - 0.1 * span,
        100 * max(0, data.ci_high.max()) + 0.1 * span,
    )
    for ax, (regime, title) in zip(axes, REGIMES.items(), strict=True):
        rows = (
            data.loc[data.regime.eq(regime)]
            .assign(order=lambda x: x["sample"].map({s: i for i, s in enumerate(SAMPLES)}))
            .sort_values(["year", "order"], ascending=[False, True])
        )
        labels = []
        for i, row in enumerate(rows.itertuples()):
            draw_interval(
                ax,
                row,
                i,
                color=BLUE if row.year == 2024 else TEAL,
                marker="o" if row.year == 2024 else "s",
                filled=True,
            )
            labels.append(
                f"{row.year} · {SAMPLES[row.sample]}\nn = {row.n}; регионов {row.regions}"
            )
        style_forest(ax, labels, title, limits, "Снижение MSE профиля CLR, %")
        ax.axhline(2.5, color="#E3E5E9", linewidth=0.8)
    includes_zero = int((data.ci_low.le(0) & data.ci_high.ge(0)).sum())
    fig.suptitle(
        "Первичное сравнение: новые признаки в модели профиля", x=0.015, ha="left", fontsize=18
    )
    fig.text(
        0.015,
        0.926,
        "База: экономика + 8 отраслей + флаги пропусков. "
        "Добавлены инвестиции, отгрузка и розница; seed 13.",
        fontsize=10.5,
    )
    fig.text(
        0.015,
        0.055,
        "95% интервалы: перевыборка регионов при фиксированных OOF-прогнозах; "
        "модели не переобучаются.",
        fontsize=10,
    )
    fig.text(
        0.015,
        0.023,
        f"Все {includes_zero} из {len(data)} интервалов включают ноль. "
        "Справа от нуля — меньшая ошибка; дополнительный сигнал для профиля не подтверждён.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.895], w_pad=3)
    save(fig, out, "profile_increment")
    return data


def control_plot(comparisons: pd.DataFrame, out: Path) -> pd.DataFrame:
    data = checked_comparisons(comparisons, "log_total", [FULL, "permuted_newvalues"])
    if not data.groupby(["year", "regime"])[["n", "regions"]].nunique().eq(1).all().all():
        raise ValueError("Real and permuted results have different denominators")
    data["series_label"] = data.alternative.map(
        {FULL: "Реальные", "permuted_newvalues": "Перемешанные"}
    )
    data.to_csv(out / "log_total_control_chart.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.3), sharex=True)
    span = 100 * (data.ci_high.max() - data.ci_low.min())
    limits = (
        100 * min(0, data.ci_low.min()) - 0.1 * span,
        100 * max(0, data.ci_high.max()) + 0.1 * span,
    )
    for ax, (regime, title) in zip(axes, REGIMES.items(), strict=True):
        rows = (
            data.loc[data.regime.eq(regime)]
            .assign(order=lambda x: x.alternative.map({FULL: 0, "permuted_newvalues": 1}))
            .sort_values(["year", "order"], ascending=[False, True])
        )
        labels = []
        for i, row in enumerate(rows.itertuples()):
            real = row.alternative == FULL
            draw_interval(
                ax, row, i, color=BLUE if real else TEAL, marker="o" if real else "D", filled=real
            )
            labels.append(f"{row.year} · {row.series_label}\nn = {row.n}; регионов {row.regions}")
        style_forest(ax, labels, title, limits, "Снижение MSE log_total, %")
        ax.axhline(1.5, color="#E3E5E9", linewidth=0.8)
    fig.suptitle(
        "Общие расходы: реальные значения и отрицательный контроль", x=0.015, ha="left", fontsize=18
    )
    fig.text(
        0.015,
        0.923,
        "Одна база: экономика + 8 отраслей + флаги пропусков; core, seed 13.",
        fontsize=10.5,
    )
    fig.text(
        0.015,
        0.889,
        "Новые значения перемешаны внутри региона, отдельно в обучении и проверке; "
        "пропуски сохранены.",
        fontsize=10.5,
    )
    fig.text(
        0.015,
        0.085,
        "95% интервалы: перевыборка регионов при фиксированных OOF-прогнозах.",
        fontsize=10,
    )
    fig.text(
        0.015,
        0.054,
        "Для новых регионов точечный выигрыш после перемешивания не меньше реального; "
        "вторичные выигрыши требуют осторожности.",
        fontsize=10,
    )
    fig.text(
        0.015,
        0.023,
        "Одна фиксированная перестановка — диагностический контроль, "
        "не распределение нулевой гипотезы и не тест причинности.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 0.855], w_pad=3)
    save(fig, out, "log_total_negative_control")
    return data


def run(root: Path, out: Path):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": "#AAB0B7",
            "svg.hashsalt": "municipal-round3-analysis",
        }
    )
    inputs = {
        "coverage": root / "audit/coverage.csv",
        "comparisons": root / "models/cv_comparisons.csv",
    }
    hashes = {k: sha(p) for k, p in inputs.items()}
    contracts = {
        "activity_coverage": {
            "question": "How many core observations have each new value?",
            "family": "grouped horizontal bars",
            "rows": 6,
            "denominator": "frozen core per year",
            "axis_start": 0,
        },
        "profile_increment": {
            "question": "Is additional profile information supported?",
            "family": "paired-regime interval forest",
            "rows": 12,
            "target": "profile_clr",
            "baseline": BASELINE,
            "alternative": FULL,
        },
        "log_total_negative_control": {
            "question": "How do secondary gains compare with shuffled values?",
            "family": "paired-regime interval forest",
            "rows": 8,
            "target": "log_total",
            "baseline": BASELINE,
            "alternatives": [FULL, "permuted_newvalues"],
        },
        "rendering": {
            "engine": "Matplotlib static PNG/SVG",
            "palette": [BLUE, TEAL, INK],
            "non_color_encoding": (
                "hatching for coverage; marker shape/fill and row labels for forests"
            ),
            "visual_review": "actual exported PNG files",
        },
    }
    write_json(out / "chart_contracts.json", contracts)
    coverage = coverage_plot(pd.read_csv(inputs["coverage"]), out)
    comparisons = pd.read_csv(inputs["comparisons"])
    primary = profile_plot(comparisons, out)
    control = control_plot(comparisons, out)
    heldout = control.loc[control.regime.eq("heldout_regions")].pivot(
        index="year", columns="alternative", values="relative_mse_reduction"
    )
    if not (heldout.permuted_newvalues >= heldout[FULL]).all():
        raise ValueError("Negative-control caption is not supported by result table")
    if not (primary.ci_low.le(0) & primary.ci_high.ge(0)).all():
        raise ValueError("Primary conclusion caption is not supported by result table")
    if {k: sha(p) for k, p in inputs.items()} != hashes:
        raise ValueError("Inputs changed while plotting")
    write_json(
        out / "QA.json",
        {
            "numeric_checks": "PASS",
            "visual_review": "PENDING actual PNG inspection",
            "coverage_cells": len(coverage),
            "primary_comparisons": len(primary),
            "negative_control_comparisons": len(control),
            "primary_intervals_include_zero": int(
                (primary.ci_low.le(0) & primary.ci_high.ge(0)).sum()
            ),
            "same_rich_flags_baseline": True,
            "seed": 13,
            "shared_limits_within_each_forest": True,
            "source_inputs_unchanged": True,
            "point_in_time_or_causal_claim": False,
        },
    )
    write_json(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "inputs": hashes,
            "output_hashes": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps({"figures": 3, "formats": ["PNG", "SVG"], "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/round3_analysis"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.root, args.out or args.root / "figures")
