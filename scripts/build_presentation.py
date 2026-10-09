"""Build the Russian 16:9 research presentation from the saved result tables.

Run from any directory: python scripts/build_presentation.py
Requires the project's core dependencies and the ``presentation`` dependency group.
No model is refitted, and no result table is modified. Figures are regenerated in
memory from their original CSV sources, using the original samples and definitions.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, PercentFormatter
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
INK = "#193441"
TEAL = "#087F82"
GOLD = "#BD7939"
BLUE = "#315AB4"
MUTED = "#60727A"
RULE = "#D7E1E4"
PALE = "#F1F6F7"
YEARS = {2023: GOLD, 2024: TEAL}
METHODS = ["leiden_hellinger", "leiden_cosine", "kmeans", "kmeans_matched_k"]
METHOD_NAMES = {
    "leiden_hellinger": "Leiden / Хеллингер",
    "leiden_cosine": "Leiden / косинус",
    "kmeans": "KMeans / K = 8",
    "kmeans_matched_k": "KMeans / сопоставимое K",
}
METHOD_COLORS = [TEAL, GOLD, BLUE, "#935B87"]
TOTAL_SLIDES = 14


def register_fonts() -> None:
    """Use portable Cyrillic fonts shipped with the locked Matplotlib package."""
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Research", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ResearchBold", str(fonts / "DejaVuSans-Bold.ttf")))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": RULE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": RULE,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def russian(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}".replace(".", ",")


def integer(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


class Deck:
    """A simple editorial grid, with top-origin coordinates and overflow checks."""

    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1, invariant=1)
        self.pdf.setTitle("Потребительские профили: экономика территорий и сети сходства")
        self.pdf.setAuthor("roshlight / sberindex-clustering-submission")
        self.pdf.setSubject("СберИндекс 2026. Экономические различия 2 016 МО в 2023-2024 годах.")
        self.page = 0

    def text(self, x, y, text, size=22, color=INK, bold=False):
        font = "ResearchBold" if bold else "Research"
        if x + pdfmetrics.stringWidth(text, font, size) > W - 48:
            raise ValueError(f"Text exceeds the right margin on slide {self.page}: {text}")
        self.pdf.setFillColor(colors.HexColor(color))
        self.pdf.setFont(font, size)
        self.pdf.drawString(x, H - y - size * 0.8, text)

    def paragraph(self, x, y, text, width, size=20, leading=None, color=INK, bold=False):
        font = "ResearchBold" if bold else "Research"
        leading = leading or size * 1.36
        offset = 0
        for segment in text.split("\n"):
            line = ""
            for word in segment.split():
                proposed = f"{line} {word}".strip()
                if pdfmetrics.stringWidth(proposed, font, size) <= width:
                    line = proposed
                else:
                    if not line:
                        raise ValueError(f"Word too wide: {word}")
                    self.text(x, y + offset, line, size, color, bold)
                    offset += leading
                    line = word
            if line:
                self.text(x, y + offset, line, size, color, bold)
                offset += leading
            else:
                offset += leading * 0.5
        max_bottom = H - 8 if y >= 672 else 670
        if y + offset > max_bottom:
            raise ValueError(f"Text would overflow slide {self.page}: {text}")
        return y + offset

    def line(self, x1, y1, x2, y2, color=RULE, width=1):
        self.pdf.setStrokeColor(colors.HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.line(x1, H - y1, x2, H - y2)

    def rect(self, x, y, width, height, color):
        self.pdf.setFillColor(colors.HexColor(color))
        self.pdf.rect(x, H - y - height, width, height, stroke=0, fill=1)

    def start(self, section, title, subtitle=None):
        self.page += 1
        self.rect(0, 0, W, H, "#FFFFFF")
        self.text(54, 29, "СБЕРИНДЕКС 2026  /  ИССЛЕДОВАНИЕ", 13, MUTED, True)
        self.text(1040, 29, section.upper(), 13, TEAL, True)
        self.line(54, 59, 1226, 59)
        self.text(54, 82, title, 36, INK, True)
        if subtitle:
            self.paragraph(54, 137, subtitle, 1172, 18, color=MUTED)

    def footer(self, source):
        self.line(54, 672, 1226, 672)
        self.paragraph(54, 684, source, 1095, 11, leading=13, color=MUTED)
        self.text(1169, 684, f"{self.page:02d} / {TOTAL_SLIDES}", 11, MUTED)
        self.pdf.showPage()

    def figure(self, fig, x, y, width, height):
        """Embed a freshly generated, high resolution chart, without disk assets."""
        data = BytesIO()
        fig.savefig(data, format="png", dpi=190)
        plt.close(fig)
        data.seek(0)
        self.pdf.drawImage(ImageReader(data), x, H - y - height, width, height, mask="auto")

    def finish(self):
        if self.page != TOTAL_SLIDES:
            raise ValueError(f"Expected {TOTAL_SLIDES} slides, got {self.page}")
        self.pdf.save()


def gradients() -> plt.Figure:
    panel = read("reports/enriched_analysis/core_v2/annual_panel.csv")
    core = panel.loc[(panel.year == 2024) & panel.core_eligible]
    if len(core) != 1998:
        raise ValueError("The original 2024 gradient sample must have 1998 rows")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.55))
    fig.subplots_adjust(left=0.071, right=0.962, bottom=0.23, top=0.9, wspace=0.37)
    settings = [
        (
            core.wage / 1000,
            core.total_mean / 1000,
            "Зарплата и уровень расходов",
            "Зарплата организаций, тыс. ₽ / месяц",
            "Оценка расходов, тыс. ₽ / месяц",
            [30, 50, 100, 200, 400],
        ),
        (
            core.population_total / 1000,
            core.p_foodservice * 100,
            "Население и компонент общепита",
            "Население, тыс. человек",
            "Общепит в профиле, %",
            [2, 10, 50, 200, 1000],
        ),
    ]
    for ax, (x, y, title, xlabel, ylabel, ticks) in zip(axes, settings, strict=True):
        points = ax.hexbin(x, y, gridsize=32, mincnt=1, cmap="Blues", xscale="log", linewidths=0)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(ticks, labels=[str(t) for t in ticks])
        ax.grid(axis="y", alpha=0.5)
        bar = fig.colorbar(points, ax=ax, shrink=0.82, pad=0.025)
        bar.set_label("МО в ячейке", fontsize=12)
        bar.ax.tick_params(labelsize=11)
    return fig


def economic_comparisons() -> pd.DataFrame:
    data = read("reports/enriched_analysis/core_v2/cv_comparisons.csv")
    selected = data.loc[
        (data.year == 2024)
        & (data["sample"] == "core")
        & (data.baseline == "context")
        & (data.alternative == "economic")
    ].set_index(["regime", "target"])
    expected = {
        ("within_region", "profile_clr"),
        ("heldout_regions", "profile_clr"),
        ("within_region", "log_total"),
        ("heldout_regions", "log_total"),
    }
    if len(selected) != 4 or set(selected.index) != expected:
        raise ValueError("Expected the four original 2024 core economic comparisons")
    return selected


def economic_increment() -> plt.Figure:
    data = economic_comparisons()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.18, top=0.83, wspace=0.22)
    for ax, regime, title in zip(
        axes,
        ["within_region", "heldout_regions"],
        ["Внутри регионов", "Регионы вне обучения"],
        strict=True,
    ):
        values = [
            float(data.loc[(regime, target), "relative_mse_reduction"]) * 100
            for target in ["profile_clr", "log_total"]
        ]
        ax.barh([0, 1], values, color=[TEAL, BLUE], height=0.47)
        for y, value in enumerate(values):
            ax.text(value + 1.5, y, russian(value, 1) + "%", va="center", weight="bold")
        ax.set(
            title=title,
            yticks=[0, 1],
            yticklabels=["Состав профиля", "Уровень расходов"],
            xlabel="Снижение квадратичной ошибки, %",
            xlim=(0, 76),
            ylim=(1.6, -0.6),
        )
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="x", alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    return fig


def sector_increment() -> plt.Figure:
    data = read("reports/round2_analysis/models/cv_comparisons.csv")
    data = data.loc[
        (data.target == "profile_clr")
        & (data.baseline == "economic_missingness")
        & (data.alternative == "economic_sectors")
        & (data.seed == 13)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), sharex=True)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.17, top=0.88, wspace=0.17)
    keys = [
        (2024, "core"),
        (2024, "stable_code"),
        (2024, "complete_sectors"),
        (2023, "core"),
        (2023, "stable_code"),
        (2023, "complete_sectors"),
    ]
    sample_names = {
        "core": "Основная",
        "stable_code": "Без смены кода / типа",
        "complete_sectors": "Все 8 отраслей раскрыты",
    }
    labels = []
    for year, sample in keys:
        n = int(data.loc[(data.year == year) & (data["sample"] == sample)].iloc[0]["n"])
        labels.append(f"{year}: {sample_names[sample]}\nn = {integer(n)}")
    for ax, regime, title in zip(
        axes,
        ["within_region", "heldout_regions"],
        ["Внутри регионов", "Целые регионы вне обучения"],
        strict=True,
    ):
        ax.axvline(0, color=MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
        for j, (year, sample) in enumerate(keys):
            row = data.loc[
                (data.year == year) & (data["sample"] == sample) & (data.regime == regime)
            ].iloc[0]
            val = float(row.relative_mse_reduction) * 100
            lo, hi = float(row.ci_low) * 100, float(row.ci_high) * 100
            ax.errorbar(
                val,
                j,
                xerr=[[val - lo], [hi - val]],
                fmt="o",
                color=YEARS[year],
                markersize=7,
                capsize=4,
                linewidth=1.7,
            )
        ax.set(title=title, xlim=(-28, 16), ylim=(5.6, -0.6))
        ax.set_xticks([-25, -15, -5, 0, 5, 15])
        ax.grid(axis="x", alpha=0.6)
        ax.set_xlabel("Снижение ошибки CLR-профиля, %")
        ax.set_yticks(range(6), labels=labels if ax is axes[0] else [""] * 6)
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.axhline(2.5, color=RULE)
    return fig


def factor_sensitivity() -> plt.Figure:
    data = read("reports/initial_study/single_factor_comparisons.csv")
    factors = ["Случайный старт", "Мера сходства", "Число соседей", "Разрешение"]
    if set(data.factor) != set(factors):
        raise ValueError("Unexpected sensitivity factor labels")
    fig, ax = plt.subplots(figsize=(12.2, 4.5))
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.93)
    for year, color in YEARS.items():
        values = [
            data.loc[(data.year == year) & (data.factor == name), "ari_to_primary"]
            for name in factors
        ]
        box = ax.boxplot(
            values,
            positions=np.arange(4) + (-0.16 if year == 2023 else 0.16),
            widths=0.25,
            orientation="horizontal",
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.8},
            whiskerprops={"color": MUTED},
            capprops={"color": MUTED},
        )
        for patch in box["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.55)
        ax.scatter([], [], color=color, label=str(year), s=65)
    ax.set(
        yticks=np.arange(4),
        yticklabels=factors,
        xlim=(0, 1),
        ylim=(3.7, -0.7),
        xlabel="ARI к основному разбиению месяца",
    )
    ax.grid(axis="x", alpha=0.65)
    ax.legend(loc="upper right", ncol=2)
    return fig


def annual_stability() -> plt.Figure:
    data = read("reports/enriched_analysis/annual_networks/temporal_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.23, top=0.87, wspace=0.39)
    for ax, field, title, factor, xmax in [
        (axes[0], "ari", "Согласие 2023 и 2024 года", 1, 0.85),
        (axes[1], "switch_share", "Сменили группу после сопоставления", 100, 70),
    ]:
        for j, method in enumerate(["leiden_hellinger", "kmeans"]):
            rows = data.loc[data.method == method].set_index("profile_basis")
            for basis, color, offset, label in [
                ("original_december", GOLD, -0.17, "Декабрь каждого года"),
                ("annual_mean_12months", TEAL, 0.17, "Среднее 12 месяцев"),
            ]:
                value = float(rows.loc[basis, field]) * factor
                ax.barh(
                    j + offset, value, height=0.29, color=color, label=label if j == 0 else None
                )
                label = russian(value, 4) if field == "ari" else russian(value, 2) + "%"
                ax.text(value + xmax * 0.023, j + offset, label, va="center", fontsize=14)
        ax.set(
            yticks=[0, 1],
            yticklabels=["Leiden", "KMeans\nK = 8"],
            xlim=(0, xmax),
            title=title,
            ylim=(1.55, -0.55),
        )
        ax.grid(axis="x", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("ARI" if field == "ari" else "Доля МО, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.02))
    return fig


def transitions() -> plt.Figure:
    data = read("reports/initial_study/temporal_summary.csv")
    data = data.loc[data.lag == 1]
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 4.75), sharex=True)
    fig.subplots_adjust(left=0.072, right=0.975, bottom=0.16, top=0.87, hspace=0.43)
    for method, color in zip(METHODS, METHOD_COLORS, strict=True):
        rows = data.loc[data.method == method].sort_values("to_month")
        x = np.arange(len(rows))
        if len(rows) != 23:
            raise ValueError("Expected 23 adjacent month pairs per method")
        for ax, field in zip(axes, ["ari", "switch_share"], strict=True):
            ax.plot(
                x,
                rows[field],
                color=color,
                linewidth=1.9,
                linestyle="-" if method in {"leiden_hellinger", "kmeans"} else "--",
                label=METHOD_NAMES[method],
            )
        if method == "leiden_hellinger":
            axes[1].plot(x, rows.robust_switch_share_of_all, color=INK, linewidth=2.5)
    axes[0].set(ylabel="ARI", ylim=(0, 0.8))
    axes[1].set(ylabel="Сменили группу", ylim=(-0.02, 0.65))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    months = sorted(data.to_month.unique())
    ticks = [0, 5, 11, 16, 22]
    axes[1].set_xticks(ticks, labels=[months[i] for i in ticks])
    for ax in axes:
        ax.grid(axis="y", alpha=0.6)
        ax.axvline(11, color=RULE, linewidth=1.3)
        ax.set_xlim(-0.4, 22.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.54, 1),
        fontsize=11.5,
        handlelength=2.5,
    )
    axes[1].text(
        0.3,
        0.025,
        "Подтверждённые переходы основного Leiden: 0",
        color=INK,
        fontsize=12,
        va="bottom",
    )
    return fig


def market_access() -> plt.Figure:
    data = read("data/derived/reviews/spatial_external_v1/group_summary.csv")
    data = data.loc[(data["sample"] == "all") & (data.hypothesis == "H2")]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.55), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.87, wspace=0.32)
    for ax, month, title in zip(
        axes, ["2024-06", "2024-12"], ["Июнь 2024", "Декабрь 2024"], strict=True
    ):
        rows = data.loc[data.month == month].sort_values("label")
        y = np.arange(len(rows))
        ax.errorbar(
            rows["median"],
            y,
            xerr=[rows["median"] - rows.q25, rows.q75 - rows["median"]],
            fmt="o",
            color=TEAL,
            capsize=3,
            markersize=5.5,
            linewidth=1.7,
        )
        ax.set(
            yticks=y,
            yticklabels=[f"{int(r.label)}   ({int(r.n_observed)})" for r in rows.itertuples()],
            title=title,
            xlabel="Индекс доступности рынков за 2024 год",
            xlim=(0, 1000),
            ylim=(len(rows) - 0.5, -0.5),
        )
        ax.grid(axis="x", alpha=0.6)
        ax.tick_params(axis="y", labelsize=12, length=0)
    axes[0].set_ylabel("Группа (число МО с индексом)")
    return fig


def build(output: Path) -> None:
    register_fonts()
    deck = Deck(output)
    economic = economic_comparisons()
    profile_gain = float(economic.loc[("within_region", "profile_clr"), "relative_mse_reduction"])
    heldout_gain = float(economic.loc[("heldout_regions", "profile_clr"), "relative_mse_reduction"])
    correlations = read("reports/enriched_analysis/core_v2/correlations.csv")
    correlations = correlations.loc[correlations.year == 2024].set_index(["feature", "outcome"])
    population_foodservice = correlations.loc[("log_population", "p_foodservice")]
    wage_spending = correlations.loc[("log_wage", "log_total")]
    spatial = read("data/derived/reviews/spatial_external_v1/spatial_checks.csv")
    spatial = spatial.loc[
        (spatial["sample"] == "all")
        & (spatial.hypothesis == "H2")
        & (spatial.method == "leiden_hellinger")
    ].set_index("month")
    if len(spatial) != 2 or set(spatial.index) != {"2024-06", "2024-12"}:
        raise ValueError("Expected the two original market-access comparisons")

    # 1: the central finding is visible before details of the protocol.
    deck.start("Результат", "Потребительские профили муниципалитетов")
    deck.text(54, 169, "Потребление отражает", 53, INK, True)
    deck.text(54, 236, "экономику территорий.", 53, TEAL, True)
    deck.paragraph(
        56,
        326,
        "Сети сходства связывают муниципалитеты с близкой структурой расходов",
        1040,
        25,
        color=MUTED,
    )
    positions = [54, 438, 830]
    for x, number, label in zip(
        positions,
        ["2 016", "24", russian(profile_gain * 100, 1) + "%"],
        ["муниципалитетов", "месяца: 2023-2024", "снижение ошибки профиля"],
        strict=True,
    ):
        deck.text(x, 423, number, 43, INK, True)
        deck.text(x, 483, label, 19, MUTED)
    deck.text(830, 512, "внутри регионов, 2024 год", 16, MUTED)
    deck.line(54, 536, 1226, 536)
    deck.paragraph(
        54,
        560,
        "Население, зарплата и концентрация рабочих мест улучшают описание потребления "
        "по сравнению с моделью, учитывающей только регион и тип муниципалитета.",
        1125,
        22,
    )
    deck.footer("Код, данные и отчёт: github.com/roshlight/sberindex-clustering-submission")

    # 2: diagram replaces the original paragraph-heavy protocol slide.
    deck.start(
        "Метод",
        "От пяти показателей к помесячной сети",
        "Разработка: 2023 год. Проверка зафиксированной процедуры: 2024 год.",
    )
    steps = [
        (
            54,
            "01",
            "Профиль",
            "Продовольствие, здоровье, маркетплейсы, общепит, транспорт.",
            "Нормировка на сумму пяти опубликованных средних.",
        ),
        (
            455,
            "02",
            "Сеть сходства",
            "20 ближайших соседей для каждого МО.",
            "Объединение списков соседей; взвешенная сеть.",
        ),
        (
            856,
            "03",
            "Сообщества",
            "Leiden: разрешение 1,0; seed 13.",
            "Сравнение: четыре метода и 21 альтернатива в каждом месяце.",
        ),
    ]
    for x, n, title, body, extra in steps:
        deck.text(x, 204, n, 35, TEAL, True)
        deck.line(x, 257, x + 355, 257, TEAL, 2)
        deck.text(x, 279, title, 25, INK, True)
        deck.paragraph(x, 329, body, 350, 20)
        deck.paragraph(x, 428, extra, 350, 18, color=MUTED)
    deck.text(54, 539, "Вес связи: сумма √(pᵢc · pⱼc) по пяти компонентам профиля", 21, TEAL)
    deck.paragraph(
        54,
        592,
        "Профиль: соотношение пяти опубликованных средних расходов. "
        "Связь в сети: сходство структуры потребления двух муниципалитетов.",
        1160,
        19,
    )
    deck.footer("Источники: reports/initial_study/REPORT.md; configs/robustness_v1.json")

    # 3: the economic gradients use the same original core sample.
    deck.start(
        "Экономика",
        "Масштаб и зарплата видны в расходах",
        "2024 год; 1 998 МО с полными базовыми признаками. По горизонтали - логарифмическая шкала.",
    )
    deck.figure(gradients(), 45, 187, 1185, 407)
    deck.paragraph(
        54,
        601,
        "Ранговая связь населения с общепитом: "
        f"ρ = {russian(population_foodservice.spearman, 3)} "
        f"({integer(population_foodservice.n)} МО). Зарплаты с уровнем расходов: "
        f"ρ = {russian(wage_spending.spearman, 3)} ({integer(wage_spending.n)} МО). "
        "Корреляции используют все парно доступные данные; на графиках 1 998 МО.",
        1160,
        18,
        leading=25,
    )
    deck.footer(
        "Источники: reports/enriched_analysis/core_v2/annual_panel.csv; core_v2/correlations.csv; "
        "2024 год"
    )

    # 4: show the economic model's gain over region and municipality type.
    deck.start(
        "Экономика",
        "Экономические признаки улучшают описание профиля",
        "2024 год, 1 998 МО. База внутри регионов: регион и тип МО; "
        "на регионах вне обучения: тип МО.",
    )
    deck.figure(economic_increment(), 35, 191, 1195, 381)
    deck.paragraph(
        54,
        594,
        "Добавлены население, зарплата организаций и работники на душу населения. "
        "Состав профиля оценён в CLR-координатах, уровень расходов в логарифмах. "
        "Признаки и расходы относятся к одному году.",
        1160,
        18,
        leading=24,
    )
    deck.footer(
        "Источник: reports/enriched_analysis/core_v2/cv_comparisons.csv; выборка core, 2024"
    )

    # 5: original sample sensitivity and both years remain visible.
    deck.start(
        "Экономика",
        "Отрасли уточняют структуру потребления",
        "2024: снижение ошибки на 1,38% внутри регионов и 7,34% при удержании целых регионов.",
    )
    deck.figure(sector_increment(), 36, 190, 1195, 411)
    deck.paragraph(
        54,
        607,
        "Точки: изменение квадратичной ошибки CLR-профиля; "
        "полосы: 95% интервалы перевыборки регионов. "
        "На малой выборке полного раскрытия качество ухудшается. "
        "Интервалы без повторного обучения.",
        1160,
        18,
        leading=24,
    )
    deck.footer(
        "Источник: reports/round2_analysis/models/cv_comparisons.csv; "
        "seed 13; контроль пропусков включён"
    )

    # 6: large table with comparable geometric and network metrics.
    deck.start(
        "Метрики",
        "Leiden выше по модульности, KMeans по силуэту",
        "Медианы по 12 месяцам 2024 года. Стрелка показывает предпочтительное направление.",
    )
    metric_table = read("reports/initial_study/method_metrics.csv")
    metric_table = metric_table.loc[metric_table.year == 2024].set_index("method")
    columns = [
        "silhouette_median",
        "calinski_harabasz_median",
        "avi_median",
        "avu_median",
        "modularity_median",
    ]
    x = [54, 545, 692, 849, 985, 1120]
    deck.rect(54, 211, 1172, 56, PALE)
    for left, label in zip(x, ["Метод", "SW ↑", "CH ↑", "AVI ↑", "AVU ↓", "MQ ↑"], strict=True):
        deck.text(left + 12, 229, label, 20, INK, True)
    for i, method in enumerate(METHODS):
        top = 281 + i * 61
        deck.text(66, top, METHOD_NAMES[method], 22, INK, method == "leiden_hellinger")
        for j, field in enumerate(columns):
            deck.text(
                x[j + 1] + 12,
                top,
                russian(float(metric_table.loc[method, field]), 2 if j == 1 else 3),
                22,
            )
        deck.line(54, top + 43, 1226, top + 43)
    counts = [int(metric_table.loc[method, "s_dbw_count"]) for method in METHODS]
    if counts != [2, 8, 8, 0]:
        raise ValueError("The original S_Dbw availability changed")
    deck.paragraph(
        54,
        548,
        "S_Dbw рассчитан для 2/12, 8/12, 8/12 и 0/12 месяцев по порядку строк. "
        "Остальные случаи имеют нулевой знаменатель плотности.",
        1160,
        19,
    )
    deck.paragraph(
        54,
        613,
        "На сети Хеллингера Leiden даёт более высокую модульность, KMeans более высокий силуэт. "
        "Косинусный Leiden использует свою сеть. По AVU контрольные перестановки "
        "опережают основной метод.",
        1160,
        18,
        leading=23,
    )
    deck.footer(
        "Источники: reports/initial_study/method_metrics.csv; "
        "docs/metrics.md (определения SW, CH, AVI, AVU, MQ)"
    )

    # 7: one-factor distributions, with their true quartile/min/max meaning.
    deck.start(
        "Устойчивость",
        "Настройки меняют границы сообществ",
        "ARI сравнивает разбиения независимо от номеров меток. 1 означает полное совпадение.",
    )
    deck.figure(factor_sensitivity(), 35, 189, 1195, 406)
    deck.paragraph(
        54,
        607,
        "Ящик: квартиль, медиана, квартиль. Усы: минимум и максимум; "
        "Смена случайного старта также меняет состав групп.",
        1160,
        19,
        leading=25,
    )
    deck.footer(
        "Источник: reports/initial_study/single_factor_comparisons.csv; "
        "основной Leiden, помесячные разбиения"
    )

    # 8: retain the operational threshold, denominator and concentration in August.
    deck.start(
        "Поддержка",
        "Строгий фильтр отбирает 66 назначений",
        "Поддержка альтернативы: Jaccard состава группы ≥ 0,8. "
        "Требуется минимум 17 из 21 альтернативы.",
    )
    support = read("reports/initial_study/monthly_support.csv")
    for left, year in [(54, 2023), (683, 2024)]:
        rows = support.loc[support.year == year]
        strong, denominator = int(rows.n_strong.sum()), int(rows.n_entities.sum())
        expected = (66, 24192) if year == 2023 else (0, 24192)
        if (strong, denominator) != expected:
            raise ValueError(f"Unexpected original support totals in {year}")
        deck.text(left, 215, str(year), 25, MUTED, True)
        deck.text(left, 264, integer(strong), 91, TEAL if strong else INK, True)
        deck.text(left, 373, f"из {integer(denominator)} назначений МО × месяц", 22)
        deck.text(left, 419, russian(strong / denominator * 100, 3) + "% назначений", 24, MUTED)
    deck.line(54, 487, 1226, 487)
    deck.paragraph(
        54, 518, "Все 66 прошедших назначений относятся к августу 2023 года.", 1120, 25, bold=True
    )
    deck.paragraph(
        54,
        579,
        "Для анализа отдельной территории используйте полный профиль и ближайших соседей. "
        "При работе с меткой группы учитывайте совпадение её состава в 21 варианте расчёта.",
        1140,
        21,
    )
    deck.footer(
        "Источники: reports/initial_study/monthly_support.csv; reports/initial_study/REPORT.md"
    )

    # 9: compare methods on the same annual representation and territories.
    deck.start(
        "Устойчивость",
        "KMeans лучше сохраняет группы между годами",
        "Сравнение 2023 и 2024 года; 2 016 МО. Оба представления используют те же территории.",
    )
    deck.figure(annual_stability(), 35, 191, 1195, 392)
    deck.paragraph(
        54,
        588,
        "На годовых профилях KMeans с K = 8 даёт ARI 0,663, Leiden 0,325. "
        "Доля смен групп: 16,1% и 54,2%. Строгий фильтр Leiden отбирает "
        "0 из 2 016 назначений в каждом году.",
        1160,
        19,
        leading=25,
    )
    deck.footer(
        "Источники: reports/enriched_analysis/annual_networks/temporal_comparison.csv; "
        "annual_networks/README.md"
    )

    # 10: distinguish group label changes from changes in the measured profile.
    deck.start(
        "Динамика",
        "Для динамики сравнивайте сами профили",
        "18 008 смен на 46 368 пар соседних месяцев. "
        "Ни одна не проходит парный фильтр основного Leiden.",
    )
    temporal = read("reports/initial_study/temporal_summary.csv")
    primary = temporal.loc[(temporal.lag == 1) & (temporal.method == "leiden_hellinger")]
    pairs = int(primary.ncommon.sum())
    switches = int(np.rint(primary.switch_share * primary.ncommon).sum())
    if (switches, pairs) != (18008, 46368):
        raise ValueError("Original adjacent-month transition totals changed")
    deck.figure(transitions(), 39, 183, 1190, 413)
    deck.paragraph(
        54,
        604,
        "Совместная поддержка концов достигает 14/21 при пороге 17/21. "
        "Для мониторинга потребления сравнивайте уровни расходов и компоненты профиля "
        "между месяцами; смену метки проверяйте отдельно.",
        1160,
        19,
        leading=25,
    )
    deck.footer(
        "Источник: reports/initial_study/temporal_summary.csv; "
        "после попарного согласования меток; 23 пары по 2 016 МО"
    )

    # 11: original IQRs, local cluster labels and retrospective index date.
    deck.start(
        "Контекст",
        "Доступность рынков различается между группами",
        "Межгрупповая доля разброса индекса: "
        f"{russian(spatial.loc['2024-06', 'observed'] * 100, 1)}% в июне и "
        f"{russian(spatial.loc['2024-12', 'observed'] * 100, 1)}% в декабре; 2 004 МО.",
    )
    deck.figure(market_access(), 30, 186, 1200, 402)
    deck.paragraph(
        54,
        590,
        "Точка: медиана; полоса: 25-75% процентили. Номера групп локальны для месяца. "
        "Региональный перестановочный контроль: "
        f"{russian(spatial.loc['2024-06', 'null_mean'] * 100, 1)}% и "
        f"{russian(spatial.loc['2024-12', 'null_mean'] * 100, 1)}%; "
        "p с поправкой Holm = 0,004 в обоих месяцах. "
        "Индекс за 2024 год; контроль сохраняет региональный состав.",
        1160,
        18,
        leading=24,
    )
    deck.footer(
        "Источники: data/derived/reviews/spatial_external_v1/group_summary.csv; "
        "spatial_checks.csv; docs/reviews/spatial_validation.md"
    )

    # 12: concrete applications of profile comparison and economic interpretation.
    deck.start(
        "Применение",
        "Как применять результат",
        "Сопоставляйте потребление с экономикой территории и её положением в сети сходства.",
    )
    rows = [
        (
            208,
            "Подбирать территории",
            "Находите муниципалитеты с близкой структурой расходов. Сравнивайте "
            "их население, зарплату и отраслевую занятость.",
        ),
        (
            349,
            "Разбирать различия",
            "Используйте уровень расходов и компоненты профиля. Сеть показывает "
            "похожих соседей, экономические признаки уточняют сравнение.",
        ),
        (
            490,
            "Следить за динамикой",
            "Сравнивайте профиль территории по месяцам. Учитывайте изменение цен "
            "при анализе расходов; сопоставляйте сдвиги с событиями в муниципалитете.",
        ),
    ]
    for top, title, body in rows:
        deck.line(54, top, 1226, top)
        deck.text(54, top + 24, title, 25, TEAL, True)
        deck.paragraph(464, top + 24, body, 740, 23)
    deck.footer(
        "Источники: reports/enriched_analysis/REPORT.md; reports/initial_study/REPORT.md; "
        "submission/METHODOLOGY.md"
    )

    # 13: name the observed relationship, its magnitude and its sample.
    deck.start("Выводы", "Что именно показали данные за 2024 год")
    findings = [
        (
            185,
            "Масштаб МО",
            "Чем крупнее муниципалитет, тем выше компонент общепита: "
            f"ρ = {russian(population_foodservice.spearman, 3)}; после регионального "
            f"центрирования {russian(population_foodservice.region_centered_rank_correlation, 3)}. "
            f"Выборка: {integer(population_foodservice.n)} МО.",
        ),
        (
            292,
            "Зарплата",
            "Выше зарплата организаций, выше уровень расходов: "
            f"ρ = {russian(wage_spending.spearman, 3)}; после регионального центрирования "
            f"{russian(wage_spending.region_centered_rank_correlation, 3)}. "
            f"Выборка: {integer(wage_spending.n)} МО.",
        ),
        (
            399,
            "Экономика и отрасли",
            "Экономические признаки снижают ошибку профиля на "
            f"{russian(profile_gain * 100, 1)}% внутри регионов и на "
            f"{russian(heldout_gain * 100, 1)}% на регионах вне обучения. "
            "Отрасли добавляют 1,38% и 7,34% в тех же режимах. Выборка: 1 998 МО.",
        ),
        (
            506,
            "Доступность рынков",
            "В декабре между группами приходится "
            f"{russian(spatial.loc['2024-12', 'observed'] * 100, 1)}% разброса индекса "
            f"против {russian(spatial.loc['2024-12', 'null_mean'] * 100, 1)}% "
            "регионального контроля; p Holm = 0,004. Выборка: 2 004 МО.",
        ),
    ]
    for top, title, body in findings:
        deck.text(54, top, title, 21, TEAL, True)
        deck.paragraph(355, top, body, 847, 20, leading=26)
        deck.line(54, top + 92, 1226, top + 92)
    deck.text(
        54,
        627,
        "Для сравнения МО учитывайте масштаб, зарплату, отрасли и доступность рынков.",
        21,
        TEAL,
        True,
    )
    deck.footer(
        "Источники: enriched_analysis/core_v2/correlations.csv; core_v2/cv_comparisons.csv; "
        "round2_analysis/REPORT.md; docs/reviews/spatial_validation.md"
    )

    # 14: proposed hypotheses have explicit predictions and a next test.
    deck.start(
        "Гипотезы",
        "Какие соотношения проверить следующим шагом",
        "Для каждой гипотезы заданы ожидаемая связь и отдельная проверка.",
    )
    for left, label in [(54, "Гипотеза"), (335, "Ожидаемая связь"), (826, "Как проверить")]:
        deck.text(left, 176, label, 18, MUTED, True)
    hypotheses = [
        (
            215,
            "01  Масштаб МО",
            "При сопоставимой зарплате в крупных МО выше CLR-компонент общепита.",
            "Регрессия с зарплатой, работниками на душу населения, регионом и типом МО. "
            "Коэффициент населения > 0.",
        ),
        (
            324,
            "02  Зарплата",
            "При сопоставимом населении выше зарплата организаций, выше расходы после "
            "поправки на цены.",
            "Региональный ИПЦ; регрессия log расходов с населением и работниками "
            "на душу населения, "
            "регионом и типом МО. Коэффициент зарплаты > 0.",
        ),
        (
            433,
            "03  Обработка",
            "При близких населении и зарплате выше доля учтённых работников обработки: "
            "выше CLR-компонент маркетплейсов, ниже транспорта.",
            "Пары МО внутри регионов с близкими базовыми признаками. "
            "Разницы CLR-компонентов: маркетплейсы > 0, транспорт < 0.",
        ),
        (
            542,
            "04  Между регионами",
            "Экономически похожие МО из разных регионов имеют более близкие "
            "потребительские профили.",
            "Подбор пар по экономике. Расстояние Хеллингера ниже контроля "
            "с перестановкой профилей внутри регионов при фиксированных парах.",
        ),
    ]
    for top, title, prediction, test in hypotheses:
        deck.line(54, top - 12, 1226, top - 12)
        deck.text(54, top, title, 18, TEAL, True)
        deck.paragraph(335, top, prediction, 458, 19, leading=25)
        deck.paragraph(826, top, test, 400, 17, leading=22)
    deck.footer(
        "Гипотезы следующего этапа. Протокол: фиксированные выборки; региональные блоки; "
        "поправка Holm, α = 0,05. Подробности: README.md"
    )
    deck.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission/PRESENTATION.pdf")
    args = parser.parse_args()
    build(args.output.resolve())
    print(f"Built {TOTAL_SLIDES} slides: {args.output.resolve()}")


if __name__ == "__main__":
    main()
