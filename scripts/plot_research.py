"""Export source-backed research figures; never fit or tune a model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

METHODS = {
    "leiden_hellinger": ("Leiden · Хеллингер", "#167c80", "-"),
    "leiden_cosine": ("Leiden · косинус", "#ca7847", "--"),
    "kmeans": ("KMeans · K=8", "#635aa7", ":"),
    "kmeans_matched_k": ("KMeans · K как у Leiden", "#536878", "-."),
}
YEARS = {"2023": "#167c80", "2024": "#ca7847"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    plt.close(fig)


def factor_rows(frame):
    core = frame.loc[frame.family.eq("core_grid")].copy()
    common = core.k.eq(20) & core.resolution.eq(1)
    definitions = {
        "Случайный старт": common & core.geometry.eq("hellinger") & core.seed.ne(13),
        "Мера сходства": common & core.geometry.eq("cosine") & core.seed.eq(13),
        "Число соседей": core.geometry.eq("hellinger")
        & core.resolution.eq(1)
        & core.seed.eq(13)
        & core.k.ne(20),
        "Разрешение": core.geometry.eq("hellinger")
        & core.k.eq(20)
        & core.seed.eq(13)
        & core.resolution.ne(1),
    }
    return pd.concat([core.loc[mask].assign(factor=name) for name, mask in definitions.items()])


def build(runs, report):
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.labelcolor": "#26343c",
            "text.color": "#26343c",
            "svg.fonttype": "none",
        }
    )

    def read(name):
        return pd.concat([pd.read_csv(run / f"{name}.csv") for run in runs], ignore_index=True)

    metrics = read("metrics")
    robust = read("robustness")
    membership = read("membership_agreement")
    for frame in (metrics, robust, membership):
        frame["year"] = frame.month.str[:4]
    months = sorted(metrics.month.unique())
    dates = pd.to_datetime(months)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, layout="constrained")
    for method, (label, color, style) in METHODS.items():
        rows = metrics.loc[metrics.method.eq(method)].set_index("month").loc[months]
        for ax, column in zip(axes, ("n_clusters", "silhouette"), strict=True):
            ax.plot(dates, rows[column], color=color, linestyle=style, label=label, linewidth=1.8)
    axes[0].set(title="Число групп меняется от месяца к месяцу", ylabel="Число групп K")
    axes[1].set(title="Компактность групп в общем пространстве √P", ylabel="Silhouette")
    axes[1].set_ylim(0, max(0.4, metrics.silhouette.max() * 1.15))
    for ax in axes:
        ax.axvline(pd.Timestamp("2024-01-01"), color="#a0a0a0", linewidth=1)
        ax.grid(axis="y", alpha=0.16)
    axes[0].legend(ncol=2, loc="upper left", fontsize=9)
    fig.suptitle(
        "2016 МО · разработка: 2023 · проверка зафиксированной процедуры: 2024", fontsize=12
    )
    save(fig, output, "monthly_structure")

    factors = factor_rows(robust)
    factors.to_csv(report / "single_factor_comparisons.csv", index=False)
    names = list(factors.factor.unique())
    fig, ax = plt.subplots(figsize=(10.5, 5.7), layout="constrained")
    for year, color in YEARS.items():
        offset = -0.17 if year == "2023" else 0.17
        values = [
            factors.loc[factors.year.eq(year) & factors.factor.eq(name), "ari_to_primary"]
            for name in names
        ]
        box = ax.boxplot(
            values,
            positions=np.arange(len(names)) + offset,
            widths=0.27,
            orientation="horizontal",
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=0.5)
        ax.scatter([], [], color=color, label=year)
    ax.set(
        yticks=np.arange(len(names)),
        yticklabels=names,
        xlim=(0, 1),
        xlabel="ARI к основному разбиению месяца · выше — больше согласие",
        title="Даже изменение одного решения меняет состав групп",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16)
    ax.legend(loc="lower right")
    fig.supxlabel(
        "Ящик: квартиль–медиана–квартиль; усы: минимум и максимум. Это не доверительные интервалы.",
        fontsize=9,
    )
    save(fig, output, "single_factor_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained", width_ratios=[1.6, 1])
    yearly_support = []
    for year, color in YEARS.items():
        rows = membership.loc[membership.year.eq(year)]
        values = np.sort(rows.mean_co_membership_jaccard.to_numpy())
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=year, color=color)
        count = int(rows.agreement.ge(0.8).sum())
        yearly_support.append(
            {
                "year": year,
                "strong_count": count,
                "denominator": len(rows),
                "strong_share": count / len(rows),
                "median_mean_co_membership_jaccard": float(np.median(values)),
            }
        )
        bar = axes[1].bar(year, 100 * count / len(rows), color=color, width=0.6)
        axes[1].bar_label(
            bar, labels=[f"{100 * count / len(rows):.3f}%\n{count} / {len(rows)}"], padding=5
        )
    axes[0].set(
        xlabel="Средний Jaccard состава группы по 21 альтернативе",
        ylabel="Доля наблюдений МО × месяц",
        title="Сохранение состава группы",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    axes[1].set(title="Прошли строгий фильтр", ylabel="% наблюдений МО × месяц")
    axes[1].set_ylim(0, max(0.5, max(row["strong_share"] for row in yearly_support) * 145))
    fig.supxlabel(
        "Фильтр: Jaccard ≥ 0,8 как минимум в 17 из 21 альтернатив.\n"
        "Порог задан до расчёта; это не вероятность истинности.",
        fontsize=9,
    )
    save(fig, output, "membership_support")
    (report / "yearly_support.json").write_text(
        json.dumps(yearly_support, indent=2), encoding="utf-8"
    )

    temporal = pd.read_csv(report / "temporal_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7), sharex=True, layout="constrained")
    for method, (label, color, style) in METHODS.items():
        rows = temporal.loc[temporal.method.eq(method) & temporal.lag.eq(1)].sort_values("to_month")
        x = pd.to_datetime(rows.to_month)
        axes[0].plot(x, rows.ari, label=label, color=color, linestyle=style)
        axes[1].plot(x, rows.switch_share, label=label, color=color, linestyle=style)
        if method == "leiden_hellinger":
            axes[1].plot(
                x,
                rows.robust_switch_share_of_all,
                color="black",
                linewidth=2.4,
                label="Подтверждённые переходы · основной Leiden",
            )
    axes[0].set(title="Согласие разбиений соседних месяцев", ylabel="ARI", ylim=(0, 1))
    axes[1].set(
        title="Смена метки после парного сопоставления групп",
        ylabel="Доля общих МО",
        ylim=(-0.015, 0.65),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(ncol=2, fontsize=9)
    # Explicitly use the black line handle; artist ordering can vary with method order.
    black = next(line for line in axes[1].lines if line.get_color() == "black")
    axes[1].legend(handles=[black], loc="upper left", fontsize=9)
    for ax in axes:
        ax.axvline(pd.Timestamp("2024-01-01"), color="#a0a0a0", linewidth=1)
        ax.grid(axis="y", alpha=0.16)
    fig.supxlabel(
        "23 соседние пары, включая декабрь 2023 → январь 2024; "
        "знаменатель каждой доли — 2016 общих МО.",
        fontsize=9,
    )
    save(fig, output, "monthly_transitions")
    print(json.dumps({"figures": 4, "formats": ["png", "svg"], "yearly_support": yearly_support}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    build(args.runs, args.report)
