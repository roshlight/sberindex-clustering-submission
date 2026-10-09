"""Standalone research figures backed only by frozen numerical result tables."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

LABELS = {
    "A": "Сельское хозяйство",
    "B": "Добыча",
    "C": "Обработка",
    "D": "Энергоснабжение",
    "H": "Транспорт и хранение",
    "O": "Госуправление",
    "P": "Образование",
    "Q": "Здравоохранение",
}
P = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
BLUE, TEAL, INK = "#4056A1", "#397A79", "#273442"


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(root):
    out = root / "figures"
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
        }
    )
    features = pd.read_csv(root / "sector/feature_panel.csv")
    coverage = pd.DataFrame(
        [
            {
                "section": s,
                "label": label,
                "n": int(features[f"sector_{s}_share"].notna().sum()),
                "denominator": len(features),
            }
            for s, label in LABELS.items()
        ]
    )
    coverage["coverage_pct"] = 100 * coverage.n / coverage.denominator
    coverage = coverage.sort_values("coverage_pct", ascending=True)
    coverage.to_csv(out / "coverage_chart.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(coverage.label, coverage.coverage_pct, color=BLUE, height=0.6)
    for i, row in enumerate(coverage.itertuples()):
        ax.text(row.coverage_pct + 1.2, i, f"{row.coverage_pct:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, 111)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("МО-годы с числовым отраслевым показателем, %")
    ax.set_title("Раскрытие восьми отраслевых показателей", loc="left", fontsize=16, pad=33)
    ax.text(
        0,
        1.035,
        "4032 МО-года • 2023 и 2024 • пропуски не равны нулю",
        transform=ax.transAxes,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out, "sector_coverage")

    c = pd.read_csv(root / "models/cv_comparisons.csv")
    c = c.loc[
        (c.baseline == "economic_missingness")
        & (c.alternative == "economic_sectors")
        & (c.target == "profile_clr")
        & (c.seed == 13)
    ].copy()
    samples = {
        "core": "Основная",
        "stable_code": "Без смены кода/типа",
        "complete_sectors": "Все 8 отраслей раскрыты",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True)
    for ax, regime, title in zip(
        axes,
        ["within_region", "heldout_regions"],
        ["Муниципалитеты внутри регионов", "Целые регионы вне обучения"],
        strict=True,
    ):
        d = c.loc[c.regime == regime].copy()
        d["order"] = d["sample"].map({"core": 0, "stable_code": 1, "complete_sectors": 2})
        d = d.sort_values(["year", "order"], ascending=[False, True])
        for i, row in enumerate(d.itertuples()):
            value = 100 * row.relative_mse_reduction
            color = BLUE if row.year == 2024 else TEAL
            ax.errorbar(
                value,
                i,
                xerr=np.array(
                    [
                        [100 * (row.relative_mse_reduction - row.ci_low)],
                        [100 * (row.ci_high - row.relative_mse_reduction)],
                    ]
                ),
                fmt="o" if row.year == 2024 else "s",
                color=color,
                capsize=4,
                markersize=6,
            )
        ax.set_yticks(
            range(len(d)),
            [f"{r.year} · {samples[r.sample]}\nn = {r.n}" for r in d.itertuples()],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.axvline(0, color=INK, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=12, pad=14)
        ax.set_xlabel("Снижение ошибки профиля, %")
        ax.grid(axis="x", color="#E3E5E9", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Дополнительная информация отраслевых чисел", x=0.015, ha="left", fontsize=18)
    fig.text(
        0.015,
        0.915,
        "База: экономика + признаки пропусков. 95% интервалы: "
        "перевыборка регионов при фиксированных OOF-прогнозах.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=3)
    save(fig, out, "sector_increment")

    correlations = pd.read_csv(root / "models/partial_correlations.csv")
    d = correlations.loc[(correlations.year == 2024) & correlations.target.ne("log_total")]
    values = d.pivot(index="sector", columns="target", values="partial_pearson").reindex(
        index=[f"sector_{s}_share" for s in LABELS], columns=[f"clr_{p}" for p in P]
    )
    cmap = LinearSegmentedColormap.from_list("industry", [BLUE, "#FFFFFF", TEAL])
    fig, ax = plt.subplots(figsize=(10, 6.3))
    vmax = max(0.3, float(abs(values).max().max()))
    image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(5), ["Продукты", "Здоровье", "Маркетплейсы", "Общепит", "Транспорт"])
    ns = d.groupby("sector").n.first()
    ax.set_yticks(
        range(8), [f"{label} (n={ns[f'sector_{s}_share']})" for s, label in LABELS.items()]
    )
    for i in range(8):
        for j in range(5):
            v = values.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(v) > 0.65 * vmax else INK,
                fontsize=12,
            )
    ax.set_title(
        "Частные связи отраслей с потребительским профилем", loc="left", fontsize=16, pad=37
    )
    ax.text(
        0,
        1.035,
        "2024 • контроль региона, типа МО и экономики • только наблюдаемые отрасли",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.colorbar(image, ax=ax, shrink=0.75, label="Частная корреляция Pearson")
    fig.text(
        0.015,
        0.01,
        "Разные строки используют разные подвыборки. Связи описательные; "
        "причинность и полная структура занятости не установлены.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, out, "sector_associations_2024")
    print("Created 3 PNG/SVG figures and source coverage table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/round2_analysis"
    )
    run(parser.parse_args().root)
