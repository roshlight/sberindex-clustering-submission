"""Export source-backed static research figures for the enrichment study."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter

BLUE, GOLD, INK, GREY = "#2364AA", "#BB7922", "#243246", "#CBD3DC"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.grid": False,
        "savefig.facecolor": "white",
    }
)


def header(fig, title, subtitle):
    fig.text(0.08, 0.958, title, fontsize=18, weight="bold", va="top")
    fig.text(0.08, 0.903, subtitle, fontsize=10.3, color="#556579", va="top")


def save(fig, out, name, footnote):
    fig.text(0.08, 0.035, footnote, fontsize=9, color="#556579", va="bottom")
    fig.savefig(out / (name + ".png"), dpi=180)
    fig.savefig(out / (name + ".svg"))
    plt.close(fig)


def make(root):
    out = root / "figures"
    out.mkdir(exist_ok=True)
    panel = pd.read_csv(root / "core_v2/annual_panel.csv")
    prices = pd.read_csv(root / "prices/municipal_price_growth.csv")
    metrics = pd.read_csv(root / "core_v2/cv_metrics.csv")
    coefs = pd.read_csv(root / "core_v2/regression_coefficients.csv")
    contracts = []

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.18, top=0.79)
    bins = np.arange(-16, 35, 1.5)
    for col, label, color, style in [
        ("nominal_growth_pct", "Номинально", GOLD, "--"),
        ("real_growth_pct", "После регионального ИПЦ", BLUE, "-"),
    ]:
        ax.hist(
            prices[col],
            bins=bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
            linestyle=style,
        )
        median = prices[col].median()
        ax.axvline(median, color=color, linestyle=style, alpha=0.8)
        ax.text(
            median + 0.5,
            290 if col.startswith("nominal") else 320,
            f"медиана {median:.2f}%".replace(".", ","),
            color=color,
            fontsize=11,
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set(
        xlabel="Изменение годового среднего месячной оценки расходов, %",
        ylabel="Количество МО",
        ylim=(0, 370),
    )
    ax.legend(frameon=False, loc="upper right")
    header(
        fig,
        "Распределение роста расходов: 2024 к 2023",
        "2016 МО · равный вес каждой территории · "
        "каждый месяц скорректирован своим региональным ИПЦ",
    )
    save(
        fig,
        out,
        "01_price_growth",
        "Источники: СберИндекс и Росстат. "
        "Поправка на инфляцию не измеряет межрегиональный уровень жизни.",
    )
    contracts.append(
        {
            "figure": "01_price_growth",
            "family": "histogram",
            "n": 2016,
            "question": "How much does inflation adjustment change growth?",
            "source": "prices/municipal_price_growth.csv",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.75, wspace=0.22)
    for ax, target, title in zip(
        axes,
        ["log_total", "profile_clr"],
        ["Уровень расходов (log)", "Потребительский профиль (CLR)"],
        strict=True,
    ):
        for j, regime in enumerate(["within_region", "heldout_regions"]):
            d = metrics.loc[
                (metrics.year == 2024)
                & (metrics["sample"] == "core")
                & (metrics.regime == regime)
                & (metrics.target == target)
            ].set_index("model")
            a, b = d.loc["context", "r2_oof"], d.loc["economic", "r2_oof"]
            ax.plot([a, b], [j, j], color=GREY, linewidth=3, zorder=1)
            ax.scatter(
                a,
                j,
                color="white",
                edgecolor=GOLD,
                s=95,
                linewidth=2,
                label="Контекст" if j == 0 else None,
                zorder=2,
            )
            ax.scatter(
                b,
                j,
                color=BLUE,
                s=85,
                label="+ экономические признаки" if j == 0 else None,
                zorder=2,
            )
            ax.text(a, j - 0.16, f"{a:.3f}", ha="center", fontsize=11, color=GOLD)
            ax.text(b, j + 0.26, f"{b:.3f}", ha="center", fontsize=11, color=BLUE)
        ax.set(
            xlim=(0, 1),
            ylim=(-0.45, 1.6),
            yticks=[0, 1],
            xlabel="R² по проверочным предсказаниям",
            title=title,
        )
        ax.set_yticklabels(
            ["МО внутри\nзнакомых регионов", "Полностью\nисключённые регионы"]
            if ax is axes[0]
            else ["", ""]
        )
        ax.grid(axis="x", color="#ECF0F4")
        ax.invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.1),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    header(
        fig,
        "Дополнительная информация в экономических признаках",
        "2024 · 1998 МО · 5 частей проверки · население, зарплата и концентрация рабочих мест",
    )
    save(
        fig,
        out,
        "02_cv_quality",
        "Контекст: тип МО; для проверки внутри регионов — также регион.\n"
        "Исследовательская проверка, не прогноз будущего.",
    )
    contracts.append(
        {
            "figure": "02_cv_quality",
            "family": "paired_dot",
            "n": 1998,
            "question": "Do economic attributes add information under both CV regimes?",
            "source": "core_v2/cv_metrics.csv",
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=0.29, right=0.94, bottom=0.21, top=0.79)
    features = ["log_population", "log_wage", "log_worker_concentration"]
    for year, color, offset, marker in [("2023", GOLD, -0.1, "s"), ("2024", BLUE, 0.1, "o")]:
        rows = (
            coefs.loc[(coefs.year.astype(str) == year) & (coefs["sample"] == "core")]
            .set_index("variable")
            .loc[features]
        )
        y = np.arange(3) + offset
        ax.errorbar(
            rows.coefficient,
            y,
            xerr=[rows.coefficient - rows.ci_low, rows.ci_high - rows.coefficient],
            color=color,
            fmt=marker,
            capsize=4,
            markersize=6,
            label=year,
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set(
        yticks=range(3),
        yticklabels=[
            "Население",
            "Средняя зарплата\nохваченных организаций",
            "Работники организаций\nна одного жителя",
        ],
        xlabel="Коэффициент при логарифме признака",
        xlim=(-0.015, 0.42),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ECF0F4")
    ax.legend(frameon=False, loc="lower right")
    header(
        fig,
        "Условные связи уровня расходов с характеристиками МО",
        "2023: 1975 МО; 2024: 1998 МО · регион и тип МО учтены · "
        "95% интервалы с группировкой ошибок по региону",
    )
    save(
        fig,
        out,
        "03_coefficients",
        "Все три признака входят в одну модель. Коэффициенты описывают ассоциации;\n"
        "причинная интерпретация не установлена.",
    )
    contracts.append(
        {
            "figure": "03_coefficients",
            "family": "coefficient_interval",
            "n": [1975, 1998],
            "question": "Which conditional associations repeat in both years?",
            "source": "core_v2/regression_coefficients.csv",
        }
    )

    core = panel.loc[(panel.year == 2024) & panel.core_eligible]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3))
    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.2, top=0.77, wspace=0.4)
    for ax, x, y, xlabel, ylabel in [
        (
            axes[0],
            core.wage / 1000,
            core.total_mean / 1000,
            "Зарплата, тыс. ₽ в месяц (лог. шкала)",
            "Оценка расходов, тыс. ₽ в месяц",
        ),
        (
            axes[1],
            core.population_total / 1000,
            core.p_foodservice * 100,
            "Население, тыс. человек (лог. шкала)",
            "Общепит в нормированном профиле, %",
        ),
    ]:
        h = ax.hexbin(x, y, gridsize=35, mincnt=1, cmap="Blues", xscale="log", linewidths=0.15)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.set_minor_formatter(NullFormatter())
        fig.colorbar(h, ax=ax, shrink=0.7, label="МО в ячейке")
    axes[0].set_xticks([30, 50, 100, 200], labels=["30", "50", "100", "200"])
    axes[1].set_xticks([5, 20, 100, 500, 2000], labels=["5", "20", "100", "500", "2000"])
    header(
        fig,
        "Непрерывные различия между муниципалитетами",
        "2024 · 1998 МО · зарплата организаций, население на 1 января и средний профиль 12 месяцев",
    )
    save(
        fig,
        out,
        "04_continuous_gradients",
        "Наблюдаемые связи без причинной трактовки. "
        "Процент профиля не равен доказанной доле бюджета домохозяйства.",
    )
    contracts.append(
        {
            "figure": "04_continuous_gradients",
            "family": "hexbin_scatter",
            "n": 1998,
            "question": "Are differences continuous rather than necessarily discrete?",
            "source": "core_v2/annual_panel.csv",
        }
    )

    correlations = pd.read_csv(root / "core_v2/correlations.csv")
    outcomes = ["log_total", "p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
    features = ["log_population", "log_wage", "log_worker_concentration", "log_market_access"]
    cmap = LinearSegmentedColormap.from_list("signed", [GOLD, "#FFFFFF", BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7))
    fig.subplots_adjust(left=0.2, right=0.88, bottom=0.24, top=0.77, wspace=0.3)
    for ax, col, title in zip(
        axes,
        ["spearman", "region_centered_rank_correlation"],
        ["Общая ранговая связь", "Ранги с вычетом среднего по региону"],
        strict=True,
    ):
        d = (
            correlations[correlations.year == 2024]
            .pivot(index="outcome", columns="feature", values=col)
            .loc[outcomes, features]
        )
        im = ax.imshow(d, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set(xticks=range(4), yticks=range(6), title=title)
        ax.set_xticklabels(
            ["Население", "Зарплата", "Рабочие места\nна жителя", "Доступность\nрынков"],
            rotation=35,
            ha="right",
            fontsize=10,
        )
        ax.set_yticklabels(
            [
                "Уровень расходов",
                "Продовольствие",
                "Здоровье",
                "Маркетплейсы",
                "Общепит",
                "Транспорт",
            ]
            if ax is axes[0]
            else [""] * 6
        )
        for i in range(6):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{d.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if abs(d.iloc[i, j]) > 0.67 else INK,
                )
    bar = fig.add_axes([0.92, 0.27, 0.015, 0.46])
    fig.colorbar(im, cax=bar)
    header(
        fig,
        "Региональная структура меняет наблюдаемые связи",
        "2024 · попарно доступные данные, 1998–2013 МО · коэффициенты от −1 до +1",
    )
    save(
        fig,
        out,
        "05_rank_correlations",
        "Справа — корреляция глобальных рангов после регионального центрирования;\n"
        "это не условная причинная оценка.",
    )
    contracts.append(
        {
            "figure": "05_rank_correlations",
            "family": "heatmap",
            "n": [1998, 2013],
            "question": "Do regional differences confound simple associations?",
            "source": "core_v2/correlations.csv",
        }
    )

    temporal = pd.read_csv(root / "annual_networks/temporal_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.76, wspace=0.35)
    for ax, metric, title, factor, xmax in [
        (axes[0], "ari", "Сходство разбиений между годами (ARI)", 1, 1),
        (axes[1], "switch_share", "Сменили группу после сопоставления, %", 100, 65),
    ]:
        for j, method in enumerate(["leiden_hellinger", "kmeans"]):
            rows = temporal[temporal.method == method].set_index("profile_basis")
            for basis, color, offset, label in [
                ("original_december", GOLD, -0.15, "Декабрь каждого года"),
                ("annual_mean_12months", BLUE, 0.15, "Среднее 12 месяцев"),
            ]:
                val = rows.loc[basis, metric] * factor
                ax.barh(j + offset, val, height=0.27, color=color, label=label if j == 0 else None)
                text_value = f"{val:.3f}" if metric == "ari" else f"{val:.1f}%"
                ax.text(val + xmax * 0.02, j + offset, text_value, va="center", fontsize=11)
        ax.set(yticks=[0, 1], yticklabels=["Leiden", "K-means\nK = 8"], xlim=(0, xmax), title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#ECF0F4")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.1), ncol=2, frameon=False
    )
    header(
        fig,
        "Годовое усреднение не укрепило сетевую типологию",
        "2023 → 2024 · 2016 МО · Leiden: 11 → 9 групп на годовых профилях",
    )
    save(
        fig,
        out,
        "06_annual_stability",
        "Годовой Leiden: 0 из 2016 назначений достигли порога поддержки "
        "17 из 21 альтернативы в каждом году.\n"
        "K-means: проверена временная согласованность при K = 8; "
        "полная устойчивость не оценивалась.",
    )
    contracts.append(
        {
            "figure": "06_annual_stability",
            "family": "grouped_bar",
            "n": 2016,
            "question": "Does annual averaging strengthen the typology?",
            "source": "annual_networks/temporal_comparison.csv",
        }
    )

    (out / "chart_contracts.json").write_text(
        json.dumps(
            {
                "surface": "standalone_static_research_figures",
                "palette": {"blue": BLUE, "gold": GOLD, "neutral": INK},
                "charts": contracts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"figures": len(contracts), "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/enriched_analysis"))
    make(parser.parse_args().report)
