"""Export the primary research comparison with conditional region-bootstrap intervals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from round2_model_analysis import sha


def plot(root, output):
    output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(root / "models/cv_comparisons.csv")
    selected = data.loc[
        data.comparison.eq("primary")
        & data.target.eq("profile_clr")
        & data.seed.eq(13)
        & data["sample"].isin(["core", "stable_code"])
    ].copy()
    selected.to_csv(output / "primary_intervals.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    colors = {"core": "#2864A8", "stable_code": "#92733A"}
    for ax, regime, title in zip(
        axes,
        ["within_region", "heldout_regions"],
        ["Внутри регионов", "Проверка на целых регионах"],
        strict=True,
    ):
        for sample, offset, marker, label in [
            ("core", -0.12, "o", "Основная выборка"),
            ("stable_code", 0.12, "s", "Без смены кода / типа"),
        ]:
            part = selected.loc[
                selected.regime.eq(regime) & selected["sample"].eq(sample)
            ].sort_values("year")
            values = 100 * part.relative_mse_reduction.to_numpy()
            intervals = np.vstack(
                [
                    values - 100 * part.ci_low.to_numpy(),
                    100 * part.ci_high.to_numpy() - values,
                ]
            )
            ax.errorbar(
                values,
                np.arange(2) + offset,
                xerr=intervals,
                fmt=marker,
                color=colors[sample],
                mfc="white" if sample == "stable_code" else colors[sample],
                capsize=4,
                lw=1.6,
                ms=7,
                label=label,
            )
        ax.axvline(0, color="#333333", lw=1)
        ax.axvline(1, color="#777777", lw=1, ls=":")
        ax.set(
            title=title,
            xlabel="Снижение ошибки профиля, %",
            xlim=(-3.0, 2.2),
            yticks=[0, 1],
            yticklabels=["2023", "2024"],
            ylim=(1.5, -0.5),
        )
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=10)
    fig.suptitle("Добавочная информация восстановленного денежного блока", y=0.97, fontsize=16)
    fig.text(
        0.5,
        0.87,
        "Значения сверх экономической базы, отраслей и актуальных флагов пропусков",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.05,
        0.035,
        "Точки: фиксированные OOF-оценки; линии: условные 95% интервалы, "
        "2000 перевыборок регионов.\n"
        "Основная выборка: 1975 / 1998 МО; без смены кода: 1897 / 1930. Пунктир: порог 1%.\n"
        "Источник: models/cv_comparisons.csv, протокол round4-v1. "
        "Причинная интерпретация не заявляется.",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.08, right=0.97, wspace=0.1)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"profile_increment.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    contract = {
        "question": "Does restored monetary information improve the primary profile target?",
        "family": "comparison",
        "variant": "point estimates with 95% interval whiskers",
        "rows": len(selected),
        "unit": "percent reduction of MSE; positive is better",
        "palette": "hard two-root cap: blue and gold plus neutrals",
        "noncolor": "circle vs open square; labeled cohort legend",
        "cohorts": "Frozen core and stable-code, two years and two split regimes",
        "limitations": "Intervals condition on fixed OOF predictions; no full-pipeline refit",
        "visual_review": "See QA.json for inspection of the exported PNG and its exact SHA256",
    }
    (output / "chart_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    manifest = {
        "input_sha256": sha(root / "models/cv_comparisons.csv"),
        "code_sha256": sha(__file__),
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.out)
