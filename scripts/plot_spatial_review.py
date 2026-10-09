"""Export the two comparisons requested by the spatial brief from reviewed CSVs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from review_spatial_transitions import sha


def plot(source, output):
    if output.exists():
        raise FileExistsError("Use a new figure directory")
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ["road_cdf.csv", "group_summary.csv", "spatial_checks.csv"]:
        if sha(source / name) != manifest["output_hashes"][name]:
            raise ValueError(f"Changed numerical source: {name}")
    cdf = pd.read_csv(source / "road_cdf.csv")
    groups = pd.read_csv(source / "group_summary.csv")
    months = manifest["config"]["months"]
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "spatial-review-v1",
        }
    )
    figures = []
    fig, axes = plt.subplots(1, len(months), figsize=(12, 4.8), sharey=True)
    for ax, month in zip(np.atleast_1d(axes), months, strict=True):
        d = cdf.loc[cdf["sample"].eq("all") & cdf.month.eq(month)].sort_values("threshold_km")
        ax.fill_between(
            d.threshold_km,
            100 * d.null_q025_cdf,
            100 * d.null_q975_cdf,
            color="#cbd5e1",
            label="2,5–97,5% перестановок",
        )
        ax.plot(
            d.threshold_km,
            100 * d.null_mean_cdf,
            "--",
            color="#475569",
            label="Среднее перестановок",
        )
        ax.plot(
            d.threshold_km, 100 * d.observed_cdf, "o-", color="#1d4ed8", label="Наблюдаемые пары"
        )
        ax.set_xscale("log")
        ax.set_xticks(
            [100, 500, 2000, 8000, 32000], labels=["100", "500", "2 000", "8 000", "32 000"]
        )
        ax.set_ylim(0, 102)
        ax.set_title(month)
        ax.set_xlabel("Автодорожное расстояние, км (лог. шкала)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Доля пар одной группы не дальше порога, %")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("H1 · Дорожная близость внутри потребительских групп", fontsize=14)
    fig.text(
        0.04,
        0.025,
        "Равный вес доступных неупорядоченных пар. "
        "Полоса — случайный ориентир, не доверительный интервал.\n"
        "Расстояния на 31.12.2024; замороженные группы Leiden; перестановки меток внутри регионов.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    figures.append(("road_comparison", fig))
    fig, axes = plt.subplots(1, len(months), figsize=(12, 5.8), sharex=True)
    for ax, month in zip(np.atleast_1d(axes), months, strict=True):
        d = groups.loc[
            groups["sample"].eq("all") & groups.month.eq(month) & groups.hypothesis.eq("H2")
        ].sort_values("label")
        y = np.arange(len(d))
        ax.hlines(y, d.q25, d.q75, color="#93c5fd", linewidth=5)
        ax.scatter(d["median"], y, color="#1d4ed8", s=26, zorder=3)
        ax.set_yticks(
            y,
            labels=[
                f"{int(r.label)}  ({int(r.n_observed)}/{int(r.n_total)})" for r in d.itertuples()
            ],
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 1000)
        ax.set_title(month)
        ax.set_xlabel("Индекс доступности рынков за 2024 год")
        ax.set_ylabel("Группа (наблюдаемые / все МО)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("H2 · Доступность рынков по потребительским группам", fontsize=14)
    fig.text(
        0.04,
        0.025,
        "Точка — медиана; отрезок — 25–75-й процентили; равный вес МО. Пропуски не заполнены.\n"
        "Номера групп локальны для месяца. Контроль регионального состава приведён в таблице H2.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    figures.append(("market_comparison", fig))
    hashes = {}
    for name, fig in figures:
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=160, metadata={"Date": None} if suffix == "svg" else {})
            hashes[path.name] = sha(path)
        plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "processor_sha256": sha(__file__),
                "matplotlib": matplotlib.__version__,
                "source_hashes": {
                    n: manifest["output_hashes"][n]
                    for n in ["road_cdf.csv", "group_summary.csv", "spatial_checks.csv"]
                },
                "output_hashes": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
