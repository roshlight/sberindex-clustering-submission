"""Run the fixed 2024 hypothesis protocol and the common five-test Holm family.

Read the saved annual economic/profile/CPI inputs; do not refit clustering.
Each component writes its sample, numerical diagnostics and resampling audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from municipal_profiles.hypothesis_matching import run_matching_tests
from municipal_profiles.hypothesis_regressions import run_regression_tests
from municipal_profiles.hypothesis_summary import TEST_IDS, summarize_tests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/hypothesis_checks_v1.json"
BASE_COMMIT = "f60455a6d82428eb519776540c6fbd32f890ba5d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, precision=4):
    return f"{value:.{precision}f}".replace(".", ",")


def validate_config(config):
    """Reject changed specifications rather than silently ignoring config fields."""
    if (
        config["protocol_version"] != "profile-hypotheses-v1"
        or config["family"] != list(TEST_IDS)
        or config["year"] != 2024
        or config["seed"] != 13
        or config["n_resamples"] != 9999
        or config["alpha"] != 0.05
        or config["regressions"]["numeric"]
        != ["log_population", "log_wage", "log_worker_concentration"]
        or config["regressions"]["fixed_effects"] != ["region_code", "municipal_district_type"]
        or config["H3"]["log_covariate_caliper_sd"] != 0.5
        or config["H3"]["minimum_sector_share_gap"] != 0
        or config["H4"]["log_covariate_caliper_sd"] != 0.5
        or config["H4"]["sector_rms_z_caliper"] != 1.0
        or config["H4"]["different_regions"] is not True
    ):
        raise ValueError("This runner implements only the declared profile-hypotheses-v1 protocol")


def write_report(result: pd.DataFrame, output: Path):
    indexed = result.set_index("test_id")
    h1, h2, hm, ht, h4 = (indexed.loc[key] for key in TEST_IDS)
    labels = {
        TEST_IDS[0]: "H1. Население → CLR общепита",
        TEST_IDS[1]: "H2. Зарплата → log скорректированных расходов",
        TEST_IDS[2]: "H3. Высокая доля обработки → CLR маркетплейсов",
        TEST_IDS[3]: "H3. Высокая доля обработки → CLR транспорта",
        TEST_IDS[4]: "H4. Межрегиональные пары → расстояние Хеллингера",
    }
    decisions = {
        "supported": "Поддержано",
        "not_supported": "Критерий не пройден",
        "unavailable": "Проверка недоступна",
    }
    table = [
        "| Проверка | Эффект | МО / регионы / пары | p исходное | p Holm | Итог |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result.itertuples():
        table.append(
            f"| {labels[row.test_id]} | {number(row.estimate, 6)} | "
            f"{row.n_entities} / {row.n_regions} / {row.n_pairs} | "
            f"{number(row.p_raw)} | {number(row.p_holm)} | {decisions[row.decision]} |"
        )
    report = [
        "# Проверка четырёх гипотез: результаты за 2024 год",
        "",
        "Проверки из заключительного слайда выполнены по фиксированному "
        "[протоколу](PROTOCOL.md). Пять направленных тестов образуют одну семью Holm "
        "с порогом 0,05; для H3 должны пройти обе компоненты. Все эффекты и p "
        "сохранены в [results.csv](results.csv).",
        "",
        *table,
        "",
        "## Размер муниципалитета и общепит",
        "",
        f"На {h1.n_entities} МО в {h1.n_regions} регионах коэффициент log населения "
        f"равен {number(h1.estimate, 6)}. При сопоставимых зарплате, концентрации "
        "рабочих мест, регионе и типе МО прирост населения на 10% связан с "
        f"изменением отношения компонента общепита к геометрическому среднему "
        f"пяти компонент на {number(100 * np.expm1(h1.estimate * np.log(1.1)), 2)}%. "
        "Это интерпретация CLR-коэффициента, а не изменение доли бюджета "
        "в процентных пунктах.",
        "",
        "## Зарплата и расходы после корректировки цен",
        "",
        f"На той же выборке коэффициент log зарплаты организаций равен "
        f"{number(h2.estimate, 6)}. Зарплата выше на 10% связана с расходами выше на "
        f"{number(100 * np.expm1(h2.estimate * np.log(1.1)), 2)}% при остальных "
        "фиксированных признаках модели. Каждый месяц расходы разделены на цепочку "
        "регионального ИПЦ до усреднения за год. База — декабрь 2022 своего региона; "
        "эта поправка учитывает инфляцию во времени, а не различия стоимости жизни "
        "между регионами. Диагностика номинальной модели и инвариантности к одному "
        "годовому множителю региона сохранена рядом с регрессиями.",
        "",
        "Основные p H1 и H2 рассчитаны restricted-null wild cluster bootstrap-t: "
        "9999 региональных Rademacher-перевыборок, CR1 studentization. "
        "95% CR1 t-интервалы и коэффициенты всех контролей сохранены "
        "в [regressions/](regressions/).",
        "",
        "## Обрабатывающая промышленность и состав профиля",
        "",
        f"Из {int(hm.n_eligible)} допустимых МО подобрано {hm.n_pairs} пар "
        f"({hm.n_entities} МО, {hm.n_regions} регионов). В каждой паре совпадают "
        "регион и тип; три экономических log-признака отличаются не более чем "
        "на 0,5 SD каждый. Высокая группа имеет долю учтённых работников обработки "
        "строго выше медианы региона × типа. Подбор максимизирует число пар, "
        "затем минимизирует экономическое расстояние и не использует потребление.",
        "",
        f"Средняя разница CLR маркетплейсов (высокая минус низкая группа): "
        f"{number(hm.estimate, 6)}; CLR транспорта: {number(ht.estimate, 6)}. "
        f"Общий критерий H3: "
        f"{'пройден' if hm.hypothesis_supported else 'не пройден'}. "
        "Каждая компонента показана отдельно; положительный знак одного эффекта "
        "не заменяет прохождение обоих тестов. Нулевое распределение меняет общий "
        "знак всех парных разниц региона, одинаково для двух исходов. "
        "Проверка использует симметрию региональных разниц под нулём.",
        "",
        "[Пары H3](matching/pairs_H3.csv) · [Баланс признаков](matching/balance.csv) "
        "· [Региональные sign-flip draws](matching/nulls_H3.csv).",
        "",
        "## Экономически близкие муниципалитеты разных регионов",
        "",
        f"Полные восемь отраслевых долей доступны у {int(h4.n_eligible)} базовых МО. "
        f"Подобрано {h4.n_pairs} непересекающихся межрегиональных пар "
        f"({h4.n_entities} МО, {h4.n_regions} регионов). Помимо caliper "
        "0,5 SD по каждому из трёх экономических логарифмов, RMS отраслевых "
        "стандартизованных разностей ограничен 1,0. Расстояние при подборе "
        "придаёт равный вес базовой экономике и восьми отраслям.",
        "",
        f"Среднее расстояние Хеллингера в парах: {number(h4.observed, 6)}; "
        f"среднее 9999 перестановок: {number(h4.null_mean, 6)}. "
        f"Профили ближе на {number(100 * (1 - h4.observed / h4.null_mean), 2)}% "
        "относительно среднего условного ориентира. Пары фиксированы; целые "
        "пятикомпонентные профили переставляются внутри регионов полной допустимой "
        "выборки. В 20 из 44 регионов есть один МО, и его профиль остаётся "
        "неподвижным. Вывод относится к этой выборке полного отраслевого раскрытия.",
        "",
        "[Пары H4](matching/pairs_H4.csv) · [Перестановочное распределение](matching/nulls_H4.csv) "
        "· [Полнота выборок](matching/coverage.json).",
        "",
        "## Воспроизведение",
        "",
        "Из корня репозитория: `uv run --frozen python scripts/test_profile_hypotheses.py`. "
        "Параметры зафиксированы в `configs/hypothesis_checks_v1.json`; исходные "
        "таблицы, старые назначения и предыдущие проверки не перезаписываются. "
        "Эти условные описательные проверки на ранее просмотренной панели "
        "оценивают связи и сходство. Контроль признаков и подбор пар не устанавливают "
        "причинный эффект населения, зарплаты или отрасли.",
        "",
        "[Машиночитаемый итог](summary.json) · [Хэши и среда](manifest.json).",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; use --output for another reproduction")
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "PROTOCOL.md").write_bytes(
        (root / "reports/hypothesis_checks/PROTOCOL.md").read_bytes()
    )
    settings = {key: config[key] for key in ("year", "seed", "n_resamples")}
    print("H1/H2: fixed 2024 regression protocol", flush=True)
    regressions = run_regression_tests(root, output / "regressions", **settings)
    print("H3/H4: fixed economic matching and regional nulls", flush=True)
    matching = run_matching_tests(root, output / "matching", **settings)
    result = summarize_tests(pd.concat([regressions, matching], ignore_index=True), config["alpha"])
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "results.csv", index=False)
    records = json.loads(result.to_json(orient="records", double_precision=15))
    summary = {
        "protocol_version": config["protocol_version"],
        **settings,
        "alpha": config["alpha"],
        "family_size": len(TEST_IDS),
        "supported_hypotheses": sorted(
            result.loc[result.hypothesis_supported, "hypothesis"].unique()
        ),
        "hypothesis_decisions": result.groupby("hypothesis").hypothesis_supported.first().to_dict(),
        "results": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(result, output)
    sources = [
        CONFIG,
        "reports/hypothesis_checks/PROTOCOL.md",
        "scripts/test_profile_hypotheses.py",
        "src/municipal_profiles/hypothesis_regressions.py",
        "src/municipal_profiles/hypothesis_matching.py",
        "src/municipal_profiles/hypothesis_summary.py",
        "reports/enriched_analysis/core_v2/annual_panel.csv",
        "reports/enriched_analysis/prices/municipal_annual_price_context.csv",
        "reports/enriched_analysis/prices/municipal_monthly_price_context.csv",
        "reports/enriched_analysis/prices/regional_cpi_chain.csv",
        "reports/round2_analysis/models/model_panel.csv",
        "reports/round2_analysis/models/selected_sector_features.csv",
    ]
    manifest = {
        "protocol_base_commit": BASE_COMMIT,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scipy"]},
        "config": config,
        "source_hashes": {path: sha256(root / path) for path in sources},
        "output_hashes": {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path not in {output / "manifest.json", output / "PROTOCOL.md", output / "QA.md"}
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        result[["test_id", "estimate", "p_raw", "p_holm", "hypothesis_supported"]].to_string(
            index=False
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/hypothesis_checks_reproduced")
    args = parser.parse_args()
    run(ROOT, args.output.resolve())


if __name__ == "__main__":
    main()
