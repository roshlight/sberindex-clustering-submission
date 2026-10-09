"""Build one canonical technical report from the frozen fourth-round evidence."""

import argparse
import json
from pathlib import Path

import pandas as pd
from build_round2_report import source, table_md

TITLE = "Что изменили восстановленные данные о муниципалитетах"
ROOT = "reports/round4_analysis"
REGIMES = {"within_region": "Внутри регионов", "heldout_regions": "Целые регионы"}
SAMPLES = {
    "core": "Основная",
    "stable_code": "Без смены кода",
    "common_complete_new3": "Прежние полные 3",
    "expanded_complete_new3": "Новые полные 3",
}


def build(repo, output):
    root = repo / ROOT
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(root / "models/cv_comparisons.csv")
    metric = pd.read_csv(root / "models/cv_metrics.csv")
    coverage = pd.read_csv(root / "audit/joint_coverage.csv")
    profile = comparison.loc[comparison.target.eq("profile_clr")]
    primary = profile.loc[profile.comparison.eq("primary")]
    core = primary.loc[primary["sample"].eq("core") & primary.seed.eq(13)]
    assert len(core) == 4 and (core.ci_low.lt(0) & core.ci_high.gt(0)).all()
    assert not json.loads((root / "readout/QA.json").read_text("utf-8"))["adoption_passed"]

    def rows(frame):
        return [
            {
                "year": str(r.year),
                "sample": SAMPLES[r.sample],
                "regime": REGIMES[r.regime],
                "seed": int(r.seed),
                "n": int(r.n),
                "regions": int(r.regions),
                "gain": round(100 * r.relative_mse_reduction, 3),
                "low": round(100 * r.ci_low, 3),
                "high": round(100 * r.ci_high, 3),
                "ci": f"[{100 * r.ci_low:.3f}; {100 * r.ci_high:.3f}]",
            }
            for r in frame.itertuples()
        ]

    main = rows(core)
    for row in main:
        selected = metric.loc[
            metric.year.eq(int(row["year"]))
            & metric["sample"].eq("core")
            & metric.seed.eq(13)
            & metric.regime.eq(next(k for k, v in REGIMES.items() if v == row["regime"]))
            & metric.target.eq("profile_clr")
        ].set_index("model")
        row["mse_base"] = round(selected.loc["new_flags", "mse_oof"], 6)
        row["mse_full"] = round(selected.loc["recovered_full", "mse_oof"], 6)
    cov = []
    for r in coverage.loc[coverage["sample"].eq("core")].itertuples():
        for state, count in [
            ("До восстановления", r.old_all3),
            ("После восстановления", r.new_all3),
        ]:
            cov.append(
                {
                    "year": str(r.year),
                    "state": state,
                    "n": int(count),
                    "denominator": int(r.denominator),
                    "pct": round(100 * count / r.denominator, 2),
                }
            )
    updates = []
    for comparison_id, label in [
        ("state_update", "Полное обновление версии"),
        ("shared_flags_recovery", "Новые числа при общих двух наборах флагов"),
    ]:
        updates.extend(
            {**r, "comparison": label}
            for r in rows(
                profile.loc[
                    profile["sample"].eq("core")
                    & profile.seed.eq(13)
                    & profile.comparison.eq(comparison_id)
                ]
            )
        )
    controls = []
    for name, label in [
        ("primary", "Исходные числа"),
        ("permuted_91013", "Перестановка 1"),
        ("permuted_91029", "Перестановка 2"),
        ("permuted_91047", "Перестановка 3"),
    ]:
        subset = comparison.loc[
            comparison.target.eq("log_total")
            & comparison["sample"].eq("core")
            & comparison.regime.eq("heldout_regions")
            & comparison.seed.eq(13)
            & comparison.comparison.eq(name)
        ]
        controls.extend({**r, "variant": label} for r in rows(subset))
    path = pd.read_csv(root / "context/investment_paths.csv")
    investment = []
    for year, group in path.groupby("year", sort=True):
        values = group.set_index("territory_id_context_candidate").value_million_rubles
        investment.append(
            {"year": str(year), "upper": float(values.loc[318]), "lower": float(values.loc[331])}
        )
    sme = pd.read_csv(root / "context/sme_entity_eligibility.csv")
    sme_rows = [
        {
            "year": str(r.year),
            "type": "ИП" if r.is_sole_trader else "Юридические лица",
            "tax_ids": int(r.tax_ids),
            "known": int(r.tax_ids_valid_known_workers),
            "known_pct": round(r.tax_ids_valid_known_workers_pct_of_tax_ids, 2),
            "record_candidate": int(r.record_candidate_known_workers),
            "inflation_pct": round(r.row_count_inflation_pct_vs_distinct_tax_ids, 2),
        }
        for r in sme.itertuples()
    ]
    strata = pd.read_csv(root / "readout/state_update_contributions.csv")
    strata = strata.loc[
        strata.slice.eq("completeness_stratum")
        & strata.target.eq("profile_clr")
        & strata.regime.eq("heldout_regions")
    ]
    strata_names = {
        "common_complete": "Полные прежде",
        "newly_complete": "Стали полными",
        "still_incomplete": "Остались неполными",
    }
    strata_rows = [
        {
            "year": str(r.year),
            "stratum": strata_names[r.group],
            "n": int(r.n),
            "gain": round(100 * r.local_relative_reduction, 3),
            "contribution": round(100 * r.contribution_to_total_relative_reduction, 3),
        }
        for r in strata.itertuples()
    ]
    definitions = [
        (
            "models",
            "Сравнения фиксированных моделей",
            "models/cv_comparisons.csv",
            "scripts/round4_model_analysis.py",
        ),
        (
            "coverage",
            "Изменение полноты признаков",
            "audit/joint_coverage.csv",
            "scripts/round4_data_audit.py",
        ),
        (
            "readout",
            "Описательные вклады в изменение ошибки",
            "readout/state_update_contributions.csv",
            "scripts/round4_model_readout.py",
        ),
        (
            "cases",
            "Проверенные инвестиционные ряды Якутиястата",
            "context/investment_paths.csv",
            "scripts/round4_context_analysis.py",
        ),
        (
            "sme",
            "Наблюдаемость работников в записях МСП",
            "context/sme_entity_eligibility.csv",
            "scripts/round4_context_analysis.py",
        ),
        (
            "geography",
            "Границы применимости географических свидетельств",
            "context/geography_evidence_decisions.csv",
            "scripts/round4_context_analysis.py",
        ),
    ]
    sources = []
    for key, label, relative, script in definitions:
        item = source(key, label, f"{ROOT}/{relative}", f"# Computation: {script}")
        item["query"]["computation_path"] = script
        item["query"]["reproduction_document"] = f"{ROOT}/REPRODUCE.md"
        sources.append(item)
    extra_tables = {
        "models": [
            f"{ROOT}/models/cv_metrics.csv",
            f"{ROOT}/models/oof_predictions.parquet",
            "configs/round4_analysis_v1.json",
            f"{ROOT}/audit/annual_features.csv",
            "reports/round3_analysis/models/model_panel.csv",
        ],
        "coverage": [f"{ROOT}/audit/QA.json", f"{ROOT}/audit/gain_by_region.csv"],
        "readout": [f"{ROOT}/models/oof_predictions.parquet", f"{ROOT}/models/model_panel.csv"],
        "cases": [
            f"{ROOT}/context/investment_longer_comparisons.csv",
            "reports/data_round4/cases/evidence_ledger.csv",
        ],
        "sme": [
            f"{ROOT}/context/sme_worker_observation_by_primary_nace2.csv",
            f"{ROOT}/context/sme_construction_routes.csv",
            "reports/data_round4/sme/manifest.json",
        ],
    }
    for src in sources:
        src["query"]["tables_used"] += extra_tables.get(src["id"], [])
    blocks, tables, charts, datasets, markdown = [], [], [], {}, []

    def prose(key, text, source_id=None):
        block = {"id": key, "type": "markdown", "body": text}
        if source_id:
            block["sourceId"] = source_id
        blocks.append(block)
        markdown.append(text)

    def table(key, title, records, columns, source_id):
        datasets[key] = records
        tables.append(
            {
                "id": key,
                "title": title,
                "dataset": key,
                "sourceId": source_id,
                "defaultSort": {"field": next(iter(columns)), "direction": "asc"},
                "columns": [
                    {
                        "field": k,
                        "label": v,
                        "type": "text" if isinstance(records[0][k], str) else "number",
                    }
                    for k, v in columns.items()
                ],
            }
        )
        blocks.append({"id": "block_" + key, "type": "table", "tableId": key, "layout": "full"})
        markdown.append(table_md(records, columns))

    def chart(key, title, records, x, value, color, unit, source_id, question):
        datasets[key] = records
        charts.append(
            {
                "id": key,
                "title": title,
                "type": "bar",
                "dataset": key,
                "sourceId": source_id,
                "encodings": {
                    "x": {"field": x, "type": "nominal"},
                    "y": {"field": value, "type": "quantitative", "label": unit},
                    "color": {"field": color, "type": "nominal"},
                },
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "intent": "comparison",
                "question": question,
                "rationale": "Дискретные сопоставимые группы; точные значения и интервалы рядом.",
            }
        )
        blocks.append({"id": "block_" + key, "type": "chart", "chartId": key, "layout": "full"})

    prose(
        "title",
        f"# {TITLE}\n\nКейс 1 СберИндекса · четвёртый раунд анализа · 16 сентября 2026 года",
    )
    prose(
        "summary",
        "## Более полные данные пока не меняют решение о модели\n\n"
        "**Восстановление денежных показателей не дало устойчивого улучшения описания "
        "потребительского профиля.** После контроля прежних экономических характеристик, "
        "отраслей и пропусков точечный выигрыш составляет +0.058% / +0.119% внутри регионов "
        "и −0.496% / −0.370% на целых регионах за 2023 / 2024 годы. Все четыре условных "
        "95% интервала включают ноль. Критерий добавления блока в ядро модели не выполнен.\n\n"
        "Сигнал номинального уровня расходов остаётся положительным на целых регионах, "
        "но сохраняется после перестановок чисел внутри регионов. Муниципальная "
        "специфичность этого сигнала не подтверждена. Это ограниченный вывод о проверенной "
        "линейной модели, а не доказательство экономической неважности инвестиций или торговли.",
        "models",
    )
    prose(
        "definitions",
        "## Сравнение сохраняет прежние муниципалитеты и цели\n\n"
        "Единица анализа: муниципалитет за год, с одинаковым весом каждого МО. Основная "
        "выборка содержит 1975 МО в 2023 году и 1998 в 2024-м. Цель — структура пяти "
        "опубликованных категорий расходов: среднее нормированных месячных показателей, "
        "выраженное в CLR-координатах, то есть логарифмах относительно геометрического "
        "среднего компонент. Это потребительский профиль опубликованных показателей, "
        "а не полный бюджет домохозяйства.\n\n"
        "Основная метрика — средняя квадратичная ошибка по пяти координатам (MSE) на МО, "
        "не использованных для обучения соответствующей модели. Положительное снижение "
        "ошибки означает улучшение. Вторичная цель — логарифм среднего месячного "
        "номинального уровня опубликованных расходов. Оба года уже изучались ранее; "
        "этот раунд является исследовательской проверкой, а не новым независимым тестом.",
        "models",
    )
    prose(
        "coverage_text",
        "## Восстановлено покрытие, а состав основной выборки сохранён\n\n"
        "779 дополнительных чисел дали 770 новых непустых значений трёх "
        "логарифмических признаков. "
        "Для девяти ячеек нет подходящего населения; они лежат вне основной выборки. "
        "В полной панели изменения затронули 445 пар «МО–год», 223 МО и 11 регионов. "
        "Тверская и Самарская области дали 432 из 779 чисел, или 55.46%. Это адресное "
        "устранение пробелов, а не случайная выборка новых наблюдений.\n\n"
        "Таблица показывает долю основной выборки со всеми тремя денежными атрибутами. "
        "Полные наблюдения выросли с 1556 до 1768 в 2023 году и с 1603 до 1814 в 2024-м. "
        "Знаменатели и годовые связи сохранены; ограничения 78 преобразований кодов "
        "и 6 календарных случаев не сняты.",
        "coverage",
    )
    chart(
        "coverage_chart",
        "Полнота трёх денежных атрибутов",
        cov,
        "year",
        "pct",
        "state",
        "Доля основной выборки, %",
        "coverage",
        "Как изменилось покрытие при том же составе МО?",
    )
    table(
        "coverage",
        "Наблюдения с тремя атрибутами",
        cov,
        {
            "year": "Год",
            "state": "Версия",
            "n": "Полных МО",
            "denominator": "Всего МО",
            "pct": "Доля, %",
        },
        "coverage",
    )
    prose(
        "primary_text",
        "## Прирост качества профиля остаётся малым и неопределённым\n\n"
        "В основной проверке модель с числами сравнивается с той же моделью, "
        "в которой уже есть актуальные флаги их пропусков. Так наличие информации "
        "отделено от добавления числовых значений. Таблица даёт четыре точечные "
        "оценки, абсолютную ошибку и условные интервалы. "
        "Во всех случаях интервал включает ноль.\n\n"
        "Во всех четырёх выборках все 16 основных "
        "интервалов seed 13 также включают ноль. Дополнительные разбиения внутри "
        "регионов не дают устойчивого выигрыша. Новые значения помогают описать МО, "
        "но оснований использовать этот блок для улучшения основной модели пока нет.",
        "models",
    )
    chart(
        "primary_chart",
        "Числа сверх базы и флагов пропусков",
        main,
        "regime",
        "gain",
        "year",
        "Снижение MSE профиля, %",
        "models",
        "Добавляют ли восстановленные атрибуты информацию?",
    )
    table(
        "main",
        "Основная проверка профиля",
        main,
        {
            "year": "Год",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение, %",
            "ci": "95% интервал, %",
            "mse_base": "MSE базы",
            "mse_full": "MSE с числами",
        },
        "models",
    )
    markdown.append("![Основные оценки и интервалы](figures/profile_increment.png)")
    table(
        "samples",
        "Чувствительность к составу выборки",
        rows(primary.loc[primary.seed.eq(13)]),
        {
            "year": "Год",
            "sample": "Выборка",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    table(
        "seeds",
        "Разбиения внутри регионов",
        rows(primary.loc[primary["sample"].eq("core") & primary.regime.eq("within_region")]),
        {"year": "Год", "seed": "Seed", "gain": "Снижение, %", "ci": "95% интервал, %"},
        "models",
    )
    prose(
        "recovery_text",
        "## Дополнение чисел и обновление признаков пропусков дают разные сравнения\n\n"
        "Прямое сравнение полной модели до и после восстановления даёт −1.491% / −1.581% "
        "на целых регионах; оба интервала включают ноль. В этом сравнении одновременно "
        "меняются значения, флаги пропусков и обученные коэффициенты. Его нельзя назвать "
        "чистым эффектом дополнительных чисел.\n\n"
        "Поэтому отдельно проверены две модели с одинаковыми наборами старых и новых "
        "флагов. При замене только версии чисел в 2023 году на целых регионах выигрыш "
        "равен +0.689% [0.145; 1.455], а в 2024 году −0.003% [−0.318; 0.370]. "
        "Это локальный исследовательский результат чувствительности; он не повторяется "
        "в обоих годах и не выполняет основной критерий.\n\n"
        "Прежние числовые значения сохранены и дают те же признаки. На прежней полной "
        "подвыборке старый и новый варианты после обучения только на этой же подвыборке "
        "совпадают. При обучении на всём core прогноз такого МО может измениться из-за "
        "новых данных других территорий.",
        "models",
    )
    table(
        "updates",
        "Изменение версии и восстановление при общих флагах",
        updates,
        {
            "year": "Год",
            "regime": "Проверка",
            "comparison": "Сравнение",
            "gain": "Снижение, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    prose(
        "strata_text",
        "Следующая таблица описательно разбивает изменение ошибки старой "
        "и новой полной модели на целых регионах. Последний столбец — вклад группы "
        "в общую относительную разность ошибок; вклады складываются в −1.491% / −1.581%. "
        "Это разложение сохранённых ошибок, не оценка причинного воздействия заполнения "
        "пропуска. Полная региональная разбивка сохранена отдельно.",
        "readout",
    )
    table(
        "strata",
        "Вклады групп наблюдений в изменение ошибки",
        strata_rows,
        {
            "year": "Год",
            "stratum": "Группа",
            "n": "МО",
            "gain": "Внутри группы, %",
            "contribution": "Вклад в общую разность, п.п.",
        },
        "readout",
    )
    prose(
        "secondary_text",
        "## Сигнал уровня расходов сохраняется без точного соответствия МО\n\n"
        "Для номинального уровня расходов на целых регионах выигрыш составляет "
        "+3.351% [0.981; 5.762] в 2023 году и +2.038% [−0.033; 4.357] в 2024-м. "
        "Внутри регионов оценки около +0.59%, а интервалы включают ноль. "
        "Вторичная цель не заменяет основной результат о составе профиля.\n\n"
        "Три заранее выбранные перестановки чисел внутри каждого региона дают "
        "2.696–3.896% в 2023 году и 1.995–2.894% в 2024-м. Таблица сопоставляет исходные "
        "данные с этими проверками. Места пропусков и региональные распределения "
        "сохранены; доноры обучения и проверки не смешиваются. Сигнал не исчезает, "
        "поэтому его муниципальная специфичность остаётся неподтверждённой.\n\n"
        "Это три диагностические перестановки, не permutation p-value и не доказательство "
        "превосходства перемешанной модели. Розница и расходы описывают близкие "
        "одновременные процессы. Среди отдельных денежных признаков положительный "
        "условный сигнал уровня расходов связан прежде всего с розницей, но такой "
        "результат нельзя считать независимой проверкой экономической структуры.",
        "models",
    )
    chart(
        "control_chart",
        "Уровень расходов: исходные числа и перестановки",
        controls,
        "variant",
        "gain",
        "year",
        "Снижение MSE лог-уровня, %",
        "models",
        "Исчезает ли сигнал при разрушении соответствия внутри региона?",
    )
    table(
        "controls",
        "Перестановки на целых регионах",
        controls,
        {"year": "Год", "variant": "Вариант", "gain": "Снижение, %", "ci": "95% интервал, %"},
        "models",
    )
    prose(
        "cases_text",
        "## Длинный ряд меняет описание нижнеколымского падения\n\n"
        "Инвестиции Нижнеколымского района в 2023 году в 41.81 раза превышали сумму "
        "2022 года. После резкого снижения сумма 2024 года оказалась на 4.25% ниже "
        "2021-го и на 22.01% ниже 2022-го в текущих ценах. Поэтому точнее говорить "
        "об отдельном высоком значении 2023 года и возврате к сопоставимому порядку "
        "сумм прежних лет. Это описание опубликованного ряда, а не установленная "
        "причина падения.\n\n"
        "У Верхнеколымского района опубликованные номинальные инвестиции увеличиваются "
        "каждый год в 2020–2024. Таблица сохраняет единицу: млн руб. без малого "
        "предпринимательства и инвестиций, не наблюдаемых прямыми статистическими "
        "методами. Исторические проценты вычислены из округлённых ежегодников; границы "
        "их округления не являются статистическими доверительными интервалами. "
        "Состав респондентов, проекты и реальные объёмы после дефляции не установлены.",
        "cases",
    )
    table(
        "investments",
        "Инвестиции двух районов, млн руб. в текущих ценах",
        investment,
        {"year": "Год", "upper": "Верхнеколымский", "lower": "Нижнеколымский"},
        "cases",
    )
    prose(
        "sme_text",
        "## МСП позволяет изучать охват, но ещё не муниципальную занятость\n\n"
        "На уровне уникальных ИНН численность работников известна у 79.28% юридических "
        "лиц в файле 2023 года и у 80.54% в 2024-м. У ИП она отсутствует полностью. "
        "Если считать строки компаниями, число юрлиц было бы завышено на 7.80% / 6.27%. "
        "Фильтр только известных работников тоже меняет состав. Среди юрлиц "
        "с однозначным двузначным основным кодом ОКВЭД в файле 2024 года "
        "наблюдаемость работников составляет 51.40% для кода 52 и 94.96% для кода 86. "
        "Представлены уникальные ИНН за год; это не число действующих фирм на конец "
        "года и не число рабочих мест в МО.\n\n"
        "После одновременной проверки известных неотрицательных работников, одного "
        "территориального кода и одного двузначного основного ОКВЭД внутри годового "
        "файла остаются 1 953 215 / 1 948 891 юрлицо. Это кандидаты на дальнейшую "
        "обработку записей. География июня 2026 года и выбор периода внутри года "
        "по-прежнему не позволяют считать их готовым историческим муниципальным слоем.",
        "sme",
    )
    table(
        "sme",
        "Охват численности работников в годовых файлах МСП",
        sme_rows,
        {
            "year": "Год",
            "type": "Тип",
            "tax_ids": "Различных ИНН",
            "known": "Работники известны",
            "known_pct": "Доля, %",
            "record_candidate": "ИНН-кандидатов",
        },
        "sme",
    )
    prose(
        "geo_text",
        "## Равенство геометрий поставщика не снимает ограничения\n\n"
        "Для четырёх проверенных ID поставщик отдаёт одинаковые геометрии за 2023 "
        "и 2024 годы. Это подтверждает воспроизводимость выбора полигона. Парных "
        "датированных координатных описаний юридических границ пока нет. Маски "
        "исторической и календарной сопоставимости остаются прежними; источник "
        "МСП с современной географией не используется для их замены.",
        "geography",
    )
    prose(
        "methods",
        "## Методика фиксирует сравнение и границы уверенности\n\n"
        "Ridge с alpha=1 обучается отдельно для каждого года на пяти разбиениях. "
        "База: логарифмы населения, зарплаты и концентрации учтённых рабочих мест; "
        "тип МО; восемь отраслевых долей и флаги их пропусков. Денежный блок "
        "включает инвестиции в основной капитал, отгрузку продукции, работ и услуг "
        "собственного производства и оборот розничной торговли; охват — без малого "
        "предпринимательства. Признаки равны log(тыс. руб. × 1000 / население "
        "на начало года) при положительных конечных числителе и знаменателе. "
        "Исходный ноль сохраняется в данных, но не даёт логарифмического признака. "
        "Это номинальная "
        "территориальная интенсивность, а не доход жителей. Три суммы не складываются.\n\n"
        "Медианы, масштабы и словарь категорий оцениваются только на обучающей части. "
        "Внутри регионов модель использует региональные индикаторы; при удержании "
        "целых регионов их нет. Основная и консервативная выборки повторяют прежние "
        "ключи разбиений. Полные подвыборки ограничивают те же core-разбиения, "
        "но обучаются на разном составе МО: разность их метрик не является чистым "
        "эффектом расширения выборки.\n\n"
        "95% интервалы построены по 2000 парным перевыборкам регионов для уже "
        "сохранённых ошибок. Они не включают повторное обучение всей процедуры "
        "и не скорректированы на множественные сравнения. Все дополнительные "
        "разбиения, перестановки и исключения отдельных признаков заданы до новых "
        "результатов. Предыдущие прогнозы воспроизведены, гиперпараметры не подбирались.\n\n"
        "Источники скачаны в 2026 году, показатели относятся к тем же годам, что "
        "и расходы. История публикаций и пересмотров неизвестна. Этот дизайн "
        "не оценивает прогноз будущего или причинное влияние.",
        "models",
    )
    prose(
        "decision",
        "## Денежный блок остаётся описательным контекстом\n\n"
        "**Основную модель и сеть оставляем в прежней спецификации.** Правило "
        "требовало не менее 1% снижения ошибки профиля в обоих годах и режимах, "
        "положительной нижней границы интервала и отсутствия разворота результата "
        "на выборке без смены кода. Новые данные этого правила не выполнили.\n\n"
        "- Для экономических карточек МО используем более полные денежные ряды "
        "с явным охватом и историческими ограничениями.\n"
        "- Для МСП сначала фиксируем момент наблюдения и проверяем исторические "
        "ключи; отдельно оцениваем отбор по наличию работников.\n"
        "- Для объяснения резких изменений ищем сведения о респондентах и проектах; "
        "для границ нужны независимые исторические описания.\n\n"
        "Открытые вопросы: меняется ли экономическая интерпретация прежних групп "
        "при исключении регионального контекста; какие муниципальные признаки МСП "
        "можно построить без подмены исторической географии; что объясняет единичный "
        "инвестиционный пик. Новый раунд уточнил эти вопросы, не устранив их автоматически.",
    )

    markdown.append(
        "Источники и определения: БДПМО Росстата; ежегодники Якутиястата; "
        "реестр МСП ФНС в обработке «Если быть точным», версия v20260605 "
        "([набор](https://tochno.st/datasets/sme)). "
        "Подробные ограничения и происхождение контекстных таблиц: "
        "[context/README.md](context/README.md). "
        "[Воспроизведение](REPRODUCE.md) · [Независимая проверка](independent_review.md)."
    )
    timestamp = json.loads((root / "models/manifest.json").read_text("utf-8"))["completed_at"]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "generatedAt": timestamp,
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": timestamp,
            "datasets": datasets,
        },
        "sources": sources,
    }
    (output / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), "utf-8"
    )
    (output / "REPORT.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    pd.DataFrame(main).to_csv(output / "headline_metrics.csv", index=False)
    pd.DataFrame(controls).to_csv(output / "negative_control_metrics.csv", index=False)
    design_record = {
        "audience": "technical",
        "delivery": "portable HTML",
        "reason": "Established repository-based research report and GitHub handoff in Codex",
        "adoption_passed": False,
        "primary": "profile CLR",
        "secondary": "log nominal level",
        "sections": [b["id"] for b in blocks if b["type"] == "markdown"],
        "spine": (
            "Coverage improves; model decision unchanged; bounded secondary and source insights"
        ),
        "chart_map": [
            {
                "id": c["id"],
                "question": c["question"],
                "family": "comparison",
                "type": "grouped bar",
            }
            for c in charts
        ],
        "repeated_chart_family_reason": (
            "All three visuals compare discrete groups, not time trajectories"
        ),
        "tables_instead_of_trends": (
            "Five annual case points retained as exact table; no shock timing claim"
        ),
        "source_data_not_changed": True,
        "static_research_figure": (
            "figures/profile_increment.png and SVG with conditional intervals"
        ),
        "structure_mapping": (
            "Summary; definitions before evidence; visual findings; dedicated methods; "
            "limitations adjacent; decision and open questions"
        ),
    }
    (output / "report_design.json").write_text(
        json.dumps(design_record, indent=2), encoding="utf-8"
    )
    print(json.dumps({"blocks": len(blocks), "charts": len(charts), "tables": len(tables)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    build(args.repo, args.out or args.repo / ROOT)
