"""Build a technical report from frozen third-round measurements, without refitting."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
from build_round2_report import source, table_md

TITLE = "Что дают инвестиции, отгрузка и розница для муниципальных профилей"
ROOT = "reports/round3_analysis"
BASE = "economic_sectors_newflags"
FULL = "economic_sectors_newvalues"
SAMPLES = {"core": "Основная", "stable_code": "Без смены кода", "complete_new3": "Полные новые 3"}
REGIMES = {"within_region": "Внутри регионов", "heldout_regions": "Целые регионы"}
NAMES = {
    318: "Верхнеколымский",
    331: "Нижнеколымский",
    883: "Свободненский, Амурская обл.",
    925: "Красноярский, Астраханская обл.",
    1180: "Светловский",
    2527: "Куркино",
}
INDICATORS = {"investment": "Инвестиции", "shipments": "Отгрузка", "retail": "Розница"}


def build(repo, out=None):
    root = repo / ROOT
    out = out or root
    out.mkdir(parents=True, exist_ok=True)

    def read(path):
        return duckdb.read_csv(str(root / path)).df()

    comparisons = read("models/cv_comparisons.csv")
    metrics = read("models/cv_metrics.csv")
    coverage = read("audit/coverage.csv")
    cases = read("context/six_cases_economy_pairs.csv")
    slices = read("models/slice_losses.csv")
    primary = comparisons.loc[
        comparisons.baseline.eq(BASE) & comparisons.alternative.eq(FULL)
    ].copy()

    def comparison_rows(frame):
        result = []
        for r in frame.itertuples():
            result.append(
                {
                    "year": str(r.year),
                    "sample": SAMPLES[r.sample],
                    "regime": REGIMES[r.regime],
                    "seed": int(r.seed),
                    "n": int(r.n),
                    "regions": int(r.regions),
                    "gain": round(100 * r.relative_mse_reduction, 3),
                    "ci": f"[{100 * r.ci_low:.3f}; {100 * r.ci_high:.3f}]",
                    "label": f"{r.year} · {REGIMES[r.regime]}",
                }
            )
        return result

    profile = primary.loc[primary.target.eq("profile_clr") & primary.seed.eq(13)]
    core_profile = profile.loc[profile["sample"].eq("core")]
    assert len(core_profile) == 4
    assert core_profile.relative_mse_reduction.lt(0).all()
    assert ((profile.ci_low < 0) & (profile.ci_high > 0)).all()
    adoption = bool(
        core_profile.relative_mse_reduction.ge(0.01).all() & core_profile.ci_low.gt(0).all()
    )
    assert not adoption
    main = comparison_rows(core_profile)
    for row in main:
        subset = metrics.loc[
            metrics.year.eq(int(row["year"]))
            & metrics["sample"].eq("core")
            & metrics.regime.eq(next(k for k, v in REGIMES.items() if v == row["regime"]))
            & metrics.seed.eq(13)
            & metrics.target.eq("profile_clr")
        ].set_index("model")
        for field, name in [("mse_oof", "mse"), ("r2_oof", "r2")]:
            row[f"{name}_base"] = round(float(subset.loc[BASE, field]), 6)
            row[f"{name}_full"] = round(float(subset.loc[FULL, field]), 6)

    cov = []
    for r in coverage.loc[coverage["sample"].eq("core")].itertuples():
        cov.append(
            {
                "year": str(r.year),
                "indicator": INDICATORS[r.prefix],
                "n": int(r.feature_available),
                "denominator": int(r.n_municipality_years),
                "pct": round(100 * r.feature_available_fraction, 2),
            }
        )
    decomposition = []
    for before, after, label in [
        ("economic_newflags", "economic_newvalues", "Числа сверх простой базы"),
        ("economic_sectors", BASE, "Только новые флаги пропусков"),
        ("economic_sectors", FULL, "Весь новый блок сверх отраслей"),
    ]:
        rows = comparison_rows(
            comparisons.loc[
                comparisons.baseline.eq(before)
                & comparisons.alternative.eq(after)
                & comparisons["sample"].eq("core")
                & comparisons.seed.eq(13)
                & comparisons.target.eq("profile_clr")
            ]
        )
        decomposition.extend([dict(r, comparison=label) for r in rows])
    ablations = []
    for prefix, label in INDICATORS.items():
        rows = comparison_rows(
            comparisons.loc[
                comparisons.baseline.eq(f"without_{prefix}") & comparisons.target.eq("profile_clr")
            ]
        )
        ablations.extend([dict(r, indicator=label) for r in rows])
    secondary = comparison_rows(primary.loc[primary.target.eq("log_total") & primary.seed.eq(13)])
    control = []
    for alternative, label in [(FULL, "Исходные числа"), ("permuted_newvalues", "Перемешанные")]:
        rows = comparison_rows(
            comparisons.loc[
                comparisons.baseline.eq(BASE)
                & comparisons.alternative.eq(alternative)
                & comparisons["sample"].eq("core")
                & comparisons.seed.eq(13)
                & comparisons.target.eq("log_total")
            ]
        )
        control.extend([dict(r, variant=label) for r in rows])
    seed_rows = comparison_rows(
        primary.loc[
            primary.target.eq("profile_clr")
            & primary["sample"].eq("core")
            & primary.regime.eq("within_region")
        ]
    )
    case_rows = []
    for territory_id, name in NAMES.items():
        d = cases.loc[cases.territory_id.eq(territory_id)].set_index("indicator_id")
        row = {"mo": name}
        for code, prefix in [(8109001, "investment"), (8401011, "shipments"), (8401003, "retail")]:
            value = d.loc[code, "nominal_growth_pct"]
            row[prefix] = "Нет пары" if pd.isna(value) else f"{value:+.2f}%"
        case_rows.append(row)
    type_rows = []
    type_names = {
        "внутригородская территория города федерального значения": "Внутригородские территории",
        "городской округ": "Городские округа",
        "муниципальный округ": "Муниципальные округа",
        "муниципальный район": "Муниципальные районы",
    }
    for r in slices.loc[
        slices["sample"].eq("core")
        & slices.target.eq("profile_clr")
        & slices.regime.eq("heldout_regions")
        & slices.slice.eq("municipal_district_type")
    ].itertuples():
        type_rows.append(
            {
                "year": str(r.year),
                "type": type_names[r.group],
                "n": int(r.n),
                "gain": round(100 * r.relative_mse_reduction, 3),
            }
        )

    source_defs = []
    for key, label, relative, script in [
        (
            "models",
            "Фиксированные сравнения моделей третьего раунда",
            "models/cv_comparisons.csv",
            "scripts/round3_model_analysis.py",
        ),
        (
            "coverage",
            "Аудит покрытия трёх атрибутов",
            "audit/coverage.csv",
            "scripts/round3_data_audit.py",
        ),
        (
            "cases",
            "Шесть ранее выбранных МО: условно сопоставимые номинальные пары",
            "context/six_cases_economy_pairs.csv",
            "scripts/round3_context_analysis.py",
        ),
        (
            "documents",
            "Проверки сопоставимости местных документов",
            "context/document_comparability_checks.csv",
            "scripts/round3_context_analysis.py",
        ),
        (
            "slices",
            "Описательные срезы фиксированных ошибок",
            "models/slice_losses.csv",
            "scripts/round3_model_analysis.py",
        ),
    ]:
        src = source(
            key,
            label,
            f"{ROOT}/{relative}",
            f"# Computation: {script}\n"
            f"import pandas as pd\nd = pd.read_csv('{ROOT}/{relative}')\nprint(d)",
        )
        src["query"]["computation_path"] = script
        source_defs.append(src)
    source_defs[0]["query"]["tables_used"] += [
        f"{ROOT}/models/cv_metrics.csv",
        f"{ROOT}/models/oof_predictions.parquet",
        "configs/round3_analysis_v1.json",
        f"{ROOT}/audit/annual_features.csv",
        "reports/round2_analysis/models/model_panel.csv",
    ]
    source_defs[1]["query"]["tables_used"] += [
        f"{ROOT}/audit/joint_coverage.csv",
        f"{ROOT}/audit/coverage_by_region.csv",
        f"{ROOT}/audit/collection_refusals.csv",
        f"{ROOT}/audit/QA.json",
        "reports/data_round3/economy/annual_candidate_panel.csv",
        "reports/data_round2/sector/bulk/year_specific_links.csv",
    ]
    blocks, md, datasets, tables, charts = [], [], {}, [], []

    def prose(key, text, src=None):
        block = {"id": key, "type": "markdown", "body": text}
        if src:
            block["sourceId"] = src
        blocks.append(block)
        md.append(text)

    def table(key, title, rows, cols, src):
        datasets[key] = rows
        tables.append(
            {
                "id": key,
                "title": title,
                "dataset": key,
                "sourceId": src,
                "columns": [
                    {
                        "field": k,
                        "label": label,
                        "type": "text" if isinstance(rows[0][k], str) else "number",
                    }
                    for k, label in cols.items()
                ],
            }
        )
        blocks.append({"id": f"block_{key}", "type": "table", "tableId": key, "layout": "full"})
        md.append(table_md(rows, cols))

    def bar(key, title, rows, x, y, color, ylabel, src, static_path, question):
        datasets[key] = rows
        charts.append(
            {
                "id": key,
                "title": title,
                "type": "bar",
                "dataset": key,
                "sourceId": src,
                "encodings": {
                    "x": {"field": x, "type": "nominal"},
                    "y": {"field": y, "type": "quantitative", "label": ylabel},
                    "color": {"field": color, "type": "nominal"},
                },
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "intent": "comparison",
                "question": question,
                "rationale": "Общая нулевая база; значения и интервалы приведены рядом в таблице.",
            }
        )
        blocks.append({"id": f"block_{key}", "type": "chart", "chartId": key, "layout": "full"})
        md.append(f"![{title}]({static_path})")

    prose(
        "title", f"# {TITLE}\n\nКейс 1 СберИндекса · третий раунд анализа · 16 сентября 2026 года"
    )
    prose(
        "summary",
        "## Технический итог: оснований расширять ядро модели нет\n\n"
        "**Инвестиции, отгрузка и розничный оборот не показали устойчивого прироста качества "
        "описания потребительского профиля после учёта прежних экономических и отраслевых "
        "характеристик.** В четырёх основных сравнениях точечная ошибка немного выросла; "
        "все 95% интервалы включают ноль. Заранее установленный критерий включения блока "
        "не выполнен. Это вывод для проверенной линейной модели и этих выборок, а не "
        "доказательство экономической незначимости показателей.\n\n"
        "Для номинального уровня расходов есть положительные оценки при проверке на "
        "целых регионах. Однако перемешивание новых чисел внутри регионов сохраняет "
        "этот сигнал. Его муниципальная специфичность не подтверждена. "
        "Сохраняем новые данные для описания отдельных территорий; не меняем на их основе "
        "основную сеть и не объявляем новые устойчивые типы экономик.",
        "models",
    )
    prose(
        "definitions",
        "## Что именно сравнивалось\n\n"
        "Единица анализа — муниципалитет за год; каждый МО получает одинаковый вес. "
        "Использованы прежние 1975 МО за 2023 год и 1998 за 2024 год. Целевая структура — "
        "среднее 12 месячных нормированных показателей пяти категорий Сбера, преобразованное "
        "в CLR-координаты (логарифм компоненты относительно геометрического среднего). "
        "Это соотношения опубликованных показателей, а не полный бюджет домохозяйства.\n\n"
        "База включает население, зарплату организаций, число учтённых работников на жителя, "
        "тип МО, восемь отраслевых долей работников и их флаги пропусков. Число работников "
        "характеризует рабочие места организаций, не долю занятых жителей. В главном сравнении "
        "обе модели также содержат три флага отсутствия новых чисел. Разница между ними — "
        "только три новых логарифма денежных объёмов на жителя.\n\n"
        "Положительное снижение ошибки означает улучшение: 100 × (1 − SSE новой модели / "
        "SSE базы). SSE суммируется по пяти CLR-координатам. В таблице абсолютная MSE "
        "дополнительно делится на число МО и на пять координат; это не меняет относительный "
        "выигрыш. R² рассчитан вне обучения по сумме координатных ошибок и вариаций. "
        "Вторичная цель — логарифм годового среднего месячного уровня опубликованных расходов.",
        "models",
    )
    prose(
        "coverage_text",
        "## Покрытие высокое, но пропуски систематичны\n\n"
        "В сетке 2016 МО × 2 года × 3 показателя принято 10 829 чисел из 12 096 позиций. "
        "Остались 1267 пропусков: 1245 не возвращены или не приняты, 22 содержат "
        "неразрешённый маркер источника. Нулевую деятельность из этих статусов вывести нельзя. "
        "Все три модельных признака доступны для 1556 из 1975 МО в 2023 году и "
        "1603 из 1998 в 2024 году.\n\n"
        "В полученных ответах за 2024 год отсутствуют все три слоя у 37 МО Тверской "
        "и 36 МО Самарской области. Это ограничение сбора принятых ответов. "
        "Поэтому наличие числа контролируется отдельно от его величины; результат "
        "нельзя автоматически распространить на непокрытые территории.",
        "coverage",
    )
    bar(
        "coverage_chart",
        "Доступность новых признаков в основной выборке",
        cov,
        "indicator",
        "pct",
        "year",
        "Доступно, % МО соответствующего года",
        "coverage",
        "figures/activity_coverage.png",
        "Насколько полны три новых атрибута?",
    )
    table(
        "coverage_table",
        "Покрытие и знаменатели",
        cov,
        {
            "year": "Год",
            "indicator": "Показатель",
            "n": "Доступно",
            "denominator": "Всего МО",
            "pct": "Покрытие, %",
        },
        "coverage",
    )
    prose(
        "measurement",
        "Денежные значения взяты из БДПМО в тысячах рублей за год; "
        "инвестиции — годовой показатель, отгрузка и розница — январь–декабрь. "
        "Для модели используется log(значение × 1000 / население на начало года). "
        "Это номинальная интенсивность учтённой деятельности на территории, с ограничением "
        "охвата малых предприятий. Инвестиции не равны доходу жителей, отгрузка не равна "
        "добавленной стоимости, а торговля по месту продажи не равна покупкам жителей. "
        "Эти объёмы нельзя складывать. Числа не дефлированы.\n\n"
        "Исторические коды соединены по прежнему годовому справочнику. Маски качества "
        "не расширялись. Вне принятой годовой связи оказались 29 исходных строк, все "
        "с нечисловыми маркерами. В полной панели ещё 108 позиций имеют денежное число, "
        "но не имеют допустимого населения для логарифмического признака; в основной "
        "и консервативной выборках такого расхождения нет.",
        "coverage",
    )
    prose(
        "main_text",
        "## Новые числа не улучшают состав профиля\n\n"
        "После контроля пропусков снижение ошибки профиля составляет −0,033% и −0,037% "
        "внутри регионов, на −0,622% и −0,328% при удержании целых регионов "
        "в 2023 и 2024 годах соответственно. Минус здесь означает ухудшение. "
        "Все интервалы пересекают ноль. Практического основания добавлять три признака "
        "в ядро модели по этому результату нет.\n\n"
        "Столбцы показывают точечные оценки основной выборки; интервалы и абсолютные "
        "метрики приведены в таблице. Статическая исследовательская фигура дополнительно "
        "показывает интервалы всех трёх выборок.",
        "models",
    )
    bar(
        "profile_chart",
        "Дополнительная информация новых чисел о профиле",
        main,
        "regime",
        "gain",
        "year",
        "Снижение ошибки профиля, %",
        "models",
        "figures/profile_increment.png",
        "Улучшается ли профиль после старых атрибутов?",
    )
    table(
        "main",
        "Основная выборка: одно и то же разбиение",
        main,
        {
            "year": "Год",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение, %",
            "ci": "95% интервал, %",
            "mse_base": "MSE базы",
            "mse_full": "MSE с числами",
            "r2_base": "R² базы",
            "r2_full": "R² с числами",
        },
        "models",
    )
    prose(
        "robustness",
        "## Смена выборки не даёт устойчивого положительного результата\n\n"
        "Сохраняем две заранее заданные проверки: МО без смены кода и типа, "
        "а также МО с полными тремя новыми показателями. Для первой точно повторены "
        "прежние разбиения; для второй построены свои общие для сравниваемых моделей "
        "разбиения. Все 12 интервалов основного сравнения профиля включают ноль. "
        "На полной подвыборке все три новых денежных показателя наблюдаются, но пропуски "
        "прежних восьми отраслей сохраняются. Изменение состава выборки и разбиения "
        "не позволяет считать разность её метрик чистым эффектом заполнения пропусков.",
        "models",
    )
    table(
        "sensitivity",
        "Все заранее заданные выборки",
        comparison_rows(profile),
        {
            "year": "Год",
            "sample": "Выборка",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    prose(
        "seed_text",
        "При двух дополнительных заранее выбранных разбиениях внутри "
        "регионов знак основной оценки также отрицателен в обоих годах. "
        "Лучшее разбиение для итогового результата не выбиралось.",
        "models",
    )
    table(
        "seeds",
        "Чувствительность к разбиению внутри регионов",
        seed_rows,
        {"year": "Год", "seed": "Seed", "gain": "Снижение ошибки, %", "ci": "95% интервал, %"},
        "models",
    )
    prose(
        "decomposition_text",
        "## Эффект наличия данных отделён от эффекта чисел\n\n"
        "Сравнение с более простой экономической базой, добавление только новых "
        "флагов пропусков и добавление всего блока не дают устойчивого выигрыша "
        "профиля. Во всех приведённых ниже сравнениях интервалы включают ноль. "
        "При последовательном исключении инвестиций, отгрузки и розницы также нет "
        "отдельного показателя с устойчивым положительным вкладом в профиль. "
        "Условные вклады зависят от остальных признаков и не обязаны складываться.",
        "models",
    )
    table(
        "decomposition",
        "Дополнительные сравнения профиля",
        decomposition,
        {
            "year": "Год",
            "regime": "Проверка",
            "comparison": "Добавление",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    table(
        "ablations",
        "Возврат одного показателя в модель с остальными",
        ablations,
        {
            "year": "Год",
            "regime": "Проверка",
            "indicator": "Возвращённый показатель",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    prose(
        "secondary_text",
        "## Сигнал уровня расходов сохраняется после перемешивания\n\n"
        "Для вторичной цели на целых регионах исходные числа дают снижение ошибки "
        "на 3,161% в 2023 году (95% интервал 0,860–5,385%) и на 1,852% в 2024 году "
        "(−0,259–4,056%). Внутри регионов интервалы включают ноль. На некоторых "
        "подвыборках оценки положительны, но это не заменяет основную проверку профиля.\n\n"
        "Диагностический контроль перемешивает наблюдаемые новые числа отдельно "
        "внутри каждого региона и отдельно в обучении и проверке, сохраняя места "
        "пропусков. Для целых регионов его точечный выигрыш равен 5,071% и 2,491% — "
        "он не исчезает и оказывается выше исходных оценок. Контроль сохраняет "
        "региональные распределения; результат не подтверждает, что выигрыш связан "
        "именно с соответствием числа конкретному МО.\n\n"
        "Это одна фиксированная перестановка, без отдельного интервала разности "
        "«исходные минус перемешанные». Она не доказывает превосходство перемешанной "
        "модели или отсутствие муниципальной связи. Розничный оборот и расходы "
        "описывают близкие одновременные процессы, поэтому такую проверку также "
        "нельзя считать независимым экономическим подтверждением.",
        "models",
    )
    bar(
        "control_chart",
        "Уровень расходов: исходные и перемешанные числа",
        control,
        "label",
        "gain",
        "variant",
        "Снижение ошибки log расходов, %",
        "models",
        "figures/log_total_negative_control.png",
        "Исчезает ли вторичный сигнал при перемешивании?",
    )
    table(
        "control",
        "Один диагностический контроль",
        control,
        {
            "year": "Год",
            "regime": "Проверка",
            "variant": "Числа",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    table(
        "secondary",
        "Вторичная цель во всех выборках",
        secondary,
        {
            "year": "Год",
            "sample": "Выборка",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    prose(
        "slices_text",
        "## Ошибки распределены неодинаково\n\n"
        "В удержанных регионах у внутригородских территорий ошибка профиля возрастает "
        "на 13,075% в 2023 году и 1,600% в 2024 году; в этой группе по 242 МО. "
        "Это описательный срез сохранённых прогнозов, без отдельного интервала и без "
        "проверки множества гипотез. Он указывает на неоднородность, но не даёт основания "
        "исключить неудобную группу и пересчитать главный результат. Срезы по региону, "
        "числу доступных признаков и квартилю населения сохранены в таблице ошибок. "
        "Квартили рассчитаны только для описания, они не участвовали в обучении.",
        "slices",
    )
    table(
        "types",
        "Профиль на целых регионах: типы МО",
        type_rows,
        {"year": "Год", "type": "Тип МО", "n": "МО", "gain": "Снижение ошибки, %"},
        "slices",
    )
    prose(
        "case_text",
        "## Отдельные территории нельзя описать одним темпом роста\n\n"
        "У шести ранее выбранных МО доступны 34 из 36 денежных значений и 17 из 18 "
        "годовых пар. Для розницы Свободненского района данных нет в обоих годах. "
        "Таблица показывает номинальное изменение 2024 года к 2023-му. Коды, периоды "
        "и единицы совпадают; неизменность границ и круга респондентов не доказана.\n\n"
        "В Верхнеколымском инвестиции выросли на 208,26%, а отгрузка снизилась на "
        "24,89%. В Нижнеколымском инвестиции упали на 98,14%, при росте розницы "
        "на 16,75%. В Куркине рост розницы на 143,56% сочетается со снижением "
        "инвестиций на 48,59%. Такие сочетания помогают формулировать адресные вопросы "
        "к источникам, но не устанавливают завершение проекта, изменение дохода "
        "жителей или причину потребительского остатка. Эти случаи ранее выбраны по "
        "остаткам модели и не являются независимой проверочной выборкой.",
        "cases",
    )
    table(
        "cases",
        "Номинальные изменения опубликованных объёмов",
        case_rows,
        {
            "mo": "Муниципалитет",
            "investment": "Инвестиции",
            "shipments": "Отгрузка",
            "retail": "Розница",
        },
        "cases",
    )
    prose(
        "documents_text",
        "Местные документы не снимают прежние противоречия. "
        "В Светловском число работников совпадает с опубликованной статистикой, "
        "но зарплата 2024 года отличается от зафиксированного ряда на 412,5 рубля; "
        "причина остаётся неизвестной. По Свободненскому нельзя смешивать декабрьское "
        "число работников со среднегодовым; два подозрительных числа из Амурского "
        "PDF остаются в карантине. В Куркине 8 — муниципальные служащие в штатном "
        "расписании на 31.12.2024, а 336 — трудоустроенные в «Жилищнике» на неустановленную "
        "дату отчёта. Эти сведения об отдельных учреждениях не заменяют общий итог территории.",
        "documents",
    )
    prose(
        "methods",
        "## Как ограничена неопределённость\n\n"
        "Использована Ridge-регрессия с фиксированным alpha = 1. Медианы заполнения, "
        "масштабы и словари категорий оцениваются только на обучающих строках. "
        "Пять разбиений основной и консервативной выборок совпадают с прошлым раундом. "
        "Внутри регионов учитывается регион; при удержании целых регионов региональные "
        "фиктивные переменные исключены. Удержание региона не означает пространственного "
        "буфера от соседей.\n\n"
        "95% интервалы получены 2000 парными перевыборками регионов из фиксированных "
        "ошибок вне обучения. Модель при этом не переобучается. Интервалы условны "
        "относительно выборки, признаков и разбиения, не включают всю неопределённость "
        "процедуры и не скорректированы за множество вторичных сравнений.\n\n"
        "Результаты обоих лет уже рассматривались в предыдущих раундах. Анализ "
        "остаётся исследовательским; 2024 год нельзя объявлять нетронутой проверкой. "
        "Годовые данные собраны в 2026 году, время первой публикации и история "
        "пересмотров неизвестны. Здесь нет утверждения о прогнозе будущего, "
        "доступности признаков на дату прогноза или причинном воздействии. "
        "Настройки после просмотра новых результатов не подбирались.",
        "models",
    )
    prose(
        "decision",
        "## Решение для первого кейса\n\n"
        "**Основную модель и сеть оставляем без нового денежного блока.** "
        "Заранее заданное правило требовало не менее 1% снижения ошибки профиля "
        "во всех четырёх основных сравнениях, положительной нижней границы "
        "интервала и согласованности на МО без смены кода. Эти условия не выполнены.\n\n"
        "Новые величины сохраняем в карточках территорий вместе с охватом и "
        "пропусками. Следующий содержательный шаг — проверка воспроизводимости "
        "экономической интерпретации прежней сети, а не расширение набора признаков "
        "ради положительной метрики. Отдельной проверки требуют отсутствие слоёв "
        "Тверской и Самарской областей, круг респондентов при резких изменениях "
        "и сопоставимость географии. До её завершения прежние ограничения "
        "по кодам, границам и несвязанной статистике МСП сохраняются.",
    )

    timestamp = json.loads((root / "models/manifest.json").read_text())["completed_at"]
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
            "sources": source_defs,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": timestamp,
            "datasets": datasets,
        },
        "sources": source_defs,
    }
    (out / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), "utf-8")
    (out / "REPORT.md").write_text("\n\n".join(md) + "\n", "utf-8")
    pd.DataFrame(main).to_csv(out / "headline_metrics.csv", index=False)
    pd.DataFrame(control).to_csv(out / "negative_control_metrics.csv", index=False)
    (out / "report_design.json").write_text(
        json.dumps(
            {
                "audience": "technical",
                "delivery": "portable HTML",
                "adoption_passed": adoption,
                "spine": "No reliable primary increment; negative control limits secondary signal",
                "primary": "profile CLR",
                "secondary": "log nominal spending",
                "sections": [b["id"] for b in blocks if b["type"] == "markdown"],
                "source_data_not_changed": True,
                "omitted": {
                    "causal_effects": "design cannot identify",
                    "new_clusters": "out of scope",
                    "forecast_metrics": "annual contemporaneous retrospective design",
                },
                "standalone_figures": "PNG/SVG with intervals; native HTML uses bars + CI tables",
            },
            ensure_ascii=False,
            indent=2,
        ),
        "utf-8",
    )
    print(f"Wrote {len(blocks)} blocks, {len(charts)} charts, {len(tables)} tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    build(args.repo, args.out)
