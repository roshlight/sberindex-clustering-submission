"""Build the fifth-round technical narrative and canonical portable report payload."""

import argparse
import json
from pathlib import Path

import pandas as pd
from build_round2_report import source, table_md

ROOT = "reports/round5_analysis"
TITLE = "Что дали ещё 64 наблюдения и уточнение географии МСП"
REGIMES = {"within_region": "Внутри регионов", "heldout_regions": "Целые регионы"}
SAMPLES = {"core": "Основная", "stable_code": "Без смены кода"}


def build(repo, root, output):
    output.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(root / "models/cv_comparisons.csv")
    metrics = pd.read_csv(root / "models/cv_metrics.csv")
    coverage = pd.read_csv(root / "audit/joint_coverage.csv")
    selection = pd.read_csv(root / "recovered/regional_selection_summary.csv")
    geo = pd.read_csv(root / "context/coverage_interpretation.csv")
    annual = pd.read_csv(root / "context/annual_population.csv")
    stock = pd.read_csv(root / "context/fns_population_contrast.csv")
    budget = pd.read_csv(root / "context/budget_revision.csv")
    readout = pd.read_csv(root / "readout/state_update_contributions.csv")
    decision = json.loads((root / "readout/QA.json").read_text("utf-8"))
    assert not decision["adoption_passed"]
    profile = cv.loc[cv.target.eq("profile_clr")]
    core = profile.loc[profile["sample"].eq("core") & profile.seed.eq(13)]
    primary = core.loc[core.comparison.eq("primary")]
    assert len(primary) == 4 and (primary.ci_low.lt(0) & primary.ci_high.gt(0)).all()

    def model_rows(frame):
        return [
            {
                "year": str(r.year),
                "regime": REGIMES[r.regime],
                "sample": SAMPLES.get(r.sample, r.sample),
                "seed": int(r.seed),
                "n": int(r.n),
                "gain": round(100 * r.relative_mse_reduction, 4),
                "low": round(100 * r.ci_low, 4),
                "high": round(100 * r.ci_high, 4),
                "ci": f"[{100 * r.ci_low:.4f}; {100 * r.ci_high:.4f}]",
            }
            for r in frame.itertuples()
        ]

    headline = model_rows(primary)
    for row in headline:
        m = metrics.loc[
            metrics.year.eq(int(row["year"]))
            & metrics["sample"].eq("core")
            & metrics.seed.eq(13)
            & metrics.regime.eq(next(k for k, v in REGIMES.items() if v == row["regime"]))
            & metrics.target.eq("profile_clr")
        ].set_index("model")
        row["mse_base"] = round(m.loc["new_flags", "mse_oof"], 7)
        row["mse_full"] = round(m.loc["recovered_full", "mse_oof"], 7)
    definitions = {
        "models": (
            "Проверка денежных признаков",
            "models/cv_comparisons.csv",
            "round5_model_analysis.py",
        ),
        "coverage": (
            "Полнота на прежней выборке",
            "audit/joint_coverage.csv",
            "round5_data_audit.py",
        ),
        "selection": (
            "Восстановленные и прежние МО",
            "recovered/regional_selection_summary.csv",
            "round5_recovered_profiles.py",
        ),
        "readout": (
            "Изменение ошибок на срезах",
            "readout/state_update_contributions.csv",
            "round5_model_readout.py",
        ),
        "geography": (
            "Детализация кодов МСП",
            "context/coverage_interpretation.csv",
            "round5_context_analysis.py",
        ),
        "workers": (
            "Наблюдаемость работников МСП",
            "context/annual_population.csv",
            "round5_context_analysis.py",
        ),
        "stocks": (
            "Разные популяции ФНС и годовых файлов",
            "context/fns_population_contrast.csv",
            "round5_context_analysis.py",
        ),
        "cases": (
            "Редакции бюджетного удорожания",
            "context/budget_revision.csv",
            "round5_context_analysis.py",
        ),
    }
    sources = []
    for key, (label, path, code) in definitions.items():
        item = source(key, label, f"{ROOT}/{path}", f"scripts/{code}")
        item["query"]["computation_path"] = f"scripts/{code}"
        item["query"]["reproduction_document"] = f"{ROOT}/REPRODUCE.md"
        sources.append(item)
    extras = {
        "models": [
            f"{ROOT}/models/cv_metrics.csv",
            f"{ROOT}/models/oof_predictions.parquet",
            "configs/round5_analysis_v1.json",
            f"{ROOT}/models/manifest.json",
            "reports/round4_analysis/models/cv_comparisons.csv",
        ],
        "coverage": [f"{ROOT}/audit/QA.json", f"{ROOT}/audit/gain_by_region.csv"],
        "selection": [
            f"{ROOT}/audit/annual_features.csv",
            f"{ROOT}/recovered/recovered_municipality_profiles.csv",
        ],
        "readout": [f"{ROOT}/models/oof_predictions.parquet", f"{ROOT}/readout/QA.json"],
        "geography": [
            f"{ROOT}/context/geography_strata.csv",
            f"{ROOT}/context/regional_code_profile.csv",
            "reports/data_round5/geography/sme_code_presence.parquet",
        ],
        "workers": [
            f"{ROOT}/context/geography_legal_workers.csv",
            "reports/data_round4/sme/manifest.json",
            "reports/data_round5/sme/README.md",
            "reports/data_round5/sme/generator_code_evidence.csv",
        ],
        "stocks": ["reports/data_round5/sme/official_stock_counts.csv"],
        "cases": [
            f"{ROOT}/context/chersky_timeline.csv",
            f"{ROOT}/context/chersky_investment_alignment.csv",
            "reports/data_round5/cases/evidence_ledger.csv",
        ],
    }
    for item in sources:
        item["query"]["tables_used"] += extras[item["id"]]
    blocks, tables, charts, datasets, markdown = [], [], [], {}, []

    def prose(key, body, source_id=None):
        block = {"id": key, "type": "markdown", "body": body}
        if source_id:
            block["sourceId"] = source_id
        blocks.append(block)
        markdown.append(body)

    def table(key, title, rows, columns, source_id):
        datasets[key] = rows
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
                        "type": "text" if isinstance(rows[0][k], str) else "number",
                    }
                    for k, v in columns.items()
                ],
            }
        )
        blocks.append({"id": f"block_{key}", "type": "table", "tableId": key, "layout": "full"})
        markdown.append(table_md(rows, columns))

    def chart(key, title, rows, x, y, series, unit, source_id, question):
        datasets[key] = rows
        charts.append(
            {
                "id": key,
                "title": title,
                "type": "bar",
                "dataset": key,
                "sourceId": source_id,
                "encodings": {
                    "x": {"field": x, "type": "nominal"},
                    "y": {"field": y, "type": "quantitative", "label": unit},
                    "color": {"field": series, "type": "nominal"},
                },
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "intent": "comparison",
                "question": question,
                "rationale": "Дискретные группы и два года; временная траектория не заявляется. "
                "Точные значения и ограничения рядом.",
            }
        )
        blocks.append({"id": f"block_{key}", "type": "chart", "chartId": key, "layout": "full"})

    prose("title", f"# {TITLE}\n\nКейс 1 СберИндекса · пятый раунд анализа · 16 сентября 2026 года")
    prose(
        "summary",
        "## Данные стали полнее, решение об основной модели прежнее\n\n"
        "**64 новых значения улучшили описание 32 муниципалитетов, но не дали "
        "устойчивого выигрыша основной модели потребительского профиля.** "
        "Внутри регионов снижение ошибки составляет +0.0585% / +0.0933%, на целых "
        "регионах — −0.5234% / −0.4533% за 2023 / 2024 годы. Все четыре условных 95% "
        "интервала включают ноль. Денежный блок остаётся описательным контекстом.\n\n"
        "**Пропуски были содержательно неслучайными.** У восстановленных МО медианная "
        "отгрузка или розница на человека примерно вдвое выше, чем у ранее "
        "наблюдавшихся МО того же региона. Это помогает понять прежнюю неполноту "
        "данных, но не измеряет причинный эффект.\n\n"
        "**Грубая география МСП сосредоточена в федеральных городах.** Они дают более "
        "96% строк с кодом уровня субъекта. Поэтому общую потерю около 20 процентных "
        "пунктов детализации нельзя одинаково переносить на каждый регион. "
        "Историческое муниципальное присоединение по-прежнему не подтверждено.",
    )
    prose(
        "definitions",
        "## Сравниваем те же муниципалитеты и тот же потребительский профиль\n\n"
        "Единица анализа — муниципалитет за год, с одинаковым весом каждого МО: 1975 "
        "наблюдений в 2023 году и 1998 в 2024-м. Основная цель — средний месячный "
        "профиль пяти опубликованных категорий расходов, нормированный по сумме и "
        "выраженный в CLR-координатах: логарифмах компонент относительно их "
        "геометрического среднего. Он не равен полному бюджету домохозяйства.\n\n"
        "Метрика — средняя квадратичная ошибка (MSE) по пяти координатам на "
        "наблюдениях вне соответствующей обучающей части. Снижение ошибки "
        "рассчитывается как 1 − MSE новой модели / MSE базы; положительное значение "
        "лучше. Вторичная цель — логарифм среднего месячного номинального уровня "
        "расходов. Внутри регионов удерживаются отдельные МО, во втором режиме целые "
        "регионы не участвуют в обучении. Оба года уже исследовались: это "
        "ретроспективный исследовательский раунд, а не новый независимый тест.",
        "models",
    )
    cov = [
        {
            "year": str(r.year),
            "n": int(r.denominator),
            "old": int(r.old_all3),
            "new": int(r.new_all3),
            "old_pct": round(100 * r.old_all3 / r.denominator, 2),
            "new_pct": round(100 * r.new_all3 / r.denominator, 2),
        }
        for r in coverage.loc[coverage["sample"].eq("core")].itertuples()
    ]
    prose(
        "coverage",
        "## Ещё 32 МО в каждом году получили все три денежных атрибута\n\n"
        "Все 64 новых числа дали допустимые логарифмические признаки: 42 значения "
        "отгрузки Пермского края и 22 значения розницы Приморья. Число МО со всеми "
        "тремя признаками выросло с 1768 до 1800 в 2023 году и с 1814 до 1846 в "
        "2024-м. Полная сетка содержит 11 672 числовые ячейки из 12 096; остаётся 424 "
        "пропуска. Прежние значения и знаменатели сохранены.\n\n"
        "Дополнения относятся к двум регионам и выбранным типам территорий из формы "
        "Росстата. Состав основной и консервативной выборок не расширялся; прежние "
        "ограничения смены кодов и календарных случаев остаются.",
        "coverage",
    )
    table(
        "coverage_table",
        "Полнота трёх признаков",
        cov,
        {
            "year": "Год",
            "n": "Всего МО",
            "old": "Полных прежде",
            "new": "Полных теперь",
            "old_pct": "Прежде, %",
            "new_pct": "Теперь, %",
        },
        "coverage",
    )
    selected = [
        {
            "year": str(r.year),
            "region": r.region_name,
            "indicator": "Отгрузка" if r.prefix == "shipments" else "Розница",
            "new_n": int(r.recovered_n),
            "old_n": int(r.old_observed_n),
            "new_median": round(r.recovered_median_rubles_per_person / 1000, 1),
            "old_median": round(r.previously_observed_median_rubles_per_person / 1000, 1),
            "ratio": round(r.median_ratio, 3),
        }
        for r in selection.itertuples()
    ]
    prose(
        "selection",
        "## Восстановленные МО имели примерно вдвое большую интенсивность деятельности\n\n"
        "В Пермском крае медианная отгрузка на человека у восстановленных МО "
        "составляет 2.10 / 2.27 от медианы ранее наблюдавшихся МО; в Приморье "
        "отношение по рознице — 2.04 / 2.18 за 2023 / 2024 годы. График сравнивает "
        "медианы двух групп в одном регионе и году. Он показывает перекос прежнего "
        "покрытия, а не рост деятельности.\n\n"
        "Каждый МО имеет одинаковый вес. Это номинальная интенсивность опубликованной "
        "деятельности без малого предпринимательства, делённая на население на 1 "
        "января; она не является доходом жителей или полной экономикой территории. У "
        "восстановленных МО также больше медианное население. Разница между группами "
        "не очищена от их экономического состава.",
        "selection",
    )
    chart(
        "selection_chart",
        "Отношение медиан восстановленных и прежних МО",
        selected,
        "region",
        "ratio",
        "year",
        "Отношение медиан, раз",
        "selection",
        "Какие территории были недопредставлены из-за пропусков?",
    )
    table(
        "selection_table",
        "Интенсивность деятельности в двух группах",
        selected,
        {
            "year": "Год",
            "region": "Регион",
            "indicator": "Показатель",
            "new_n": "Новых МО",
            "old_n": "Прежних МО",
            "new_median": "Новые, тыс. руб./чел.",
            "old_median": "Прежние, тыс. руб./чел.",
            "ratio": "Отношение медиан",
        },
        "selection",
    )
    prose(
        "primary",
        "## Добавочная информация денежного блока остаётся неопределённой\n\n"
        "Основное сравнение добавляет числовые значения к модели, где уже есть прежние "
        "экономические характеристики, отраслевые доли и актуальные флаги пропусков. "
        "Это проверка всего денежного блока сверх известной доступности данных. На "
        "графике показаны точечные оценки; таблица рядом содержит условные интервалы и "
        "абсолютные ошибки.\n\n"
        "Ни одна из четырёх основных оценок не достигает заранее заданного порога 1%; "
        "все интервалы включают ноль. На выборке без смены кода все четыре точечные "
        "оценки отрицательны. Дополнительные разбиения внутри регионов также не "
        "снимают неопределённость. Оснований менять основную спецификацию модели и "
        "сети не появилось.",
        "models",
    )
    chart(
        "primary_chart",
        "Снижение ошибки профиля от денежного блока",
        headline,
        "regime",
        "gain",
        "year",
        "Снижение MSE, %",
        "models",
        "Проходит ли блок прежний критерий принятия?",
    )
    table(
        "primary_table",
        "Основные оценки и условные интервалы",
        headline,
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
    sensitivity = model_rows(
        profile.loc[
            profile.comparison.eq("primary") & profile.seed.eq(13) & profile["sample"].isin(SAMPLES)
        ]
    )
    table(
        "sensitivity_table",
        "Проверка на выборке без смены кода",
        sensitivity,
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
    update_rows = []
    for name, label in [
        ("state_update", "Полное обновление"),
        ("shared_flags_recovery", "Числа при общих флагах"),
    ]:
        update_rows += [
            {**r, "comparison": label} for r in model_rows(core.loc[core.comparison.eq(name)])
        ]
    prose(
        "update",
        "## Улучшение новой версии нельзя целиком приписать восстановленным числам\n\n"
        "При удержании целых регионов полное обновление модели относительно четвёртого "
        "раунда снижает ошибку профиля на 1.3778% / 0.5527%. Но при общих старых и "
        "новых флагах пропусков замена чисел даёт только 0.0035% / 0.0012%. Все эти "
        "интервалы включают ноль. График сопоставляет две разные проверки; их разность "
        "не является причинной оценкой вклада флагов.\n\n"
        "Полное обновление одновременно меняет значения, кодирование пропусков и "
        "обученную модель. Проверка при общих флагах сохраняет обе версии флагов и "
        "одинаковые разбиения, но заново оценивает подготовку признаков и "
        "коэффициенты. Поэтому даже она описывает конкретный вычислительный вариант, а "
        "не чистый экономический эффект.",
        "models",
    )
    chart(
        "update_chart",
        "Две оценки обновления на целых регионах",
        [r for r in update_rows if r["regime"] == "Целые регионы"],
        "comparison",
        "gain",
        "year",
        "Снижение MSE, %",
        "models",
        "Отличается ли замена чисел от полного обновления модели?",
    )
    table(
        "update_table",
        "Полное обновление и общие флаги",
        update_rows,
        {
            "year": "Год",
            "regime": "Проверка",
            "comparison": "Сравнение",
            "gain": "Снижение, %",
            "ci": "95% интервал, %",
        },
        "models",
    )
    slice_rows = [
        {
            "year": str(r.year),
            "group": "Получили данные" if str(r.group) == "True" else "Остальные",
            "n": int(r.n),
            "regions": int(r.regions),
            "local": round(100 * r.local_relative_reduction, 3),
            "contribution": round(100 * r.contribution_to_total_relative_reduction, 4),
        }
        for r in readout.loc[
            readout.target.eq("profile_clr")
            & readout.slice.eq("recovered_any")
            & readout.regime.eq("heldout_regions")
        ].itertuples()
    ]
    prose(
        "slices",
        "## Сильное локальное изменение относится только к 32 МО\n\n"
        "На целых регионах ошибка полной обновлённой модели для 32 МО с дополнениями "
        "снизилась на 41.443% / 46.697%. Их вклад в общий относительный выигрыш — "
        "0.6906 / 0.7532 процентного пункта. У остальных МО вклад составляет +0.6872 / "
        "−0.2005 пункта. Эти вклады суммируются в общий результат, но локальные "
        "проценты между собой складывать нельзя.\n\n"
        "Это описательный срез только двух региональных групп, для которого отдельный "
        "доверительный интервал не рассчитывается. Неизменённые строки тоже могут "
        "получить другие прогнозы после обновления обучающих строк. Переобучение без "
        "обоих регионов дало ровно нулевое различие старой и новой версий: это "
        "ожидаемая проверка неизменности входов, а не независимое подтверждение пользы "
        "дополнений.",
        "readout",
    )
    table(
        "slice_table",
        "Срезы полного обновления на целых регионах",
        slice_rows,
        {
            "year": "Год",
            "group": "Группа",
            "n": "МО",
            "regions": "Регионов",
            "local": "Локальное снижение, %",
            "contribution": "Вклад в общий выигрыш, п.п.",
        },
        "readout",
    )
    georows = []
    for r in geo.loc[geo.source_region_scope.eq("ALL")].itertuples():
        group = r.federal_city_group_rows + r.other_region_group_rows
        for label, count, share in [
            ("Код ниже уровня субъекта", r.subregional_code_rows, r.subregional_code_share_pct),
            (
                "Субъект: федеральные города",
                r.federal_city_group_rows,
                r.federal_city_group_share_pct,
            ),
            ("Субъект: остальные", r.other_region_group_rows, r.other_region_group_share_pct),
            ("Нет точного кода", r.unmatched_rows, r.unmatched_share_pct),
        ]:
            georows.append(
                {
                    "year": str(r.year),
                    "kind": label,
                    "rows": int(count),
                    "denominator": int(r.source_rows),
                    "share": round(share, 4),
                    "federal_share_of_grouped": round(100 * r.federal_city_group_rows / group, 3),
                }
            )
    prose(
        "geo",
        "## Почти вся грубая география МСП приходится на федеральные города\n\n"
        "Использованы годовые файлы tochno v20260605 и версия ОКТМО с меткой 01.06.2026. "
        "Из строк с группировочным кодом субъекта федеральные города дают 97.061% / "
        "96.292% за 2023 / 2024 годы. Это 19.6701% / 19.6552% всех строк; прочие "
        "группировки уровня субъекта — только 0.5956% / 0.7570%. График разделяет "
        "точное наличие кода и его детализацию; знаменатель — 7 555 943 / 7 884 985 "
        "строк периодов, не уникальные фирмы.\n\n"
        "Код ниже уровня субъекта имеется у 79.6864% / 79.5481% всех строк. Среди "
        "строк с указанным исходным регионом, кроме 77, 78 и 92, эта доля достигает "
        "99.1900% / 98.9977%; 4 / 3 строки без кода исходного региона исключены "
        "из этого частного знаменателя. Значит, "
        "общую потерю около 20 п.п. детализации нельзя одинаково приписать всей "
        "стране. Но код населённого пункта или промежуточной группы ещё не является "
        "проверенным историческим ключом нужного МО: происхождение геокодировки и "
        "адрес на дату наблюдения остаются открытыми.",
        "geography",
    )
    chart(
        "geo_chart",
        "Детализация кодов в строках МСП",
        georows,
        "kind",
        "share",
        "year",
        "Доля строк периода, %",
        "geography",
        "Где сосредоточено отсутствие муниципальной детализации?",
    )
    table(
        "geo_table",
        "Точные знаменатели географического аудита",
        georows,
        {
            "year": "Год",
            "kind": "Уровень",
            "rows": "Строк",
            "denominator": "Всего строк",
            "share": "Доля, %",
        },
        "geography",
    )
    workers = [
        {
            "year": str(r.year),
            "kind": "ИП" if r.legal_kind == "IP" else "ЮЛ",
            "rows": int(r.source_rows),
            "known": int(r.nonnegative_employee_rows),
            "pct": round(r.nonnegative_employee_row_share_pct, 4),
            "missing": int(r.missing_employee_rows),
            "negative": int(r.negative_employee_rows),
        }
        for r in annual.loc[annual.legal_kind.ne("ALL")].itertuples()
    ]
    prose(
        "workers",
        "## Отбор по работникам исключает все записи ИП\n\n"
        "Неотрицательное значение работников есть у 79.9425% / 81.1275% строк "
        "юридических лиц. У всех 4 805 572 / 5 223 902 строк ИП поле отсутствует. "
        "Поэтому фильтр «есть работники» меняет состав исследуемой популяции, а не "
        "просто удаляет случайные пропуски. Обнаружено по одному отрицательному "
        "значению у ЮЛ в каждом году; они отдельно помечены недопустимыми.\n\n"
        "Таблица считает строки, а не работников: численность работников по повторным "
        "состояниям ИНН не суммируется, пропуски ИП не заменяются нулями. "
        "Наличие числа не устанавливает "
        "его исторический год или место работы сотрудников. Лаг текущего месячного "
        "реестра ФНС нельзя автоматически переносить на год в обработанном наборе "
        "tochno.",
        "workers",
    )
    table(
        "workers_table",
        "Наблюдаемость поля работников",
        workers,
        {
            "year": "Год",
            "kind": "Тип",
            "rows": "Строк",
            "known": "Неотрицательное значение",
            "pct": "Доля, %",
            "missing": "Пропуск",
            "negative": "Отрицательное",
        },
        "workers",
    )
    stockrows = [
        {
            "year": str(r.year),
            "date": r.fns_snapshot_date,
            "stock": int(r.fns_stock_subjects),
            "rows": int(r.tochno_period_rows),
            "ids": int(r.tochno_annual_distinct_tax_ids),
        }
        for r in stock.loc[stock.legal_kind.eq("ALL")].itertuples()
    ]
    prose(
        "stock",
        "## Численность на дату и годовые состояния — разные популяции\n\n"
        "Контроль ФНС на 10 декабря считает субъекты, учтённые в реестре на эту дату. "
        "Годовой файл содержит состояния в течение года и повторения одного ИНН. Даже "
        "число уникальных ИНН за год не обязано совпадать с численностью в реестре на эту "
        "дату. Таблица сохраняет оба определения, без расчёта «ошибки покрытия» или "
        "темпа роста из их разности. Территориальный состав источников также не "
        "гармонизирован.",
        "stocks",
    )
    table(
        "stock_table",
        "Сопоставление определений численности",
        stockrows,
        {
            "year": "Год",
            "date": "Дата ФНС",
            "stock": "ФНС на дату",
            "rows": "Годовых строк",
            "ids": "Уникальных ИНН за год",
        },
        "stocks",
    )
    budgetrows = [
        {
            "year": str(r.year_column),
            "old": str(r.old_value_as_published),
            "new": str(r.new_value_as_published),
            "delta": "—"
            if pd.isna(r.revision_delta_million_rub)
            else float(r.revision_delta_million_rub),
        }
        for r in budget.itertuples()
    ]
    prose(
        "cases",
        "## Документы уточняют инвестиционную гипотезу, но не раскладывают пик по объектам\n\n"
        "Ввод аэропорта Черский, создание ТЛЦ и ввод амбулатории Андрюшкино отнесены "
        "источниками к 2023 году — году прежнего пика 1329.7 млн руб. в Нижнеколымском "
        "районе. Совпадение периода делает проекты кандидатами для предметной "
        "проверки, но не определяет их вклад в агрегат. Публикации и годы событий "
        "отдельно сохранены в хронологии.\n\n"
        "В двух редакциях правительственного приложения величина удорожания "
        "аэропортовых контрактов по колонке 2023 года уменьшилась с 134642.6 до 126838 "
        "тыс. руб., то есть на 7.8046 млн руб. Это пересмотр бюджетного значения между "
        "редакциями от 16.12.2022 и 21.04.2023, а не изменение фактических годовых "
        "инвестиций. Прочерк 2024 года не превращён в ноль. Доля аэропорта в "
        "муниципальном пике не рассчитывается.",
        "cases",
    )
    table(
        "budget_table",
        "Две редакции бюджетного удорожания аэропорта",
        budgetrows,
        {
            "year": "Колонка года",
            "old": "16.12.2022, тыс. руб.",
            "new": "21.04.2023, тыс. руб.",
            "delta": "Разность редакций, млн руб.",
        },
        "cases",
    )
    secondary = model_rows(
        cv.loc[
            cv.target.eq("log_total")
            & cv["sample"].eq("core")
            & cv.seed.eq(13)
            & cv.regime.eq("heldout_regions")
            & cv.comparison.eq("primary")
        ]
    )
    prose(
        "secondary",
        "## Номинальный уровень расходов остаётся вторичной задачей\n\n"
        "Для логарифма номинального уровня расходов весь денежный блок даёт "
        "положительные оценки на целых регионах. Эта вторичная цель не заменяет "
        "основной потребительский профиль и не меняет правило принятия. В предыдущем "
        "раунде сигнал сохранялся после перестановок внутри регионов; в этом раунде "
        "такие проверки заново не выполнялись, поэтому нового доказательства "
        "муниципальной специфичности нет.\n\n"
        "В дополнительном сравнении чисел при общих флагах на целых регионах интервал 2023 "
        "года для этой вторичной цели исключает ноль, а интервал 2024 года включает "
        "его. Результаты исследовательские и не скорректированы на множественные "
        "сравнения. Таблица ниже показывает весь денежный блок сверх актуальных флагов.",
        "models",
    )
    table(
        "secondary_table",
        "Весь денежный блок сверх актуальных флагов: номинальный уровень",
        secondary,
        {"year": "Год", "n": "МО", "gain": "Снижение MSE, %", "ci": "95% интервал, %"},
        "models",
    )
    prose(
        "methods",
        "## Протокол сохраняет разбиения и ограничивает силу вывода\n\n"
        "Использована Ridge-регрессия с alpha=1. Медианы, масштабирование и словарь "
        "категорий оценены только по обучающей части. Folds перенесены по точным "
        "ключам из четвёртого раунда. Основной seed — 13, дополнительные внутри "
        "регионов — 29 и 47. Выполнено 640 обучений; прежние прогнозы на основной и "
        "консервативной выборках совпали точно.\n\n"
        "95% интервалы построены по 2000 парным перевыборкам регионов из сохранённых "
        "ошибок. Они условны по обученным прогнозам, не включают повторное обучение "
        "всей процедуры и не скорректированы на множественные проверки. На прежних "
        "полных наблюдениях и после удаления двух регионов восстановления старые и "
        "новые варианты совпадают. Эти проверки подтверждают исполнение протокола, а "
        "не экономическую причинность.\n\n"
        "Перестановки, абляции и ещё одна расширенная полная выборка не запускались "
        "заново: признаки и алгоритм прежние, вопрос ограничен 64 дополнениями. "
        "Критерий принятия требует не менее 1% выигрыша по профилю в обоих годах и "
        "режимах, положительной нижней границы и отсутствия разворота на стабильных "
        "кодах. Техническое уточнение «без разворота» как >=0 вместо >0 не меняет и "
        "прежнее решение. Источники получены в 2026 году; доступность в исторический "
        "момент и неизменность юридических границ не доказаны.",
        "models",
    )
    prose(
        "decision",
        "## Используем дополнения для описания МО, МСП готовим отдельно\n\n"
        "**Основную модель и сеть оставляем в прежней спецификации.** Более полные "
        "денежные показатели полезны для экономических карточек территорий; их "
        "применение не означает, что выделенные группы стали точнее.\n\n"
        "- Для карточек МО сохраняем явные единицы, охват без малого "
        "предпринимательства, год и население в знаменателе.\n"
        "- Для МСП отдельно проверяем историческую адресную привязку вне федеральных "
        "городов и внутри них. Высокое наличие кода ниже субъекта не снимает этой "
        "задачи.\n"
        "- Показатель работников строим только после подтверждения года и источника; "
        "отсутствие работников у ИП учитываем как ограничение состава.\n"
        "- Для Черского нужны фактическое финансовое исполнение по объектам и правила "
        "территориального учёта, прежде чем распределять муниципальный инвестиционный "
        "пик.\n\n"
        "Открытые вопросы: удастся ли подтвердить исторические адреса МСП; какие "
        "популяции сопоставимы с нашей панелью; какие расходы по проектам фактически "
        "учтены в муниципальном агрегате. Новый раунд уточняет эти условия, но не "
        "подменяет недостающие свидетельства расчётами.",
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
    markdown.append(
        "[Воспроизведение](REPRODUCE.md) · [Независимая "
        "проверка](independent_review.md) · [Контекст и ограничения "
        "источников](context/README.md)."
    )
    (output / "REPORT.md").write_text("\n\n".join(markdown) + "\n", "utf-8")
    pd.DataFrame(headline).to_csv(output / "headline_metrics.csv", index=False)
    design = {
        "audience": "technical",
        "delivery": "portable HTML",
        "reason": "Continuation of the repository-based research report in Codex",
        "adoption_passed": False,
        "structure_mapping": (
            "Summary; definitions before evidence; coverage and selection; model findings; "
            "update diagnostics; SME/context; dedicated methods and limitations; actions "
            "and open questions"
        ),
        "chart_map": [
            {
                "id": c["id"],
                "question": c["question"],
                "family": "discrete grouped comparison",
                "palette": "shared native categorical palette; year is the only color dimension",
                "surface": "native artifact bar",
                "row_count": len(datasets[c["dataset"]]),
            }
            for c in charts
        ],
        "repeated_chart_family_reason": (
            "All charts compare discrete groups and two separately assessed years; no "
            "temporal trajectory or unsupported confidence whiskers implied"
        ),
        "tables_without_charts": (
            "Coverage is a small exact lookup; workers and population definitions need "
            "precise denominators; budget revision includes a nonnumeric dash. No invented "
            "trend points."
        ),
        "snapshot_limits": (
            "Conditional retrospective exploratory evidence, no new holdout or causal claim"
        ),
    }
    (output / "report_design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({"blocks": len(blocks), "charts": len(charts), "tables": len(tables)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    build(args.repo, args.root or args.repo / ROOT, args.out or args.repo / ROOT)
