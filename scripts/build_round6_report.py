"""Canonical sixth-analysis narrative and portable Data Analytics payload."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from build_round2_report import source, table_md

ROOT = "reports/round6_analysis"
TITLE = "Новые данные: уровень расходов и проверка групп"
STAMP = "2026-09-16T20:58:19Z"
REGIONS = {26: "Ставрополье", 64: "Саратовская область"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repo, root, out):
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "regions": "regions/associations.csv",
        "coverage": "regions/coverage.csv",
        "panel": "regions/regional_panel.csv",
        "clusters": "regions/cluster_alignment.csv",
        "cash": "context/airport_comparison.csv",
        "changes": "context/airport_changes.csv",
        "queue": "context/geography_queue_summary.csv",
        "admission": "admission/ratio_summary.csv",
    }
    frames = {k: pd.read_csv(root / p) for k, p in paths.items()}
    assoc, coverage, panel = (frames[k] for k in ["regions", "coverage", "panel"])
    primary = assoc.loc[assoc["sample"].eq("paired_available")]
    level = primary.loc[primary.outcome.eq("log_total")]
    urban = assoc.loc[assoc["sample"].eq("paired_urban_complete") & assoc.outcome.eq("log_total")]
    assert len(primary) == 8 and len(level) == 4 and len(urban) == 4
    assert ((urban.urban_partial_rho_ci_low < 0) & (urban.urban_partial_rho_ci_high > 0)).all()
    scripts = {k: "analyze_round6_regions.py" for k in ["regions", "coverage", "panel", "clusters"]}
    scripts.update({k: "analyze_round6_context.py" for k in ["cash", "changes", "queue"]})
    scripts["admission"] = "analyze_round6_admission.py"
    sources = [source(k, p, f"{ROOT}/{p}", "scripts/" + scripts[k]) for k, p in paths.items()]
    for s in sources:
        s["query"]["reproduction_document"] = f"{ROOT}/REPRODUCE.md"
        s["query"]["computation_path"] = "scripts/" + scripts[s["id"]]
    sources.extend(
        [
            source(
                "stability",
                "Устойчивость годовой сети",
                "reports/enriched_analysis/annual_networks/yearly_support.csv",
                "reports/enriched_analysis/annual_networks/verify_results.py",
            ),
        ]
    )
    stability_source = next(s for s in sources if s["id"] == "stability")
    stability_source["query"]["reproduction_document"] = f"{ROOT}/REPRODUCE.md"
    stability_source["query"]["computation_path"] = (
        "reports/enriched_analysis/annual_networks/verify_results.py"
    )
    next(s for s in sources if s["id"] == "admission")["query"]["tables_used"] += [
        "reports/data_round6/economy/regional_context_candidate.csv",
        f"{ROOT}/admission/QA.json",
    ]
    blocks, tables, charts, datasets, markdown = [], [], [], {}, []

    def prose(key, body, source_id=None):
        block = {"id": key, "type": "markdown", "body": body}
        if source_id:
            block["sourceId"] = source_id
        blocks.append(block)
        markdown.append(body)

    def table(key, title, rows, columns, sid):
        datasets[key] = rows
        tables.append(
            {
                "id": key,
                "title": title,
                "dataset": key,
                "sourceId": sid,
                "defaultSort": {
                    "field": "year" if "year" in columns else next(iter(columns)),
                    "direction": "asc",
                },
                "columns": [
                    {
                        "field": k,
                        "label": label,
                        # Exact lookup: the native numeric formatter rounds small p to zero.
                        # Keep numeric dataset values, render their reviewed decimal precision.
                        "type": "number" if isinstance(rows[0][k], int) else "text",
                    }
                    for k, label in columns.items()
                ],
            }
        )
        blocks.append({"id": "block_" + key, "type": "table", "tableId": key, "layout": "full"})
        markdown.append(table_md(rows, columns))

    def bar(key, title, rows, x, y, group, unit, sid):
        datasets[key] = rows
        charts.append(
            {
                "id": key,
                "title": title,
                "type": "bar",
                "dataset": key,
                "sourceId": sid,
                "encodings": {
                    "x": {
                        "field": x,
                        "type": "nominal",
                        "label": "Год" if x == "year" else "Регион и показатель",
                    },
                    "y": {"field": y, "type": "quantitative", "label": unit},
                    "color": {"field": group, "type": "nominal"},
                },
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "intent": "comparison",
                "question": title,
                "rationale": "Сравнение отдельных групп/лет, не непрерывный временной тренд.",
            }
        )
        blocks.append({"id": "block_" + key, "type": "chart", "chartId": key, "layout": "full"})

    def ci(row, prefix):
        return f"[{getattr(row, prefix + '_ci_low'):.3f}; {getattr(row, prefix + '_ci_high'):.3f}]"

    prose(
        "title", f"# {TITLE}\n\nКейс 1 СберИндекса · шестой раунд анализа · данные 2023–2024 годов"
    )
    prose(
        "summary",
        "## Краткие выводы · Executive Summary\n\n"
        "- **Новые показатели связаны с уровнем расходов.** В двух регионах ранговая связь "
        "положительна в обоих годах и сохраняет знак при исключении любого одного МО. "
        "Простой связи с долей питания в пяти категориях не обнаружено.\n"
        "- **Состав выборки и урбанизация ограничивают вывод.** На подвыборках с полными "
        "данными об урбанизации после её учёта вместе с населением все интервалы связи с уровнем "
        "расходов включают ноль. Это не доказывает отсутствия связи.\n"
        "- **Касса аэропорта не является долей муниципальных инвестиций.** В 2022 году "
        "она больше муниципального итога, а в 2023 году ряды меняются в разные стороны. "
        "Вклад аэропорта в пик остаётся неустановленным.\n"
        "- **Сеть и основную панель сохраняем.** Дополнительные суммы полезны как "
        "контекст; устойчивые экономические типы и исторические адреса МСП ими не доказаны.",
    )
    prose(
        "definitions",
        "## Сравниваем одинаковые территории внутри каждого региона\n\n"
        "В основной проверке — **33 МО Ставрополья и 38 МО Саратовской области в каждом "
        "году**, с одинаковым весом. Для Ставрополья берём отгрузку по хозяйственным "
        "видам деятельности без малого бизнеса; для Саратовской области — реализацию "
        "продовольственных товаров без скрытой деятельности. Годовые суммы делим на "
        "население на начало года. Это интенсивность деятельности территории, не доход "
        "жителей и не точный аналог потребительского показателя.\n\n"
        "Уровень расходов — средняя месячная сумма пяти опубликованных категорий СберИндекса. "
        "Доля питания — средняя месячная доля среди этих пяти категорий; это не весь бюджет "
        "домохозяйства. Ранговая корреляция оценивает совпадение порядка территорий: "
        "от −1 до +1. Она не измеряет, на сколько процентов вырастут расходы.\n\n"
        "Один саратовский МО исключён из обоих годов из-за отсутствия населения в 2024-м. "
        "Прежняя модельная маска требовала также зарплату и работников; для данного "
        "описательного вопроса эти ограничения не нужны. Соединение новых публикаций "
        "остаётся условным: точное название, регион и год не доказывают исторические границы.",
        "coverage",
    )
    covrows = []
    for region in [26, 64]:
        c = coverage.loc[coverage.region_code.eq(region) & coverage.year.eq(2023)].set_index(
            "sample"
        )
        covrows.append(
            {
                "region": REGIONS[region],
                "paired": int(c.loc["paired_available", "n"]),
                "stable": int(c.loc["paired_stable_code", "n"]),
                "urban": int(c.loc["paired_urban_complete", "n"]),
                "stable_urban": int(c.loc["paired_stable_urban_complete", "n"]),
            }
        )
    table(
        "coverage",
        "Парные выборки: одно и то же число МО в каждом году",
        covrows,
        {
            "region": "Регион",
            "paired": "Основная",
            "stable": "Без смены кода",
            "urban": "Полные данные об урбанизации",
            "stable_urban": "Оба ограничения",
        },
        "coverage",
    )
    prose(
        "level",
        "## Связь с уровнем расходов сильнее, чем с долей питания\n\n"
        "Чем выше отгрузка или продовольственная торговля на жителя, тем выше обычно "
        "наблюдаемый уровень расходов. После учёта численности населения связь остаётся "
        "положительной в обоих годах. График сравнивает коэффициенты по двум целям; "
        "возле нуля находятся связи с долей питания. Следовательно, новый контекст лучше "
        "характеризует уровень активности, чем именно эту компоненту структуры расходов. "
        "Это не проверка всех возможных нелинейных связей или качества прогноза.",
        "regions",
    )
    corrrows = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "measure": REGIONS[r.region_code]
            + (": уровень" if r.outcome == "log_total" else ": доля питания"),
            "rho": round(r.rho, 4),
            "n": int(r.n),
            "interval": ci(r, "rho"),
        }
        for r in primary.itertuples()
    ]
    bar(
        "primary_correlations",
        "Ранговая связь с расходами и долей питания",
        corrrows,
        "measure",
        "rho",
        "year",
        "Корреляция Спирмена",
        "regions",
    )
    levelrows = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "rho": round(r.rho, 3),
            "rho_ci": ci(r, "rho"),
            "partial": round(r.partial_rho, 3),
            "partial_ci": ci(r, "partial_rho"),
        }
        for r in level.itertuples()
    ]
    table(
        "level_coefficients",
        "Связь с уровнем расходов и условные 95% интервалы",
        levelrows,
        {
            "region": "Регион",
            "year": "Год",
            "n": "МО",
            "rho": "Исходная",
            "rho_ci": "95% интервал",
            "partial": "С учётом населения",
            "partial_ci": "95% интервал",
        },
        "regions",
    )
    prose(
        "influence",
        "## Положительный знак не объясняется одним крупнейшим городом\n\n"
        "Удаление любого одного МО сохраняет положительный знак исходной связи с "
        "уровнем расходов: диапазоны коэффициентов по четырём проверкам лежат между "
        "0.518 и 0.788. Отдельно удалены крупнейшие по населению 2023 года территории "
        "с теми же идентификаторами в обоих годах. Это проверка влияния одиночного "
        "наблюдения, а не пространственной зависимости или репрезентативности двух регионов.\n\n"
        "На диаграмме каждая точка — один из 38 саратовских МО в 2024 году. Оси имеют "
        "разные временные определения: годовая продовольственная торговля на жителя и "
        "средний месячный уровень пяти категорий расходов. Их абсолютные значения "
        "нельзя приравнивать.",
        "regions",
    )
    scatter = panel.loc[panel.region_code.eq(64) & panel.year.eq(2024) & panel.paired_available]
    datasets["saratov_scatter"] = [
        {
            "municipality": r.panel_name,
            "territory_id": int(r.territory_id),
            "food_thousand_per_person": round(r.rubles_per_person / 1000, 4),
            "monthly_spending_thousand": round(r.total_mean / 1000, 4),
            "population": int(r.population_total),
        }
        for r in scatter.itertuples()
    ]
    charts.append(
        {
            "id": "saratov_scatter",
            "title": "Саратовская область, 2024: 38 муниципалитетов",
            "type": "scatter",
            "dataset": "saratov_scatter",
            "sourceId": "panel",
            "encodings": {
                "x": {
                    "field": "food_thousand_per_person",
                    "type": "quantitative",
                    "label": "Продовольственная торговля, тыс. руб. за год на жителя",
                },
                "y": {
                    "field": "monthly_spending_thousand",
                    "type": "quantitative",
                    "label": "Расходы, тыс. руб. в месяц",
                },
            },
            "intent": "relationship",
            "question": "Как выглядят уровни внутри региональной выборки?",
            "rationale": (
                "38 сопоставимых точек, одна территория на точку, идентификаторы сохранены."
            ),
        }
    )
    blocks.append(
        {
            "id": "block_saratov_scatter",
            "type": "chart",
            "chartId": "saratov_scatter",
            "layout": "full",
        }
    )
    prose(
        "urban",
        "## Урбанизация и пропуски не позволяют выделить самостоятельную связь\n\n"
        "Для учёта урбанизации доступны лишь 19 ставропольских и 32 саратовских МО. "
        "В Ставрополье исходная корреляция уже на этой подвыборке падает до 0.060 / 0.102; "
        "это изменение состава, которое нельзя назвать эффектом контроля урбанизации. "
        "На тех же строках сравниваем последовательно исходную связь, контроль населения "
        "и контроль населения вместе с урбанизацией.\n\n"
        "Во всех четырёх последних оценках интервал включает ноль. Поэтому самостоятельная "
        "связь после этих контролей не подтверждена имеющимися данными. Малая и отобранная "
        "по полноте выборка также не позволяет утверждать, что связь отсутствует. "
        "Ставропольская подвыборка одновременно без смены кода и с полными данными об урбанизации "
        "содержит 9 МО: она не оценивалась при заранее заданном минимуме 12.",
        "regions",
    )
    urbanrows = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "raw": round(r.rho, 3),
            "pop": round(r.partial_rho, 3),
            "urban": round(r.urban_partial_rho, 3),
            "interval": ci(r, "urban_partial_rho"),
        }
        for r in urban.itertuples()
    ]
    table(
        "urban_comparison",
        "Оценки на одинаковых строках с полными данными об урбанизации",
        urbanrows,
        {
            "region": "Регион",
            "year": "Год",
            "n": "МО",
            "raw": "Исходная",
            "pop": "Население",
            "urban": "Население + урбанизация",
            "interval": "95% интервал последней оценки",
        },
        "regions",
    )
    prose(
        "clusters",
        "## Различия между группами не доказывают устойчивые экономические типы\n\n"
        "В замороженных годовых группах различия продовольственной торговли Саратовской "
        "области заметны в обоих годах относительно условной случайной перестановки меток. "
        "Для отгрузки Ставрополья такое различие есть в 2023 году и не подтверждается в "
        "2024-м. Статистика ниже — доля вариации рангов нового показателя между группами "
        "внутри региона; это не качество прогноза и не устойчивость членства.\n\n"
        "Поправка BH охватывает четыре исследовательских сравнения. Перестановки предполагают "
        "обменность МО и не учитывают пространственную зависимость; q не даёт гарантии "
        "контроля ошибок при этой зависимости. Метки годов не совмещались и не трактуются "
        "как одни и те же типы.",
        "clusters",
    )
    clusterrows = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "groups": int(r.clusters),
            "eta": round(r.rank_eta_squared, 3),
            "null": round(r.null_mean, 3),
            "p": round(r.permutation_p, 4),
            "q": round(r.permutation_q_bh, 4),
        }
        for r in frames["clusters"].itertuples()
    ]
    table(
        "cluster_alignment",
        "Соответствие нового показателя замороженным годовым группам",
        clusterrows,
        {
            "region": "Регион",
            "year": "Год",
            "n": "МО",
            "groups": "Групп в регионе",
            "eta": "Доля вариации рангов",
            "null": "Среднее при перестановках",
            "p": "p",
            "q": "q BH",
        },
        "clusters",
    )
    prose(
        "stability",
        "Прежняя проверка годовой сети не обнаружила сильных назначений "
        "муниципалитетов в группы ни в одном году: 0 из 2016 в 2023-м и 0 из 2016 в "
        "2024-м. Новая ассоциация с внешним показателем не устраняет эту неустойчивость. "
        "Поэтому экономические названия устойчивых типов на основании данного раунда не вводим.",
        "stability",
    )
    prose(
        "admission",
        "## Определения показателей важнее простого прироста числа ячеек\n\n"
        "Сохранены 276 новых значений, но ни одно не подставлено в 424 прежних пропуска. "
        "Продовольственная торговля и продажа товаров несобственного производства не "
        "равны прежнему показателю розницы. Отгрузка по чистым видам деятельности отличается "
        "от хозяйственной по охвату; два столбца не заменяют друг друга. Оперативные "
        "выпуски не гарантируют совпадение с текущей редакцией БДПМО.\n\n"
        "Прямого пересечения с наблюдаемыми значениями того же показателя БДПМО здесь "
        "нет: другие показатели той же территории не являются независимой сверкой. "
        "Печатные проценты источника проверены отдельно против отношений номинальных "
        "сумм двух выпусков с учётом округления. Несовпадение не объясняется автоматически "
        "дефляцией или пересмотром: причина требует источника. Подробный допуск и проверки "
        "сохранены в отдельной таблице исследования.",
        "admission",
    )
    ratio_names = {
        "shipments_economic_activity": "Отгрузка: хозяйственные виды",
        "shipments_pure_activity": "Отгрузка: чистые виды",
        "resale_non_own_goods": "Товары несобственного производства",
    }
    ratiorows = [
        {
            "variable": ratio_names[r.variable],
            "pairs": int(r.pairs),
            "compatible": int(r.rounding_compatible),
            "different": int(r.different_beyond_rounding),
            "maximum": round(r.max_absolute_difference_pp, 3),
        }
        for r in frames["admission"].itertuples()
    ]
    table(
        "source_percentages",
        "Проценты в выпуске 2024 и отношения сумм двух выпусков",
        ratiorows,
        {
            "variable": "Показатель",
            "pairs": "Пар МО",
            "compatible": "Совместимы с округлением",
            "different": "Различаются",
            "maximum": "Макс. разница, п.п.",
        },
        "admission",
    )
    prose(
        "source_percentages_readout",
        "Для хозяйственной отгрузки все 33 пары совместимы "
        "с печатным округлением. Для чистой отгрузки — лишь 10 из 33, для перепродажи — "
        "31 из 33. Это аргумент сохранять определения столбцов и редакции выпусков. "
        "Разность хозяйственной и чистой отгрузки нельзя приписывать только организациям "
        "до 15 работников: различается также способ группировки деятельности.",
        "admission",
    )
    prose(
        "cash",
        "## Касса аэропорта не воспроизводит инвестиционный пик Нижнеколымского района\n\n"
        "В 2022 году федеральная касса аэропорта составила 939.908 млн рублей при "
        "муниципальных инвестициях 31.8 млн рублей: отношение разных показателей — "
        "29.557 раза. В 2023-м касса снизилась на 16.104%, а муниципальный ряд вырос "
        "с 31.8 до 1329.7 млн рублей. Это исключает прямую подстановку кассы вместо "
        "статистических инвестиций, но не исключает вклад проекта при ином порядке учёта.\n\n"
        "График показывает два самостоятельных номинальных ряда, а не части одного "
        "итога. Отношение за 2023 год 0.593 нельзя объявить долей аэропорта. Разность "
        "рядов не распределяется по остальным проектам. За 2024 год касса отсутствует; "
        "муниципальные 24.8 млн рублей не восполняют этот пропуск.",
        "cash",
    )
    cashrows = []
    cashframe = frames["cash"]
    for r in cashframe.loc[cashframe.year.isin([2022, 2023])].itertuples():
        for field, label in [
            ("airport_cash_million_rubles", "Касса аэропорта"),
            ("municipal_investment_million_rubles", "Муниципальные инвестиции"),
        ]:
            cashrows.append(
                {"year": str(r.year), "series": label, "value": float(getattr(r, field))}
            )
    bar(
        "airport_money",
        "Касса объекта и статистические инвестиции муниципалитета",
        cashrows,
        "year",
        "value",
        "series",
        "Млн рублей, разные определения",
        "cash",
    )
    prose(
        "queue",
        "## Коды МСП образуют очередь проверки, а не готовое покрытие\n\n"
        "После устранения повторного счёта по рёбрам получены 38262 строки МСП за "
        "2023 и 36724 за 2024 год. Это около 0.506% / 0.466% всех периодных строк "
        "соответствующего годового файла, а не доля уникальных предприятий или принятых "
        "исторических присоединений. Наивная сумма по рёбрам завысила бы очередь на "
        "781 / 823 строки.\n\n"
        "Несколько предшественников могут вести к одному родительскому МО-кандидату. "
        "Отсутствие нескольких родителей внутри выбранной очереди не доказывает адрес "
        "предприятия в прошлом. Принятых исторических присоединений по-прежнему 0; "
        "полученные сведения работников за 2025 год не используются в 2023–2024.",
        "queue",
    )
    queue = frames["queue"].loc[frames["queue"].code_region.eq("ALL_CANDIDATE_CODE_REGIONS")]
    queuerows = [
        {
            "year": str(r.year),
            "groups": int(r.source_code_groups),
            "rows": int(r.source_rows_once),
            "denominator": int(r.denominator_all_sme_year_rows),
            "share": round(r.share_all_sme_year_rows_pct, 3),
            "excess": int(r.naive_excess_rows),
        }
        for r in queue.itertuples()
    ]
    table(
        "queue",
        "Размер очереди проверки без повторного счёта",
        queuerows,
        {
            "year": "Год",
            "groups": "Групп кодов",
            "rows": "Строк один раз",
            "denominator": "Все строки года",
            "share": "% строк года",
            "excess": "Лишних при счёте по рёбрам",
        },
        "queue",
    )
    prose(
        "decision",
        "## Сохраняем контекст и проверяем источники перед расширением модели\n\n"
        "1. Используем новые показатели в экономических карточках территорий с явным "
        "охватом, годом, населением и условием сопоставления по названию.\n"
        "2. Для проверки самостоятельной связи с расходами прежде всего восстанавливаем "
        "урбанизацию на недостающих МО. Сокращённая выборка не заменяет полную.\n"
        "3. По Черскому запрашиваем сверку проектов, респондентов и времени статистического "
        "признания инвестиций. Касса уже найдена, но арифметика показывает несопоставимость "
        "прямой подстановки.\n"
        "4. По МСП проверяем датированные адреса и историю объектов; расширение списка "
        "кодов само по себе не расширяет подтверждённую панель.\n\n"
        "Открытые вопросы: сохранится ли связь после восстановления урбанизации; чем "
        "объясняются различия процентов и номинальных отношений выпусков; где и когда "
        "затраты аэропорта отражены в статистике; как устроено геокодирование конкретной "
        "сборки МСП. Внешние запросы владельцам источников в этом раунде не отправлялись.",
    )
    prose(
        "limits",
        "## Ограничения: два региона, условная география, повторно изученные годы\n\n"
        "Это ретроспективный исследовательский анализ, не новый независимый тест. "
        "Кластеры не переобучались, качество прогнозов не оценивалось заново, причинные "
        "эффекты не заявляются. Интервалы — 2000 перевыборок МО внутри каждой региональной "
        "подвыборки; те же идентификаторы используются в обоих годах. В каждой "
        "перевыборке заново считаются ранги и поправки. Интервалы не учитывают "
        "пространственную зависимость и множественность исследованных ассоциаций.\n\n"
        "Данные номинальные; два годовых наблюдения не дают временного причинного "
        "дизайна. Некоторые группы внутри регионов малы. Источники, опубликованные "
        "позднее отчётного года, не считаются доступными в исторический момент. "
        "Подтверждение совпадения кода или названия не равно проверке юридических границ.",
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "generatedAt": STAMP,
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {"version": 1, "status": "ready", "generatedAt": STAMP, "datasets": datasets},
        "sources": sources,
    }
    (out / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    markdown.append(
        "[Воспроизведение](REPRODUCE.md) · [Проверки](QA.md) · "
        "[Независимый отзыв](independent_review.md)."
    )
    (out / "REPORT.md").write_text("\n\n".join(markdown) + "\n", "utf-8")
    claim_values = {
        "level": levelrows,
        "urban": urbanrows,
        "clusters": clusterrows,
        "queue": queuerows,
        "coverage": covrows,
        "admission": ratiorows,
    }
    (out / "claim_values.json").write_text(
        json.dumps(claim_values, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    design = {
        "audience": "product stakeholders",
        "delivery": "portable HTML",
        "reason": "Continuation of the repository-based research report in Codex Desktop",
        "structure_mapping": (
            "Executive Summary; definitions; evidence; recommendations; questions; limitations"
        ),
        "chart_map": [
            {
                "id": c["id"],
                "family": c["type"],
                "rows": len(datasets[c["dataset"]]),
                "palette": (
                    "native restrained palette; years or financial definition are meaningful groups"
                ),
            }
            for c in charts
        ],
        "tables_without_charts": (
            "Exact coefficients with intervals, changing denominators, rank variance "
            "and code queues need exact lookup"
        ),
        "prohibited_claims": (
            "Causality, nationwide effects, stable clusters, historical SME adoption"
        ),
        "inputs": {p: digest(root / p) for p in paths.values()},
        "builder_sha256": digest(Path(__file__)),
    }
    (out / "report_design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({"blocks": len(blocks), "charts": len(charts), "tables": len(tables)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    build(args.repo.resolve(), args.root or args.repo / ROOT, args.out or args.repo / ROOT)
