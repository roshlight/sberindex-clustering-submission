"""One reviewed narrative and canonical portable artifact from saved round2 results."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

REPO_URL = "https://github.com/sberindex-hackathon-2026/municipal-economic-profiles"
TITLE = "Что новые данные добавляют к муниципальным профилям"
SECTORS = {
    "A": "Сельское хозяйство",
    "B": "Добыча",
    "C": "Обработка",
    "D": "Энергоснабжение",
    "H": "Транспорт и хранение",
    "O": "Госуправление",
    "P": "Образование",
    "Q": "Здравоохранение",
}
NAMES = {
    318: "Верхнеколымский район",
    331: "Нижнеколымский район",
    883: "Свободненский район",
    925: "Красноярский район, Астраханская область",
    1180: "Светловский округ",
    2527: "Куркино",
}


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def table_md(data, columns):
    return "\n".join(
        [
            "| " + " | ".join(columns.values()) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(str(row[k]) for k in columns) + " |" for row in data],
        ]
    )


def source(key, label, path, script):
    return {
        "id": key,
        "label": label,
        "path": path,
        "href": f"{REPO_URL}/blob/main/{path}",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": label
            + ". Чтение сохранённого научного расчёта; исходное преобразование в original_python.",
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "original_python": script,
            "tables_used": [path],
        },
    }


def build(repo):
    root = repo / "reports/round2_analysis"
    p = root / "prices/canonical"

    def read_result(relative):
        # Read materialized scientific outputs, preserving upstream code separately in provenance.
        return duckdb.read_csv(str(repo / relative)).df()

    m = read_result("reports/round2_analysis/models/cv_metrics.csv")
    c = read_result("reports/round2_analysis/models/cv_comparisons.csv")
    primary = c.loc[
        (c.baseline == "economic_missingness")
        & (c.alternative == "economic_sectors")
        & (c.target == "profile_clr")
        & (c.seed == 13)
    ].copy()
    coverage = read_result("reports/round2_analysis/figures/coverage_chart.csv")
    prices = json.loads((p / "summary.json").read_text())
    iszh = read_result("reports/round2_analysis/prices/canonical/iszh_associations.csv")
    headline_price = iszh.loc[(iszh["sample"] == "core_complete") & (iszh.target == "log_total")]
    examples = read_result("reports/round2_analysis/sector/examples.csv")
    price_context = read_result("reports/round2_analysis/prices/canonical/annual_context_panel.csv")
    six_prices = price_context.loc[price_context.territory_id.isin(NAMES)].dropna(subset=["iszh"])
    assert len(six_prices) == 2 and set(six_prices.territory_id) == {318}
    assert six_prices.iszh.eq(1.44).all() and set(six_prices.year) == {2023, 2024}
    ex24 = examples.loc[examples.year == 2024].copy()
    ex23 = examples.loc[examples.year == 2023].set_index("territory_id")
    model_rows = []
    for r in primary.itertuples():
        subset = m.loc[
            (m.year == r.year)
            & (m["sample"] == r.sample)
            & (m.regime == r.regime)
            & (m.seed == 13)
            & (m.target == "profile_clr")
        ].set_index("model")
        model_rows.append(
            {
                "year": str(r.year),
                "sample": {
                    "core": "Основная",
                    "stable_code": "Без смены кода",
                    "complete_sectors": "Полные 8 отраслей",
                }[r.sample],
                "regime": "Внутри регионов" if r.regime == "within_region" else "Новые регионы",
                "n": r.n,
                "gain": round(100 * r.relative_mse_reduction, 2),
                "ci": f"[{100 * r.ci_low:.2f}; {100 * r.ci_high:.2f}]",
                "r2_base": round(subset.loc["economic_missingness", "r2_oof"], 4),
                "r2_sector": round(subset.loc["economic_sectors", "r2_oof"], 4),
            }
        )
    main_rows = [r for r in model_rows if r["sample"] == "Основная"]
    exrows = []
    for r in ex24.itertuples():
        old = ex23.loc[r.territory_id]
        exrows.append(
            {
                "mo": NAMES[r.territory_id],
                "sector": SECTORS[r.max_among_observed_sections],
                "workers": int(r.max_observed_workers),
                "total": int(r.workers_total_source),
                "share_2023": f"{100 * old.max_observed_share_of_official_total:.2f}%",
                "share_2024": f"{100 * r.max_observed_share_of_official_total:.2f}%",
                "observed": int(r.n_observed_sectors),
            }
        )
    pricerows = [
        {
            "year": int(r.year),
            "n": r.n,
            "regions": r.regions,
            "raw_r": round(r.raw_pearson, 3),
            "partial_r": round(r.partial_pearson, 3),
            "ci": f"[{r.partial_r_ci_low:.3f}; {r.partial_r_ci_high:.3f}]",
        }
        for r in headline_price.itertuples()
    ]
    source_defs = [
        source(
            "models",
            "Дополнительная информация отраслей: фиксированный CV",
            "reports/round2_analysis/models/cv_comparisons.csv",
            "from pathlib import Path\nimport pandas as pd\n"
            "d = pd.read_csv(Path('reports/round2_analysis/models/cv_comparisons.csv'))\n"
            "print(d.query(\"baseline == 'economic_missingness' and "
            "alternative == 'economic_sectors' and target == "
            "'profile_clr' and seed == 13\"))",
        ),
        source(
            "correlations",
            "Описательные частные связи отраслей с пятью CLR-компонентами и уровнем расходов",
            "reports/round2_analysis/models/partial_correlations.csv",
            "import pandas as pd\n"
            "d = pd.read_csv('reports/round2_analysis/models/partial_correlations.csv')\n"
            "print(d.query('year == 2024'))",
        ),
        source(
            "sector",
            "Отраслевые доли и раскрытие: проверенные годовые значения БДПМО",
            "reports/round2_analysis/figures/coverage_chart.csv",
            "import pandas as pd\nd = "
            "pd.read_csv('reports/round2_analysis/sector/feature_panel.csv')\n"
            "print(d.filter(regex='sector_.*_share').notna().sum())",
        ),
        source(
            "prices",
            "Городские цены: сопоставимые товары и условные связи с расходами",
            "reports/round2_analysis/prices/canonical/iszh_associations.csv",
            "import pandas as pd\nd = "
            "pd.read_csv('reports/round2_analysis/prices/canonical/iszh_associations.csv')\n"
            "print(d.query(\"sample == 'core_complete' and target == 'log_total'\"))",
        ),
        source(
            "price_context",
            "Принятый городской ценовой контекст отдельных муниципалитетов",
            "reports/round2_analysis/prices/canonical/annual_context_panel.csv",
            "import pandas as pd\n"
            "d = pd.read_csv('reports/round2_analysis/prices/canonical/annual_context_panel.csv')\n"
            "print(d.query('territory_id == 318')[['year', 'source_city_name', 'iszh']])",
        ),
        source(
            "ratios",
            "Городские отношения средних цен одинаковых товаров и единиц",
            "reports/round2_analysis/prices/canonical/primary_city_price_ratios.csv",
            "import pandas as pd\nd = "
            "pd.read_csv('reports/round2_analysis/prices/canonical/primary_city_price_ratios.csv')\n"
            "print(d.city_median_ratio.median())",
        ),
        source(
            "examples",
            "Шесть ранее выбранных муниципалитетов: максимальные раскрытые отрасли",
            "reports/round2_analysis/sector/examples.csv",
            "import pandas as pd\nd = "
            "pd.read_csv('reports/round2_analysis/sector/examples.csv')\n"
            "print(d.query('year == 2024'))",
        ),
    ]
    source_defs[0]["query"]["tables_used"] += [
        "reports/round2_analysis/models/cv_metrics.csv",
        "configs/round2_analysis_v1.json",
    ]
    source_defs[0]["query"]["computation_path"] = "scripts/round2_model_analysis.py"
    blocks, md = [], []

    def prose(key, text, src=None):
        block = {"id": key, "type": "markdown", "body": text}
        if src:
            block["sourceId"] = src
        blocks.append(block)
        md.append(text)

    def chart(key, img, alt):
        blocks.append({"id": f"block_{key}", "type": "chart", "chartId": key, "layout": "full"})
        md.append(f"![{alt}]({img})")

    def table(key, data, columns):
        blocks.append({"id": f"block_{key}", "type": "table", "tableId": key, "layout": "full"})
        md.append(table_md(data, columns))

    prose(
        "title",
        f"# {TITLE}\n\nПервый кейс СберИндекса. "
        "Анализ данных за 2023 и 2024 годы, 15 сентября 2026 года.",
    )
    prose(
        "summary",
        "## Основные результаты\n\n"
        "Отраслевые показатели дают небольшую дополнительную информацию о "
        "составе потребительского профиля: "
        "ошибка уменьшается примерно на 1,4% внутри регионов и на 7–8% "
        "при проверке на целых регионах вне обучения. "
        "Сравнение уже учитывает сам факт отсутствия отраслевых чисел.\n\n"
        "На небольшой выборке с полными восемью отраслевыми показателями результат ухудшается. "
        "Поэтому отрасли стоит сохранить как дополнительные атрибуты для "
        "интерпретации и дальнейшей проверки; "
        "универсальная польза и надёжная типология муниципальных экономик пока не установлены.\n\n"
        "Городские цены помогают интерпретировать номинальный уровень "
        "расходов на узкой городской подвыборке. "
        "Убедительной отдельной связи с составом пятикомпонентного профиля не обнаружено. "
        "Новые данные не устраняют прежние ограничения по географии и устойчивости кластеров.",
        "models",
    )
    prose(
        "definition",
        "## Как читать результаты\n\n"
        "Основная панель содержит 2016 муниципалитетов за два года. Из "
        "неё по прежним правилам качества "
        "в модели входят 1975 МО за 2023 год и 1998 за 2024-й. Каждый МО имеет одинаковый вес. "
        "Потребительский профиль — среднее 12 месячных нормированных "
        "значений пяти категорий Сбера. "
        "Эти значения характеризуют соотношения опубликованных "
        "показателей и не являются полной структурой бюджета домохозяйства.\n\n"
        "Базовые характеристики: регион и тип МО, население, зарплата "
        "организаций и число учтённых работников на жителя. "
        "В проверке целых регионов региональные признаки исключены. Затем "
        "добавлены восемь индикаторов пропусков "
        "и восемь отношений отраслевых работников к официальному общему итогу. "
        "Исходная статистика исключает малое предпринимательство; рабочие "
        "места в МО и занятые жители различаются.\n\n"
        "Все варианты сравниваются на одинаковых строках и пяти "
        "разбиениях. Медианы для модельного заполнения "
        "пропусков, масштабы и категории оцениваются только на обучающей части. "
        "Положительное снижение ошибки означает улучшение; R² показывает "
        "долю вариации пяти CLR-координат, "
        "объяснённую вне обучающей выборки. Это ретроспективное описание, не прогноз будущего.",
        "models",
    )
    prose(
        "coverage_text",
        "## Раскрытие отраслей сильно различается\n\n"
        "Образование наблюдается в 99,0% из 4032 МО-лет, госуправление в 96,6%, добыча в 15,5%. "
        "В 174 МО-годах добыча вообще не предлагается в форме источника. "
        "Такой пропуск не означает нулевую добычу. "
        "Восемь отраслей выбраны до просмотра модельных результатов: "
        "сельское хозяйство, добыча, обработка, "
        "энергоснабжение, транспорт, госуправление, образование и здравоохранение.\n\n"
        "Столбцы показывают наличие числовой доли, а не размер отрасли. "
        "Для интерпретации сети нужно хранить "
        "показатели раскрытия вместе с числовыми атрибутами. Отраслевые "
        "числа не перенормируются до 100%; "
        "в 88 МО-годах арифметическая сумма раскрытых чисел слегка выше "
        "официального итога, причина не установлена.",
        "sector",
    )
    chart("coverage", "figures/sector_coverage.png", "Раскрытие восьми отраслевых показателей")
    prose(
        "models_text",
        "## Прирост есть, но его величина зависит от проверки\n\n"
        "После контроля пропусков отраслевые числа снижают ошибку профиля "
        "на 1,40% в 2023 году и на 1,38% "
        "в 2024-м при проверке муниципалитетов внутри регионов. При "
        "удержании целых регионов снижение составляет "
        "8,27% и 7,34%. В этих четырёх сравнениях интервалы перевыборки "
        "регионов остаются выше нуля. "
        "Это дополнительная информация сверх населения, зарплаты и "
        "концентрации рабочих мест, а не причинный эффект отраслей.\n\n"
        "Таблица показывает базу с контролем пропусков и вариант с числовыми отраслевыми долями. "
        "Проценты относятся к уменьшению квадратичной ошибки профиля, а "
        "не к росту потребления или точности в процентах.",
        "models",
    )
    columns_main = {
        "year": "Год",
        "regime": "Проверка",
        "n": "МО",
        "r2_base": "R² базы",
        "r2_sector": "R² с отраслями",
        "gain": "Снижение ошибки, %",
        "ci": "95% интервал, %",
    }
    table("main_metrics", main_rows, columns_main)
    prose(
        "sensitivity_text",
        "## Полное раскрытие пока не подтверждает улучшение\n\n"
        "После исключения территорий со сменой кода или типа прирост "
        "сохраняется: 1,92% и 1,64% внутри регионов, "
        "9,06% и 7,72% при удержании целых регионов за 2023 и 2024 годы соответственно. "
        "При дополнительных разбиениях внутри регионов прирост основной "
        "выборки составляет 1,40–1,65% "
        "в 2023-м и 1,38–1,87% в 2024-м.\n\n"
        "Но одновременно все восемь отраслей раскрыты только у 134 и 127 МО основной выборки. "
        "Здесь ошибка внутри регионов увеличивается на 12,91% и 7,08%; "
        "при удержании регионов — на 5,49% и 13,18%. "
        "Это другая, маленькая и отобранная по раскрытию выборка. По "
        "этому опыту нельзя отделить влияние её состава "
        "от размера обучения и ограничений модели. Полный отраслевой "
        "профиль пока не получил универсального подтверждения.\n\n"
        "На графике показаны все три выборки, включая неудачный "
        "результат. Нулевая линия отделяет улучшение "
        "от ухудшения. Интервалы получены из 2000 парных перевыборок "
        "регионов по фиксированным OOF-ошибкам; "
        "они не включают неопределённость повторного обучения всего процесса.",
        "models",
    )
    chart("gains", "figures/sector_increment.png", "Проверки отраслевого прироста с интервалами")
    table(
        "sensitivity",
        model_rows,
        {
            "year": "Год",
            "sample": "Выборка",
            "regime": "Проверка",
            "n": "МО",
            "gain": "Снижение ошибки, %",
            "ci": "95% интервал, %",
        },
    )
    prose(
        "negative_control",
        "Одна заранее заданная перестановка наблюдаемых отраслевых значений внутри регионов "
        "при сохранении пропусков даёт для профиля −0,53% и −0,22% внутри "
        "регионов, +1,29% и +0,77% "
        "при удержании регионов. Во всех четырёх случаях интервалы "
        "включают ноль. Это диагностический контроль, "
        "а не перестановочный p-value.\n\n"
        "Для общего номинального уровня расходов отрасли не дают столь же устойчивого улучшения: "
        "−0,20% и −0,11% внутри регионов, +2,49% и +2,63% при удержании регионов. "
        "Все четыре интервала включают ноль. Основная обнаруженная польза "
        "относится к составу профиля.",
        "models",
    )
    prose(
        "interpretation",
        "## Отрасли уточняют непрерывные различия между территориями\n\n"
        "Например, в 2024 году большая наблюдаемая доля работников "
        "образования связана с более высокой "
        "транспортной координатой профиля (+0,17) и более низкой координатой маркетплейсов (−0,16) "
        "после учёта региона, типа МО и базовых экономических "
        "показателей. Для обработки наблюдаются "
        "связи +0,13 с маркетплейсами и −0,18 с транспортом. Это частные корреляции, а не эффекты "
        "увеличения занятости в отрасли. Для разных отраслей доступны "
        "разные наборы муниципалитетов.\n\n"
        "Полная таблица содержит все 96 связей: восемь отраслей, шесть исходов, два года. "
        "Выбор примеров в тексте описательный; подтверждающих тестов "
        "отдельных отраслевых гипотез здесь нет. "
        "Эти результаты поддерживают интерпретацию непрерывного сходства "
        "профилей. Они не доказывают "
        "существование устойчивых дискретных типов муниципальной экономики.",
        "correlations",
    )
    md.append("![Частные связи отраслей](figures/sector_associations_2024.png)")
    prose(
        "prices_text",
        "## Цены связаны с уровнем расходов на городской подвыборке\n\n"
        "Годовой индекс стоимости жизни доступен для 215 МО в 2023 году и "
        "214 в 2024-м, около 10,6% "
        "исходной панели. В 2024 году медианное население выбранных "
        "территорий — 106466 человек против "
        "20981 вне этой подвыборки; городские округа составляют 75,2% "
        "выбранных МО против 21,5% всей панели. "
        "Это не представительная выборка российских муниципалитетов.\n\n"
        "После ограничения полными базовыми признаками остаются 207 и 212 МО. "
        "Частная корреляция логарифмов ИСЖ и расходов составляет 0,406 и 0,349. "
        "Интервалы из 999 перевыборок регионов с повторной оценкой "
        "контрольных зависимостей выше нуля. "
        "При этом интервалы для каждой из пяти координат состава профиля "
        "включают ноль в обоих годах. "
        "Это отсутствие убедительного сигнала в данной выборке, не "
        "доказательство отсутствия связи вообще.\n\n"
        "Таблица сопоставляет исходную и скорректированную связь. Регион, "
        "тип МО и экономика снимают большую "
        "часть вариации городского ИСЖ; оставшиеся различия не позволяют "
        "трактовать коэффициенты причинно. "
        "Цена наблюдается в городе, связь с родительским МО не устанавливает равенство их границ.",
        "prices",
    )
    table(
        "price_associations",
        pricerows,
        {
            "year": "Год",
            "n": "МО",
            "regions": "Регионов",
            "raw_r": "Исходная r",
            "partial_r": "Частная r",
            "ci": "95% интервал частной r",
        },
    )
    ratio_gain = 100 * (prices["median_city_price_ratio"] - 1)
    prose(
        "ratios_text",
        "## Сопоставимые товарные цены дают полезную диагностику\n\n"
        f"Строго сопоставлены {prices['matched_positive_month_pairs']} "
        "положительных месячных пар одинаковых "
        "товаров и единиц за 2023 и 2024 годы. Полные 12 пар есть у 102759 сочетаний город–товар. "
        "В 207 городах остаётся от 317 до 564 таких товаров; "
        "медиана городских медиан отношений средних "
        f"годовых цен равна {prices['median_city_price_ratio']:.5f}, или +{ratio_gain:.2f}%. "
        "У всех товаров здесь одинаковый вес, а наборы между городами "
        "различаются. Это описательная "
        "характеристика наблюдаемых цен, не ИПЦ, не муниципальная "
        "инфляция и не дефлятор пяти категорий Сбера.\n\n"
        "Общий набор из 377 товаров полностью наблюдается в 90 городах; "
        "для него медиана составляет "
        "1,07000. На тех же 90 городах с их собственными полными наборами она равна 1,07241. "
        "Близость этого контрольного расчёта не исправляет географическую и товарную селекцию. "
        "Обнаружены 290 городских товарных отношений ниже 0,5 или выше 2. "
        "Они сохранены для проверки "
        "исходных ячеек; причины экстремальных значений не установлены.\n\n"
        "На графике каждый город представлен одной медианой. Источники исключений сохранены: "
        "неоднозначный код Вышнего Волочка в январе 2023 года, временно "
        "не подтверждённые связи городов, "
        "нулевые цены и отсутствующие месячные товарные позиции не восстанавливались.",
        "ratios",
    )
    chart(
        "price_distribution",
        "prices/canonical/matched_price_ratios.png",
        "Распределение сопоставимых городских товарных отношений",
    )
    prose(
        "cases_text",
        "## Шесть примеров получили более конкретный отраслевой контекст\n\n"
        "Все шесть территорий были выбраны по крайним остаткам прежней модели, поэтому не являются "
        "независимой проверочной выборкой. Ниже указана крупнейшая среди "
        "раскрытых отраслей в 2024 году. "
        "Во всех шести она совпадает с крупнейшей раскрытой отраслью 2023 "
        "года. Доля делится на официальный "
        "общий итог работников организаций, не на сумму доступных отраслей.",
        "examples",
    )
    table(
        "cases",
        exrows,
        {
            "mo": "Муниципалитет",
            "sector": "Максимум среди раскрытых",
            "workers": "Работники 2024",
            "total": "Общий итог 2024",
            "share_2023": "Доля 2023",
            "share_2024": "Доля 2024",
            "observed": "Раскрыто отраслей 2024",
        },
    )
    prose(
        "cases_implication",
        "В Светловском округе опубликованные работники обработки выросли с 1723 до 2864, "
        "а её доля в общем итоге с 31,93% до 44,37%. В Свободненском "
        "районе работники энергоснабжения выросли "
        "с 2549 до 2937, но доля снизилась с 40,56% до 32,90% при росте общего итога. "
        "В Куркине работники образования сократились с 1201 до 1184, а "
        "доля выросла с 31,51% до 35,35%. "
        "Последний пример показывает, почему рост доли сам по себе не "
        "доказывает расширение отрасли.\n\n"
        "Такие наблюдения дают конкретные вопросы для проверки организаций и охвата источника. "
        "Они не устанавливают причину потребительского остатка, инвестиционный проект, миграцию "
        "или экономический шок. Перепись 2021 года пока не имеет "
        "принятого географического соединения "
        "с нашей панелью 2023/2024, поэтому в эти расчёты не включена.",
        "examples",
    )
    prose(
        "case_price_context",
        "Из шести примеров принятый городской ценовой контекст есть только у "
        "Верхнеколымского района: для пгт Зырянка ИСЖ равен 1,44 в 2023 и 2024 годах. "
        "Это стоимость сопоставляемой годовой корзины относительно России в том же году, "
        "а не цена для всего района. Равенство индекса между годами не означает "
        "неизменность цен. Оно даёт контекст высокого номинального уровня расходов, "
        "но не устанавливает причину отклонения от модели. Для остальных пяти примеров "
        "подстановка ближайшего города не выполнялась.",
        "price_context",
    )
    prose(
        "actions",
        "## Что использовать в работе над первым кейсом\n\n"
        "1. Сохранить отраслевые доли и признаки раскрытия как отдельный "
        "слой атрибутов муниципалитетов. "
        "Основную сеть сходства потребительских профилей пока не менять; новые данные не содержат "
        "наблюдаемых межмуниципальных экономических потоков.\n"
        "2. В итоговой работе показать и прирост на основной панели, и "
        "ухудшение на полной отраслевой "
        "подвыборке. Следующий узкий эксперимент должен разделить влияние "
        "размера выборки, её состава "
        "и редких отраслевых признаков, с новым заранее зафиксированным протоколом.\n"
        "3. Для экономического разбора приоритетны Светловский округ, "
        "Свободненский район и Куркино: "
        "проверить изменения числителей, общего итога и состава учтённых организаций. "
        "Названия предприятий и причины изменений потребуют первичных подтверждений.\n"
        "4. Использовать цены как городской контекст номинальных уровней "
        "и для проверки подозрительных "
        "товарных рядов. Для дефлятора нужны обоснованные веса и соответствие товарных категорий, "
        "которых данный расчёт не устанавливает.",
    )
    prose(
        "limits",
        "## Что остаётся открытым\n\n"
        "Одинаковый муниципальный код на 1 января не доказывает "
        "неизменность границ и состава организаций "
        "за весь год. Исключение 78 сменивших код или тип МО остаётся проверкой чувствительности, "
        "а не сертификатом географической эквивалентности остальных.\n\n"
        "Пропуски отраслей могут зависеть от структуры экономики, формы "
        "публикации и охвата организаций. "
        "Индикаторы пропусков уменьшают смешение этих факторов в "
        "сравнении моделей, но не устраняют "
        "селекцию. Все данные ретроспективны, оба года уже изучены, модели и отдельные примеры "
        "не дают причинных выводов. Неопределённость границ, источников и доступности данных "
        "не покрывается статистическими интервалами.\n\n"
        "Ограничения прежних дискретных кластеров сохраняются. Новые "
        "результаты усиливают проверяемую "
        "часть исследования о непрерывных различиях муниципальных профилей и уточняют, "
        "какие экономические атрибуты пригодны для их интерпретации.",
    )
    prose(
        "evidence",
        "## Источники и воспроизведение\n\n"
        "Исходный пакет второго раунда зафиксирован в коммите "
        "`ac535334432bca4f99c30cd430f5b5cf4bd0811a` "
        f"репозитория [municipal-economic-profiles]({REPO_URL}). "
        f"[Полный анализ и таблицы]({REPO_URL}/tree/main/reports/round2_analysis), "
        f"[протокол]({REPO_URL}/blob/main/configs/round2_analysis_v1.json), "
        f"[независимая проверка]({REPO_URL}"
        "/blob/main/reports/round2_analysis/independent_review.md), "
        f"[команды воспроизведения]({REPO_URL}/blob/main/reports/round2_analysis/REPRODUCE.md). "
        "Отраслевой и ценовой блоки содержат собственные описания, QA, исходные ссылки и хеши.",
    )
    gains = pd.DataFrame(model_rows)
    gains["comparison"] = gains["sample"] + " · " + gains["regime"]
    dist = read_result("reports/round2_analysis/prices/canonical/primary_city_price_ratios.csv")
    dist["growth_pct"] = 100 * (dist.city_median_ratio - 1)
    charts = [
        {
            "id": "coverage",
            "title": "Раскрытие отраслевых показателей",
            "type": "bar",
            "dataset": "coverage",
            "sourceId": "sector",
            "encodings": {
                "x": {"field": "label", "type": "nominal"},
                "y": {"field": "coverage_pct", "type": "quantitative", "label": "% из 4032 МО-лет"},
            },
            "options": {"orientation": "horizontal"},
            "intent": "comparison",
            "question": "Насколько различается числовое раскрытие восьми отраслей?",
            "rationale": "Восемь сопоставимых долей покрытия из одного знаменателя.",
        },
        {
            "id": "gains",
            "title": "Отраслевое снижение ошибки профиля",
            "type": "bar",
            "dataset": "gains",
            "sourceId": "models",
            "encodings": {
                "x": {"field": "comparison", "type": "nominal"},
                "y": {"field": "gain", "type": "quantitative", "label": "Снижение ошибки, %"},
                "color": {"field": "year", "type": "nominal", "label": "Год"},
            },
            "options": {"orientation": "horizontal", "grouping": "grouped"},
            "intent": "comparison",
            "question": "Сохраняется ли дополнительная информация при "
            "смене выборки и режима проверки?",
            "rationale": "Группированные бары показывают знак и масштаб; "
            "точные95%интервалы в соседней таблице.",
        },
        {
            "id": "price_distribution",
            "title": "Городские медианы товарных отношений цен",
            "type": "histogram",
            "dataset": "price_distribution",
            "sourceId": "ratios",
            "encodings": {
                "x": {
                    "field": "growth_pct",
                    "type": "quantitative",
                    "label": "Отношение минус1, %",
                },
                "y": {"field": "growth_pct", "type": "quantitative"},
            },
            "intent": "distribution",
            "question": "Как распределены207городских описательных отношений цен?",
            "rationale": "Одно наблюдение на город; распределение "
            "показывает разброс без инфляционной интерпретации.",
        },
    ]
    datasets = {
        "coverage": records(coverage),
        "gains": records(gains),
        "main_metrics": main_rows,
        "sensitivity": model_rows,
        "price_associations": pricerows,
        "cases": exrows,
        "price_distribution": records(dist),
    }
    tables = []
    for key, cols, src in [
        ("main_metrics", columns_main, "models"),
        (
            "sensitivity",
            {
                "year": "Год",
                "sample": "Выборка",
                "regime": "Проверка",
                "n": "МО",
                "gain": "Снижение ошибки, %",
                "ci": "95% интервал, %",
            },
            "models",
        ),
        (
            "price_associations",
            {
                "year": "Год",
                "n": "МО",
                "regions": "Регионов",
                "raw_r": "Исходная r",
                "partial_r": "Частная r",
                "ci": "95% интервал",
            },
            "prices",
        ),
        (
            "cases",
            {
                "mo": "Муниципалитет",
                "sector": "Отрасль",
                "workers": "Работники2024",
                "total": "Итог2024",
                "share_2023": "Доля2023",
                "share_2024": "Доля2024",
                "observed": "Раскрытоотраслей",
            },
            "examples",
        ),
    ]:
        tables.append(
            {
                "id": key,
                "title": {
                    "main_metrics": "Сравнение на основной выборке",
                    "sensitivity": "Результаты проверок чувствительности",
                    "price_associations": "Связь ИСЖ с уровнем расходов",
                    "cases": "Шесть отраслевых примеров",
                }[key],
                "dataset": key,
                "sourceId": src,
                "defaultSort": {"field": next(iter(cols)), "direction": "asc"},
                "columns": [
                    {
                        "field": k,
                        "label": v,
                        "type": "text" if isinstance(datasets[key][0][k], str) else "number",
                    }
                    for k, v in cols.items()
                ],
            }
        )
    timestamp = datetime.now(UTC).isoformat()
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
    (root / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / "REPORT.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(main_rows).to_csv(root / "headline_metrics.csv", index=False)
    print("Wrote artifact.json, REPORT.md and headline_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().repo)
