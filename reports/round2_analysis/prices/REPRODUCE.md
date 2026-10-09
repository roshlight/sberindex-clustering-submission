# Воспроизведение и проверка

Из корня репозитория, в существующем корневом окружении:

```powershell
uv run --frozen python scripts/round2_price_analysis.py --repo . --out work/round2-price-replay --bootstrap 999
uv run --frozen ruff check scripts/round2_price_analysis.py
uv run --frozen ruff format --check scripts/round2_price_analysis.py
```

Команда требует новую пустую выходную папку, записывает протокол перед вычислениями и сохраняет `failed` manifest при ошибке. Нужны `reports/enriched_analysis/core_v2/annual_panel.parquet` и его manifest, а также полный frozen `reports/data_round2/prices/` с `bridge_v2`, `monthly_v3`, manifest и результатом XML-проверки. Входные пути и SHA перечислены в каноническом manifest. Для чтения и записи Parquet используется DuckDB; `pyarrow` не нужен. Интернет, модели API и сырые XLSX для этого этапа не требуются: их прежняя полная XML-проверка и SHA цепочки сохраняются как входная предпосылка, а не объявляются новым независимым обследованием.

Скрипт повторно проверяет все 29 файлов исходного ценового manifest и hash годовой core-панели. Коды города читаются строками; соединения контролируются на уникальность. Встроенные содержательные проверки подтверждают удаление региональных эффектов и известный частный коэффициент на синтетике. Реальный SQL сопоставления прогоняется на примерах смены городского кода, исходного кода и единицы, нулевой цены и неполного года: они не превращаются в допустимые полные пары. Все 999 региональных повторов дали конечные оценки для каждого исхода и сценария.

В `independent_arithmetic_qa.json` сохранён результат отдельной арифметической проверки: коэффициенты полного OLS с явными региональными и типовыми dummy совпали с последовательным удалением эффектов с максимальной разницей менее `3e-13`; 240 чисел и их локаторы из диагностических примеров сверены с исходным frozen Parquet. Это альтернативный расчёт и проверка переноса тем же агентом, а не независимое измерение качества исходных цен. Программные проверки и оба сохранённых графика проверены; графики просмотрены после построения.

Два ранних технических завершения из-за зарезервированных SQL-псевдонимов и расчёт до добавления финальных проверок сохранены вне публикуемого отчёта в `work/round2-analysis-price-history/`. Дизайн, пороги и признаки между ними не выбирались по качеству. Канонический результат только один: `canonical/`. При воспроизведении сравнивайте ключи, статусы и числа с числовым допуском; порядок суммирования плавающей точки и контейнер Parquet могут менять байты без изменения результата.

## Основные схемы для общего отчёта

| Таблица | Зерно | Основные поля |
|---|---|---|
| `canonical/iszh_associations.csv` | год × sample × target; 24 строки | `n`, `regions`, `raw_pearson`, `partial_pearson`, `partial_r_ci_low/high`, `adjusted_slope`, `slope_ci_low/high`, `bootstrap_finite` |
| `canonical/coverage.csv` | год × группа | `n`, `regions`, `core_eligible`, `conservative_eligible` |
| `canonical/selection_numeric.csv` | год × группа × признак | `n_numeric`, `median`, `mean`, `q25`, `q75` |
| `canonical/selection_categories.csv` | год × регион/тип | `full_n`, `selected_n`, `within_category_coverage`, `share_of_full`, `share_of_selected` |
| `canonical/primary_city_price_ratios.csv` | МО/наблюдаемый город; 207 строк | `territory_id`, `source_city_name`, `n_products`, `city_median_ratio`, `product_q10/q90`, `min_month_pairs=12`, `min_products=100` |
| `canonical/ratio_threshold_sensitivity.csv` | порог месяцев × порог товаров; 8 строк | `cities`, `products_min/median/max`, `median_city_ratio`, `q10/q90_city_ratio` |
| `canonical/city_product_pairs.parquet` | МО × товар; 103 242 строки | точные коды города, полное название, `positive_month_pairs`, `mean2023/2024`, `annual_ratio`; основная оценка требует `positive_month_pairs=12` |
| `canonical/city_pair_eligibility.csv` | МО; 2016 строк | разрешённые связи по годам, `same_approved_city_code`, полнота, `pair_status` |
| `canonical/source_cell_examples.csv` | крайний диагностический город × товар × месяц; 120 строк | обе цены, отношение, точные исходные строки и столбцы |
| `canonical/summary.json` | общий итог | coverage и price ratios; без причинных/CPI трактовок |

`annual_ratio` для чувствительности с 10 или 11 парами — отношение средних **только одинаковых наблюдаемых месяцев**, а не полный год. Эти строки сохраняются для проверки полноты и не входят в основную оценку 12/12. Пропуски `city_pair_eligibility.csv` не означают нулевые цены или нулевое число опубликованных товаров вне принятой схемы.
