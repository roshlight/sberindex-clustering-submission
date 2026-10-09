# Воспроизведение анализа второго раунда

Все команды выполняются из корня `municipal-economic-profiles`. Python 3.12 и библиотеки зафиксированы существующим `uv.lock`. Интернет для повторения числового анализа не требуется после установки зависимостей; используются сохранённые данные из Git. Сырые XLSX/CSV доступны в release `data-round2-2026-09-15`, но повторное сетевое получение для этого анализа не требуется.

```powershell
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check scripts src tests
uv run --frozen ruff format --check scripts src tests
uv run --frozen python scripts/verify_round2_bundle.py
```

## Числовые результаты в новых папках

```powershell
uv run --frozen python scripts/round2_sector_analysis.py --repo . --out work/round2-sector-replay
uv run --frozen python scripts/round2_model_analysis.py --repo . --out work/round2-model-replay
uv run --frozen python scripts/round2_price_analysis.py --repo . --out work/round2-price-replay --bootstrap 999
```

Выходные папки должны быть новыми. Исходные отчёты и данные не перезаписываются. Модельный скрипт самостоятельно строит восемь долей из frozen отраслевого Parquet; результат проверен на точное совпадение с отдельной реализацией полного отраслевого слоя. Во всех сравнениях используются прежние `core_eligible` и `conservative_eligible`. Восстановленные итоги ID2710 не расширяют выборку задним числом.

`models/manifest.json` содержит хеши исходной панели, отраслевого Parquet, протокола, общего модуля расчётов и lockfile; код имеет собственный SHA256. `models/preprocessing.json` содержит параметры всех обучающих фолдов. `models/oof_predictions.parquet` позволяет пересчитать ошибки без обучения. Полные словари, ограничения и команды проверок отраслей и цен находятся в `sector/README.md` и `prices/REPRODUCE.md`.

Независимая проверка опубликованного модельного расчёта:

```powershell
uv run --frozen python scripts/verify_round2_analysis.py --repo .
```

Точные параметры verifier и альтернативного каталога вывода приведены в `independent_review.md` и `--help`. Проверка использует независимый расчёт признаков, ошибок, bootstrap и части Ridge-моделей через линейную алгебру. В `report_claims_review.json` дополнительно сохранены проверка утверждений отчёта и независимый пересчёт 96 частных корреляций. Ценовая ветка имеет отдельную проверку коэффициентов и исходных ячеек; это не новый аудит всей первичной статистики.

## Отчёт и изображения

```powershell
uv run --frozen python scripts/plot_round2_analysis.py
uv run --frozen python scripts/build_round2_report.py
```

Эти две команды обновляют только изображения, `artifact.json`, `REPORT.md` и сводную таблицу в существующем `reports/round2_analysis`; для побайтовой проверки опубликованного пакета сначала используйте хеши, не пересобирая файлы. Время генерации, метаданные SVG и упаковка Parquet могут изменять байты при совпадающих числах. Статистические результаты сравниваются по ключам и числам с допуском `1e-10`.

Portable HTML собирается из `artifact.json` средствами установленного плагина Data Analytics версии `0.2.10`. Скрипт `scripts/build_round2_portable.mjs` вызывает штатные builder, SVG extractor и verifier. Он исправляет только ширину стандартной шапки при классической полосе прокрутки Windows; исходный payload и runtime побайтово сохраняются. Подробности и причина изменения находятся в `portable_compatibility.md`.

```powershell
$analyticsPluginPath = 'C:/Users/ghora/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599'
node scripts/build_round2_portable.mjs --plugin $analyticsPluginPath --input reports/round2_analysis/artifact.json --output work/round2-report-replay.html --receipt work/round2-report-replay.json --work-dir work/round2-html-build
```

На другом компьютере укажите фактический путь установленного плагина; выходные файлы должны быть новыми. Готовый `report.html` самодостаточен и открывается без плагина, сервера и сети; `REPORT.md`, CSV и научные PNG/SVG также доступны независимо от него. Python-код не требует данного плагина.

Данные в HTML — сохранённый снимок. Источники визуализаций указывают точные материализованные таблицы, запрос чтения DuckDB и исходное Python-преобразование; код научных расчётов хранится в репозитории. Проверки упаковщика и визуальная проверка экспортированных научных графиков описаны в `QA.md`.
