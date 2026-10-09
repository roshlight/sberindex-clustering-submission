# Как повторить новую аналитику

Используется **корневое** окружение проекта: Python 3.12 и неизменённый `uv.lock`. Здесь нет обращения к LLM, платным API или серверному поиску. Проверено повторение в установленном Windows-окружении; отдельное чистое развёртывание с нуля и другая ОС в этой поставке не проверялись.

## Входы и сохранённые результаты

Клонировать приватный репозиторий и установить `uv sync --frozen`. Таблицы `reports/enrichment_v2/`, `reports/enrichment_v3/` и исходные назначения `reports/initial_study/assignments.parquet` включены в Git. Единственный не включённый в Git вход основной модели — `data/raw/consumption.parquet`.

Восстановление оригинального файла потребления описано в [корневом README](../../README.md) и `scripts/import_archive.py`. Канонический SHA-256 этого Parquet: `9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61`. Если доступный официальный архив изменился, нельзя молча заменить данные и назвать новые числа воспроизведением. Первичные архивные снимки контекста и команды их восстановления описаны в [enrichment_v3/REPRODUCE.md](../enrichment_v3/REPRODUCE.md); сам пересчёт моделей не требует распаковки всего контекстного архива.

`core_v2` — канонический успешный запуск новой основной аналитики. Это третья техническая попытка с тем же дизайном, а не третий подобранный вариант модели. Первая остановилась на отсутствии необязательного pandas Parquet engine; запись переведена на уже установленный DuckDB. Вторая остановилась на readonly-представлении NumPy в pandas 3; добавлена явная копия перед изменением массива. Оба завершения отмечены как failed в [history](history/); частичные таблицы не входят в аналитическую поставку. До завершения CV не выбирались модели по их качеству.

## Основная модель и цены

Из корня репозитория выполнять в **новые** каталоги:

```powershell
uv run --frozen python scripts/analyze_enriched_profiles.py --repo . --out work/replay/core
uv run --frozen python scripts/analyze_price_context.py --repo-root . --output-dir work/replay/prices
uv run --frozen python scripts/verify_enriched_bundle.py --repo . --compare-core work/replay/core --output work/replay/bundle_qa.json
```

Для core сравниваются SHA всех десяти содержательных файлов. В повторении поставки они совпали побайтно. Служебные время запуска и завершения в manifest не обязаны совпадать. Для цен доступны программные проверки цепи ИПЦ, регионального соединения, арифметики и сокращения дефлятора в месячном профиле.

Модельная конфигурация: [enriched_analysis_v1.json](../../configs/enriched_analysis_v1.json). Параметры этого выпуска также явно заданы в `src/municipal_profiles/enriched_analysis.py`; JSON служит зафиксированным контрактом исследования, а не универсальным конфигуратором. Изменение контракта требует отдельной реализации/версии и нового output. Редактирование JSON само по себе не меняет оценивание.

## Контекст групп и годовая сеть

```powershell
uv run --frozen python scripts/analyze_cluster_context.py --self-test
uv run --frozen python scripts/analyze_cluster_context.py --output work/replay/cluster_context --permutations 499 --seed 13
uv run --frozen python scripts/analyze_annual_networks.py --self-test
uv run --frozen python scripts/analyze_annual_networks.py --prepare --output work/replay/annual_networks
uv run --frozen python scripts/analyze_annual_networks.py --output work/replay/annual_networks
```

`--prepare` фиксирует контракт годовой сети до обучения. Его хеши и точный состав 21 альтернативы должны совпадать при последующем запуске. Платформа и версии библиотек могут влиять на оптимизацию Leiden; сохранённые labels/support — авторитетный снимок этого запуска. Дополнительные проверяющие скрипты рядом с frozen результатами восстанавливают суммы квадратов, перестановочные сравнения, peer-support и парные переходы. Их запуск пишет соответствующую QA-квитанцию; для сохранения оригинальной поставки запускать в отдельной копии репозитория.

```powershell
uv run --frozen python reports/enriched_analysis/cluster_context/verify_results.py
uv run --frozen python reports/enriched_analysis/annual_networks/verify_results.py
```

Переносимые проверки входов описаны в [input_audit/REPRODUCE.md](input_audit/REPRODUCE.md); проверка шести примеров непосредственно по архивным ячейкам — в [examples/REPRODUCE.md](examples/REPRODUCE.md). Для последней нужны сохранённые ZIP предыдущих выпусков; числовая аналитика уже поставлена отдельно от них.

## Графики, тесты и карта файлов

```powershell
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python scripts/plot_enriched_analysis.py --report reports/enriched_analysis
```

Последняя команда заново экспортирует PNG/SVG из frozen CSV; для неизменности байтов поставки использовать копию каталога отчёта. В `figures/chart_contracts.json` записаны источник, размер выборки и вопрос каждого графика. PNG проверены визуально; SVG предоставлены для экспорта и редактирования.

| Каталог | Что в нём |
|---|---|
| `core_v2` | 4032 годовые строки, маски, регрессии, 68 строк метрик, 40 сравнений, 60 690 OOF-строк |
| `prices` | Цепи ИПЦ, месячные/годовые значения, темпы и региональные сводки |
| `cluster_context` | Замороженные группы, контекст, 5988 перестановочных результатов |
| `annual_networks` | Годовые профили, 44 варианта Leiden и два K-means, метки и поддержка |
| `examples` | Шесть механически отобранных крайних остатков с происхождением чисел |
| `input_audit` | Ключи, даты, маски, диагностические случаи и отдельная проверка core |
| `figures` | Шесть исследовательских графиков в PNG и SVG |
| `history` | Явно отмеченные технические неудачные попытки |

Проверка SHA подтверждает тождество файлов, а не достоверность экономического смысла. Ограничения источников и интерпретации сохраняются при идеально воспроизведённых числах.
