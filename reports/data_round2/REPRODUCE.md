**Воспроизведение второго раунда сбора**

Код использует существующее окружение `enrichment` с Python 3.12 и его `uv.lock`. Все повторные расчёты следует направлять в новую папку. Исходные файлы и прежние исследовательские результаты не перезаписываются.

Архивы исходников публикуются в [релизе data-round2-2026-09-15](https://github.com/sberindex-hackathon-2026/municipal-economic-profiles/releases/tag/data-round2-2026-09-15). Сначала скачайте их и распакуйте содержимое в один каталог `work/data-round2` в клоне репозитория. У каждого архива есть собственный `MANIFEST.<имя архива>.json` с путями и SHA-256 всех включённых файлов. Хеши самих архивов и результаты обратного чтения приведены в `delivery_manifest.json`.

```powershell
uv sync --project enrichment --frozen
gh release download data-round2-2026-09-15 --repo sberindex-hackathon-2026/municipal-economic-profiles --pattern '*.zip' --dir work/round2-archives
```

Пути архивов разделены по веткам. Для воспроизведения используйте распакованные пути `work/data-round2/<ветка>`; исходные журналы могут содержать пути рабочей машины, начинающиеся с `../../work/data-round2`.

Перепись:

```powershell
uv run --project enrichment --frozen python enrichment/scripts/collect_round2_census.py --sources configs/data_round2_census_sources.json --raw work/data-round2/commuting/raw --out work/data-round2/replayed-census --offline
uv run --project enrichment --frozen python enrichment/scripts/verify_round2_census.py --raw work/data-round2/commuting/raw --report work/data-round2/replayed-census --out work/data-round2/replayed-census-qa
```

Ожидаются 15 скачанных исходников, 257 листов, 350 850 непустых и 263 406 числовых ячеек; независимый проверяющий должен сверить все числовые ячейки и 12 тождеств сложения в четырёх разобранных строках. Принятых исторических соединений с основной панелью: 0. Сравнивать нужно исходные и производные данные; время запуска и абсолютные пути в метаданных на другой машине могут отличаться.

Точные команды остальных веток и описания входов: [отрасли](sector/REPRODUCE.md), [география](geography/README.md), [цены](prices/README.md). [Выводы отраслевой проверки](sector/FINDINGS.md) и эти инструкции определяют, какие таблицы являются кандидатами, какие допущены только как контекст города и какие пока не соединены с панелью.

Серверный поиск воспроизводим по конфигурациям `configs/data_round2_needs.json`, `configs/data_round2_sources.json` и снимку реализации в архиве `server-discovery/implementation_snapshot`. Живой повтор потребует прежних настроек сервера и ключа OpenRouter, которые в пакет не включены. Результат поисковой выдачи и ответы модели могут меняться. Сохранённые публичные документы и их разбор позволяют проверить выполненный запуск без ключа; [отчёт о поиске](discovery/README.md) явно указывает ограничения движков и результаты попыток загрузки.

Проверка кода из корня репозитория:

```powershell
uv run --project enrichment --frozen pytest enrichment/tests -q
uv run --project enrichment --frozen ruff check enrichment
uv run --project enrichment --frozen ruff format --check enrichment
```

Сбор не меняет результаты `reports/initial_study`, `reports/enriched_analysis` и прежние наборы `reports/enrichment_v2`, `reports/enrichment_v3`. Новая модельная аналитика должна брать только разрешённые соответствующим README строки, фиксировать версию входов и сохраняться отдельным запуском.
