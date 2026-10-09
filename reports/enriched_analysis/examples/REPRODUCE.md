Переносимый повтор расчёта шести примеров

Скрипт [scripts/build_enriched_examples.py](../../../scripts/build_enriched_examples.py) повторяет механический выбор и проверку исходных клеток, описанные в [FINDINGS.md](FINDINGS.md). Он принимает три обязательных аргумента: `--repo` — корень репозитория с зафиксированными входами; `--archive-dir` — папка с двумя ZIP; `--out` — новая или пустая папка результатов. Пути задаются пользователем и не зависят от расположения исходной исследовательской среды.

Поместите в одну папку, например `../release-assets`, два файла релизов этого репозитория:

| Release asset | Релиз | SHA256 |
|---|---|---|
| `server-economy-snapshots.zip` | [context-data-2026-09-15](https://github.com/roshlight/sberindex-clustering-submission/releases/tag/context-data-2026-09-15) | `83cc1b420e22fccc9e51af4f3b5441ea58202b767f49cce46a81ff3454993f7f` |
| `server-population-snapshots.zip` | [population-data-2026-09-15](https://github.com/roshlight/sberindex-clustering-submission/releases/tag/population-data-2026-09-15) | `02b785cc9ab6d0dfd6bce6e6096cf7fc7a4149efa14a19f0483d495def647fde` |

Распаковывать эти ZIP для данного скрипта не требуется. Он читает `ARCHIVE_MANIFEST.json`, находит исходные ответы по SHA256 и проверяет конкретные клетки внутри архива. В новом манифесте сохраняются имена, размеры и хеши обоих assets. В `source_cell_checks.csv` поле `raw_archive` сохраняет исторические **логические обозначения** `work/enrichment-v3-release/...` и `work/enrichment-v2-release/...`, чтобы таблица совпадала с исходным запуском. Фактическое расположение файлов определяется только `--archive-dir` и именем ZIP; историческая структура папок не требуется.

В репозитории должны присутствовать шесть исходных таблиц, перечисленных в неизменённом [manifest.json](manifest.json):

- `reports/enriched_analysis/core_v2/annual_panel.csv`;
- `reports/enriched_analysis/core_v2/oof_predictions.parquet`;
- `reports/enriched_analysis/prices/municipal_price_growth.csv`;
- `data/raw/consumption.parquet`;
- `reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv`;
- `reports/enrichment_v2/population/municipal_population.csv`.

Также нужен `reports/enriched_analysis/core_v2/manifest.json`. Исходный Parquet расходов восстанавливается по [основной инструкции проекта](../../../README.md); два перечисленных ZIP служат проверке клеток населения и экономики и не содержат этот Parquet. Все шесть входов должны совпасть по SHA256 с манифестом исходного запуска примеров. Скрипт останавливается при расхождении; новая версия данных требует отдельного анализа, а не незаметной замены зафиксированных входов.

Из корня репозитория:

```powershell
uv sync --frozen
uv run --frozen python scripts/build_enriched_examples.py --repo . --archive-dir ../release-assets --out ../examples-reproduced
```

Не используйте Python `-O`: проверка исходных клеток опирается на assertions, и скрипт явно отклоняет этот режим. Повторная запись в непустую папку запрещена. Папка `reports/enriched_analysis/examples` и её подпапки защищены независимо от содержимого. Для следующего запуска укажите другую новую папку.

Проверочный запуск в отдельную scratch-папку дал побайтовое совпадение трёх CSV и `qa.json` с каноническими файлами:

| Файл | SHA256 обоих экземпляров |
|---|---|
| `six_oof_examples_2024.csv` | `229228616861a4818438ade314c7eddf61c2a9675fb853f62c9ea0cbfe8b160d` |
| `selected_consumption_source_rows.csv` | `f18283497cef2c3fa6d8ccd49bbedd689de948ccaa0a72a80334c0f122995324` |
| `source_cell_checks.csv` | `17e0ba0b9cfa4574b3bd967befd9bf0221973846ca9c17a37446359ab437e312` |
| `qa.json` | `2aeb94cab084171c7897e62217e0f28131bab6e2a21f85848c87b73378f66d12` |

Повторная QA подтвердила тот же выбор из 1998 OOF-наблюдений, шесть ID и все 18 исходных клеток. Попытки записи в каноническую и уже заполненную scratch-папки завершились с кодом `2`, не изменив канонические файлы. Ruff — PASS. Подробный журнал локальной проверки сохранён в `work/enriched-analysis/examples/portable-comparison.json` исходной рабочей поставки.

Манифест нового запуска отличается закономерно: он содержит хеш переносимого скрипта, хеш исходного манифеста и явную атрибуцию архивов. Исходный `manifest.json` сохраняет хеш scratch-скрипта первоначального исполнения; он и фактические выводы `FINDINGS.md` не переписаны. Перенос не меняет модель, выбор примеров, числовые результаты или ограничения их интерпретации.
