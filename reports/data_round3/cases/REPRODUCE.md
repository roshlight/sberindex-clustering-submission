# Воспроизведение

Нужны репозиторий и неизменённые файлы из `raw_inventory.json`. Их рабочий корень относительно workspace: `work/data-round3/cases/`. В релиз следует включать только перечисленные 155 файлов; модели OCR, виртуальное окружение и ключи в этом каталоге отсутствуют. Размер на момент фиксации — 81 897 740 байт. Имена и SHA‑256 в реестре являются точным allowlist.

Все команды ниже выполняются из `outputs/municipal-economic-profiles/enrichment` в PowerShell. Используется существующий `uv.lock`.

```powershell
uv sync --frozen
uv run --frozen python scripts/collect_round3_cases.py --finalize --raw-dir ../../../work/data-round3/cases --evidence-recipe ../reports/data_round3/cases/evidence_recipe.json --output ../reports/data_round3/cases
uv run --frozen pytest tests/test_round3_cases.py -q
uv run --frozen ruff check scripts/collect_round3_cases.py tests/test_round3_cases.py
uv run --frozen ruff format --check scripts/collect_round3_cases.py tests/test_round3_cases.py
```

`--finalize` работает без сети, проверяет хеши и локаторы и заново создаёт `factual_evidence.csv`, `source_inventory.json`, `raw_inventory.json` и `summary.json`. Рецепт — результат ручного чтения источников, а не универсальный распознаватель отчётов. Изменение числа в рецепте само по себе не проходит семантическую экспертизу; для нового рецепта нужна повторная проверка страницы и единиц.

Необязательное повторное скачивание выполняется в **новый** каталог, чтобы сохранить текущий снимок:

```powershell
uv run --frozen python scripts/collect_round3_cases.py --sources ../reports/data_round3/cases/sources.json --raw-dir ../../../work/data-round3/cases-redownload --interval 1.5
```

Новые байты могут отличаться: сайты обновляются, некоторые ссылки уже недоступны. Скрипт делает один запрос на URL, проверяет разрешённые хосты перенаправлений и не повторяет ошибочные ответы. В прежнем каталоге повторный запуск использует сохранённые метаданные и проверяет хеши. Для Росстата при необходимости можно передать `--ca-bundle` с публичным доверенным PEM; отключения TLS-проверки нет.

Стратегия Свободненского района находится в `svob_strategy2035.rar`. В поставку включены сам архив, оба его извлечённых документа и `svob_strategy2035_members.json`. DOCX используется по точному пути и хешу из рецепта. Для проверки происхождения членов архива можно распаковать его в отдельный каталог и сравнить хеши:

```powershell
tar -tf ../../../work/data-round3/cases/svob_strategy2035.rar
New-Item -ItemType Directory ../../../work/data-round3/strategy-replay -Force
tar -xf ../../../work/data-round3/cases/svob_strategy2035.rar -C ../../../work/data-round3/strategy-replay
```

PDF извлекались `pypdf`; физические страницы сохранены в `*.extracted.json`. Для визуальной проверки использовался Poppler `pdftoppm -f N -l N -scale-to 1600 -png INPUT OUTPUT_PREFIX` (для страницы 14 сборника и страницы 4 Светловского отчёта — масштаб 1800). Результаты рендеринга включены в сырьё. OCR слайдов управы выполнен Tesseract 5 с русской моделью: `tesseract IMAGE OUTPUT_PREFIX -l rus --tessdata-dir MODEL_DIRECTORY --psm 3`. Русская модель использована из установленного ранее рабочего каталога, не входит в поставку; для числового replay она не требуется.

`existing_baseline_reference.csv` — зафиксированный срез шести строк `reports/enriched_analysis/core_v2/annual_panel.csv`. Его исходный SHA‑256 указан в каждой строке. Это справка для сравнения, а не вход извлечения новых фактов. README, QA и роли источников являются ручной проверкой; их содержание не генерируется из чисел автоматически.

Статус воспроизводимости: извлечение 53 записей воспроизводимо по сохранённым источникам и рецепту; актуальный повторный веб-поиск, доступность сайтов и независимая повторная экспертиза всех формулировок не гарантируются.
