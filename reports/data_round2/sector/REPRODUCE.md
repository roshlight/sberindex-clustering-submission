# Воспроизведение отраслевого сбора

Все команды запускаются из `enrichment/` существующего репозитория. Используется его `uv.lock`; отдельной установки библиотек и ключей API не требуется. `--repo`, `--output`, `--archive`, `--pilot` принимают пользовательские пути. Ни один скрипт не меняет прежние v1/v2/v3 или результаты исследования.

Нужны неизменённый `reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv`, исходный `reports/enrichment_v3/economy/source_capture_inventory.json` и новый каталог `work/data-round2/sector/` с оригинальными `cache/*.raw`, метаданными и checkpoint. Все 140 форм находятся в новом кэше: старый release asset `server-economy-snapshots.zip` для повторения не требуется. Необязательный `--archive` нужен только для первоначального восстановления отсутствующих форм из v3. HTTP payload входит в метаданные: один URL `DBInet.cgi` без него не задаёт таблицу.

Пилот шести фиксированных ID, исходные байты и итоговая таблица:

```powershell
uv run --frozen python scripts/run_round2_sector.py --repo .. --output ../../../work/data-round2/sector --interval 2.5
```

Та же команда с `--cache-only` пересчитывает пилот по сохранённым байтам, без сетевых запросов. У пилота 22 уникальных ответа; повторное чтение обоих годов использует кэш. В `qa.json` число запросов относится только к конкретному вызову, а не ко всей истории сбора.

Возобновление массового сбора по сохранённым checkpoint:

```powershell
uv run --frozen python scripts/run_round2_sector_bulk.py --repo .. --pilot ../../../work/data-round2/sector --output ../../../work/data-round2/sector/bulk --workers 2 --cache-only
```

Для предварительной проверки плана без запросов добавьте `--prepare-only`. `--limit-jobs` ограничивает текущий проход; `--workers` допускает только 1 или 2. Повторный запуск проверяет хеши выполненных checkpoint и использует сохранённые ответы. Для явно разрешённого возобновления сетевого сбора уберите `--cache-only`: незавершённый запрос может потребовать повторного получения, завершённые снимки не обновляются автоматически. `progress.json` обновляется после каждого задания, `jobs.json` фиксирует payload и целевые коды, `deferred_targets.json` сохраняет исходную причину исключения ID2710 из групп. Итоговые CSV, `source_inventory.json` и `qa.json` собираются после завершения прохода.

Полное повторное чтение всех checkpoint по первичным CSV, без сети, и воспроизведение отдельного решения ID2710:

```powershell
uv run --frozen python scripts/repair_round2_sector.py --bulk-dir ../../../work/data-round2/sector/bulk --check-all
uv run --frozen python scripts/review_round2_sector_caption.py --bulk-dir ../../../work/data-round2/sector/bulk --out ../../../work/data-round2/sector/caption_2710 --decision ../reports/data_round2/sector/caption_2710/accepted_caption_decisions.json
```

Решение ограничено сохранёнными хешами; новый или изменённый ответ не принимается по сходству подписи. Документы исторического реестра, на которые оно ссылается, уже опубликованы в пакете enrichment_v2 и не повторяются в новом архиве. Их точные URL и SHA указаны в журнале решения.

Локальная сверка общих итогов с v3 и одиночных пилотных ячеек с групповыми:

```powershell
uv run --frozen python scripts/audit_round2_sector.py --repo .. --source-dir ../../../work/data-round2/sector --out ../reports/data_round2/sector
uv run --frozen pytest tests/test_round2_sector.py -q
uv run --frozen ruff check scripts/run_round2_sector.py scripts/run_round2_sector_bulk.py scripts/audit_round2_sector.py tests/test_round2_sector.py
uv run --frozen ruff format --check scripts/run_round2_sector.py scripts/run_round2_sector_bulk.py scripts/audit_round2_sector.py tests/test_round2_sector.py
```

Финальное покрытие: 162210 кандидатных ячеек по 2016 МО, 4032 МО-года; 111384 числовых, 4105 маркеров, 46721 не возвращена. Parquet сохраняет строковые муниципальные коды, включая ведущие нули; `source_activity_code` — внутренний код измерения `okved2`. `actual_source_oktmo_label` — подпись текущего источника, `source_oktmo_name` также сохраняет прежнюю ссылку в обычных группах; соединение выполняется по кодам. `caption_decision` отделяет точное групповое сопоставление от исключения ID2710, `geographic_equivalence_confirmed` везде NULL. Значения работников и зарплаты не превращаются в показатели жителей.

QA имеет итоговый `WARN`: все 488 пилотных значения/NULL совпали, но 22 NULL представлены разными статусами в двух разворотах CSV. Это сохранённое ограничение источника; числовых расхождений нет. В 8064 общих позициях четыре прежних пропуска ID2710 заполнены по новому отдельному доказательству, остальные 8060 совпадают с v3.

В 2023 году общие работники и зарплата числовые для всех 2016 МО; в 2024 — для 2001 МО каждый показатель. Оставшиеся 30 позиций для 15 МО перечислены в [missing_aggregate_values.csv](missing_aggregate_values.csv). В полном слое сохранены два явно опубликованных нуля; отрицательных значений и нарушений восьмизначной длины ОКТМО нет. Итоговые проверки и воспроизводимые SQL-запросы — в [FINAL_QA.json](FINAL_QA.json); 15 тестов и финальные Ruff check/format check прошли. Точная спецификация файлов release-архива — [raw_package_allowlist.json](raw_package_allowlist.json): 803 файла, 557414492 байта до сжатия.

Аудит сверяет реальные raw SHA256 и размеры. `status=observed` означает полученную числовую ячейку. `source_marker_unresolved` в пилоте и `unresolved_source_marker` в массовом экспорте обозначают один класс нерасшифрованных маркеров; при сверке применяется только эта явная текстовая замена. `not_returned` означает, что ожидаемая по форме ячейка не пришла; `year_not_offered` — что запрошенный год отсутствует в форме. Ни один статус не преобразуется в ноль. Показатель успешного парсинга не удостоверяет полноту отраслевого охвата, точность статистического измерения или географическую сопоставимость.
