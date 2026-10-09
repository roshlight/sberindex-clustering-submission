# Воспроизведение четвёртого раунда анализа

Статус: **ready for independent reproduction**. В существующем окружении Windows независимо пересчитаны **700 Ridge fit через NumPy**, проверены исходные значения, разбиения, метрики и интервалы. Запуск всего пути отдельным получателем в новом чистом окружении — **NOT RUN**. Это готовность к такому воспроизведению, а не заявление о завершённой проверке с нуля.

Вопрос раунда: меняют ли восстановленные денежные наблюдения решение о добавлении этого блока в объяснение муниципальных потребительских профилей. Кейс 1; 2023/2024 годы; прежние 2016 МО, основная выборка 1975/1998. Порог добавления и все варианты расчёта зафиксированы в [конфигурации](../../configs/round4_analysis_v1.json). Результат: критерий не выполнен; денежные показатели остаются описательным контекстом. Новый муниципальный слой МСП, причинные оценки и новые кластеры не заявляются.

## Входы и окружение

Все команды выполняются из корня репозитория. Анализ использует **Python 3.12 и корневой `uv.lock`**. `enrichment/uv.lock` нужен только для отдельного извлечения исходников МСП. GPU, сервер, Telegram, SearXNG и OpenRouter для воспроизведения не нужны; платные API не вызываются. После установки зависимостей и получения сырья числовые этапы работают без сети.

Снимок сбора зафиксирован тегом [data-round4-2026-09-16](https://github.com/sberindex-hackathon-2026/municipal-economic-profiles/releases/tag/data-round4-2026-09-16); базовый коммит протокола — `f443a93d77ca4149b2e49e517b053c2f6e5c7814`. Для кода самого анализа сохраните идентификатор полученного коммита через `git rev-parse HEAD`. Не переходите на тег сбора для запуска анализа: новых аналитических скриптов на нём ещё нет.

Канонические результаты — `reports/round4_analysis/`; прежние `reports/round3_analysis/` и входы `reports/data_round4/` сохраняются. Выходы в `work/` являются повторными расчётами, а не новой принятой версией. Хеши и зависимости перечислены в манифестах компонентов и поставки.

```powershell
uv sync --frozen
uv run --frozen python scripts/verify_round4_bundle.py
uv run --frozen python scripts/verify_round4_analysis.py --repo . --out work/round4-independent-replay.json
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Проверка пакета сверяет опубликованные байты; независимый валидатор отдельно восстанавливает вычисления. Он подтверждает 266 426 строк OOF, 280 строк метрик, 256 сравнений и 2052 строки описательных срезов. Зафиксированный максимум расхождения Ridge — **7.105427357601002e-14**, эффекта и границ интервала — **6.453171330633722e-16**. Код допускает абсолютную ошибку менее `1e-8` для Ridge и срезов, менее `1e-10` для остальных проверяемых числовых групп. Подробности: [independent_QA.json](independent_QA.json) и [independent_review.md](independent_review.md).

Модельная проверка читает сохранённые таблицы репозитория и не требует сырых Parquet МСП. Для повторения контекстного аудита нужен следующий отдельный шаг.

## Получение исходников МСП

Нужен доступ участника команды к частному репозиторию через `gh`. Если `../../work/data-round4/sme/` уже содержит проверенный снимок, повторную загрузку пропустите. Для новой загрузки используйте пустой каталог:

```powershell
New-Item -ItemType Directory -Force ../../work/round4-analysis-assets
gh release download data-round4-2026-09-16 --repo sberindex-hackathon-2026/municipal-economic-profiles --pattern data-round4-sme.zip --pattern SHA256SUMS.txt --dir ../../work/round4-analysis-assets
$smeAssets = (Resolve-Path ../../work/round4-analysis-assets).Path
$smeChecksum = @(Get-Content (Join-Path $smeAssets SHA256SUMS.txt) | Where-Object { $_ -match '\sdata-round4-sme\.zip$' })
if ($smeChecksum.Count -ne 1) { throw 'Expected one SME archive checksum' }
$smeExpected = ($smeChecksum[0] -split '\s+', 2)[0]
$smeArchive = Join-Path $smeAssets data-round4-sme.zip
if ((Get-FileHash -LiteralPath $smeArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $smeExpected) { throw 'SME archive checksum mismatch' }
Expand-Archive -LiteralPath $smeArchive -DestinationPath ../..
uv sync --project enrichment --frozen
uv run --project enrichment --frozen python enrichment/scripts/extract_round4_sme.py --raw ../../work/data-round4/sme
```

Внешний ZIP содержит префикс `work/data-round4/sme/`; его нельзя убирать. Внутри лежат оригинальные ZIP поставщика. Экстрактор проверяет SHA-256, CRC и пути, создаёт только отсутствующие Parquet и не меняет метаданные получения. `--sme-raw` в аналитическом скрипте указывает на каталог с `retrievals.json`, `raw/` и `extracted/`, а не на ZIP или отдельный Parquet. Источник: ФНС; обработка [«Если быть точным»](https://tochno.st/datasets/sme), `v20260605`; ограничения и атрибуция — в [описании поставки](../data_round4/sme/README.md).

## Повтор отдельных этапов без изменения канонических результатов

Каталоги вывода должны быть новыми или пустыми; ряд скриптов намеренно отказывается перезаписывать результат.

```powershell
uv run --frozen python scripts/round4_data_audit.py --repo . --out work/round4-replay/audit
uv run --frozen python scripts/round4_context_analysis.py --repo . --sme-raw ../../work/data-round4/sme --out work/round4-replay/context
uv run --frozen python scripts/round4_model_analysis.py --repo . --out work/round4-replay/models
uv run --frozen python scripts/round4_model_readout.py --root reports/round4_analysis --out work/round4-replay/readout
uv run --frozen python scripts/plot_round4_analysis.py --root reports/round4_analysis --out work/round4-replay/figures
uv run --frozen python scripts/build_round4_report.py --repo . --out work/round4-replay/report
```

Эти команды повторяют каждый этап от опубликованных входов. `--out` **не перенаправляет входы последующих этапов**: модель читает `reports/round4_analysis/audit/annual_features.csv`; readout и график читают `<--root>/models/`; сборщик отчёта читает канонические `audit/`, `models/`, `readout/` и `context/` внутри `reports/round4_analysis/`. Исходные цели, прежние признаки и folds приходят из третьего раунда. Поэтому список выше не следует выдавать за последовательный расчёт от только что созданных файлов.

Для проверки всей цепочки создайте **отдельный checkout того же коммита**. Следующие команды создают соседний worktree и переименовывают только его опубликованный каталог. Они не меняют основной рабочий каталог; имя нового worktree должно быть свободно.

```powershell
git worktree add --detach ../municipal-economic-profiles-round4-replay HEAD
Set-Location ../municipal-economic-profiles-round4-replay
uv sync --frozen
$round4ReplayRoot = (Resolve-Path .).Path
$round4PublishedPath = (Resolve-Path reports/round4_analysis).Path
if (-not $round4PublishedPath.StartsWith($round4ReplayRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Published path is outside the replay checkout' }
Rename-Item -LiteralPath $round4PublishedPath -NewName round4_analysis_published
```

Затем выполните этапы по порядку с каноническими путями вывода:

```powershell
uv run --frozen python scripts/round4_data_audit.py --repo . --out reports/round4_analysis/audit
uv run --frozen python scripts/round4_context_analysis.py --repo . --sme-raw ../../work/data-round4/sme --out reports/round4_analysis/context
Copy-Item -LiteralPath reports/round4_analysis_published/context/README.md -Destination reports/round4_analysis/context/README.md
Copy-Item -LiteralPath reports/round4_analysis_published/REPRODUCE.md -Destination reports/round4_analysis/REPRODUCE.md
uv run --frozen python scripts/round4_model_analysis.py --repo . --out reports/round4_analysis/models
uv run --frozen python scripts/round4_model_readout.py --root reports/round4_analysis --out reports/round4_analysis/readout
uv run --frozen python scripts/plot_round4_analysis.py --root reports/round4_analysis --out reports/round4_analysis/figures
uv run --frozen python scripts/build_round4_report.py --repo . --out reports/round4_analysis
uv run --frozen python scripts/verify_round4_analysis.py --repo . --out work/round4-independent-chain.json
uv run --frozen python scripts/verify_round4_report_claims.py
```

Если отдельный checkout расположен иначе, передайте фактический путь к прежнему снимку через `--sme-raw`. Текстовые документы, например `context/README.md` и эта инструкция, сохраняются из опубликованного пакета: вычислительные скрипты не генерируют их заново. `verify_round4_report_claims.py` всегда пишет `reports/round4_analysis/report_claims_QA.json`; отдельный checkout нужен и для сохранения прежнего QA без перезаписи.

Аудит проверяет 4032 муниципалитета–года, неизменность старых значений и масок, 779 новых числовых ячеек и 770 новых непустых значений логарифмических признаков. Контекстный повтор дал побайтовое совпадение девяти машинных выходов. При новом обучении сравниваются ключи, определения, значения с численным допуском и итоговое решение. Метки времени, сериализация Parquet и служебные поля SVG не обязаны совпадать побайтово. Не заменяйте ожидаемые хеши опубликованного пакета хешами повторного расчёта ради прохождения проверки.

## HTML и визуальная проверка

`artifact.json` — канонические блоки и данные отчёта; `REPORT.md` — версия для GitHub; `report.html` — автономный интерактивный файл. Для чтения готового HTML плагин не требуется. Для пересборки использован установленный Data Analytics 0.2.10; его версия и хеши runtime записаны в [portable_delivery.json](portable_delivery.json).

```powershell
node scripts/build_round2_portable.mjs --plugin <DATA_ANALYTICS_PLUGIN> --input reports/round4_analysis/artifact.json --output work/round4-replay/report.html --receipt work/round4-replay/portable_delivery.json --work-dir work/round4-native-build
```

`<DATA_ANALYTICS_PLUGIN>` заменяется локальным корнем установленного плагина; это отдельная зависимость сборки HTML, не часть `uv.lock`. Штатный сборщик обнаружил переполнение при полосе прокрутки Windows. Прежний проверенный wrapper применил одну CSS-поправку ширины заголовка; payload и runtime не менялись. Штатный валидатор подтвердил **25 блоков, 3 графика, 9 таблиц**, меню источника и ширины **1440/390 px**, без внешних запросов. Это не проверка всех браузеров и печати. Экспортированный исследовательский PNG осмотрен отдельно; границы такого осмотра записаны в `figures/QA.json`. При новой сборке её визуальную проверку следует записать заново.

## Передача и ограничения

Владелец результата — команда `sberindex-hackathon-2026/municipal-economic-profiles`; разбор расхождений ведётся в репозитории и существующем чате. Перед изменением версии сохраните старые входы, конфигурацию и результаты, затем задайте новый ограниченный протокол. Отсутствующий файл, изменившийся хеш или непрошедший численный допуск — основание разобраться в версии или вычислении, а не подменить эталон.

Оба года уже изучались: весь раунд остаётся исследовательским и ретроспективным. Региональные интервалы условны по сохранённым OOF-прогнозам; полного переобучения bootstrap и поправки на множественные проверки нет. Доступность источника в 2026 году не устанавливает историческую доступность показателя. Совпадающие полигоны не доказывают юридическую сопоставимость; МСП с географией июня 2026 года не присоединён к панели; отсутствующие работники ИП не превращены в нули. Успешное исполнение кода не снимает этих ограничений и не меняет решение о денежном блоке.
