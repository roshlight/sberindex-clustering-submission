# Воспроизведение шестого анализа

Проверки `verify_*_bundle.py` сверяют исходные тексты документов. Выполняйте их в [копии с первоначальной документацией](../../submission/REPRODUCE.md#проверка-исторических-контрольных-сумм); расчётные команды и данные от редакционных изменений не зависят.

Владелец — команда `sberindex-hackathon-2026/municipal-economic-profiles`. Анализ выполнен на переходе 16–17 сентября 2026 года по московскому времени. Исходный коммит перед анализом: `c39666da2b4a35223d3f3be96f6e1dfcc2456545`. Сохранённые источники относятся к шестому сбору; тег `data-round6-2026-09-16` содержит сбор, но ещё не этот анализ. Для анализа используйте коммит, содержащий данную папку, и сохраните `git rev-parse HEAD`.

Канонические результаты — `reports/round6_analysis/`, протокол — `configs/round6_analysis_v1.json`. Отчёт представлен в одной вычислительной цепочке: `artifact.json` содержит блоки и данные, `report.html` — автономный интерактивный документ, `REPORT.md` — представление для GitHub. Результаты прошлых раундов не заменяются.

## Контракт исследования

Вопрос: что новые региональные показатели и документы позволяют добавить к экономической интерпретации уже построенной сети? Новые модели и кластеры не обучаются, прежнее качество прогнозов не переоценивается, исторические адреса предприятий не восстанавливаются автоматически.

Единица — муниципалитет за год с равным весом. Основная парная выборка: 33 МО Ставрополья и 38 МО Саратовской области в каждом году. В неё входят только одинаковые идентификаторы с доступным населением, годовым профилем и уровнем расходов. Прежняя модельная маска не используется: она требует также зарплату и работников, которые не нужны для этого вопроса. Пропуски урбанизации не заменяются нулями.

Сравниваются ранговые связи территориальной интенсивности отгрузки / продовольственной торговли с уровнем расходов и долей питания. Основная поправка — численность населения. Урбанизация проверяется отдельно на 19 / 32 МО, с пересчётом исходной и скорректированной связи на тех же строках. Подвыборка стабильных кодов с урбанизацией в Ставрополье содержит 9 МО и не оценивается при минимуме 12.

2000 перевыборок идентификаторов внутри каждой региональной выборки используют одинаковые случайные индексы для обоих годов. Ранги и регрессии пересчитываются внутри каждого повтора. Интервалы условны, не учитывают пространственную зависимость и множественность исследуемых ассоциаций. Сопоставление с замороженными метками использует 4999 перестановок и BH по четырём сравнениям; этот условный тест не проверяет пространственную независимость или устойчивость кластеров. Годовые метки не сопоставляются по номеру между годами.

Протокол уточнён после проверки структуры, до результатов ассоциаций: удалены ненужные требования зарплаты/работников, контроль урбанизации вынесен в отдельную подвыборку из-за пропусков. Это явно записано в JSON. Оба года и прежние группы уже изучались, нового независимого holdout нет.

## Окружение и сырьё

Нужны Python 3.12, `uv`, Git и доступ команды к частному GitHub через `gh`. Использован существующий Windows runtime Python 3.12.10 и корневой `uv.lock`. GPU, Microsoft Word, сервер, SearXNG и ключи API для анализа не нужны. Ориентир свободного диска сверх окружения — 1 ГБ; это запас, не измеренный максимум памяти.

Если уже есть проверенный каталог `work/data-round6/economy`, новую загрузку не делайте. Иначе из корня репозитория:

```powershell
uv sync --python 3.12.10 --frozen
New-Item -ItemType Directory -Force ../../work/round6-analysis-assets
gh release download data-round6-2026-09-16 --repo roshlight/sberindex-clustering-submission --pattern data-round6-economy.zip --dir ../../work/round6-analysis-assets
$collectionSeal = Get-Content -Raw reports/data_round6/delivery_manifest.json | ConvertFrom-Json
$expectedArchive = @($collectionSeal.raw_releases | Where-Object { $_.path -eq 'data-round6-economy.zip' })
if ($expectedArchive.Count -ne 1) { throw 'Expected one pinned economy archive' }
$archivePath = (Resolve-Path ../../work/round6-analysis-assets/data-round6-economy.zip).Path
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedArchive[0].sha256) { throw 'Archive checksum mismatch' }
New-Item -ItemType Directory ../../work/round6-analysis-inputs
Expand-Archive -LiteralPath $archivePath -DestinationPath ../../work/round6-analysis-inputs
```

Префикс внутри архива — `raw6/economy`. Проверяется хеш, закреплённый в репозитории. Анализ использует сохранённые оригиналы и конверсии Word; заново конвертировать DOC не требуется. Остальные входы — проверенные таблицы и манифесты, уже находящиеся в репозитории.

## Проверка и полный вычислительный повтор

```powershell
uv run --frozen --offline python scripts/verify_round6_bundle.py --repo .
uv run --frozen --offline python scripts/verify_round6_analysis.py --repo . --economy-raw ../../work/round6-analysis-inputs/raw6/economy --out ../../work/round6-independent-QA.json
uv run --frozen --offline python scripts/replay_round6_analysis.py --repo . --economy-raw ../../work/round6-analysis-inputs/raw6/economy --out ../../work/round6-independent-replay
```

Выходные пути должны быть новыми. Полный повтор заново строит `admission`, `regions`, `context`, затем запускает независимый проверяющий код на этих свежих таблицах через `--root`. Он не заменяет расчёт чтением канонических результатов.

Допуск и контекст сравниваются побайтно, включая манифесты. В региональных таблицах идентификаторы, ключи, схема, флаги и пропуски сравниваются точно; вещественные числа допускают `atol=1e-9`, `rtol=1e-9`. Независимые формулы используют `1e-10` для ранговых статистик и `atol=1e-8`, `rtol=1e-12` для денежных сопоставлений. Первичные денежные литералы и проверки печатного округления используют Decimal.

Проверка включает все 40 строк ассоциаций, в том числе четыре намеренно не оценённые строки; пересчёт bootstrap, leave-one-out, перестановок, BH, региональных сводок, процентов источника и контекстных отношений. Она не повторяет независимое визуальное чтение всех PDF, конвертацию Word или скачивание страниц. Чистое окружение другого получателя ещё не проверено.

## Отчёт и проверки кода

```powershell
uv run --frozen --offline python scripts/build_round6_report.py --repo . --root ../../work/round6-independent-replay --out ../../work/round6-independent-report
uv run --frozen --offline python scripts/verify_round6_report_claims.py --repo . --root ../../work/round6-independent-replay --report ../../work/round6-independent-report --out ../../work/round6-independent-report-QA.json
uv run --frozen --offline pytest -q
uv run --frozen --offline ruff check .
uv run --frozen --offline ruff format --check .
```

Сборка создаёт машинную часть отчёта; ручные `REPRODUCE.md`, `QA.md` и `independent_review.md` при отдельной передаче скопируйте с сохранением имён. В отчётном JSON закреплены источники и путь вычисления.

Для автономного HTML используется установленный Data Analytics 0.2.10; это отдельная зависимость сборки, не Python lock. Готовый HTML работает без плагина и сети. Укажите фактический корень плагина вместо `<DATA_ANALYTICS_PLUGIN>`:

```powershell
node scripts/build_round2_portable.mjs --plugin <DATA_ANALYTICS_PLUGIN> --input ../../work/round6-independent-report/artifact.json --output ../../work/round6-independent-report/report.html --receipt ../../work/round6-independent-report/portable_delivery.json --work-dir ../../work/round6-independent-native
```

Используется прежний неизменённый помощник упаковки с ограниченной CSS-поправкой ширины заголовка Windows. Он вызывает штатные построение, извлечение SVG и браузерную проверку плагина. Для браузерной проверки нужен уже установленный Chromium; при необходимости задайте `CHROMIUM_EXECUTABLE_PATH` или `PLAYWRIGHT_BROWSERS_PATH`. HTML не требует соседних файлов данных для чтения.

Состав поставки и SHA фиксирует `delivery_manifest.json`; статусы — [QA.md](QA.md). Для нового источника или нового определения создавайте отдельный раунд. Нельзя менять ожидаемые хеши только ради прохождения проверки, принимать кассу за долю инвестиций, переносить работников 2025 года назад или объявлять условные совпадения кодов доказанными историческими адресами.
