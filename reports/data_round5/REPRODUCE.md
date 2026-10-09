# Воспроизведение пятого сбора

Статус: **ready for independent reproduction**. Все 48 машинных результатов повторены из сохранённых источников в существующем окружении Windows. Полное воспроизведение новым получателем в чистом окружении ещё не выполнено. Поиск и ответы моделей изменчивы; повторяется сохранённый снимок, а не будущая выдача.

Владелец поставки — команда `sberindex-hackathon-2026/municipal-economic-profiles`. Дата: 16.09.2026. Исходный коммит до сбора: `8a7e3cf94272b2b5425e3ec891ed962c8ea6ea32`. Тег поставки: `data-round5-2026-09-16`. Контрольные суммы кода, результатов и документов перечислены в [delivery_manifest.json](delivery_manifest.json).

## Что воспроизводится

Пакет закрывает часть пробелов после четвёртого анализа первого кейса: 64 дополнительные денежные ячейки, аудит исторических источников МСП, проверка кодов по шести версиям ОКТМО и документы по инвестиционным объектам. Канонический результат находится в `reports/data_round5/`; прежние панели, маски и модели сохранены. Ограничения применения перечислены в [REPORT.md](REPORT.md).

| Ветвь | Входы | Результат и проверка |
|---|---|---|
| Экономика | HTTP-ответы БДПМО, предыдущий кандидат и прежние таблицы соответствия | 15 файлов; 64 новых числа; сохранность прежних 11 608 чисел и реквизитов |
| МСП | Паспорта, схемы, выбранные XML и Range-ответы, XLSX ФНС, PDF, исходный код поставщика | 15 машинных файлов; время, схема, контрольные численности; присоединение отключено |
| География | Шесть полных CSV ОКТМО, полные файлы МСП из четвёртого раунда | 9 файлов; точное наличие кода, уровень и даты; историческая адресная связь не доказана |
| Объекты | Документы, просмотренные страницы, сохранённый ручной рецепт | 7 файлов; реестр свидетельств и проверенных гипотез; причинность не установлена |
| Поиск | Полный серверный снимок и снимок реализации | 2 файла; проверка сохранённых хешей и счётчиков |

Все 48 файлов сравниваются побайтно. Ручные README, рецепты и отдельные аудиторские заключения не выдаются за автоматически восстановленные результаты; список сохранён в [offline_replay_QA.json](offline_replay_QA.json). Отдельный аудит экономики использует самостоятельный парсер. Независимая проверка географии выполнена разовой командой; её журнал сохранён, отдельный исполняемый проверяющий скрипт не сохранён.

## Окружение и получение сырья

Нужны доступ команды к частному GitHub через `gh`, Git, Python 3.12 и `uv`. Зависимости зафиксированы в `enrichment/uv.lock`. Для полного повтора нужны также исходники МСП четвёртого раунда: пятая поставка не дублирует большие архивы. Ориентир для свободного диска — 6 ГБ с учётом архивов, распаковки и результатов; желательно 8 ГБ ОЗУ. DuckDB ограничен 2 потоками и 2 ГБ. GPU не требуется. Replay не вызывает OpenRouter, SearXNG или Telegram и не требует их ключей.

В новом пустом рабочем каталоге:

```powershell
New-Item -ItemType Directory outputs
gh repo clone sberindex-hackathon-2026/municipal-economic-profiles outputs/municipal-economic-profiles
Set-Location outputs/municipal-economic-profiles
git switch --detach data-round5-2026-09-16
uv sync --project enrichment --frozen
New-Item -ItemType Directory -Force ../../work/data-round5/release
New-Item -ItemType Directory -Force ../../work/data-round4/release
gh release download data-round5-2026-09-16 --pattern '*.zip' --pattern SHA256SUMS.txt --dir ../../work/data-round5/release
gh release download data-round4-2026-09-16 --pattern data-round4-sme.zip --pattern SHA256SUMS.txt --dir ../../work/data-round4/release
```

Проверить каждый скачанный архив до распаковки. В четвёртом раунде скачивается только ветвь МСП, поэтому остальные строки её общего списка сумм не используются:

```powershell
foreach ($releaseFolder in @('../../work/data-round5/release', '../../work/data-round4/release')) {
    $assetRoot = (Resolve-Path $releaseFolder).Path
    $expectedHashes = @{}
    Get-Content (Join-Path $assetRoot SHA256SUMS.txt) | ForEach-Object {
        $parts = $_ -split '\s+', 2
        $expectedHashes[$parts[1]] = $parts[0]
    }
    Get-ChildItem -LiteralPath $assetRoot -Filter '*.zip' | ForEach-Object {
        $actualHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $expectedHashes.ContainsKey($_.Name) -or $actualHash -ne $expectedHashes[$_.Name]) {
            throw "Archive checksum mismatch: $($_.Name)"
        }
        Expand-Archive -LiteralPath $_.FullName -DestinationPath ../..
    }
}
uv run --project enrichment --frozen python enrichment/scripts/extract_round4_sme.py --raw ../../work/data-round4/sme
```

Внутри новых ZIP сохранён префикс `work/data-round5/`; не убирайте его. Каждый ZIP содержит собственный `MANIFEST.data-round5-<branch>.json`. Архив МСП четвёртого раунда хранит оригинальные ZIP поставщика; `extract_round4_sme.py` проверяет их SHA/CRC, безопасные пути и извлекает только отсутствующие Parquet. Метаданные скачивания не переписываются.

## Единый повтор

Из корня репозитория:

```powershell
uv run --project enrichment --frozen python enrichment/scripts/verify_round5_delivery.py --repo .
uv run --project enrichment --frozen python enrichment/scripts/replay_round5.py --repo . --raw ../../work/data-round5 --prior-sme-raw ../../work/data-round4/sme --output ../../work/round5-independent-replay
uv run --project enrichment --frozen python enrichment/scripts/verify_round5_economy_independent.py --repo . --raw ../../work/data-round5/economy --output ../../work/round5-independent-economy.json
```

Каталог повторного результата должен быть новым и находиться вне репозитория и входных каталогов. Первый проверяющий скрипт сверяет файлы поставки и заявленное происхождение; он не заменяет повтор расчётов или проверку сырья. Вторая команда запускает пять ветвей, проверяет исходные хеши, сравнивает результаты, сохраняет журналы и проверяет отсутствие изменений входных каталогов. Ошибка шага или несовпадение хеша останавливает повтор; подробности остаются в `QA.json` и журналах. Третья команда повторяет независимую проверку экономики.

Рецепт чтения документов — отдельный вход. Побайтовое совпадение не заменяет визуальный просмотр заголовков, сносок и соседних строк. В пакете сохранены документы, страницы и локаторы, позволяющие выполнить эту проверку заново. Для четырёх выбранных XML ФНС проверены CRC и локальные SHA; полные исторические ZIP ФНС не скачаны, полная XSD-валидация не выполнялась.

## Проверки и обновления

```powershell
uv run --project enrichment --frozen pytest enrichment/tests -q
uv run --project enrichment --frozen ruff check enrichment
uv run --project enrichment --frozen ruff format --check enrichment
```

Результаты выполненных проверок — в [QA.md](QA.md). Код упаковки и явные списки включаемых файлов сохранены в `enrichment/scripts/package_round4_delivery.py`, `package_round5_delivery.py` и `delivery/`. ZIP содержит время упаковки, поэтому его повторная сборка не обещает одинаковый хеш; проверяется равенство включённых исходных файлов. Второй упаковщик отличается точечными, проверенными исключениями для байтов JPEG DHT в одном PDF; доказательство находится в `delivery/scanner_false_positive_review.json`.

Для следующего сетевого сбора используйте новые каталоги и заново проверьте единицы, охват, годы и ключи. Отличие хеша означает новую версию источника, а не разрешение заменить ожидаемый хеш старого отчёта. Пакеты прошлых раундов сохраняются.

Сырые файлы содержат открытые регистрационные записи и документы источников; в отчётных таблицах используются агрегаты качества. Доступ к поставке ограничен частным репозиторием команды. Атрибуция, исходные URL и ограничения перечислены в предметных README и реестрах. Запросы владельцам данных подготовлены, но не отправлялись. Для следующего анализа отдельно потребуется решение команды о применении нового денежного кандидата; готовая историческая муниципальная переменная МСП и доли отдельных объектов в инвестициях пока отсутствуют.
