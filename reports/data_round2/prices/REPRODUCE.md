# Воспроизведение

Рабочая директория команд — `outputs/municipal-economic-profiles/enrichment`. Используется отдельный существующий `enrichment/uv.lock`; исходный эксперимент и его окружение не меняются. Исходники перечислены с SHA в `raw_inventory.json`. Архив нужно распаковать с сохранением путей относительно рабочего каталога пользователя: `work/data-round2/prices/` остаётся вне репозитория. Модельные API и интернет для повторного извлечения не нужны.

Новые исходники: две книги `sred_potreb_cen_2023.xlsx` и `sred_potreb_cen_2024.xlsx`, копия `isg_2012-2025.xlsx`, приказы 975/665/657, методология440, официальный ОКТМО2024, страницы и метаданные загрузки. Для воспроизведения OCR-движок не требуется: точные тексты страниц `975-60.txt`…`975-69.txt` и `665-13.txt`…`665-22.txt` сохранены вместе с PDF; исправления распознавания заданы явно в коде. `rus.traineddata` является инструментальным файлом и не входит в набор данных.

Старые требуемые входы сохранены небольшими точными копиями внутри `work/data-round2/prices/repo_inputs/` с их исходными путями относительно репозитория: `reports/enrichment_v3/prices/city_cost_of_living_2023_2024.csv`, `reports/initial_study/assignments.parquet`, оба годовых `data/derived/enrichment/municipal_crosswalk_YYYY-01-01.csv`. Если рабочая копия репозитория уже содержит эти файлы, используются они; проверьте SHA по инвентарю. Если их нет, восстановите копии по указанным относительным путям. Не подменяйте существующие файлы другой версии. Исходные CSV ОКТМО за 08.02.2023 и 25.01.2024, а также их download metadata, находятся в новом raw-подкаталоге `registry/`. Другой старый релиз для этой процедуры не требуется.

```powershell
uv sync --frozen

uv run --frozen python scripts/prepare_round2_prices.py `
  --repo-root .. `
  --raw-dir ../../../work/data-round2/prices `
  --registry-dir ../../../work/data-round2/prices/registry `
  --workbook ../../../work/data-round2/prices/isg_2012-2025.xlsx `
  --output ../reports/data_round2/prices_replay/bridge_v2

uv run --frozen python scripts/extract_round2_prices.py `
  --repo-root .. `
  --raw-dir ../../../work/data-round2/prices `
  --bridge-dir ../reports/data_round2/prices_replay/bridge_v2 `
  --output ../reports/data_round2/prices_replay/monthly_v3

uv run --frozen python scripts/verify_round2_prices.py `
  --raw-dir ../../../work/data-round2/prices `
  --monthly-dir ../reports/data_round2/prices_replay/monthly_v3 `
  --output ../reports/data_round2/prices_replay/source_cell_verification_v3.json

uv run --frozen pytest tests/test_round2_prices.py -q
uv run --frozen python scripts/summarize_round2_prices.py `
  --repo-root .. `
  --report-dir ../reports/data_round2/prices_replay `
  --raw-inventory ../reports/data_round2/prices/raw_inventory.json

uv run --frozen ruff check scripts/prepare_round2_prices.py scripts/extract_round2_prices.py scripts/verify_round2_prices.py scripts/summarize_round2_prices.py tests/test_round2_prices.py
uv run --frozen ruff format --check scripts/prepare_round2_prices.py scripts/extract_round2_prices.py scripts/verify_round2_prices.py scripts/summarize_round2_prices.py tests/test_round2_prices.py
```

Три команды извлечения и XML-проверки отказывают в перезаписи существующего выходного каталога или файла. Для второго повтора выберите новое имя. `summarize_round2_prices.py` повторно рассчитывает только производные QA-таблицы, ограничения, инвентарь и итоговый manifest в указанной новой папке; исходники, мост и числовые слои не меняются. Сравнивайте числовые таблицы, ключи, статусы и SHA исходников; время генерации, путь входа и служебный порядок строк JSON не являются результатом исследования. Если требуется побайтовое сравнение Parquet, используйте тот же lock и платформу: другой writer может сохранить те же значения иначе.

Два больших канонических Parquet находятся в `monthly_v3/`: `city_monthly_price_cells.parquet` и `primary_city_price_context.parquet`. Их можно доставлять через архив релиза; относительные пути должны сохраниться. Из `raw_inventory.json` следует брать только явно перечисленные файлы, а не весь рабочий каталог: в нём остаются промежуточные пробные извлечения и OCR-runtime, не относящиеся к итоговой поставке.
