# Чтение результатов и запуск проекта

## Просмотр без вычислений

Начните с корневого README и `submission/METHODOLOGY.md`. Файл `index.html` открывается в браузере без сервера. Графики встроены в него; ссылки на подробные отчёты требуют полной папки репозитория. Готовые `reports/round2_analysis/report.html` и отчёты раундов 3–6 содержат сохранённые интерактивные представления.

Для таблиц основного исследования используйте `reports/initial_study/`. `assignments.parquet` хранит назначения, `cluster_profiles.csv` хранит профили групп, `membership_agreement.parquet` хранит поддержку, `transitions.parquet` и `flows.csv` описывают динамику. `method_metrics.csv` содержит сравнение методов. Метки локальны для месяца и метода.

## Основной эксперимент

Требуются Python 3.12 и uv. Зависимости закреплены в `uv.lock`; основной конфиг находится в `configs/robustness_v1.json`.

Из корня репозитория в macOS/Linux:

```sh
uv sync --frozen
mkdir -p work
curl -fL "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip" -o work/hackathonlicence.zip
uv run --frozen python scripts/import_archive.py --archive work/hackathonlicence.zip
uv run --frozen python scripts/run_study.py --phase development --run-id development-submission
uv run --frozen python scripts/freeze_evaluation.py --development-run results/development-submission --output work/submission_evaluation_lock.json
uv run --frozen python scripts/run_study.py --phase evaluation --run-id evaluation-submission --lock work/submission_evaluation_lock.json
uv run --frozen python scripts/verify_results.py --run-dir results/development-submission
uv run --frozen python scripts/verify_results.py --run-dir results/evaluation-submission
uv run --frozen python scripts/build_research_report.py --runs results/development-submission results/evaluation-submission --output reports/reproduced-submission
uv run --frozen python scripts/plot_research.py --runs results/development-submission results/evaluation-submission --report reports/reproduced-submission
```

В Windows PowerShell замените `mkdir -p work` на `New-Item -ItemType Directory -Force work`, а `curl` на `curl.exe`. Остальные команды совпадают. Новый запуск требует новых идентификаторов и выходных путей. Сохранённые результаты не перезаписываются. Импорт проверяет закреплённую контрольную сумму источника; изменившийся файл требует отдельного аудита.

Проверки проекта, предусмотренные авторами:

```sh
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

## Пересборка презентации

Исходник оформления находится в `scripts/build_presentation.py`. Он читает сохранённые таблицы исследования, строит графики и собирает 12 слайдов формата 16:9. Шрифты с поддержкой русского языка поставляются вместе с закреплённой версией Matplotlib.

```sh
uv sync --frozen --group presentation
uv run --frozen --group presentation python scripts/build_presentation.py --output submission/PRESENTATION.pdf
```

Для отдельной копии укажите другой путь в `--output`. Группа `presentation` добавляет зависимости сборки PDF к окружению анализа.

## Дополнительные исследования

Следуйте исходным инструкциям [обогащённого анализа](../reports/enriched_analysis/REPRODUCE.md), [раунда 2](../reports/round2_analysis/REPRODUCE.md), [раунда 3](../reports/round3_analysis/REPRODUCE.md), [раунда 4](../reports/round4_analysis/REPRODUCE.md), [раунда 5](../reports/round5_analysis/REPRODUCE.md), [раунда 6](../reports/round6_analysis/REPRODUCE.md) и [пространственного трека](../docs/reviews/spatial_validation.md).

Первичные архивы доступны в [релизах этого репозитория](https://github.com/roshlight/sberindex-clustering-submission/releases). [Каталог данных](DATA.md) связывает каждый этап с его архивами. Имена файлов, теги и контрольные суммы сохранены.

Пример скачивания архива:

```sh
gh release download data-round6-2026-09-16 --repo roshlight/sberindex-clustering-submission --pattern data-round6-economy.zip --dir work/round6-assets
```

Команды скачивания в подробных инструкциях уже указывают на этот репозиторий. Полный список архивов приведён в каталоге данных.

Сборка прежних интерактивных HTML использует отдельный плагин Data Analytics, не включённый в Python lock. Готовые HTML уже сохранены и читаются без этого плагина. Сводная `index.html` не зависит от него.

Статусы QA относятся к указанным в них средам и датам; подробности проверок приведены рядом с результатами каждого исследования.

## Проверка исторических контрольных сумм

Скрипты `verify_enriched_bundle.py` и `verify_round3_bundle.py`–`verify_round6_bundle.py` сверяют также байты документации на момент исследования. Ссылки в текущих инструкциях обновлены для репозитория решения. Для проверки исходных QA используйте отдельную копию с первоначальными документами; оригиналы включены в [source_documentation.zip](source_documentation.zip), их реестр — в [source_documentation.json](source_documentation.json).

Из корня клонированного репозитория, macOS/Linux:

```sh
mkdir -p work/source-snapshot
git archive HEAD | tar -x -C work/source-snapshot
python3 -m zipfile -e submission/source_documentation.zip work/source-snapshot
cd work/source-snapshot
uv sync --frozen
uv run --frozen python scripts/verify_round6_bundle.py --repo .
```

Архив документов восстанавливает точные байты исследовательского снимка. Код, конфигурации и числовые результаты уже совпадают с ним; старые QA-записи не переписаны под новые тексты. Остальные проверки исторических пакетов запускаются из этой же копии по соответствующей инструкции. Для скачивания данных используйте каталог текущей поставки: исторические документы в копии содержат исходные адреса.
