# Чтение результатов и запуск проекта

## Просмотр без вычислений

Начните с корневого README и `submission/METHODOLOGY.md`. Файл `index.html` открывается в браузере без сервера. Графики встроены в него; ссылки на подробные отчёты требуют полной папки репозитория. Готовые `reports/round2_analysis/report.html` и отчёты раундов 3–6 содержат сохранённые интерактивные представления.

Для таблиц основного исследования используйте `reports/initial_study/`. `assignments.parquet` хранит назначения, `cluster_profiles.csv` хранит профили групп, `membership_agreement.parquet` хранит поддержку, `transitions.parquet` и `flows.csv` описывают динамику. `method_metrics.csv` содержит сравнение методов. Метки локальны для месяца и метода.

## Основной эксперимент

Ниже перенесена последовательность из исходного проекта. При подготовке сводной поставки эти команды не запускались. Требуются Python 3.12 и uv. Зависимости закреплены в `uv.lock`; основной конфиг находится в `configs/robustness_v1.json`.

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

## Дополнительные исследования

Следуйте исходным инструкциям [обогащённого анализа](../reports/enriched_analysis/REPRODUCE.md), [раунда 2](../reports/round2_analysis/REPRODUCE.md), [раунда 3](../reports/round3_analysis/REPRODUCE.md), [раунда 4](../reports/round4_analysis/REPRODUCE.md), [раунда 5](../reports/round5_analysis/REPRODUCE.md), [раунда 6](../reports/round6_analysis/REPRODUCE.md) и [пространственного трека](../docs/reviews/spatial_validation.md).

Часть сырья опубликована в Assets релизов исходного частного репозитория. Эти архивы не входят в локальную сводную поставку. Для полного повторного извлечения нужно либо предоставить проверяющему доступ к ним, либо перенести закреплённые архивы в новый репозиторий с сохранением SHA-256 и обновить инструкции получения. Наличие производных таблиц не заменяет доступ к исходным архивам.

Сборка прежних интерактивных HTML использует отдельный плагин Data Analytics, не включённый в Python lock. Готовые HTML уже сохранены и читаются без этого плагина. Сводная `index.html` не зависит от него.

Статусы в прежних QA относятся к указанным в них средам и датам. Повторное выполнение на чистой машине при подготовке этой поставки не проводилось.
