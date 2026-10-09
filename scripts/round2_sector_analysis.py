"""Audit published worker sections and build descriptive, outcome-independent shares.

The denominator is the official annual workers total, never the sum of disclosed
sections. Source activity keys are internal BDPMO option keys, not OKVED class
codes. The A/B/C/D/H/O/P/Q model set was specified before outcome inspection.
No target, prediction, wage aggregation, imputation or model is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

WORKERS = "8423005"
WAGES = "8423007"
TOTAL = "101"
SECTIONS = {
    "10000": ("A", "Сельское, лесное хозяйство, охота, рыболовство и рыбоводство"),
    "50000": ("B", "Добыча полезных ископаемых"),
    "20000": ("C", "Обрабатывающие производства"),
    "30000": ("D", "Электричество, газ, пар и кондиционирование воздуха"),
    "40000": ("E", "Водоснабжение, отходы и ликвидация загрязнений"),
    "41000": ("F", "Строительство"),
    "45000": ("G", "Оптовая и розничная торговля, ремонт автотранспорта"),
    "49000": ("H", "Транспортировка и хранение"),
    "55000": ("I", "Гостиницы и общественное питание"),
    "63000": ("J", "Информация и связь"),
    "60000": ("K", "Финансовая и страховая деятельность"),
    "68000": ("L", "Операции с недвижимым имуществом"),
    "70000": ("M", "Профессиональная, научная и техническая деятельность"),
    "80000": ("N", "Административная деятельность и сопутствующие услуги"),
    "84000": ("O", "Государственное управление, военная безопасность и социальное обеспечение"),
    "85000": ("P", "Образование"),
    "86000": ("Q", "Здравоохранение и социальные услуги"),
    "90000": ("R", "Культура, спорт, досуг и развлечения"),
    "95000": ("S", "Прочие услуги"),
    "73000": ("U", "Экстерриториальные организации и органы"),
}
SELECTED = ("A", "B", "C", "D", "H", "O", "P", "Q")
LETTERS = tuple(value[0] for value in SECTIONS.values())
EXAMPLE_IDS = (331, 318, 2527, 925, 1180, 883)
INDEX = ["territory_id", "year"]
CORE_COLUMNS = [
    *INDEX,
    "official_oktmo_8",
    "region_code",
    "region_name",
    "municipal_district_name",
    "municipal_district_type",
    "workers",
    "calendar_review",
    "code_type_changed",
    "core_eligible",
    "conservative_eligible",
    "geography_status",
]
PREFIX_TRANSLATION = str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")
SOURCE_PATH = "reports/data_round2/sector/bulk/annual_candidate_sector.parquet"
OPTIONS_PATH = "reports/data_round2/sector/activity_options.csv"
CORE_PATH = "reports/enriched_analysis/core_v2/annual_panel.csv"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def label_dictionary(source: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Use exact captured form keys/labels; preserve all source label variants."""
    keys = [
        "source_region_url",
        "indicator_id",
        "source_activity_code",
        "source_activity_label",
        "options_sha256",
    ]
    require(
        not options.duplicated([keys[0], keys[1], keys[2], keys[4]]).any(),
        "Form option keys must be unique; ambiguous labels cannot be collapsed",
    )
    counts = source.groupby(keys, dropna=False).size().rename("source_rows").reset_index()
    matched = counts.merge(
        options[keys], on=keys, how="left", indicator=True, validate="one_to_one"
    )
    require(matched["_merge"].eq("both").all(), "Source label is absent from exact captured form")
    records = []
    for row in matched.drop(columns="_merge").to_dict("records"):
        code = row["source_activity_code"]
        require(code == TOTAL or code in SECTIONS, f"Unreviewed activity key: {code}")
        letter, label = ("TOTAL", "Официальный итог") if code == TOTAL else SECTIONS[code]
        prefix = re.match(r"^раздел\s+([A-Za-zА-Яа-я])", row["source_activity_label"], re.I)
        observed_prefix = prefix[1].upper().translate(PREFIX_TRANSLATION) if prefix else ""
        require(not prefix or observed_prefix == letter, f"Conflicting section prefix: {row}")
        records.append(
            {
                **row,
                "canonical_section": letter,
                "canonical_short_label": label,
                "literal_section_prefix": prefix[1] if prefix else "",
                "prefix_check": "matched" if prefix else "no_prefix_reviewed_source_key",
                "selected_for_model": letter in SELECTED,
                "mapping_basis": "reviewed_internal_option_key_and_exact_captured_form_label",
            }
        )
    return pd.DataFrame(records).sort_values(["canonical_section", *keys])


def audit_source(source: pd.DataFrame) -> dict:
    """Fail closed on structural defects; retain all published missing statuses."""
    require(
        not source.duplicated([*INDEX, "indicator_id", "source_activity_code"]).any(),
        "Duplicate annual source cell",
    )
    require(set(source["indicator_id"]) == {WORKERS, WAGES}, "Unexpected indicators")
    require(set(source["source_activity_code"]) == {TOTAL, *SECTIONS}, "Unexpected sections")
    require(source["year"].isin([2023, 2024]).all(), "Unexpected observation year")
    require(source["period_code"].eq(79).all(), "Expected annual January-December period")
    require(
        source["period_start"].eq(source["year"].astype(str) + "-01-01").all(),
        "Incorrect period start",
    )
    require(
        source["period_end"].eq(source["year"].astype(str) + "-12-31").all(), "Incorrect period end"
    )
    require(source["official_oktmo_8"].str.fullmatch(r"\d{8}").all(), "Invalid official code")
    require(
        source.loc[source.indicator_id.eq(WORKERS), "unit"].eq("человек").all(),
        "Worker numerator and denominator must use people",
    )
    require(
        source.loc[source.indicator_id.eq(WAGES), "unit"].eq("рубль").all(), "Unexpected wage units"
    )
    observed = source["status"].eq("observed")
    require(
        source["status"].isin(["observed", "not_returned", "unresolved_source_marker"]).all(),
        "Unexpected source status",
    )
    require(
        (np.isfinite(source.loc[observed, "value"]) & source.loc[observed, "value"].ge(0)).all(),
        "Invalid observed numeric cell",
    )
    require(source.loc[~observed, "value"].isna().all(), "Non-observed value must stay missing")
    require(source["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all(), "Missing source hash")
    require(
        source["geographic_equivalence_confirmed"].isna().all(),
        "New geography evidence needs explicit review",
    )
    return {
        "source_rows": len(source),
        "source_csv_hashes": int(source.source_sha256.nunique()),
        "years": sorted(source.year.unique().tolist()),
        "source_status_counts": [
            {"indicator_id": indicator, "status": status, "rows": int(rows)}
            for (indicator, status), rows in source.groupby(["indicator_id", "status"])
            .size()
            .items()
        ],
        "published_zero_cells": int(source.value.eq(0).sum()),
        "geographic_equivalence_confirmed_nonnull": 0,
    }


def derive_shares(values: pd.Series, status: pd.Series, totals: pd.Series) -> pd.DataFrame:
    """No renormalization, clipping or zero imputation, including in sparse disclosures."""
    reason = status.fillna("not_offered").copy()
    observed = reason.eq("observed")
    finite = pd.Series(np.isfinite(values), index=values.index) & values.ge(0)
    total_valid = pd.Series(np.isfinite(totals), index=totals.index) & totals.gt(0)
    reason.loc[observed & ~finite] = "invalid_numerator"
    reason.loc[observed & finite & ~total_valid] = "invalid_total"
    reason.loc[observed & finite & total_valid & values.gt(totals)] = "exceeds_total"
    valid = observed & finite & total_valid & values.le(totals)
    reason.loc[valid] = "valid"
    return pd.DataFrame({"share": (values / totals).where(valid), "share_status": reason})


def build_features(source: pd.DataFrame, core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the frozen core cohort while retaining newly available source totals."""
    require(not core.duplicated(INDEX).any(), "Duplicate core annual key")
    workers = source.loc[source.indicator_id.eq(WORKERS)].copy()
    require(not workers.duplicated([*INDEX, "source_activity_code"]).any(), "Duplicate worker cell")
    code_links = workers[[*INDEX, "official_oktmo_8"]].drop_duplicates()
    linked = core[[*INDEX, "official_oktmo_8"]].merge(
        code_links,
        on=INDEX,
        suffixes=("_core", "_source"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    require(linked["_merge"].eq("both").all(), "Source/core keys differ")
    require(
        linked.official_oktmo_8_core.eq(linked.official_oktmo_8_source).all(),
        "Source/core annual codes differ",
    )
    totals = workers.loc[workers.source_activity_code.eq(TOTAL)].copy()
    totals = totals.set_index(INDEX)
    require(len(totals) == len(core), "A source total row must exist even when value is missing")
    features = core.rename(columns={"workers": "workers_total_frozen_core"}).set_index(INDEX).copy()
    features["workers_total_source"] = totals.value
    features["workers_total_source_status"] = totals.status
    features["workers_total_source_sha256"] = totals.source_sha256
    features["workers_total_source_csv_row"] = totals.source_csv_row
    features["workers_total_source_url"] = totals.source_url
    features["source_temporal_review_required"] = totals.temporal_review_required
    features["geographic_equivalence_confirmed"] = pd.array(
        [pd.NA] * len(features), dtype="boolean"
    )
    # Source metadata can distinguish new totals, but cannot revise old analytic eligibility.
    features["newly_available_workers_total"] = (
        features.workers_total_frozen_core.isna() & features.workers_total_source.notna()
    )
    sections = workers.loc[workers.source_activity_code.ne(TOTAL)].copy()
    sections["canonical_section"] = sections.source_activity_code.map(
        lambda code: SECTIONS[code][0]
    )
    grid = features.reset_index()[INDEX].merge(
        pd.DataFrame({"canonical_section": LETTERS}), how="cross"
    )
    long = grid.merge(sections, on=[*INDEX, "canonical_section"], how="left", validate="one_to_one")
    long = long.merge(
        features.reset_index()[
            [
                *INDEX,
                "workers_total_source",
                "workers_total_source_status",
                "workers_total_source_sha256",
                "workers_total_source_csv_row",
                "workers_total_source_url",
                "core_eligible",
                "conservative_eligible",
                "calendar_review",
                "code_type_changed",
                "geography_status",
            ]
        ],
        on=INDEX,
        how="left",
        validate="many_to_one",
    )
    long["offered_by_source_form"] = long.source_activity_code.notna()
    # Backfill only the reviewed dimension key on unoffered grid rows, never values/status.
    inverse = {letter: code for code, (letter, _) in SECTIONS.items()}
    long["canonical_internal_key"] = long.canonical_section.map(inverse)
    derived = derive_shares(long.value, long.status, long.workers_total_source)
    long[["share", "share_status"]] = derived
    long["selected_for_model"] = long.canonical_section.isin(SELECTED)
    wide = long.pivot(index=INDEX, columns="canonical_section", values="share")
    for letter in LETTERS:
        features[f"sector_{letter}_share"] = wide[letter]
    grouped = long.groupby(INDEX)
    features["n_offered_sectors"] = grouped.offered_by_source_form.sum()
    features["n_observed_sectors"] = (
        long.status.eq("observed").groupby([long.territory_id, long.year]).sum()
    )
    features["n_valid_sector_shares"] = grouped.share.count()
    features["n_exceeds_total_sectors"] = (
        long.share_status.eq("exceeds_total").groupby([long.territory_id, long.year]).sum()
    )
    features["observed_workers_sum_diagnostic"] = grouped.value.sum(min_count=1)
    features["observed_sum_over_official_total_diagnostic"] = (
        features.observed_workers_sum_diagnostic / features.workers_total_source
    ).where(features.workers_total_source.gt(0))
    features["observed_sum_exceeds_official_total"] = (
        features.observed_sum_over_official_total_diagnostic.gt(1)
        .astype("boolean")
        .where(features.observed_sum_over_official_total_diagnostic.notna(), pd.NA)
    )
    selected_columns = [f"sector_{letter}_share" for letter in SELECTED]
    features["n_selected_shares_available"] = features[selected_columns].notna().sum(axis=1)
    features["selected8_complete"] = features.n_selected_shares_available.eq(len(SELECTED))
    return features.reset_index().sort_values(INDEX), long.sort_values(
        [*INDEX, "canonical_section"]
    )


def coverage_tables(
    features: pd.DataFrame, long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for (year, letter), rows in long.groupby(["year", "canonical_section"]):
        counts = rows.share_status.value_counts()
        records.append(
            {
                "year": int(year),
                "canonical_section": letter,
                "selected_for_model": letter in SELECTED,
                "cohort_municipalities_denominator": len(rows),
                "offered": int(rows.offered_by_source_form.sum()),
                "observed_numerators": int(rows.status.eq("observed").sum()),
                "valid_shares": int(rows.share.notna().sum()),
                "valid_share_fraction_of_cohort": float(rows.share.notna().mean()),
                **{
                    f"n_{status}": int(counts.get(status, 0))
                    for status in [
                        "not_offered",
                        "not_returned",
                        "unresolved_source_marker",
                        "invalid_total",
                        "invalid_numerator",
                        "exceeds_total",
                    ]
                },
            }
        )
    samples = []
    for year, rows in features.groupby("year"):
        for sample, eligible in [
            ("full_source_cohort", pd.Series(True, index=rows.index)),
            ("frozen_core", rows.core_eligible),
            ("frozen_conservative", rows.conservative_eligible),
        ]:
            keep = rows.loc[eligible]
            samples.append(
                {
                    "year": int(year),
                    "sample": sample,
                    "n_municipalities": len(keep),
                    "workers_total_available": int(keep.workers_total_source.gt(0).sum()),
                    "selected8_complete": int(keep.selected8_complete.sum()),
                    "selected8_complete_fraction": float(keep.selected8_complete.mean()),
                    "median_n_observed_sectors": float(keep.n_observed_sectors.median()),
                    "observed_sum_exceeds_total": int(
                        keep.observed_sum_exceeds_official_total.sum()
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(samples)


def example_tables(features: pd.DataFrame, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Maxima describe disclosed values only; same codes do not prove equal boundaries."""
    selected = features.loc[features.territory_id.isin(EXAMPLE_IDS)].copy()
    records = []
    for row in selected.to_dict("records"):
        sectors = long.loc[long.territory_id.eq(row["territory_id"]) & long.year.eq(row["year"])]
        disclosed = sectors.loc[sectors.status.eq("observed")]
        top = disclosed.loc[disclosed.value.eq(disclosed.value.max())].sort_values(
            "canonical_section"
        )
        records.append(
            {
                **row,
                "max_among_observed_sections": "|".join(top.canonical_section),
                "max_observed_workers": float(top.value.iloc[0]) if len(top) else np.nan,
                "max_observed_share_of_official_total": float(top.share.iloc[0])
                if len(top)
                else np.nan,
                "max_source_sha256": "|".join(top.source_sha256),
                "max_source_csv_row": "|".join(top.source_csv_row.astype("Int64").astype(str)),
                "max_source_url": "|".join(top.source_url),
            }
        )
    summary = pd.DataFrame(records)
    earlier = long.loc[long.territory_id.isin(EXAMPLE_IDS) & long.year.eq(2023)]
    later = long.loc[long.territory_id.isin(EXAMPLE_IDS) & long.year.eq(2024)]
    trace_columns = [
        "territory_id",
        "canonical_section",
        "official_oktmo_8",
        "value",
        "status",
        "share",
        "share_status",
        "source_sha256",
        "source_csv_row",
        "source_url",
        "source_activity_code",
        "source_activity_label",
        "workers_total_source",
        "workers_total_source_sha256",
        "workers_total_source_csv_row",
        "calendar_review",
        "code_type_changed",
    ]
    pairs = earlier[trace_columns].merge(
        later[trace_columns],
        on=["territory_id", "canonical_section"],
        suffixes=("_2023", "_2024"),
        how="outer",
        validate="one_to_one",
    )
    # Use annual core codes for absent source sections too; geography remains legally unconfirmed.
    codes = selected.pivot(index="territory_id", columns="year", values="official_oktmo_8")
    pairs["same_annual_code"] = pairs.territory_id.map(codes[2023].eq(codes[2024]))
    pairs["calendar_flags_clear"] = ~(pairs.calendar_review_2023 | pairs.calendar_review_2024)
    pairs["code_type_change_flag_clear"] = ~(
        pairs.code_type_changed_2023 | pairs.code_type_changed_2024
    )
    pairs["independent_boundary_equivalence_confirmed"] = pd.array(
        [pd.NA] * len(pairs), dtype="boolean"
    )
    both_valid = pairs.share_status_2023.eq("valid") & pairs.share_status_2024.eq("valid")
    pair_candidate = (
        pairs.same_annual_code & pairs.calendar_flags_clear & pairs.code_type_change_flag_clear
    )
    pairs["conditional_same_code_share_change_pp"] = (
        100 * (pairs.share_2024 - pairs.share_2023)
    ).where(both_valid & pair_candidate)
    pairs["comparison_status"] = np.select(
        [~pair_candidate, ~both_valid],
        ["time_or_code_review_required", "one_or_both_sector_values_unavailable"],
        default="conditional_same_code_not_independent_boundary_proof",
    )
    return summary, pairs


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    con = duckdb.connect()
    try:
        con.register("derived_frame", frame)
        con.execute("COPY derived_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()


def verify_example_raw(
    source: pd.DataFrame, examples: pd.DataFrame, raw_dir: Path, inventory: list[dict]
) -> pd.DataFrame:
    """Recheck headline source cells, using explicit years and exact stored line locators."""
    index = {row["sha256"]: row for row in inventory}
    records = []
    cached = {}
    for example in examples.to_dict("records"):
        target = source.loc[
            source.territory_id.eq(example["territory_id"])
            & source.year.eq(example["year"])
            & source.indicator_id.eq(WORKERS)
        ]
        top = example["max_among_observed_sections"].split("|")
        wanted = {TOTAL, *(code for code, (letter, _) in SECTIONS.items() if letter in top)}
        for row in target.loc[target.source_activity_code.isin(wanted)].to_dict("records"):
            digest = row["source_sha256"]
            if digest not in cached:
                path = (raw_dir / index[digest]["path"]).resolve()
                require(path.is_relative_to(raw_dir.resolve()), "Unsafe raw inventory path")
                require(sha256(path) == digest, "Raw response hash differs from source lineage")
                body = path.read_bytes()
                try:
                    decoded = body.decode("utf-8-sig")
                except UnicodeDecodeError:
                    decoded = body.decode("cp1251")
                cached[digest] = decoded.splitlines()
            lines = cached[digest]
            offset = int(row["source_csv_row"]) - 1
            require(offset >= 2, "Missing source header locator")
            year_headers = [
                position
                for position, line in enumerate(lines[:offset])
                if re.fullmatch(r";(?:20\d{2};)+", line.strip())
            ]
            require(bool(year_headers), "Explicit source year header is absent before value")
            header_offset = year_headers[-1]
            section_offset = header_offset - 1
            while section_offset > 0 and lines[section_offset - 1].strip():
                section_offset -= 1
            # Moscow source headers wrap over physical lines. Join only these header lines;
            # exact text must match the stored section, without substitutions or fuzzy matching.
            section_text = " ".join(line.strip() for line in lines[section_offset:header_offset])
            require(section_text == row["source_section"], "Exact source section changed")
            require(
                all(line.strip() for line in lines[header_offset:offset]),
                "A block separator intervenes after the exact source section",
            )
            years = lines[header_offset].strip().strip(";").split(";")
            require(
                years and all(re.fullmatch(r"20\d{2}", year) for year in years),
                "Expected explicit year header immediately after exact source section",
            )
            require(str(row["year"]) in years, "Observation year absent from explicit header")
            cells = lines[offset].strip().rstrip(";").split(";")
            caption, numbers = ";".join(cells[: -len(years)]), cells[-len(years) :]
            require(caption == row["actual_source_oktmo_label"], "Exact source caption changed")
            raw_value = numbers[years.index(str(row["year"]))]
            value = float(raw_value.replace(" ", "").replace("\xa0", "").replace(",", "."))
            require(value == row["value"], "Raw source value differs from derived value")
            records.append(
                {
                    "territory_id": row["territory_id"],
                    "year": row["year"],
                    "source_activity_code": row["source_activity_code"],
                    "official_oktmo_8": row["official_oktmo_8"],
                    "value": value,
                    "source_sha256": digest,
                    "source_csv_row": offset + 1,
                    "source_year_header_row": header_offset + 1,
                    "source_section_start_row": section_offset + 1,
                    "source_section_end_row": header_offset,
                    "source_path_within_raw_dir": index[digest]["path"],
                    "source_url": row["source_url"],
                    "raw_hash_and_cell_check": "PASS",
                }
            )
    return pd.DataFrame(records).sort_values([*INDEX, "source_activity_code"])


def report_text(qa: dict, coverage: pd.DataFrame, examples: pd.DataFrame) -> str:
    lines = [
        "# Отраслевые доли работников: аудит и шесть описательных примеров",
        "",
        "Опубликованные доли характеризуют работников "
        "учтённых организаций и пригодны для анализа с ограничениями. "
        "Они не описывают отрасли занятости всех жителей и не объясняют "
        "потребительские остатки причинно. Анализ исследовательский: 2024 год уже был просмотрен.",
        "",
        "## Что измерено",
        "",
        "Единица таблицы — МО × календарный год, 2023 или 2024. Числитель — среднесписочная "
        "численность работников организаций по опубликованному разделу, показатель БДПМО "
        "8423005; знаменатель — официальный итог 101 того же показателя, МО и года. "
        "Это работники организаций без субъектов малого предпринимательства, а не число "
        "занятых жителей. Данные могут пересматриваться: сохранены снимки, полученные в 2026 году. "
        "Зарплаты 8423007 проходят проверку структуры и кодов; они здесь не суммируются.",
        "",
        "[Методология Росстата](https://rosstat.gov.ru/storage/mediabank/bdmo_met2024.docx) "
        "и [разъяснения по труду](https://35.rosstat.gov.ru/folder/120693) сохранены в предыдущем "
        "пакете. Адреса выгрузок, SHA256 и номера строк сохранены "
        "в worker_shares.parquet и example_sector_pairs.csv; снимки форм перечислены "
        "в исходном raw_source_inventory.json. Здесь нет новых сетевых запросов.",
        "",
        "Ключи 10000, 50000 и другие — внутренние значения опций БДПМО, а не формальные "
        "пятизначные коды классов ОКВЭД. Проверенное соответствие широким разделам A–S и U "
        "записано в section_dictionary.csv. Все варианты исходных подписей сохранены в "
        "source_label_dictionary.csv: сопоставление с формой точное по региону, показателю, "
        "коду, подписи и SHA256. Варианты пунктуации, сокращения и опечатки не исправляют "
        "исходник. Раздел T не получен. U предложен частью форм, "
        "но числовых значений численности по нему нет.",
        "",
        "## Покрытие и ограничения",
        "",
        f"Полная таблица содержит {qa['feature_rows']} МО-лет для {qa['municipalities']} МО. "
        f"Положительный официальный итог доступен в {qa['workers_total_available']} МО-лет. "
        "Для отдельных отраслей превышений над итогом найдено "
        f"{qa['sector_cells_exceeding_total']}; "
        "защитное правило исключает такие доли, если они появятся, без обрезания до 1. "
        f"Сумма раскрытых отраслей превышает итог в {qa['observed_sums_exceeding_total']} МО-лет; "
        "она сохранена только как диагностическая сумма. Недостающая до итога величина "
        "не называется ни отдельной отраслью, ни экономическим остатком.",
        "",
        f"В {qa['observed_sums_exceeding_total']} случаях превышение суммы составляет "
        f"{qa['sum_excess_min_workers']:g}–{qa['sum_excess_max_workers']:g} работника, "
        f"максимум отношения — {qa['maximum_observed_sum_over_total']:.9f}. "
        "Такой масштаб совместим с округлением опубликованных средних, но причина "
        "по самим ячейкам не установлена. Числа и флаги превышения сохранены.",
        "",
        "При недоступной диагностической сумме флаг превышения равен NULL: "
        f"{qa['sum_diagnostic_unavailable_mo_years']} МО-лет. В "
        f"{qa['total_available_no_disclosed_sections_mo_years']} из них итог работников есть, "
        "но не раскрыто ни одного раздела. Эти случаи не превращены в нулевую "
        "отраслевую структуру.",
        "",
        "Для анализа связи с потребительскими профилями заранее согласованы "
        "восемь признаков: A, B, C, D, H, O, P, Q. "
        "Пропуски сохраняются; модельные признаки отсутствия готовит основной анализ. "
        "Охват ниже имеет знаменателем все 4032 МО-года, а не только формы, предложившие раздел.",
        "",
        "| Раздел | Валидные доли / 4032 | Доля |",
        "|---|---:|---:|",
    ]
    for letter in SELECTED:
        rows = coverage.loc[coverage.canonical_section.eq(letter)]
        n = int(rows.valid_shares.sum())
        lines.append(f"| {letter} | {n} | {100 * n / 4032:.1f}% |")
    lines += [
        "",
        f"Все восемь долей раскрыты в {qa['selected8_complete_mo_years']} МО-лет; "
        "выборка с полным набором имеет отдельную таблицу sample_coverage.csv. "
        "Особенно редка добыча B. Отсутствие опции в форме, неполученная строка и "
        "неразрешённый маркер источника разделены в coverage.csv и worker_shares.parquet. "
        "Ни один из этих статусов не заменяется нулём. Только явный опубликованный ноль "
        "остаётся числовым нулём.",
        "",
        "Флаги core_eligible, conservative_eligible, calendar_review и code_type_changed "
        "перенесены из frozen core без пересчёта. Новые итоги для ID2710 сохранены отдельно "
        "от старого workers_total_frozen_core; они не меняют прежний допуск в модель. "
        "78 смен кодов/типов не считаются доказательством неизменных границ. Независимая "
        "юридическая и геометрическая эквивалентность границ во всём слое остаётся неизвестной. "
        "Предыдущая сверка пилота сохранила предупреждение о 22 различиях представления "
        "пропуска при совпадении чисел/NULL; этот аудит его не устраняет.",
        "",
        "## Шесть заранее выбранных примеров",
        "",
        "Это прежние шесть крайних остатков линейной модели; новые модели здесь не "
        "строились. Максимум относится только к наблюдаемым разделам, поэтому не задаёт "
        "полную отраслевую типологию. У всех шести совпадают годовые коды и отсутствуют "
        "календарный флаг и флаг смены кода/типа. Это позволяет показать условное сравнение "
        "одинаковых кодов, но не заменяет независимую проверку границ. Для доли, отсутствующей "
        "хотя бы в одном году, изменение не вычисляется.",
        "",
        "| ID / МО | Год | Официальный итог | Раскрыто разделов | Максимум среди наблюдаемых | "
        "Работников | Доля официального итога |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in examples.sort_values(INDEX).itertuples():
        lines.append(
            f"| {row.territory_id} / {row.municipal_district_name} | {row.year} | "
            f"{row.workers_total_source:g} | {row.n_observed_sectors} | "
            f"{row.max_among_observed_sections} | {row.max_observed_workers:g} | "
            f"{100 * row.max_observed_share_of_official_total:.2f}% |"
        )
    lines.append("")
    for territory_id in EXAMPLE_IDS:
        pair = examples.loc[examples.territory_id.eq(territory_id)].set_index("year")
        before, after = pair.loc[2023], pair.loc[2024]
        letter = after.max_among_observed_sections
        require(
            before.max_among_observed_sections == letter,
            "Example prose needs review if the leading disclosed section changes",
        )
        label = next(label for section, label in SECTIONS.values() if section == letter)
        lines.append(
            f"ID {territory_id}: наибольший раскрытый раздел в 2024 году — {letter}, "
            f"{label.lower()}: {after.max_observed_workers:g} из "
            f"{after.workers_total_source:g} работников "
            f"({100 * after.max_observed_share_of_official_total:.2f}%). "
            f"В 2023 году тот же раздел был первым среди раскрытых: "
            f"{before.max_observed_workers:g} из {before.workers_total_source:g} "
            f"({100 * before.max_observed_share_of_official_total:.2f}%). "
            f"Число раскрытых разделов: {before.n_observed_sectors} → "
            f"{after.n_observed_sectors}; сравнение условно по одинаковому годовому коду."
        )
        lines.append("")
    lines += [
        "",
        "В examples.csv указаны числитель и итог, их исходные SHA256 и строки; "
        "example_sector_pairs.csv содержит все разделы, оба года и сопоставимые раскрытые "
        "доли. Различия охвата организаций, нераскрытые отрасли, рабочие места нерезидентов "
        "и условное сопоставление географии ограничивают интерпретацию. Эти цифры не "
        "устанавливают причины потребительских расходов, экономический шок или ошибку данных.",
        "",
        "## Воспроизведение",
        "",
        "Из корня репозитория:",
        "",
        "```powershell",
        "uv run --frozen python scripts/round2_sector_analysis.py --repo . "
        "--out work/round2-sector-replay",
        "uv run --frozen pytest tests/test_round2_sector_analysis.py",
        "uv run --frozen ruff check scripts/round2_sector_analysis.py "
        "tests/test_round2_sector_analysis.py",
        "```",
        "",
        "Указанная папка результата должна быть новой или пустой. Нужны три frozen-входа "
        "из manifest.json: годовой Parquet отраслей, activity_options.csv "
        "и core_v2/annual_panel.csv; "
        "для ссылок также используются FINAL_QA.json и methodology_source_inventory.json. "
        "Скрипт читает из core только ключи, названия, старый итог работников и флаги. "
        "Потребительские доли, расходы, предсказания и остатки не загружаются. "
        "Все вычисления локальные. QA.json отделяет пройденные проверки от ограничений источника.",
        "",
        "Для повторной проверки 24 исходных ячеек примеров добавьте "
        "`--raw-source-dir <папка work/data-round2/sector из raw release>`. "
        "Нужны файлы из raw_source_inventory.json предыдущего пакета. Проверяются SHA256, "
        "точная исходная подпись и раздел, номер строки, явный заголовок года и числовая ячейка. "
        "Без этого аргумента повторная проверка raw отмечается NOT_RUN; числовые таблицы "
        "воспроизводятся из frozen Parquet. Результат текущего запуска — "
        f"{qa['raw_example_check_status']}; проверенных ячеек — "
        f"{qa['raw_example_checked_cells']}. "
        "raw_example_trace_check.csv сохраняет эти локаторы без копирования исходных файлов.",
        "",
    ]
    return "\n".join(lines)


def run(repo: Path, out: Path, raw_source_dir: Path | None = None) -> dict:
    repo = repo.resolve()
    out = out.resolve()
    require(not out.exists() or not any(out.iterdir()), "Output directory must be new or empty")
    inputs = [
        SOURCE_PATH,
        OPTIONS_PATH,
        CORE_PATH,
        "reports/data_round2/sector/FINAL_QA.json",
        "reports/data_round2/sector/methodology_source_inventory.json",
        "reports/data_round2/sector/raw_source_inventory.json",
        "uv.lock",
    ]
    hashes = {path: sha256(repo / path) for path in inputs}
    source = (
        duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(repo / SOURCE_PATH)]).df()
    )
    source["territory_id"] = source.territory_id.astype(int)
    source["year"] = source.year.astype(int)
    core = pd.read_csv(
        repo / CORE_PATH, usecols=CORE_COLUMNS, dtype={"official_oktmo_8": str, "region_code": str}
    )
    options = pd.read_csv(repo / OPTIONS_PATH, dtype=str)
    audit = audit_source(source)
    dictionary = label_dictionary(source, options)
    features, long = build_features(source, core)
    coverage, sample_coverage = coverage_tables(features, long)
    examples, pairs = example_tables(features, long)
    raw_check = None
    if raw_source_dir is not None:
        inventory = json.loads(
            (repo / "reports/data_round2/sector/raw_source_inventory.json").read_text("utf-8")
        )
        raw_check = verify_example_raw(source, examples, raw_source_dir, inventory)
    shares = features[[f"sector_{letter}_share" for letter in LETTERS]]
    require(
        ((shares >= 0) & (shares <= 1) | shares.isna()).all().all(), "Out-of-range derived shares"
    )
    require(len(features) == 4032 and features.territory_id.nunique() == 2016, "Unexpected cohort")
    require(features.groupby("year").size().eq(2016).all(), "Unbalanced year cohort")
    require(
        core.set_index(INDEX)["core_eligible"]
        .sort_index()
        .equals(features.set_index(INDEX)["core_eligible"].sort_index()),
        "Frozen core eligibility changed",
    )
    common = features.workers_total_frozen_core.notna() & features.workers_total_source.notna()
    require(
        np.allclose(
            features.loc[common, "workers_total_frozen_core"],
            features.loc[common, "workers_total_source"],
            rtol=0,
            atol=0,
        ),
        "Existing total changed versus frozen core",
    )
    excess = features.loc[features.observed_sum_exceeds_official_total]
    excess_workers = excess.observed_workers_sum_diagnostic - excess.workers_total_source
    qa = {
        "status": "PASS_WITH_SOURCE_LIMITATIONS",
        "readiness": "ready_with_conditions",
        **audit,
        "feature_rows": len(features),
        "municipalities": int(features.territory_id.nunique()),
        "worker_section_grid_rows": len(long),
        "actual_canonical_sections": list(LETTERS),
        "not_obtained_section": "T",
        "precommitted_model_sections": list(SELECTED),
        "workers_total_available": int(features.workers_total_source.gt(0).sum()),
        "worker_section_cells_observed": int(long.status.eq("observed").sum()),
        "valid_worker_shares": int(long.share.notna().sum()),
        "sector_cells_exceeding_total": int(long.share_status.eq("exceeds_total").sum()),
        "observed_sums_exceeding_total": int(features.observed_sum_exceeds_official_total.sum()),
        "maximum_observed_sum_over_total": float(
            features.observed_sum_over_official_total_diagnostic.max()
        ),
        "sum_excess_min_workers": float(excess_workers.min()),
        "sum_excess_max_workers": float(excess_workers.max()),
        "sum_diagnostic_unavailable_mo_years": int(
            features.observed_sum_over_official_total_diagnostic.isna().sum()
        ),
        "total_available_no_disclosed_sections_mo_years": int(
            (features.workers_total_source.gt(0) & features.n_observed_sectors.eq(0)).sum()
        ),
        "raw_example_check_status": "PASS" if raw_check is not None else "NOT_RUN",
        "raw_example_checked_cells": len(raw_check) if raw_check is not None else 0,
        "raw_example_checked_source_hashes": (
            int(raw_check.source_sha256.nunique()) if raw_check is not None else 0
        ),
        "selected8_complete_mo_years": int(features.selected8_complete.sum()),
        "new_workers_totals": features.loc[
            features.newly_available_workers_total,
            [*INDEX, "workers_total_source", "core_eligible"],
        ].to_dict("records"),
        "frozen_eligibility_preserved": True,
        "known_common_workers_totals_exactly_reconciled": int(common.sum()),
        "exact_form_label_match": True,
        "source_label_variants": int(
            dictionary[["source_activity_code", "source_activity_label"]].drop_duplicates().shape[0]
        ),
        "changed_code_type_municipalities": int(
            features.loc[features.code_type_changed, "territory_id"].nunique()
        ),
        "example_ids": list(EXAMPLE_IDS),
        "example_annual_rows": len(examples),
        "example_conditional_valid_sector_pairs": int(
            pairs.comparison_status.eq("conditional_same_code_not_independent_boundary_proof").sum()
        ),
        "consumer_outcome_columns_loaded": [],
        "tests_and_lint": "separate command results; see validation.json",
        "limitations": [
            "Observed organizations exclude small enterprises; workers are not resident employment",
            "Missing, not offered and unresolved source marker are not zero",
            "Shares use official total; observed sum ratio is only a disclosure diagnostic",
            "Source captures from 2026 do not prove contemporaneous availability in 2023/2024",
            "Independent legal boundary equivalence remains unknown, including unchanged codes",
            "U has no observed worker values; T was not obtained",
            "Prior pilot has 22 NULL representation differences; values/NULLs reconcile",
            "Exploratory analysis after previous inspection of 2024; no causal inference",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "feature_panel.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    sample_coverage.to_csv(out / "sample_coverage.csv", index=False)
    dictionary.to_csv(out / "source_label_dictionary.csv", index=False)
    pd.DataFrame(
        [
            {
                "canonical_section": letter,
                "source_activity_code": code,
                "label": label,
                "feature": f"sector_{letter}_share",
                "selected_for_model": letter in SELECTED,
                "denominator": "8423005:101:same municipality and calendar year",
                "unit": "dimensionless ratio of average annual organizational workers",
            }
            for code, (letter, label) in SECTIONS.items()
        ]
    ).to_csv(out / "section_dictionary.csv", index=False)
    examples.to_csv(out / "examples.csv", index=False)
    pairs.to_csv(out / "example_sector_pairs.csv", index=False)
    if raw_check is not None:
        raw_check.to_csv(out / "raw_example_trace_check.csv", index=False)
    excess.to_csv(out / "observed_sum_exceeds_total.csv", index=False)
    long.loc[long.share_status.eq("exceeds_total")].to_csv(out / "exceeds_total.csv", index=False)
    write_parquet(out / "worker_shares.parquet", long)
    write_json(out / "QA.json", qa)
    write_json(
        out / "feature_dictionary.json",
        {
            "grain": (
                "one territory_id/year; source cohort includes 2016 municipalities in each year"
            ),
            "source_scope": "organizational average annual workers, excluding small enterprises",
            "section_features": {
                f"sector_{letter}_share": {
                    "canonical_section": letter,
                    "label": label,
                    "source_activity_key": code,
                    "formula": (
                        "observed 8423005 section / observed 8423005 total101 in same ID/year"
                    ),
                    "validity": "finite numerator>=0, finite official total>0, numerator<=total",
                    "missing": (
                        "not offered, not returned, unresolved marker or invalid ratio; never0"
                    ),
                    "selected_for_model": letter in SELECTED,
                }
                for code, (letter, label) in SECTIONS.items()
            },
            "fields": {
                "territory_id": (
                    "SberIndex internal identifier, joined exactly to frozen annual panel"
                ),
                "year": "January-December observation year; source capture was in2026",
                "official_oktmo_8": (
                    "same annual code verified on both input sides; retain leading zeros"
                ),
                "workers_total_source": "published total101; people, not resident employment",
                "workers_total_frozen_core": (
                    "previous core total, unchanged even when new total exists"
                ),
                "newly_available_workers_total": (
                    "old total NULL and current source total available"
                ),
                "n_offered_sectors": (
                    "number of non-total dimensions offered in source form, including U"
                ),
                "n_observed_sectors": "number of non-total sections with observed numeric values",
                "n_valid_sector_shares": "number of sections satisfying the individual ratio rules",
                "n_exceeds_total_sectors": (
                    "number of individual observed numerators exceeding total"
                ),
                "observed_workers_sum_diagnostic": (
                    "sum of observed section numerators only; no wages"
                ),
                "observed_sum_over_official_total_diagnostic": (
                    "observed sum/total; no economic residual"
                ),
                "observed_sum_exceeds_official_total": (
                    "nullable Boolean; NULL if ratio unavailable"
                ),
                "n_selected_shares_available": "available share count among fixed A,B,C,D,H,O,P,Q",
                "selected8_complete": "all eight shares available; not an eligibility override",
                "core_eligible": "copied from frozen core, never recomputed from new source values",
                "conservative_eligible": (
                    "copied from frozen core; includes original geography guards"
                ),
                "calendar_review": (
                    "original core temporal review flag, not silently adjudicated here"
                ),
                "source_temporal_review_required": "upstream sector candidate temporal flag",
                "code_type_changed": "original annual code/type change flag;78municipalities",
                "geography_status": "original conditional linkage status, not legal boundary proof",
                "geographic_equivalence_confirmed": "all NULL: independent equivalence unconfirmed",
                "workers_total_source_*": (
                    "SHA256, source CSV row and official request URL of denominator"
                ),
                "region_code,region_name,municipal_district_name,municipal_district_type": (
                    "unchanged descriptors from frozen core; never used for joins"
                ),
            },
            "long_table_statuses": {
                "valid": "ratio satisfies rules; explicit source zero remains zero",
                "not_offered": "the section is absent from the source form",
                "not_returned": "no cell returned for an offered section; not a zero",
                "unresolved_source_marker": (
                    "source marker or blank could not be treated as numeric"
                ),
                "invalid_total": "observed numerator but denominator is missing, nonfinite or<=0",
                "invalid_numerator": (
                    "observed status but numeric cell is invalid; production audit rejects"
                ),
                "exceeds_total": "numerator>official total; ratio excluded without clipping",
            },
        },
    )
    (out / "README.md").write_text(report_text(qa, coverage, examples), "utf-8")
    require(
        all(sha256(repo / path) == digest for path, digest in hashes.items()),
        "An input changed during execution",
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/round2_sector_analysis.py",
        "script_sha256": sha256(Path(__file__)),
        "command": "uv run --frozen python scripts/round2_sector_analysis.py "
        "--repo . --out <new-dir>",
        "optional_raw_source_dir_provided": raw_source_dir is not None,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
        },
        "input_hashes": hashes,
        "inputs_unchanged": True,
        "input_core_columns_loaded": CORE_COLUMNS,
        "output_hashes": {
            path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()
        },
        "post_run_validation_file": "validation.json (not part of numerical output hashes)",
    }
    write_json(out / "manifest.json", manifest)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-source-dir", type=Path)
    args = parser.parse_args()
    qa = run(args.repo, args.out, args.raw_source_dir)
    print(
        json.dumps(
            {
                key: qa[key]
                for key in [
                    "status",
                    "feature_rows",
                    "workers_total_available",
                    "valid_worker_shares",
                    "selected8_complete_mo_years",
                    "observed_sums_exceeding_total",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
