"""Read-only admission audit of regional monetary context; no model or panel update."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
VARIABLES = ["shipments_economic_activity", "shipments_pure_activity", "resale_non_own_goods"]
COUNTERPART = {
    VARIABLES[0]: "8401011",
    VARIABLES[1]: "8401011",
    VARIABLES[2]: "8401003",
    "food_sales_all_visible_activity": "8401003",
}
SCOPES = {
    VARIABLES[
        0
    ]: "without_small_business_including_organizations_under_15_workers_excluding_VAT_excise",
    VARIABLES[
        1
    ]: "without_small_business_excluding_organizations_under_15_workers_excluding_VAT_excise",
    VARIABLES[
        2
    ]: "without_small_business_including_organizations_under_15_workers_not_retail_turnover",
    "food_sales_all_visible_activity": "food_only_without_hidden_activity",
}
PUBLICATION = {
    (26, 2023): "2024-07-31",
    (26, 2024): "2025-07-03",
    (64, 2023): "2024-04-23",
    (64, 2024): "2025-04-22",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_name(value: str) -> str:
    value = value.casefold().replace("ё", "е").removeprefix("г. ")
    for token in [
        "муниципальный район",
        "муниципальный округ",
        "городской округ",
        "город-курорт",
        "город",
    ]:
        value = re.sub(rf"\b{re.escape(token)}\b", "", value)
    return re.sub(r"\s+", " ", value).strip(" .")


def numeric(value) -> Decimal | None:
    text = str(value).strip().replace("\u00a0", " ")
    if text in {"", "...", "…", "-", "–", "—", "Х", "X"}:
        return None
    if not re.fullmatch(r"[0-9][0-9 ]*(?:[.,][0-9]+)?", text):
        raise ValueError("Unreviewed source numeric literal")
    return Decimal(text.replace(" ", "").replace(",", "."))


def rounding_comparison(old_literal: str, new_literal: str, pct_literal: str) -> dict:
    old, new, pct = map(numeric, [old_literal, new_literal, pct_literal])
    if old is None or new is None or pct is None or old <= 0:
        return {"status": "uncomputable", "ratio_percent": None, "difference_pp": None}
    old_half = Decimal(1).scaleb(old.as_tuple().exponent) / 2
    new_half = Decimal(1).scaleb(new.as_tuple().exponent) / 2
    pct_half = Decimal(1).scaleb(pct.as_tuple().exponent) / 2
    if old - old_half <= 0:
        raise ValueError("Prior rounded amount cannot support a positive denominator")
    low = 100 * (new - new_half) / (old + old_half)
    high = 100 * (new + new_half) / (old - old_half)
    compatible = low <= pct + pct_half and high >= pct - pct_half
    ratio = 100 * new / old
    return {
        "status": "rounding_compatible" if compatible else "different_beyond_rounding",
        "ratio_percent": float(ratio),
        "published_percent": float(pct),
        "difference_pp": float(ratio - pct),
        "ratio_rounding_low": float(low),
        "ratio_rounding_high": float(high),
        "published_rounding_halfwidth": float(pct_half),
        "difference_cause": "unknown",
        "same_definition_confirmed": False,
    }


def read_docx_totals(path: Path) -> tuple[list[str], list[str], list[str]]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables = [
        [
            [
                "".join(t.text or "" for t in cell.findall(".//w:t", NS)).strip()
                for cell in row.findall("w:tc", NS)
            ]
            for row in table.findall("w:tr", NS)
        ]
        for table in root.findall(".//w:tbl", NS)
    ]
    indexes = [
        i
        for i, table in enumerate(tables)
        if any(
            "По хозяйственным" in " ".join(row) and "По чистым" in " ".join(row) for row in table
        )
    ]
    if len(indexes) != 1:
        raise ValueError("Ambiguous shipment table")
    i = indexes[0]
    headers = " ".join(" ".join(row) for row in tables[i][:3])
    for token in ["2024", "2023", "тыс. рублей", "включая организации", "без организаций"]:
        if token not in headers:
            raise ValueError(f"Missing DOCX period, unit or scope: {token}")
    total = [row[1:] for row in tables[i] if row and row[0] == "Всего"]
    resale = [row[1:] for row in tables[i + 1] if row and row[0] == "Всего"]
    if len(total) != 1 or len(resale) != 1 or len(total[0]) != 4 or len(resale[0]) != 2:
        raise ValueError("Unexpected total-row shape")
    resale_headers = " ".join(" ".join(r) for r in tables[i + 1][:2])
    for token in ["2024", "2023", "тыс. рублей"]:
        if token not in resale_headers:
            raise ValueError(f"Missing resale period or unit: {token}")
    return total[0], resale[0], [headers, resale_headers]


def admission_for(variable: str) -> dict:
    if variable not in SCOPES:
        raise ValueError("Unknown metric cannot inherit admission")
    near = variable.startswith("shipments_")
    return {
        "variable": variable,
        "source_scope": SCOPES[variable],
        "descriptive_context_status": "ready_with_conditions",
        "counterpart_indicator_id": COUNTERPART[variable],
        "construct_relation": "shipment_related_not_equivalent_proven"
        if near
        else "different_construct_from_retail",
        "frozen_value_replacement": False,
        "missing_cell_fill": False,
        "historical_boundary_equivalence": "unknown",
        "vintage_equivalence": "unknown",
        "model_admission": False,
        "join_admission": "name_candidate_only",
        "conditions": (
            "Preserve source scope, operational edition, date and unit; separate table only"
        ),
    }


def run(repo: Path, raw: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Choose a fresh analysis output directory")
    repo, raw = repo.resolve(), raw.resolve()
    inputs = {}

    def pin(path: Path, root="repo") -> Path:
        base = repo if root == "repo" else raw
        if not path.resolve().is_relative_to(base):
            raise ValueError("Input path outside its declared root")
        inputs[f"{root}/{path.relative_to(base).as_posix()}"] = sha(path)
        return path

    def read(relative: str) -> pd.DataFrame:
        return pd.read_csv(pin(repo / relative), dtype=str, keep_default_na=False)

    prefix = "reports/data_round6/economy/"
    candidate = read(prefix + "regional_context_candidate.csv")
    documents = read(prefix + "document_inventory.csv")
    catalog = read(prefix + "catalog_publication_labels.csv")
    reported_food = read(prefix + "food_reconciliation.csv")
    source_manifest = json.loads(pin(repo / (prefix + "manifest.json")).read_text("utf-8"))
    reviewed = json.loads(pin(repo / (prefix + "independent_QA.json")).read_text("utf-8"))
    if reviewed["candidate_csv_sha256"] != sha(repo / (prefix + "regional_context_candidate.csv")):
        raise ValueError("Prior independent review does not refer to these candidate bytes")
    if reviewed["raw_inputs_sha256"] != source_manifest["raw_inputs"]:
        raise ValueError("Prior independent review does not refer to the current frozen sources")
    for name, digest in source_manifest["raw_inputs"].items():
        if sha(pin(raw / name, "raw")) != digest:
            raise ValueError("Frozen raw source hash differs")
    old = read("reports/data_round5/economy/annual_candidate_panel.csv")
    missing = read("reports/data_round5/economy/target_cell_decisions.csv")
    audit = read("reports/round5_analysis/audit/annual_features.csv")
    pin(repo / "uv.lock")
    for frame, key in [
        (candidate, ["territory_id", "year", "variable"]),
        (old, ["territory_id", "year", "indicator_id"]),
        (audit, ["territory_id", "year"]),
    ]:
        if frame.duplicated(key).any():
            raise ValueError(f"Nonunique join keys: {key}")
    if len(candidate) != 276 or set(candidate.variable) != set(SCOPES):
        raise ValueError("Candidate population changed")
    for row in candidate.to_dict("records"):
        if row["source_scope"] != SCOPES[row["variable"]] or row["unit"] != "thousand_rubles":
            raise ValueError("Metric meaning changed")
        if numeric(row["source_value_literal"]) != numeric(row["value"]):
            raise ValueError("Literal and machine amount differ")
        if any(
            row[col].lower() != "false"
            for col in [
                "source_has_oktmo",
                "historical_boundary_equivalence_proven",
                "replace_frozen_bdpmo_value",
            ]
        ):
            raise ValueError("Prior admission restriction was changed")
    meta_cols = [
        "territory_id",
        "year",
        "region_code",
        "municipal_district_name",
        "municipal_district_type",
        "official_oktmo_8",
        "core_eligible",
        "conservative_eligible",
        "calendar_review",
        "code_type_changed",
    ]
    metadata = audit[meta_cols].rename(columns={"region_code": "frozen_region_code"})
    rows = candidate.merge(
        metadata, on=["territory_id", "year"], how="left", validate="many_to_one", indicator=True
    )
    if (rows._merge != "both").any() or len(rows) != len(candidate):
        raise ValueError("Candidate-to-panel join lost or multiplied rows")
    rows = rows.drop(columns="_merge")
    lookup = audit[audit.region_code.isin(["26", "64"])][meta_cols].copy()
    lookup["normalized_name"] = lookup.municipal_district_name.map(normalized_name)
    if lookup.duplicated(["region_code", "year", "normalized_name"]).any():
        raise ValueError("Name matching is ambiguous")
    name_ids = {
        (r.region_code, r.year, r.normalized_name): r.territory_id for r in lookup.itertuples()
    }
    for row in rows.itertuples():
        if (
            row.region_code != row.frozen_region_code
            or name_ids.get((row.region_code, row.year, normalized_name(row.source_name)))
            != row.territory_id
        ):
            raise ValueError("Independent exact name/region/year join disagrees")
    # Catalog timestamps are observed labels, not proof of initial release dates.
    for (region, year), date in PUBLICATION.items():
        label = pd.Timestamp(date).strftime("%d.%m.%Y")
        selector = catalog.catalog.str.contains("saratov" if region == 64 else "stavropol")
        title = (
            f"за {year} финансовый год"
            if region == 64
            else ("Итоги года" if year == 2023 else "Итоги 2024 года")
        )
        selector &= catalog.title.str.contains(title, regex=False)
        if not catalog.loc[selector, "publication_label"].str.contains(label, regex=False).any():
            raise ValueError("Expected catalog date label is absent")
    rows["catalog_date_label"] = [
        PUBLICATION[int(r.region_code), int(r.year)] for r in rows.itertuples()
    ]
    rows["availability_by_observation_year_end_proven"] = False
    rows["first_publication_date_proven"] = False
    rows["candidate_indicator_id"] = rows.variable.map(COUNTERPART)
    counterpart = old[
        ["territory_id", "year", "indicator_id", "value", "value_status", "publication_vintage"]
    ].rename(
        columns={
            "indicator_id": "candidate_indicator_id",
            "value": "frozen_counterpart_value",
            "value_status": "frozen_counterpart_status",
            "publication_vintage": "frozen_counterpart_vintage",
        }
    )
    rows = rows.merge(
        counterpart, on=["territory_id", "year", "candidate_indicator_id"], validate="many_to_one"
    )
    if len(rows) != len(candidate):
        raise ValueError("Counterpart join lost candidate rows")
    reasons = missing.rename(
        columns={"indicator_id": "candidate_indicator_id", "reason": "frozen_missing_reason"}
    )
    rows = rows.merge(
        reasons[["territory_id", "year", "candidate_indicator_id", "frozen_missing_reason"]],
        on=["territory_id", "year", "candidate_indicator_id"],
        how="left",
        validate="many_to_one",
    )
    rows["direct_numeric_reconciliation_possible"] = rows.frozen_counterpart_value.ne(
        ""
    ) & rows.variable.str.startswith("shipments_")
    rows["descriptive_context_status"] = "ready_with_conditions"
    rows["frozen_value_replacement"] = False
    rows["model_admission"] = False
    rows["retail_equivalence_confirmed"] = False
    # Re-read 2024 Word tables without importing the collection parser. Published
    # 2023 percentages are retained from the pinned prior independent QA only.
    published = []
    review_by_key = {
        (str(r["territory_id"]), str(r["year"])): r for r in reviewed["stavropol_documents"]
    }
    raw_docx_checks = 0
    for doc in documents.to_dict("records"):
        key = doc["territory_id"], doc["year"]
        evidence = review_by_key[key]
        shipment = evidence["independent_shipment_tokens"]
        resale = evidence["independent_resale_tokens"]
        mode = "pinned_prior_independent_pdf_review"
        headers = ["", ""]
        source_path = evidence["pdf_path"]
        if doc["year"] == "2024":
            source_path = str(Path(doc["pdf_path"]).with_suffix(".docx")).replace("\\", "/")
            shipment_new, resale_new, headers = read_docx_totals(raw / source_path)
            if list(map(numeric, shipment_new + resale_new)) != list(
                map(numeric, shipment + resale)
            ):
                raise ValueError("New DOCX read differs from prior independently reviewed PDF")
            shipment, resale = shipment_new, resale_new
            mode = "fresh_stdlib_DOCX_XML_plus_pinned_PDF_review"
            raw_docx_checks += 1
        for variable, amount, percent, header in [
            (VARIABLES[0], shipment[0], shipment[1], headers[0]),
            (VARIABLES[1], shipment[2], shipment[3], headers[0]),
            (VARIABLES[2], resale[0], resale[1], headers[1]),
        ]:
            value = rows.loc[
                (rows.territory_id == key[0]) & (rows.year == key[1]) & (rows.variable == variable),
                "source_value_literal",
            ]
            if len(value) != 1 or numeric(value.iloc[0]) != numeric(amount):
                raise ValueError("Source total does not match candidate key/value")
            published.append(
                {
                    "territory_id": key[0],
                    "year": key[1],
                    "variable": variable,
                    "amount_literal": amount,
                    "published_percent_literal": percent,
                    "published_percent": float(numeric(percent)),
                    "comparison_prior_year": int(key[1]) - 1,
                    "raw_path": source_path,
                    "source_sha256": sha(raw / source_path),
                    "source_header": header,
                    "extraction_method": mode,
                    "real_or_nominal_index_definition": "not_established",
                }
            )
    pub = pd.DataFrame(published)
    ratios = []
    for record in pub[pub.year == "2024"].to_dict("records"):
        prior = pub[
            (pub.territory_id == record["territory_id"])
            & (pub.variable == record["variable"])
            & (pub.year == "2023")
        ].iloc[0]
        compared = rounding_comparison(
            prior.amount_literal, record["amount_literal"], record["published_percent_literal"]
        )
        ratios.append(
            {
                "territory_id": record["territory_id"],
                "variable": record["variable"],
                "old_amount_literal": prior.amount_literal,
                "new_amount_literal": record["amount_literal"],
                "published_percent_literal": record["published_percent_literal"],
                **compared,
                "old_source_sha256": prior.source_sha256,
                "new_source_sha256": record["source_sha256"],
            }
        )
    ratios = pd.DataFrame(ratios)
    relation = (
        rows[rows.variable.isin(VARIABLES[:2])]
        .pivot(index=["territory_id", "year", "panel_name"], columns="variable", values="value")
        .reset_index()
    )
    for col in VARIABLES[:2]:
        relation[col] = relation[col].astype(float)
    relation["economic_minus_pure_thousand_rubles"] = (
        relation[VARIABLES[0]] - relation[VARIABLES[1]]
    )
    relation["pure_div_economic_percent"] = 100 * relation[VARIABLES[1]] / relation[VARIABLES[0]]
    relation["difference_interpretation"] = (
        "definition_and_population_difference_not_small_firm_contribution"
    )
    relation["nested_population_or_additivity_proven"] = False
    relation["boundary_equivalence_proven"] = False
    food_rows = []
    for record in reported_food.to_dict("records"):
        values = rows[
            (rows.variable == "food_sales_all_visible_activity") & (rows.year == record["year"])
        ].source_value_literal
        total = sum((numeric(v) for v in values), Decimal(0))
        region = numeric(record["regional_total_thousand_rubles"])
        residual = region - total
        if total != numeric(record["municipal_sum_thousand_rubles"]):
            raise ValueError("Food source reconciliation changed")
        food_rows.append(
            {
                "year": record["year"],
                "municipal_rows": len(values),
                "municipal_sum_thousand_rubles": str(total),
                "regional_total_thousand_rubles": str(region),
                "regional_minus_municipal_thousand_rubles": str(residual),
                "residual_percent_of_region": float(100 * residual / region),
                "residual_cause": "unknown",
                "allocated_to_municipalities": False,
            }
        )
    coverage = []
    for (region, year, variable), group in rows.groupby(
        ["region_code", "year", "variable"], sort=True
    ):
        regional_denominator = int(((audit.region_code == region) & (audit.year == year)).sum())
        coverage.append(
            {
                "region_code": region,
                "year": year,
                "variable": variable,
                "candidate_rows": len(group),
                "distinct_municipalities": group.territory_id.nunique(),
                "regional_panel_municipalities": regional_denominator,
                "regional_panel_coverage_percent": 100
                * group.territory_id.nunique()
                / regional_denominator,
                "national_panel_municipalities": 2016,
                "national_coverage_percent": 100 * group.territory_id.nunique() / 2016,
                **{
                    c: int(group[c].eq("True").sum())
                    for c in [
                        "core_eligible",
                        "conservative_eligible",
                        "code_type_changed",
                        "calendar_review",
                    ]
                },
            }
        )
    summaries = []
    for variable, group in ratios.groupby("variable", sort=True):
        summaries.append(
            {
                "variable": variable,
                "pairs": len(group),
                "rounding_compatible": int(group.status.eq("rounding_compatible").sum()),
                "different_beyond_rounding": int(
                    group.status.eq("different_beyond_rounding").sum()
                ),
                "median_absolute_difference_pp": group.difference_pp.abs().median(),
                "max_absolute_difference_pp": group.difference_pp.abs().max(),
                "mean_difference_pp": group.difference_pp.mean(),
                "explanation_identified": False,
            }
        )
    intersections = (
        rows.groupby(
            [
                "territory_id",
                "year",
                "candidate_indicator_id",
                "frozen_counterpart_status",
                "frozen_missing_reason",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            candidate_variables=("variable", lambda v: ";".join(sorted(v))),
            candidate_numeric_rows=("variable", "size"),
        )
        .reset_index()
    )
    intersections["missing_cell_filled"] = False
    keys = rows[["territory_id", "year"]].drop_duplicates()
    cooccurrence = old.merge(
        keys, on=["territory_id", "year"], how="inner", validate="many_to_one"
    )[["territory_id", "year", "indicator_id", "value", "value_status", "unit"]]
    cooccurrence["same_metric_reconciliation"] = False
    unique = rows.drop_duplicates(["territory_id", "year"])
    qa = {
        "status": "READY_WITH_CONDITIONS_FOR_SEPARATE_CONTEXT_ONLY",
        "rows": len(rows),
        "municipalities": int(rows.territory_id.nunique()),
        "municipality_years": len(unique),
        "unique_candidate_keys": True,
        "candidate_to_metadata_multiplier": len(rows) / len(candidate),
        "unmatched_candidate_rows": 0,
        "name_region_year_independent_matches": len(rows),
        "distinct_missing_old_positions_with_related_candidate": len(intersections),
        "direct_numeric_same_construct_reconciliations": int(
            rows.direct_numeric_reconciliation_possible.sum()
        ),
        "other_frozen_numeric_cooccurrences": int(cooccurrence.value.ne("").sum()),
        "old_panel_rows": len(old),
        "old_numeric_values": int(old.value.ne("").sum()),
        "old_missing_values": int(old.value.eq("").sum()),
        "missing_cells_filled": 0,
        "new_panel_values": 0,
        "models_fit": 0,
        "core_municipality_years": int(unique.core_eligible.eq("True").sum()),
        "conservative_municipality_years": int(unique.conservative_eligible.eq("True").sum()),
        "code_changed_municipality_years": int(unique.code_type_changed.eq("True").sum()),
        "code_changed_municipalities": int(
            unique.loc[unique.code_type_changed.eq("True"), "territory_id"].nunique()
        ),
        "fresh_DOCX_rechecks": raw_docx_checks,
        "published_percent_rows": len(pub),
        "two_vintage_percent_checks": len(ratios),
        "source_input_hashes_verified": len(source_manifest["raw_inputs"]),
        "numerical_source_validation": (
            "Reused pinned prior independent PDF review for 276 amounts; "
            "fresh XML read of 33 DOCX/198 total cells in 2024"
        ),
        "boundary_equivalence": "unknown",
        "vintage_equivalence": "unknown",
        "cross_vintage_discrepancy_cause": "unknown",
        "network_requests": 0,
    }
    out.mkdir(parents=True)
    tables = {
        "row_admission.csv": rows,
        "coverage.csv": pd.DataFrame(coverage),
        "missing_cell_intersections.csv": intersections,
        "frozen_indicator_cooccurrence.csv": cooccurrence,
        "shipment_definition_comparison.csv": relation,
        "published_percentages.csv": pub,
        "cross_vintage_ratio_checks.csv": ratios,
        "ratio_summary.csv": pd.DataFrame(summaries),
        "food_reconciliation.csv": pd.DataFrame(food_rows),
        "admission_decisions.csv": pd.DataFrame([admission_for(v) for v in SCOPES]),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False, lineterminator="\n")
    save(out / "QA.json", qa)
    save(
        out / "analysis_contract.json",
        {
            "decision": (
                "Separate descriptive context only; no replacement of the frozen monetary panel"
            ),
            "grain": "territory_id x year x source-specific variable",
            "observations": [2023, 2024],
            "normalization": (
                "Exact region/year/name with explicit type-token removal; "
                "no fuzzy match; no boundary proof"
            ),
            "percentage_check": (
                "Cross-edition amount ratio versus source percentage; interval overlap allowing "
                "half a unit in the final printed decimal of every input; "
                "compatibility is arithmetic only"
            ),
            "no_claims": [
                "real growth",
                "retail substitution",
                "small-firm contribution from two shipment definitions",
                "revision cause",
                "causal economic change",
            ],
            "source_review_independence": (
                "New analysis reuses prior independent PDF evidence; "
                "it is not a second independent reading of all PDFs"
            ),
        },
    )
    findings = [
        "# Допуск региональных показателей\n",
        (
            "**Решение: использовать отдельным описательным контекстом с сохранением определения "
            "и редакции источника.** Новые числа не заменяют БДПМО и не закрывают её пропуски.\n"
        ),
        (
            f"Проверены {len(rows)} значений для {qa['municipalities']} МО и {len(unique)} МО-лет. "
            "Соединение по точному нормализованному названию, региону и году повторено: потерь "
            "и размножения строк нет. Прежние core/conservative содержат "
            f"{qa['core_municipality_years']}/{qa['conservative_municipality_years']} этих МО-лет. "
            f"Для {qa['code_changed_municipalities']} МО "
            f"({qa['code_changed_municipality_years']} МО-лет) сохранён флаг изменения кода/типа; "
            "он не доказывает изменение границ, а отсутствие флага не доказывает их неизменность.\n"
        ),
        (
            f"Кандидаты относятся к {len(intersections)} уникальным незаполненным позициям "
            "прежнего набора, но ни одна позиция не восстановлена. Для ставропольской отгрузки "
            "нет принятого "
            "числового значения БДПМО того же ключа; для продовольственных продаж и перепродажи "
            "отсутствует подтверждение эквивалентности полной рознице. "
            f"{qa['other_frozen_numeric_cooccurrences']} числовых совпадений по территории и году "
            "относятся к другим показателям и не являются сверкой одного экономического понятия.\n"
        ),
        (
            "Ставропольские две отгрузки различаются классификацией деятельности и охватом "
            "организаций до 15 человек. Их разность нельзя приписывать исключительно небольшим "
            "организациям. Продажа товаров несобственного производства не равна обороту розницы. "
            "Саратовский показатель охватывает продовольственные товары и исключает скрытую "
            "деятельность; это тоже не полная розница.\n"
        ),
    ]
    for item in summaries:
        findings.append(
            f"Для `{item['variable']}` отношение сумм 2024/2023 совместимо с опубликованным "
            f"процентом при учёте округления в {item['rounding_compatible']} из {item['pairs']} "
            f"случаев; в {item['different_beyond_rounding']} случаях различие больше "
            "этого допуска. "
            f"Максимальный модуль разницы — {item['max_absolute_difference_pp']:.6f} п.п. "
            "Это сравнение разных выпусков: причина различий не установлена, ни дефляция, "
            "ни пересмотр базы не подтверждены.\n"
        )
    for item in food_rows:
        findings.append(
            f"Саратов, {item['year']}: региональный итог превышает сумму "
            f"{item['municipal_rows']} строк на {item['regional_minus_municipal_thousand_rubles']} "
            f"тыс. руб. ({item['residual_percent_of_region']:.6f}% итога). "
            "Причина остатка неизвестна; распределения между МО нет.\n"
        )
    findings.extend(
        [
            (
                "Каталожные метки для годовых данных относятся к следующему году: Ставрополь — "
                "31.07.2024 и 03.07.2025, Саратов — 23.04.2024 и 22.04.2025. Это наблюдённые метки "
                "сайта, а не доказанная история первой публикации. "
                "Доступность оперативных выпусков "
                "в конце наблюдаемого года не установлена. Их нельзя автоматически смешивать "
                "с БДПМО, выгруженной в 2026 году.\n"
            ),
            (
                "Все 144 исходных SHA проверены. Из 33 сохранённых DOCX 2024 года заново прочитаны "
                "198 полей итоговых строк; они совпали с ранее независимо прочитанными PDF. Для "
                "2023 года использован закреплённый независимый QA предыдущего раунда. Нового "
                "независимого чтения всех PDF в этом анализе нет.\n"
            ),
            (
                "Воспроизведение из корня репозитория: `uv run --frozen --offline python "
                "scripts/analyze_round6_admission.py --repo . --raw <raw6>/economy --out <NEW>`. "
                "Новые зависимости, сеть и модели не используются.\n"
            ),
        ]
    )
    (out / "findings.md").write_text("\n".join(findings), encoding="utf-8")
    for name, digest in inputs.items():
        root, relative = name.split("/", 1)
        if sha((repo if root == "repo" else raw) / relative) != digest:
            raise ValueError("Read-only input changed during analysis")
    save(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "inputs": inputs,
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.raw, args.out)))


if __name__ == "__main__":
    main()
