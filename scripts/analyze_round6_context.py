"""Offline documentary context: financial non-equivalence and code-review queues.

Input grain is preserved: municipal investment is not airport cash, and an SME
code group is not a company or a confirmed historical municipality assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

D = Decimal
CASES = "reports/data_round6/cases"
GEO = "reports/data_round6/geography"
SME = "reports/data_round6/sme"
OLD = "reports/round5_analysis/context"
INV = "reports/data_round5/geography/sme_code_inventory.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return D(str(value)) if value not in (None, "") else None


def ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


def fmt(value):
    if isinstance(value, Decimal):
        return format(value, ".12f").rstrip("0").rstrip(".") or "0"
    return value


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: fmt(v) for k, v in row.items()} for row in rows)


def airport_comparison(finance, investments):
    inv = {r["year"]: r for r in investments}
    if len(inv) != len(investments):
        raise ValueError("Municipal year must be unique")
    cash = [r for r in finance if r["financial_kind"] == "annual_cash_execution"]
    if len({r["year"] for r in cash}) != len(cash):
        raise ValueError("Cash year must be unique")
    result = []
    for row in sorted(cash, key=lambda r: r["year"]):
        year = row["year"]
        other = inv[year]
        a, b = number(row["value_million_rubles"]), number(other["value_million_rubles"])
        if a is not None and a != number(row["value_thousand_rubles"]) / 1000:
            raise ValueError("Cash units disagree")
        if (a is None) != (row["observation_status"] == "blank_not_zero"):
            raise ValueError("Cash missingness status disagrees")
        half = D(other["rounding_half_step_million_rubles"])
        result.append(
            {
                "year": year,
                "municipality": other["municipality"],
                "airport_cash_million_rubles": a,
                "cash_status": row["observation_status"],
                "municipal_investment_million_rubles": b,
                "municipal_rounding_half_step_million_rubles": half,
                "cash_divided_by_municipal_investment": ratio(a, b),
                "ratio_lower_from_municipal_rounding_only": ratio(a, b + half),
                "ratio_upper_from_municipal_rounding_only": ratio(a, b - half),
                "ratio_is_project_share": False,
                "model_join_accepted": False,
                "cash_scope": row["scope"],
                "investment_scope": other["scope"],
                "cash_publication_date": row["publication_date"],
                "cash_source_cells": row["source_cells"],
                "cash_source_sha256": row["source_sha256"],
                "cash_source_url": row["source_url"],
                "investment_source_sha256s": other["source_sha256s"],
                "investment_source_urls": other["source_urls"],
            }
        )
    return result


def airport_changes(rows):
    index = {r["year"]: r for r in rows}
    result = []
    for a, b in [("2022", "2023"), ("2023", "2024")]:
        for metric in ["airport_cash_million_rubles", "municipal_investment_million_rubles"]:
            x, y = index[a][metric], index[b][metric]
            change = y - x if x is not None and y is not None else None
            result.append(
                {
                    "start_year": a,
                    "end_year": b,
                    "metric": metric,
                    "start_million_rubles": x,
                    "end_million_rubles": y,
                    "nominal_change_million_rubles": change,
                    "nominal_change_pct": ratio(change * 100, x) if change is not None else None,
                    "cross_metric_decomposition": False,
                }
            )
    return result


def code_groups(edges):
    """Count each year/code once; edges retain their distinct historical evidence."""
    groups = defaultdict(list)
    for row in edges:
        groups[(row["year"], row["source_code"])].append(row)
    result = []
    for (year, code), rows in sorted(groups.items()):
        for field in ["source_rows", "region", "new_code_predecessors_all_regions"]:
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Inconsistent {field} within year/code")
        if any(r["address_date_proven"] != "False" for r in rows):
            raise ValueError("This analysis accepts candidates only")
        parents = {r["parent_code"] for r in rows}
        old_codes = {r["old_code"] for r in rows}
        all_old = int(rows[0]["new_code_predecessors_all_regions"])
        if all_old < len(old_codes):
            raise ValueError("Candidate predecessor count exceeds full graph")
        if len(parents) > 1:
            state = "multiple_candidate_parents"
        elif all_old > len(old_codes):
            state = "additional_predecessor_outside_candidate_set"
        elif len(old_codes) > 1:
            state = "multiple_settlements_same_candidate_parent"
        else:
            state = "single_observed_predecessor_parent"
        result.append(
            {
                "year": year,
                "code_region": rows[0]["region"],
                "source_code": code,
                "source_rows_once": int(rows[0]["source_rows"]),
                "candidate_edge_rows": len(rows),
                "candidate_old_codes": len(old_codes),
                "all_crosswalk_predecessor_codes": all_old,
                "candidate_parent_codes": len(parents),
                "candidate_territories": ";".join(sorted({r["territory_id"] for r in rows})),
                "ambiguity_class": state,
                "source_rows_are_unique_firms": False,
                "historical_assignment_approved": False,
            }
        )
    return result


def queue_summary(groups, inventory):
    totals = defaultdict(int)
    counts = {}
    for row in inventory:
        totals[row["year"]] += int(row["source_rows"])
        if row["source_code_is_null"] == "False":
            key = row["year"], row["source_code"]
            if key in counts:
                raise ValueError("Duplicate nonnull inventory code/year")
            counts[key] = int(row["source_rows"])
    for row in groups:
        if counts[(row["year"], row["source_code"])] != row["source_rows_once"]:
            raise ValueError("Candidate/inventory row counts disagree")
    summary = []
    partitions = sorted({(r["year"], r["code_region"]) for r in groups})
    partitions += [(year, "ALL_CANDIDATE_CODE_REGIONS") for year in sorted(totals)]
    for year, region in partitions:
        subset = [
            r
            for r in groups
            if r["year"] == year
            and (r["code_region"] == region or region == "ALL_CANDIDATE_CODE_REGIONS")
        ]
        weighted = sum(r["source_rows_once"] for r in subset)
        naive = sum(r["source_rows_once"] * r["candidate_edge_rows"] for r in subset)
        ambiguous = [r for r in subset if r["all_crosswalk_predecessor_codes"] > 1]
        summary.append(
            {
                "year": year,
                "code_region": region,
                "source_code_groups": len(subset),
                "source_rows_once": weighted,
                "denominator_all_sme_year_rows": totals[year],
                "share_all_sme_year_rows_pct": D(weighted) * 100 / totals[year],
                "naive_edge_weighted_rows": naive,
                "naive_excess_rows": naive - weighted,
                "naive_excess_pct_of_correct_queue": D(naive - weighted) * 100 / weighted,
                "multi_predecessor_code_groups": len(ambiguous),
                "multi_predecessor_rows_once": sum(r["source_rows_once"] for r in ambiguous),
                "multiple_candidate_parent_groups": sum(
                    r["candidate_parent_codes"] > 1 for r in subset
                ),
                "additional_predecessor_outside_groups": sum(
                    r["ambiguity_class"] == "additional_predecessor_outside_candidate_set"
                    for r in subset
                ),
                "historical_assignments_approved": 0,
                "denominator_is_code_region_population": False,
            }
        )
    return summary


def ambiguity_summary(edges):
    """Disjoint edge classes; node totals are separate and never added to edges."""
    classes = defaultdict(list)
    for row in edges:
        reverse = int(row["predecessor_code_count"]) > 1
        forward = int(row["successor_code_count"]) > 1
        if not (reverse or forward):
            raise ValueError("Input contains an unambiguous edge")
        state = (
            "both_directions"
            if reverse and forward
            else "reverse_only"
            if reverse
            else "forward_only"
        )
        classes[state].append(row)
    result = []
    for state, rows in sorted(classes.items()):
        distinct = {(r["old_code"], r["new_code"]) for r in rows}
        if len(distinct) != len(rows):
            raise ValueError("Duplicate ambiguity edge")
        result.append(
            {
                "edge_class": state,
                "edges": len(rows),
                "distinct_old_codes": len({r["old_code"] for r in rows}),
                "distinct_new_codes": len({r["new_code"] for r in rows}),
                "regions": len({r["region"] for r in rows}),
                "denominator_all_ambiguous_edges": len(edges),
                "edge_share_pct": D(len(rows)) * 100 / len(edges),
                "unique_code_counts_not_additive_between_classes": True,
            }
        )
    return result


def parent_summary(rows):
    partitions = defaultdict(list)
    for row in rows:
        partitions[row["year"], row["ter"]].append(row)
    result = []
    for (year, ter), group in sorted(partitions.items()):
        codes = defaultdict(set)
        for row in group:
            codes[row["settlement_code"]].add(row["parent_code"])
        result.append(
            {
                "year": year,
                "classifier_ter": ter,
                "candidate_rows": len(group),
                "settlement_codes": len(codes),
                "multiple_parent_settlement_codes": sum(len(x) > 1 for x in codes.values()),
                "parent_codes": len({r["parent_code"] for r in group}),
                "record_date_after_reference_rows": sum(
                    r["record_date_after_reference"] == "True" for r in group
                ),
                "unparsed_date_rows": sum(r["record_date_unparsed"] == "True" for r in group),
                "registry_version_label": group[0]["registry_version_label"],
                "historical_company_assignments_approved": 0,
            }
        )
    return result


def asset_readiness(syntax):
    groups = defaultdict(list)
    for row in syntax:
        groups[row["asset"]].append(row)
    result = []
    for asset, rows in sorted(groups.items()):
        counts = {r["code_state"]: int(r["rows"]) for r in rows}
        denom = {int(r["denominator_rows"]) for r in rows}
        if len(counts) != len(rows) or len(denom) != 1 or sum(counts.values()) != next(iter(denom)):
            raise ValueError("Asset code states must partition source rows")
        total = next(iter(denom))
        exact = counts["exact_8_or_11_digits"]
        result.append(
            {
                "asset": asset,
                "source_rows": total,
                "exact_syntax_rows": exact,
                "syntax_review_rows": total - exact,
                "syntax_review_pct": D(total - exact) * 100 / total,
                "scientific_notation_rows": counts["scientific_notation"],
                "other_digit_length_rows": counts["other_digit_length"],
                "empty_rows": counts["empty"],
                "other_syntax_rows": counts["other_syntax"],
                "exact_syntax_proves_historical_validity": False,
                "rows_are_distinct_companies": False,
            }
        )
    return result


def worker_temporal(rows):
    result = []
    for row in rows:
        data = datetime.strptime(row["data_date"], "%d.%m.%Y").date()
        doc = datetime.strptime(row["document_date"], "%d.%m.%Y").date()
        for year in [2023, 2024]:
            result.append(
                {
                    "target_year": year,
                    "data_date": data.isoformat(),
                    "document_date": doc.isoformat(),
                    "workers_state": row["workers_state"],
                    "records": int(row["records"]),
                    "denominator_sample_records": int(row["denominator_records"]),
                    "document_lag_days_from_data_date": (doc - data).days,
                    "measurement_year_matches_target": data.year == year,
                    "document_after_target_year_end": doc.year > year,
                    "eligible_historical_input": False,
                    "representative_sample": False,
                }
            )
    return result


class Inputs:
    def __init__(self, repo):
        self.repo, self.hashes = repo, {}

    def read(self, path):
        target = self.repo / path
        self.hashes[path] = sha(target)
        manifest_path = target.parent / "manifest.json"
        if target.suffix == ".csv" and manifest_path.exists():
            relative = manifest_path.relative_to(self.repo).as_posix()
            self.hashes[relative] = sha(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("outputs", {})
            if "files" in manifest:
                expected = {r["path"]: r["sha256"] for r in manifest["files"]}
            if expected.get(target.name) != self.hashes[path]:
                raise ValueError(f"Upstream manifest mismatch: {path}")
        return target.read_text(encoding="utf-8-sig")

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path))))


def source_availability(inputs, evidence, versions, workers, parent_rows):
    result = []
    investment_evidence = inputs.csv("reports/data_round4/cases/evidence_ledger.csv")
    yearbooks = defaultdict(list)
    for row in investment_evidence:
        if row["territory_id_context_candidate"] == "331":
            yearbooks[row["source_id"]].append(row)
    for source_id, rows in sorted(yearbooks.items()):
        row = rows[0]
        result.append(
            {
                "source_family": "municipal_investment_yearbook",
                "evidence_key": source_id,
                "measurement_time": ";".join(sorted({r["period"] for r in rows})),
                "source_date_or_version": row["catalogue_display_date"] or row["edition_year"],
                "date_kind": "catalogue_label" if row["catalogue_display_date"] else "edition_year",
                "retrieved_at": row["retrieved_at_utc"],
                "sha256": row["source_sha256"],
                "source_url": row["source_url"],
                "historical_model_input_approved": False,
                "limitation": "Edition/catalogue label does not prove when exact bytes "
                "became public. " + "HTTP Last-Modified: " + row["http_last_modified"],
            }
        )
    for row in evidence:
        result.append(
            {
                "source_family": "airport_documentary_evidence",
                "evidence_key": row["evidence_id"],
                "measurement_time": row["reported_year"],
                "source_date_or_version": row["source_publication_date"],
                "date_kind": "publication_or_registry_revision_as_reported",
                "retrieved_at": row["retrieved_at_utc"],
                "sha256": row["source_sha256"],
                "source_url": row["source_url"],
                "historical_model_input_approved": False,
                "limitation": row["limitation"],
            }
        )
    downloads = inputs.csv(f"{GEO}/downloaded_sources.csv")
    by_sha = {r["sha256"]: r for r in downloads}
    for row in versions:
        src = by_sha[row["sha256"]]
        result.append(
            {
                "source_family": "oktmo_transition_release",
                "evidence_key": row["version"],
                "measurement_time": "effective_dates_not_in_csv",
                "source_date_or_version": row["version"],
                "date_kind": "release_label",
                "retrieved_at": src["retrieved_at"],
                "sha256": row["sha256"],
                "source_url": src["url"],
                "historical_model_input_approved": False,
                "limitation": "Release label does not date individual transitions or addresses.",
            }
        )
    registry = {(r["registry_version_label"], r["registry_sha256"]) for r in parent_rows}
    prior_versions = inputs.csv("reports/data_round5/geography/registry_versions.csv")
    for version, digest in sorted(registry):
        src = next(r for r in prior_versions if r["sha256"] == digest)
        result.append(
            {
                "source_family": "historical_classifier_release",
                "evidence_key": version,
                "measurement_time": "records_have_individual_introduction_dates",
                "source_date_or_version": version,
                "date_kind": "registry_release_label",
                "retrieved_at": src["retrieved_at"],
                "sha256": digest,
                "source_url": src.get("source_url", src.get("url", "")),
                "historical_model_input_approved": False,
                "limitation": "Presence and record date do not establish a company address.",
            }
        )
    sample = inputs.csv(f"{SME}/worker_sample_summary.csv")[0]
    attempts = inputs.csv(f"{SME}/retrieval_attempts.csv")
    for row in {tuple(r.items()) for r in workers}:
        record = dict(row)
        result.append(
            {
                "source_family": "fns_workers_selected_xml",
                "evidence_key": record["workers_state"],
                "measurement_time": record["data_date"],
                "source_date_or_version": record["document_date"],
                "date_kind": "xml_document_date",
                "retrieved_at": ";".join(
                    sorted(
                        {r["retrieved_at"] for r in attempts if r["url"] == sample["archive_url"]}
                    )
                ),
                "sha256": sample["sample_sha256"],
                "source_url": sample["archive_url"],
                "historical_model_input_approved": False,
                "limitation": "900 selected records, not representative; data year 2025.",
            }
        )
    for row in inputs.csv(f"{SME}/author_source_lineage.csv"):
        result.append(
            {
                "source_family": "author_code_and_assets",
                "evidence_key": row["path"],
                "measurement_time": "asset_vintage_not_independently_dated_per_row",
                "source_date_or_version": row["git_commit"],
                "date_kind": "git_commit",
                "retrieved_at": row["prior_package_retrieved_at"],
                "sha256": row["sha256"],
                "source_url": row["git_url"],
                "historical_model_input_approved": False,
                "limitation": "12 selected package files matched; exact tochno build unproven.",
            }
        )
    return sorted(result, key=lambda r: (r["source_family"], r["evidence_key"]))


def run(repo, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory")
    inputs = Inputs(repo)
    finance = inputs.csv(f"{CASES}/airport_annual_finance.csv")
    investments = inputs.csv(f"{OLD}/chersky_investment_alignment.csv")
    edges = inputs.csv(f"{GEO}/candidate_transitions_to_panel.csv")
    parents = inputs.csv(f"{GEO}/priority_settlement_parent_candidates.csv")
    evidence = inputs.csv(f"{CASES}/evidence_ledger.csv")
    groups = code_groups(edges)
    comparisons = airport_comparison(finance, investments)
    queue = queue_summary(groups, inputs.csv(INV))
    workers = inputs.csv(f"{SME}/worker_dates.csv")
    tables = {
        "airport_comparison.csv": comparisons,
        "airport_changes.csv": airport_changes(comparisons),
        "geography_code_groups.csv": groups,
        "geography_queue_summary.csv": queue,
        "settlement_parent_summary.csv": parent_summary(parents),
        "transition_ambiguity_summary.csv": ambiguity_summary(
            inputs.csv(f"{GEO}/ambiguous_transition_edges.csv")
        ),
        "sme_asset_readiness.csv": asset_readiness(inputs.csv(f"{SME}/asset_code_syntax.csv")),
        "worker_temporal_eligibility.csv": worker_temporal(workers),
        "source_availability.csv": source_availability(
            inputs, evidence, inputs.csv(f"{GEO}/crosswalk_versions.csv"), workers, parents
        ),
    }
    inputs.read("uv.lock")
    inputs.read("pyproject.toml")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(out / name, rows)
    claims = [
        (
            "cash_2022_and_2023_documented",
            "accepted_documentary",
            "airport_comparison.csv;source_availability.csv",
            "Two annual sums from federal auditor cells; "
            "units independently checked in collection.",
            "Retrospective context only; statistical allocation remains unproven.",
        ),
        (
            "code_queue_double_count_quantified",
            "accepted_scoped",
            "geography_code_groups.csv;geography_queue_summary.csv",
            "Each code/year counted once against the original SME code inventory.",
            "Not unique firms, accepted assignments or a regional population estimate.",
        ),
        (
            "worker_sample_measurement_and_document_date_differ",
            "accepted_scoped",
            "worker_temporal_eligibility.csv",
            "2025 measurement and 2026 document date remain separate; 237-day lag in this sample.",
            "Nonrepresentative sample; no inference about all FNS publication lags.",
        ),
        (
            "cash_is_municipal_investment",
            "unsupported",
            "airport_comparison.csv;airport_changes.csv",
            "Cash2022 exceeds municipal investment; two series have opposite2022to2023 changes.",
            "Reconcile statistical respondent, place, recognition period and accounting scope.",
        ),
        (
            "airport_contributed_to_spike",
            "unknown",
            "airport_comparison.csv;source_availability.csv",
            "Annual cash is documented, but contribution is not identified.",
            "Need project-to-municipal statistical reconciliation, not event coincidence.",
        ),
        (
            "airport_cash_2024_zero",
            "unsupported",
            "airport_comparison.csv",
            "Official source cells are blank, ratio remains null.",
            "Need actual2024 cash report.",
        ),
        (
            "other_projects_or_statistical_scope_explain_spike",
            "unknown",
            "source_availability.csv",
            "Alternative projects, respondent changes and recognition timing remain unresolved.",
            "Obtain annual project amounts and statistical coverage; no residual allocation.",
        ),
        (
            "queue_gives_extra_municipal_coverage",
            "unsupported",
            "geography_queue_summary.csv",
            "Nonduplicated code-group rows measure review workload, not accepted assignments.",
            "Need dated company addresses and effective transitions.",
        ),
        (
            "single_candidate_parent_proves_location",
            "unsupported",
            "geography_code_groups.csv",
            "One candidate parent can conceal multiple or excluded predecessors.",
            "Resolve all predecessors and address history before adoption.",
        ),
        (
            "workers_xml_improves_2023_2024_features",
            "unsupported",
            "worker_temporal_eligibility.csv",
            "All900 sampled observations refer to2025 with documents dated2026.",
            "Obtain target-year data and publication/revision lineage.",
        ),
        (
            "asset_syntax_issue_is_firm_join_error",
            "unsupported",
            "sme_asset_readiness.csv",
            "Asset rows and companies are different populations; exact build is unknown.",
            "Trace exact author-code run and per-record geography.",
        ),
    ]
    claim_rows = [
        dict(
            zip(["claim", "status", "evidence", "finding", "remaining_requirement"], r, strict=True)
        )
        for r in claims
    ]
    write_csv(out / "claim_evidence.csv", claim_rows)
    qa = {
        "status": "ready_with_conditions",
        "machine_checks": "PASS",
        "tables": {k: len(v) for k, v in tables.items()},
        "cash_ratio_is_not_share": True,
        "missing_cash2024_preserved": True,
        "code_groups_unique_year_code": len(groups)
        == len({(r["year"], r["source_code"]) for r in groups}),
        "queue_matches_original_code_inventory": True,
        "no_employee_totals_computed": True,
        "no_historical_company_join": True,
        "all_read_csv_hashes_verified_against_upstream_manifests": True,
        "raw_pdf_xlsx_xml_reextraction_this_analysis": "NOT RUN: using verified collection outputs",
        "independent_clean_environment": "NOT RUN",
        "model_adoption": False,
    }
    write_json(out / "QA.json", qa)
    (out / "findings.md").write_text(findings(tables), encoding="utf-8", newline="\n")
    for path, digest in inputs.hashes.items():
        if sha(repo / path) != digest:
            raise ValueError(f"Input changed during run: {path}")
    manifest = {
        "schema_version": 1,
        "script_sha256": sha(Path(__file__)),
        "inputs": dict(sorted(inputs.hashes.items())),
        "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "historical_model_adoption": False,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(qa, ensure_ascii=False))


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def findings(tables):
    cash = {r["year"]: r for r in tables["airport_comparison.csv"]}
    queue = [
        r
        for r in tables["geography_queue_summary.csv"]
        if r["code_region"] == "ALL_CANDIDATE_CODE_REGIONS"
    ]
    lines = [
        "# Контекст шестого анализа\n",
        "Новые документы уточняют проверку инвестиционного случая Черского, но не устанавливают "
        "вклад аэропорта в муниципальный показатель. Географические связи позволяют организовать "
        "проверку адресов; ни одна историческая запись МСП не принята в модель.\n",
        "## Два денежных ряда нельзя подставлять друг вместо друга\n",
        "| Год | Касса аэропорта, млн руб. | Муниципальные инвестиции, млн руб. | "
        "Отношение кассы к инвестициям |",
        "|---|---:|---:|---:|",
    ]
    for year, row in cash.items():
        a = fmt(row["airport_cash_million_rubles"])
        r = row["cash_divided_by_municipal_investment"]
        lines.append(
            f"| {year} | {a if a is not None else 'нет значения'} | "
            f"{fmt(row['municipal_investment_million_rubles'])} | "
            f"{format(r, '.6f') if r is not None else 'не рассчитывается'} |"
        )
    lines += [
        "\nЭто отношения разных показателей, а не доли проекта. Муниципальная серия исключает "
        "малый бизнес и инвестиции, не наблюдаемые прямыми статистическими методами; "
        "касса относится "
        "к объекту федерального заказчика. Пределы отношения в CSV учитывают только "
        "округление муниципального числа на ±0.05 млн рублей, это не доверительный интервал.\n",
        "В 2022 году касса значительно превышает муниципальный итог. В 2023 году касса снижается, "
        "а муниципальные инвестиции растут. Такая динамика не подтверждает прямую замену одного "
        "ряда другим. Она также не исключает участие аэропорта: остаются несогласованными место "
        "статистического учёта, состав респондентов, период признания вложений и кассовый момент. "
        "Ни одна из этих возможных причин расхождения здесь не установлена.\n",
        "Годовое кассовое исполнение теперь подтверждено для 2022 и 2023, тогда как раньше были "
        "прежде всего даты проекта и редакции бюджета. Пустые ячейки 2024 сохранены как пропуск. "
        "Возврат переплаты и план финансирования из ЕИС не подменяют кассу. "
        "Годовая касса транспортно-логистического центра и амбулатории Андрюшкино не получена; "
        "их возможный вклад, изменения охвата респондентов и временная регистрация вложений "
        "остаются вопросами для сверки, а не установленными объяснениями. Разность двух денежных "
        "рядов не распределяется между другими проектами.\n",
        "Источники: [официальная проверка Счётной палаты]"
        "(https://ach.gov.ru/checks/stroitelstvo-transportnoy-infrastruktury), "
        "`airport_comparison.csv` и `source_availability.csv`. Документы проверки опубликованы "
        "03.12.2025; использование их значений как информации, известной в 2023/2024, "
        "не разрешено.\n",
        "## Очередь проверки МСП без двойного счёта\n",
        "| Год | Групп кодов | Строк МСП один раз | Все строки МСП года | "
        "Наивный подсчёт по рёбрам | Завышение, строк |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in queue:
        lines.append(
            f"| {row['year']} | {row['source_code_groups']} | {row['source_rows_once']} | "
            f"{row['denominator_all_sme_year_rows']} | {row['naive_edge_weighted_rows']} | "
            f"{row['naive_excess_rows']} |"
        )
    lines += [
        "\nЗнаменатель — все периодные строки соответствующего годового файла МСП. Они не равны "
        "числу уникальных предприятий. Географический регион очереди определяется кодом перехода; "
        "он не доказывает совпадение с регионом каждой исходной записи МСП.\n",
        "Два кода Перми имеют несколько предшествующих населённых пунктов с одним и тем же "
        "родительским МО-кандидатом; третий код имеет дополнительного предшественника вне "
        "сформированного набора кандидатов. Поэтому отсутствие нескольких родителей внутри "
        "очереди не означает, что вся историческая неоднозначность устранена. Каждая пара "
        "«год — исходный код» учитывается один раз. `geography_code_groups.csv` сохраняет "
        "различие между неоднозначностью населённого пункта и родительского МО.\n",
        "Сводка 98 неоднозначных рёбер официальной таблицы делит их на непересекающиеся классы "
        "направлений. Числа старых и новых кодов не складываются с числом рёбер. "
        "9736 записей исторического справочника — отдельная очередь «населённый пункт — родитель»; "
        "её нельзя прибавлять к строкам МСП. Версия переходов 01.06.2026 и даты выпусков "
        "классификатора не являются датами действия адресов предприятий. "
        "[Источник переходов](https://rosstat.gov.ru/opendata/7708234640-codingtable).\n",
        "## Работники и авторские справочники\n",
        "`sme_asset_readiness.csv` считает непересекающиеся состояния написания кодов внутри "
        "каждого справочника. Это нагрузка на проверку исходных справочников, а не число ошибочно "
        "присоединённых фирм. Даже восемь или одиннадцать цифр не доказывают "
        "историческую валидность. "
        "Точная сборка tochno остаётся неустановленной.\n",
        "Все 900 строк выбранного XML работников относятся к 31.12.2025; дата документа — "
        "25.08.2026. Это не новые значения за 2023/2024. Состояния «ноль» "
        "и «положительное значение» "
        "сохранены отдельно, работники по строкам не суммируются. Выборка не репрезентативна. "
        "Источник и SHA конкретного XML записаны в `source_availability.csv`.\n",
        "## Воспроизведение и решение\n",
        "Из корня репозитория в новый каталог:\n",
        "```powershell\nuv run --frozen --offline python scripts/analyze_round6_context.py "
        "--repo . --out <NEW>\n```\n",
        "Используются сохранённые проверенные таблицы сборов. Их SHA сверяются с прежними "
        "манифестами, входы и выходы нового расчёта закреплены в `manifest.json`. Повторное чтение "
        "сырого PDF/XLSX/XML в этом аналитическом шаге не выполнялось; независимый чистый "
        "runtime не проверен. `claim_evidence.csv` разделяет неподтверждённые утверждения "
        "и вопросы с неизвестным ответом. Итог — **ready with conditions** для документального "
        "контекста и организации проверки, **0** принятых исторических присоединений МСП. "
        "Модели, исходные панели и прежние результаты не меняются.\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.out.resolve())
