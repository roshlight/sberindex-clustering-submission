"""Reconcile round-five report datasets/tables with frozen CSV; no production imports."""

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGIMES = {"within_region": "Внутри регионов", "heldout_regions": "Целые регионы"}
SAMPLES = {"core": "Основная", "stable_code": "Без смены кода"}
SOURCES = {
    "models": "models/cv_comparisons.csv",
    "coverage": "audit/joint_coverage.csv",
    "selection": "recovered/regional_selection_summary.csv",
    "readout": "readout/state_update_contributions.csv",
    "geography": "context/coverage_interpretation.csv",
    "workers": "context/annual_population.csv",
    "stocks": "context/fns_population_contrast.csv",
    "cases": "context/budget_revision.csv",
}
DATASET_SOURCE = {
    **dict.fromkeys(
        [
            "primary_chart",
            "primary_table",
            "sensitivity_table",
            "update_chart",
            "update_table",
            "secondary_table",
        ],
        "models",
    ),
    "coverage_table": "coverage",
    "selection_chart": "selection",
    "selection_table": "selection",
    "slice_table": "readout",
    "geo_chart": "geography",
    "geo_table": "geography",
    "workers_table": "workers",
    "stock_table": "stocks",
    "budget_table": "cases",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(frame, **conditions):
    for key, value in conditions.items():
        frame = frame.loc[frame[key].eq(value)]
    return frame


def equal_value(got, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(got, (int, float)) and math.isclose(
            got, expected, rel_tol=0, abs_tol=1e-11
        )
    return got == expected


def equal_rows(got, expected):
    """Preserve row order, key set, strings and units; tolerate only float serialization."""
    return len(got) == len(expected) and all(
        set(a) == set(b) and all(equal_value(a[k], b[k]) for k in b)
        for a, b in zip(got, expected, strict=True)
    )


def markdown_tables(text):
    result, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            if not re.fullmatch(r"[| :\-]+", line):
                current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            result.append(current)
            current = []
    return result


def expected_datasets(root):
    """Independent report-scale projection from already independently verified calculations."""
    comparisons = pd.read_csv(root / SOURCES["models"])
    metrics = pd.read_csv(root / "models/cv_metrics.csv")
    result = {}

    def effects(rows, with_mse=False):
        items = []
        for row in rows.itertuples():
            low, high = row.ci_low * 100, row.ci_high * 100
            item = dict(
                year=str(row.year),
                regime=REGIMES[row.regime],
                sample=SAMPLES[row.sample],
                seed=int(row.seed),
                n=int(row.n),
                gain=round(row.relative_mse_reduction * 100, 4),
                low=round(low, 4),
                high=round(high, 4),
                ci=f"[{low:.4f}; {high:.4f}]",
            )
            if with_mse:
                for key, model in [("mse_base", "new_flags"), ("mse_full", "recovered_full")]:
                    record = subset(
                        metrics,
                        year=row.year,
                        regime=row.regime,
                        sample=row.sample,
                        seed=row.seed,
                        target=row.target,
                        model=model,
                    )
                    assert len(record) == 1
                    item[key] = round(float(record.iloc[0].mse_oof), 7)
            items.append(item)
        return items

    primary = subset(
        comparisons, sample="core", target="profile_clr", comparison="primary", seed=13
    )
    result["primary_chart"] = result["primary_table"] = effects(primary, True)
    sensitivity = subset(comparisons, target="profile_clr", comparison="primary", seed=13)
    result["sensitivity_table"] = effects(sensitivity.loc[sensitivity["sample"].isin(SAMPLES)])
    result["secondary_table"] = effects(
        subset(
            comparisons,
            sample="core",
            target="log_total",
            comparison="primary",
            seed=13,
            regime="heldout_regions",
        )
    )
    update = []
    for comparison, label in [
        ("state_update", "Полное обновление"),
        ("shared_flags_recovery", "Числа при общих флагах"),
    ]:
        rows = effects(
            subset(comparisons, sample="core", target="profile_clr", seed=13, comparison=comparison)
        )
        update.extend([{**row, "comparison": label} for row in rows])
    result["update_table"] = update
    result["update_chart"] = [row for row in update if row["regime"] == "Целые регионы"]
    coverage = subset(pd.read_csv(root / SOURCES["coverage"]), sample="core")
    result["coverage_table"] = [
        dict(
            year=str(r.year),
            n=int(r.denominator),
            old=int(r.old_all3),
            new=int(r.new_all3),
            old_pct=round(100 * r.old_all3 / r.denominator, 2),
            new_pct=round(100 * r.new_all3 / r.denominator, 2),
        )
        for r in coverage.itertuples()
    ]
    selection = pd.read_csv(root / SOURCES["selection"])
    result["selection_chart"] = result["selection_table"] = [
        dict(
            year=str(r.year),
            region=r.region_name,
            indicator={"shipments": "Отгрузка", "retail": "Розница"}[r.prefix],
            new_n=int(r.recovered_n),
            old_n=int(r.old_observed_n),
            new_median=round(r.recovered_median_rubles_per_person / 1000, 1),
            old_median=round(r.previously_observed_median_rubles_per_person / 1000, 1),
            ratio=round(
                r.recovered_median_rubles_per_person
                / r.previously_observed_median_rubles_per_person,
                3,
            ),
        )
        for r in selection.itertuples()
    ]
    slices = subset(
        pd.read_csv(root / SOURCES["readout"], dtype={"group": str}),
        target="profile_clr",
        regime="heldout_regions",
        slice="recovered_any",
    )
    result["slice_table"] = [
        dict(
            year=str(r.year),
            group={"False": "Остальные", "True": "Получили данные"}[r.group],
            n=int(r.n),
            regions=int(r.regions),
            local=round(r.local_relative_reduction * 100, 3),
            contribution=round(r.contribution_to_total_relative_reduction * 100, 4),
        )
        for r in slices.itertuples()
    ]
    geography = subset(pd.read_csv(root / SOURCES["geography"]), source_region_scope="ALL")
    geo_rows = []
    for r in geography.itertuples():
        federal_share = (
            100
            * r.federal_city_group_rows
            / (r.federal_city_group_rows + r.other_region_group_rows)
        )
        for column, label in [
            ("subregional_code_rows", "Код ниже уровня субъекта"),
            ("federal_city_group_rows", "Субъект: федеральные города"),
            ("other_region_group_rows", "Субъект: остальные"),
            ("unmatched_rows", "Нет точного кода"),
        ]:
            n = int(getattr(r, column))
            geo_rows.append(
                dict(
                    year=str(r.year),
                    kind=label,
                    rows=n,
                    denominator=int(r.source_rows),
                    share=round(100 * n / r.source_rows, 4),
                    federal_share_of_grouped=round(federal_share, 3),
                )
            )
    result["geo_chart"] = result["geo_table"] = geo_rows
    workers = pd.read_csv(root / SOURCES["workers"])
    workers = workers.loc[workers.legal_kind.ne("ALL")]
    result["workers_table"] = [
        dict(
            year=str(r.year),
            kind={"IP": "ИП", "legal_entity": "ЮЛ"}[r.legal_kind],
            rows=int(r.source_rows),
            known=int(r.nonnegative_employee_rows),
            pct=round(100 * r.nonnegative_employee_rows / r.source_rows, 4),
            missing=int(r.missing_employee_rows),
            negative=int(r.negative_employee_rows),
        )
        for r in workers.itertuples()
    ]
    stocks = subset(pd.read_csv(root / SOURCES["stocks"]), legal_kind="ALL")
    result["stock_table"] = [
        dict(
            year=str(r.year),
            date=r.fns_snapshot_date,
            stock=int(r.fns_stock_subjects),
            rows=int(r.tochno_period_rows),
            ids=int(r.tochno_annual_distinct_tax_ids),
        )
        for r in stocks.itertuples()
    ]
    budget = pd.read_csv(root / SOURCES["cases"], dtype={"year_column": str})
    result["budget_table"] = [
        dict(
            year=r.year_column.replace("-", "–"),
            old=r.old_value_as_published,
            new=r.new_value_as_published,
            delta="—"
            if pd.isna(r.revision_delta_million_rub)
            else round(r.revision_delta_million_rub, 4),
        )
        for r in budget.itertuples()
    ]
    return result, comparisons


def verify(repo, root):
    record = {
        "status": "RUNNING",
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": [],
        "model_refits": 0,
        "production_functions_imported": False,
        "scope": "Numerical reconciliation and local source resolution, not full prose/visual QA",
        "report_sha256": sha(root / "REPORT.md"),
        "artifact_sha256": sha(root / "artifact.json"),
        "verifier_sha256": sha(Path(__file__)),
    }

    def check(name, condition, **details):
        record["checks"].append(
            {"check": name, "status": "PASS" if condition else "FAIL", **details}
        )
        if not condition:
            raise AssertionError(name)

    try:
        artifact = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
        expected, comparisons = expected_datasets(root)
        datasets = artifact["snapshot"]["datasets"]
        check("dataset_names", set(datasets) == set(expected))
        for name, rows in expected.items():
            check(f"dataset:{name}", equal_rows(datasets[name], rows), rows=len(rows))
        headline = pd.read_csv(root / "headline_metrics.csv", dtype={"year": str})
        check("headline_csv", equal_rows(headline.to_dict("records"), expected["primary_table"]))
        report = (root / "REPORT.md").read_text(encoding="utf-8")
        tables = markdown_tables(report)
        definitions = artifact["manifest"]["tables"]
        check("markdown_table_count", len(tables) == len(definitions))
        for table, definition in zip(tables, definitions, strict=True):
            rows = expected[definition["dataset"]]
            columns = definition["columns"]
            check(f"markdown_headers:{definition['id']}", table[0] == [c["label"] for c in columns])
            check(f"markdown_shape:{definition['id']}", len(table) == len(rows) + 1)
            okay = True
            for cells, row in zip(table[1:], rows, strict=True):
                okay &= len(cells) == len(columns)
                for cell, column in zip(cells, columns, strict=True):
                    value = row[column["field"]]
                    if isinstance(value, (float, int)):
                        okay &= equal_value(float(cell), value)
                    else:
                        okay &= cell == value
            check(f"markdown_values:{definition['id']}", bool(okay))
        source_rows = artifact["sources"]
        sources = {s["id"]: s for s in source_rows}
        check("source_inventory", len(source_rows) == len(sources) and set(sources) == set(SOURCES))
        check("source_copies_match", artifact["manifest"]["sources"] == source_rows)
        for source_id, relative in SOURCES.items():
            source = sources[source_id]
            canonical = "reports/round5_analysis/" + relative
            check(
                f"source_primary:{source_id}",
                source["path"] == canonical and source["href"].endswith("/blob/main/" + canonical),
            )
            query = source["query"]
            check(
                f"source_tables:{source_id}",
                canonical in query["tables_used"]
                and all((repo / path).is_file() for path in query["tables_used"]),
            )
            check(
                f"source_computation:{source_id}",
                (repo / query["computation_path"]).is_file()
                and (repo / query["reproduction_document"]).is_file(),
            )
        for definition in definitions + artifact["manifest"]["charts"]:
            dataset = definition["dataset"]
            check(
                f"dataset_source:{definition['id']}",
                definition["sourceId"] == DATASET_SOURCE[dataset],
            )
        primary = subset(
            comparisons, sample="core", seed=13, target="profile_clr", comparison="primary"
        )
        stable = subset(
            comparisons, sample="stable_code", seed=13, target="profile_clr", comparison="primary"
        )
        extra = subset(comparisons, sample="core", target="profile_clr", comparison="primary")
        extra = extra.loc[extra.seed.isin([29, 47])]
        check(
            "primary_four_below_threshold_and_ci_includes_zero",
            len(primary) == 4
            and primary.relative_mse_reduction.lt(0.01).all()
            and (primary.ci_low.le(0) & primary.ci_high.ge(0)).all(),
        )
        check(
            "stable_four_negative", len(stable) == 4 and stable.relative_mse_reduction.lt(0).all()
        )
        check(
            "four_extra_seed_intervals_include_zero",
            len(extra) == 4 and (extra.ci_low.le(0) & extra.ci_high.ge(0)).all(),
        )
        shared = subset(
            comparisons,
            sample="core",
            seed=13,
            target="log_total",
            regime="heldout_regions",
            comparison="shared_flags_recovery",
        )
        check(
            "secondary_shared_2023_positive_2024_crosses_zero",
            len(shared) == 2
            and subset(shared, year=2023).iloc[0].ci_low > 0
            and subset(shared, year=2024).iloc[0].ci_low < 0
            and subset(shared, year=2024).iloc[0].ci_high > 0,
        )
        independent = json.loads((root / "independent_QA.json").read_text())
        readout = json.loads((root / "readout/QA.json").read_text())
        check(
            "independent_model_qa_current",
            independent["status"] == "PASS"
            and independent["model_manifest_sha256"] == sha(root / "models/manifest.json"),
        )
        check(
            "decision_unchanged_false",
            independent["independent_adoption_criteria_met"] is False
            and readout["adoption_passed"] is False
            and readout["stable_operator_clarification_changes_prior_decision"] is False,
        )
        record["source_sha256"] = {
            relative: sha(root / relative)
            for relative in [
                *SOURCES.values(),
                "models/cv_metrics.csv",
                "independent_QA.json",
                "readout/QA.json",
                "headline_metrics.csv",
            ]
        }
        record["status"] = "PASS"
    except Exception as exc:
        record["status"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root or args.repo / "reports/round5_analysis"
    out = args.out or root / "report_claims_QA.json"
    record = verify(args.repo, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": record["status"],
                "checks": len(record["checks"]),
                "output": str(out),
                "error": record.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
