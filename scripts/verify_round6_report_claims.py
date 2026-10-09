"""Independent source-to-report reconciliation; no builder or analytical imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REGIONS = {26: "Ставрополье", 64: "Саратовская область"}
PATHS = {
    "regions": "regions/associations.csv",
    "coverage": "regions/coverage.csv",
    "panel": "regions/regional_panel.csv",
    "clusters": "regions/cluster_alignment.csv",
    "cash": "context/airport_comparison.csv",
    "changes": "context/airport_changes.csv",
    "queue": "context/geography_queue_summary.csv",
    "ratios": "admission/ratio_summary.csv",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(got, expected):
    if isinstance(expected, dict):
        return (
            isinstance(got, dict)
            and set(got) == set(expected)
            and all(equal(got[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(got, list)
            and len(got) == len(expected)
            and all(equal(a, b) for a, b in zip(got, expected, strict=True))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(got, (int, float))
            and not isinstance(got, bool)
            and math.isclose(got, expected, rel_tol=0, abs_tol=1e-11)
        )
    return got == expected


def ci(row, stem):
    return f"[{row[stem + '_ci_low']:.3f}; {row[stem + '_ci_high']:.3f}]"


def expected_datasets(frames):
    """Report-scale projections derived directly from independently audited tables."""
    result = {}
    assoc = frames["regions"]
    primary = assoc[assoc["sample"] == "paired_available"]
    level = primary[primary.outcome == "log_total"]
    urban = assoc[(assoc["sample"] == "paired_urban_complete") & (assoc.outcome == "log_total")]
    result["coverage"] = []
    for region in [26, 64]:
        group = frames["coverage"]
        group = group[(group.region_code == region) & (group.year == 2023)].set_index("sample")
        result["coverage"].append(
            {
                "region": REGIONS[region],
                **{
                    key: int(group.loc[sample, "n"])
                    for key, sample in {
                        "paired": "paired_available",
                        "stable": "paired_stable_code",
                        "urban": "paired_urban_complete",
                        "stable_urban": "paired_stable_urban_complete",
                    }.items()
                },
            }
        )
    result["primary_correlations"] = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "measure": REGIONS[r.region_code]
            + (": уровень" if r.outcome == "log_total" else ": доля питания"),
            "rho": round(r.rho, 4),
            "n": int(r.n),
            "interval": ci(r, "rho"),
        }
        for _, r in primary.iterrows()
    ]
    result["level_coefficients"] = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "rho": round(r.rho, 3),
            "rho_ci": ci(r, "rho"),
            "partial": round(r.partial_rho, 3),
            "partial_ci": ci(r, "partial_rho"),
        }
        for _, r in level.iterrows()
    ]
    result["urban_comparison"] = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "raw": round(r.rho, 3),
            "pop": round(r.partial_rho, 3),
            "urban": round(r.urban_partial_rho, 3),
            "interval": ci(r, "urban_partial_rho"),
        }
        for _, r in urban.iterrows()
    ]
    result["cluster_alignment"] = [
        {
            "region": REGIONS[r.region_code],
            "year": str(r.year),
            "n": int(r.n),
            "groups": int(r.clusters),
            "eta": round(r.rank_eta_squared, 3),
            "null": round(r.null_mean, 3),
            "p": round(r.permutation_p, 4),
            "q": round(r.permutation_q_bh, 4),
        }
        for r in frames["clusters"].itertuples()
    ]
    panel = frames["panel"]
    scatter = panel[(panel.region_code == 64) & (panel.year == 2024) & panel.paired_available]
    result["saratov_scatter"] = [
        {
            "municipality": r.panel_name,
            "territory_id": int(r.territory_id),
            "food_thousand_per_person": round(r.rubles_per_person / 1000, 4),
            "monthly_spending_thousand": round(r.total_mean / 1000, 4),
            "population": int(r.population_total),
        }
        for r in scatter.itertuples()
    ]
    cash = frames["cash"]
    result["airport_money"] = [
        {"year": str(r.year), "series": label, "value": float(getattr(r, field))}
        for r in cash[cash.year.isin([2022, 2023])].itertuples()
        for field, label in [
            ("airport_cash_million_rubles", "Касса аэропорта"),
            ("municipal_investment_million_rubles", "Муниципальные инвестиции"),
        ]
    ]
    queue = frames["queue"]
    queue = queue[queue.code_region == "ALL_CANDIDATE_CODE_REGIONS"]
    result["queue"] = [
        {
            "year": str(r.year),
            "groups": int(r.source_code_groups),
            "rows": int(r.source_rows_once),
            "denominator": int(r.denominator_all_sme_year_rows),
            "share": round(r.share_all_sme_year_rows_pct, 3),
            "excess": int(r.naive_excess_rows),
        }
        for r in queue.itertuples()
    ]
    labels = {
        "shipments_economic_activity": "Отгрузка: хозяйственные виды",
        "shipments_pure_activity": "Отгрузка: чистые виды",
        "resale_non_own_goods": "Товары несобственного производства",
    }
    result["source_percentages"] = [
        {
            "variable": labels[r.variable],
            "pairs": int(r.pairs),
            "compatible": int(r.rounding_compatible),
            "different": int(r.different_beyond_rounding),
            "maximum": round(r.max_absolute_difference_pp, 3),
        }
        for r in frames["ratios"].itertuples()
    ]
    return result


def markdown_tables(report):
    result, current = [], []
    for line in report.splitlines() + [""]:
        if line.startswith("|"):
            if not re.fullmatch(r"[| :\-]+", line):
                current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            result.append(current)
            current = []
    return result


def verify(repo, root, report_dir):
    record = {
        "status": "RUNNING",
        "checks": [],
        "production_functions_imported": False,
        "model_refits": 0,
        "network_requests": 0,
        "scope": "Independent CSV-to-report numbers and explicit semantic guards; not visual QA",
        "verifier_sha256": sha(Path(__file__)),
    }

    def check(name, condition, **details):
        record["checks"].append(
            {"check": name, "status": "PASS" if condition else "FAIL", **details}
        )
        if not condition:
            raise AssertionError(name)

    try:
        artifact = json.loads((report_dir / "artifact.json").read_text("utf8"))
        report = (report_dir / "REPORT.md").read_text("utf8")
        claims = json.loads((report_dir / "claim_values.json").read_text("utf8"))
        record.update(
            {
                "report_sha256": sha(report_dir / "REPORT.md"),
                "artifact_sha256": sha(report_dir / "artifact.json"),
                "claim_values_sha256": sha(report_dir / "claim_values.json"),
            }
        )
        frames = {k: pd.read_csv(root / p) for k, p in PATHS.items()}
        expected = expected_datasets(frames)
        datasets = artifact["snapshot"]["datasets"]
        check("dataset_names", set(datasets) == set(expected))
        for name, rows in expected.items():
            check(f"dataset:{name}", equal(datasets[name], rows), rows=len(rows))
        expected_claims = {
            key: expected[dataset]
            for key, dataset in {
                "level": "level_coefficients",
                "urban": "urban_comparison",
                "clusters": "cluster_alignment",
                "queue": "queue",
                "coverage": "coverage",
                "admission": "source_percentages",
            }.items()
        }
        check("claim_values", equal(claims, expected_claims))
        definitions = artifact["manifest"]["tables"]
        tables = markdown_tables(report)
        check("markdown_table_count", len(tables) == len(definitions))
        for table, definition in zip(tables, definitions, strict=True):
            columns = definition["columns"]
            rows = expected[definition["dataset"]]
            expected_table = [[c["label"] for c in columns]] + [
                [str(row[c["field"]]) for c in columns] for row in rows
            ]
            check(f"markdown_table:{definition['id']}", table == expected_table)
        blocks = {
            b["id"]: b["body"] for b in artifact["manifest"]["blocks"] if b["type"] == "markdown"
        }
        for key, body in blocks.items():
            check(f"prose_shared:{key}", body in report)
        level = frames["regions"]
        level = level[(level["sample"] == "paired_available") & (level.outcome == "log_total")]
        loo_min, loo_max = level.rho_loo_min.min(), level.rho_loo_max.max()
        food = frames["regions"]
        food = food[(food["sample"] == "paired_available") & (food.outcome == "p_food")]
        check(
            "food_raw_correlation_unconfirmed_not_all_profile_claim",
            len(food) == 4
            and ((food.rho_ci_low < 0) & (food.rho_ci_high > 0)).all()
            and "долей питания в пяти категориях" in blocks["summary"],
        )
        check(
            "four_positive_level_and_population_adjusted",
            len(level) == 4 and level.rho.gt(0).all() and level.partial_rho.gt(0).all(),
        )
        check(
            "loo_range_exact_rounding",
            f"{loo_min:.3f} и {loo_max:.3f}" in blocks["influence"] and loo_min > 0,
            minimum=loo_min,
            maximum=loo_max,
        )
        urban = frames["regions"]
        urban = urban[(urban["sample"] == "paired_urban_complete") & (urban.outcome == "log_total")]
        check(
            "urban_four_intervals_cover_zero",
            len(urban) == 4
            and (
                (urban.urban_partial_rho_ci_low < 0) & (urban.urban_partial_rho_ci_high > 0)
            ).all(),
        )
        check(
            "urban_selection_counts",
            urban.groupby("region_code").n.unique().map(list).to_dict() == {26: [19], 64: [32]},
        )
        check(
            "urban_selection_not_control_effect",
            "изменение состава" in blocks["urban"]
            and "нельзя назвать эффектом контроля" in blocks["urban"]
            and "не позволяет утверждать, что связь отсутствует" in blocks["urban"],
        )
        cluster = frames["clusters"]
        stav_urban = urban[urban.region_code == 26]
        check(
            "urban_raw_coefficients_in_prose",
            " / ".join(f"{v:.3f}" for v in stav_urban.rho) in blocks["urban"]
            and "19 ставропольских и 32 саратовских" in blocks["urban"],
        )
        small = frames["regions"]
        small = small[
            (small.region_code == 26) & (small["sample"] == "paired_stable_urban_complete")
        ]
        protocol = json.loads((repo / "configs/round6_analysis_v1.json").read_text("utf8"))
        check(
            "nine_municipalities_not_estimated_under_protocol_twelve",
            len(small) == 4
            and small.n.eq(9).all()
            and small.status.str.startswith("NOT_ESTIMATED").all()
            and protocol["minimum_n"] == 12
            and "содержит 9 МО" in blocks["urban"]
            and "минимуме 12" in blocks["urban"],
        )
        significant = set(
            zip(
                cluster.loc[cluster.permutation_q_bh < 0.05, "region_code"],
                cluster.loc[cluster.permutation_q_bh < 0.05, "year"],
                strict=True,
            )
        )
        check("cluster_bh_conclusion", significant == {(26, 2023), (64, 2023), (64, 2024)})
        check(
            "cluster_bh_and_exchangeability_caveats",
            "четыре" in blocks["clusters"]
            and "обменность" in blocks["clusters"]
            and "не учитывают пространственную зависимость" in blocks["clusters"]
            and "Метки годов не совмещались" in blocks["clusters"],
        )
        support_path = repo / "reports/enriched_analysis/annual_networks/yearly_support.csv"
        support = pd.read_csv(support_path)
        check(
            "frozen_strong_memberships_zero",
            support.n_strong_memberships.tolist() == [0, 0]
            and support.n.tolist() == [2016, 2016]
            and "0 из 2016 в 2023" in blocks["stability"]
            and "0 из 2016 в" in blocks["stability"],
        )
        cash = frames["cash"].set_index("year")
        cash_ratio = (
            cash.loc[2022, "airport_cash_million_rubles"]
            / cash.loc[2022, "municipal_investment_million_rubles"]
        )
        change = 100 * (
            cash.loc[2023, "airport_cash_million_rubles"]
            / cash.loc[2022, "airport_cash_million_rubles"]
            - 1
        )
        check(
            "cash_ratio_and_decline",
            f"{cash_ratio:.3f} раза" in blocks["cash"] and f"{abs(change):.3f}%" in blocks["cash"],
        )
        check(
            "cash_ratio_not_share",
            not cash.ratio_is_project_share.any()
            and "нельзя объявить долей" in blocks["cash"]
            and "два самостоятельных номинальных ряда" in blocks["cash"],
        )
        check(
            "cash_2024_missing_not_zero",
            pd.isna(cash.loc[2024, "airport_cash_million_rubles"])
            and "За 2024 год касса отсутствует" in blocks["cash"],
        )
        queue = frames["queue"]
        queue = queue[queue.code_region == "ALL_CANDIDATE_CODE_REGIONS"]
        check(
            "cash_amounts_and_other_year_ratio_in_prose",
            all(
                str(v) in blocks["cash"]
                for v in [
                    cash.loc[2022, "airport_cash_million_rubles"],
                    *cash.municipal_investment_million_rubles.tolist(),
                ]
            )
            and f"{cash.loc[2023, 'cash_divided_by_municipal_investment']:.3f}" in blocks["cash"],
        )
        check(
            "queue_percentages_in_prose",
            all(f"{v:.3f}" in blocks["queue"] for v in queue.share_all_sme_year_rows_pct),
        )
        check(
            "queue_counts_and_denominator",
            queue.source_rows_once.tolist() == [38262, 36724]
            and all(str(n) in blocks["queue"] for n in [38262, 36724, 781, 823])
            and "не доля уникальных предприятий" in blocks["queue"],
        )
        check(
            "no_historical_sme_join",
            queue.historical_assignments_approved.eq(0).all()
            and "по-прежнему 0" in blocks["queue"]
            and "за 2025 год не используются в 2023–2024" in blocks["queue"],
        )
        ratios = frames["ratios"].set_index("variable")
        check(
            "ratio_readout_and_nonadditivity",
            all(
                token in blocks["source_percentages_readout"]
                for token in [
                    "все 33 пары",
                    "10 из 33",
                    "31 из 33",
                    "нельзя приписывать только",
                    "различается также способ группировки деятельности",
                ]
            ),
        )
        admission = json.loads((root / "admission/QA.json").read_text("utf8"))
        check(
            "admission_counts_from_qa",
            admission["rows"] == 276
            and admission["old_missing_values"] == 424
            and admission["missing_cells_filled"] == 0
            and admission["direct_numeric_same_construct_reconciliations"] == 0,
        )
        check(
            "ratio_count_evidence",
            ratios.rounding_compatible.to_dict()
            == {
                "shipments_economic_activity": 33,
                "shipments_pure_activity": 10,
                "resale_non_own_goods": 31,
            }
            and ratios.pairs.eq(33).all(),
        )
        check(
            "admission_construct_limits",
            "не подставлено в 424" in blocks["admission"]
            and "причина требует источника" in blocks["admission"]
            and "276" in blocks["admission"],
        )
        check(
            "limits_no_causal_or_new_independent_test",
            "не новый независимый тест" in blocks["limits"]
            and "причинные эффекты не заявляются" in blocks["limits"]
            and "2000" in blocks["limits"]
            and "пространственную зависимость и множественность" in blocks["limits"],
        )
        source_rows = artifact["sources"]
        check("bootstrap_count_protocol", protocol["bootstrap_draws"] == 2000)
        check("source_copies_equal", source_rows == artifact["manifest"]["sources"])
        check("source_ids_unique", len({s["id"] for s in source_rows}) == len(source_rows))
        dataset_sources = {
            "coverage": "coverage",
            "primary_correlations": "regions",
            "level_coefficients": "regions",
            "urban_comparison": "regions",
            "cluster_alignment": "clusters",
            "saratov_scatter": "panel",
            "airport_money": "cash",
            "queue": "queue",
            "source_percentages": "admission",
        }
        for definition in definitions + artifact["manifest"]["charts"]:
            check(
                f"dataset_source:{definition['id']}",
                definition["sourceId"] == dataset_sources[definition["dataset"]],
            )
        for source in source_rows:
            path = source["path"]
            check(
                f"source_exists:{source['id']}",
                (repo / path).is_file() and source["href"].endswith("/blob/main/" + path),
            )
            check(
                f"source_code:{source['id']}", (repo / source["query"]["original_python"]).is_file()
            )
            check(
                f"source_tables:{source['id']}",
                path in source["query"]["tables_used"]
                and all((repo / p).is_file() for p in source["query"]["tables_used"]),
            )
        design = json.loads((report_dir / "report_design.json").read_text("utf8"))
        for name, digest in design["inputs"].items():
            check(f"design_source_sha:{name}", sha(root / name) == digest)
        check(
            "design_builder_sha",
            design["builder_sha256"] == sha(repo / "scripts/build_round6_report.py"),
        )
        record["source_sha256"] = {relative: sha(root / relative) for relative in PATHS.values()}
        record["source_sha256"]["admission/QA.json"] = sha(root / "admission/QA.json")
        record["source_sha256"]["repo/configs/round6_analysis_v1.json"] = sha(
            repo / "configs/round6_analysis_v1.json"
        )
        record["source_sha256"][
            "repo/reports/enriched_analysis/annual_networks/yearly_support.csv"
        ] = sha(support_path)
        record["sources"] = {
            (
                relative.removeprefix("repo/")
                if relative.startswith("repo/")
                else "reports/round6_analysis/" + relative
            ): digest
            for relative, digest in record["source_sha256"].items()
        }
        record["status"] = "PASS"
    except Exception as exc:
        record["status"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve(), args.root.resolve(), args.report.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "error": result.get("error"),
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
