"""Reconcile the final fourth-round readout with frozen OOF, without model refits."""

import hashlib
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/round4_analysis"
KEY = ["territory_id", "year"]
TARGETS = ["log_total"] + [
    "clr_p_" + name for name in ["food", "health", "marketplaces", "foodservice", "transport"]
]
NEW = [name + "_log_rubles_per_person" for name in ["investment", "shipments", "retail"]]
REGIME = {"Внутри регионов": "within_region", "Целые регионы": "heldout_regions"}
SAMPLE = {
    "Основная": "core",
    "Без смены кода": "stable_code",
    "Прежние полные 3": "common_complete_new3",
    "Новые полные 3": "expanded_complete_new3",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(frame, **conditions):
    result = frame
    for name, value in conditions.items():
        result = result.loc[result[name].eq(value)]
    return result


def tables(text):
    result, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            if not re.fullmatch(r"[| :\-]+", line):
                current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            result.append(current)
            current = []
    return result


def main():
    record = {"status": "RUNNING", "model_refits": 0, "checks": []}

    def check(name, condition, **details):
        record["checks"].append(
            {"check": name, "status": "PASS" if condition else "FAIL", **details}
        )
        if not condition:
            raise AssertionError(name)

    def close(values, expected, atol=1e-10):
        return bool(
            np.allclose(
                np.array(values, float),
                np.array(expected, float),
                rtol=0,
                atol=atol,
                equal_nan=True,
            )
        )

    try:
        model = REPORT / "models"
        panel = pd.read_csv(model / "model_panel.csv")
        oof = duckdb.read_parquet(str(model / "oof_predictions.parquet")).df()
        comparisons = pd.read_csv(model / "cv_comparisons.csv")
        metrics = pd.read_csv(model / "cv_metrics.csv")
        contributions = pd.read_csv(
            REPORT / "readout/state_update_contributions.csv", dtype={"group": str}
        )
        keys = ["year", "regime", "target", "slice", "group"]
        check("contribution_unique_keys", not contributions.duplicated(keys).any())
        expected_keys, worst = set(), 0.0
        for year in [2023, 2024]:
            data = panel.loc[panel.year.eq(year) & panel.core_eligible].copy()
            old_count = data[["old_" + c for c in NEW]].notna().sum(axis=1)
            new_count = data[NEW].notna().sum(axis=1)
            data["recovered_any"] = new_count.gt(old_count)
            data["completeness_stratum"] = "still_incomplete"
            data.loc[new_count.eq(3) & old_count.lt(3), "completeness_stratum"] = "newly_complete"
            data.loc[new_count.eq(3) & old_count.eq(3), "completeness_stratum"] = "common_complete"
            for regime in ["within_region", "heldout_regions"]:
                source = subset(oof, year=year, sample="core", seed=13, regime=regime)
                for target, fields in [("log_total", TARGETS[:1]), ("profile_clr", TARGETS[1:])]:
                    joined = data.copy()
                    for arm in ["legacy_full", "recovered_full"]:
                        rows = subset(source, model=arm)
                        rows = rows.set_index(KEY).loc[pd.MultiIndex.from_frame(data[KEY])]
                        check(
                            "exact_oof_alignment",
                            len(rows) == len(data) and not rows.index.duplicated().any(),
                        )
                        actual = rows[["actual_" + c for c in fields]].to_numpy(float)
                        prediction = rows[["predicted_" + c for c in fields]].to_numpy(float)
                        joined[arm] = np.einsum(
                            "ij,ij->i", actual - prediction, actual - prediction
                        )
                    total = float(joined.legacy_full.sum())
                    expected_total = 1 - joined.recovered_full.sum() / total
                    claim = subset(
                        comparisons,
                        year=year,
                        sample="core",
                        seed=13,
                        regime=regime,
                        target=target,
                        comparison="state_update",
                    ).iloc[0]
                    check("state_update_total", close(expected_total, claim.relative_mse_reduction))
                    for scope in ["region_code", "recovered_any", "completeness_stratum"]:
                        used = []
                        for value, group in joined.groupby(scope):
                            k = (year, regime, target, scope, str(value))
                            expected_keys.add(k)
                            saved = subset(contributions, **dict(zip(keys, k, strict=True)))
                            check(
                                "contribution_group_denominators",
                                len(saved) == 1
                                and saved.iloc[0].n == len(group)
                                and saved.iloc[0].regions == group.region_code.nunique(),
                            )
                            old, new = (
                                float(group.legacy_full.sum()),
                                float(group.recovered_full.sum()),
                            )
                            expected = [
                                old,
                                new,
                                new - old,
                                total,
                                (old - new) / old,
                                (old - new) / total,
                            ]
                            numeric = [
                                "legacy_sse",
                                "recovered_sse",
                                "sse_change",
                                "legacy_total_sse",
                                "local_relative_reduction",
                                "contribution_to_total_relative_reduction",
                            ]
                            got = saved[numeric].iloc[0].to_numpy(float)
                            worst = max(worst, float(np.max(np.abs(got - expected))))
                            check("contribution_arithmetic", close(got, expected))
                            used.append(saved.iloc[0])
                        used = pd.DataFrame(used)
                        check(
                            "partition_complete_and_additive",
                            used.n.sum() == len(data)
                            and close(used.legacy_sse.sum(), total)
                            and close(used.recovered_sse.sum(), joined.recovered_full.sum())
                            and close(
                                used.contribution_to_total_relative_reduction.sum(), expected_total
                            ),
                        )
        check(
            "contribution_complete_grid",
            expected_keys == set(contributions[keys].itertuples(index=False, name=None)),
        )
        headline = pd.read_csv(REPORT / "headline_metrics.csv")
        check(
            "headline_four_unique_rows",
            len(headline) == 4 and not headline.duplicated(["year", "regime"]).any(),
        )
        for row in headline.to_dict("records"):
            context = dict(
                year=row["year"],
                sample="core",
                seed=13,
                regime=REGIME[row["regime"]],
                target="profile_clr",
            )
            compare = subset(comparisons, **context, comparison="primary").iloc[0]
            check(
                "headline_rounded_effect_ci",
                close(
                    [row[c] for c in ["gain", "low", "high"]],
                    (
                        compare[["relative_mse_reduction", "ci_low", "ci_high"]].astype(float) * 100
                    ).round(3),
                    atol=1e-12,
                )
                and row["n"] == compare.n
                and row["regions"] == compare.regions,
            )
            for field, arm in [("mse_base", "new_flags"), ("mse_full", "recovered_full")]:
                mse = subset(metrics, **context, model=arm).iloc[0].mse_oof
                check("headline_rounded_mse", close(row[field], round(mse, 6), atol=1e-12))
        prose = (REPORT / "REPORT.md").read_text(encoding="utf-8")
        blocks = tables(prose)
        check("report_table_count", len(blocks) == 9)

        def verify_effect(row, context, effect_index, interval_index, comparison="primary"):
            evidence = subset(comparisons, **context, comparison=comparison).iloc[0]
            interval = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", row[interval_index])]
            check(
                "report_table_effect_ci",
                close(
                    float(row[effect_index]),
                    round(evidence.relative_mse_reduction * 100, 3),
                    atol=1e-12,
                )
                and close(
                    interval,
                    np.round([evidence.ci_low * 100, evidence.ci_high * 100], 3),
                    atol=1e-12,
                ),
            )
            return evidence

        for row in blocks[0][1:]:
            data = panel.loc[panel.year.eq(int(row[0])) & panel.core_eligible]
            fields = ["old_" + c for c in NEW] if row[1] == "До восстановления" else NEW
            complete = int(data[fields].notna().all(axis=1).sum())
            check(
                "report_coverage_table",
                int(row[2]) == complete
                and int(row[3]) == len(data)
                and close(row[4], round(complete / len(data) * 100, 2), atol=1e-12),
            )
        for row in blocks[1][1:]:
            ctx = dict(
                year=int(row[0]),
                regime=REGIME[row[1]],
                sample="core",
                seed=13,
                target="profile_clr",
            )
            evidence = verify_effect(row, ctx, 3, 4)
            check("report_primary_sample_n", int(row[2]) == evidence.n)
            for pos, arm in [(5, "new_flags"), (6, "recovered_full")]:
                expected = subset(metrics, **ctx, model=arm).iloc[0].mse_oof
                check("report_primary_mse", close(row[pos], round(expected, 6), atol=1e-12))
        for row in blocks[2][1:]:
            ctx = dict(
                year=int(row[0]),
                sample=SAMPLE[row[1]],
                regime=REGIME[row[2]],
                seed=13,
                target="profile_clr",
            )
            evidence = verify_effect(row, ctx, 4, 5)
            check("report_sensitivity_n", int(row[3]) == evidence.n)
        for row in blocks[3][1:]:
            ctx = dict(
                year=int(row[0]),
                sample="core",
                regime="within_region",
                seed=int(row[1]),
                target="profile_clr",
            )
            verify_effect(row, ctx, 2, 3)
        for row in blocks[4][1:]:
            ctx = dict(
                year=int(row[0]),
                sample="core",
                regime=REGIME[row[1]],
                seed=13,
                target="profile_clr",
            )
            comparison = (
                "state_update" if row[2] == "Полное обновление версии" else "shared_flags_recovery"
            )
            verify_effect(row, ctx, 3, 4, comparison)
        group_names = {
            "Полные прежде": "common_complete",
            "Стали полными": "newly_complete",
            "Остались неполными": "still_incomplete",
        }
        for row in blocks[5][1:]:
            evidence = subset(
                contributions,
                year=int(row[0]),
                regime="heldout_regions",
                target="profile_clr",
                slice="completeness_stratum",
                group=group_names[row[1]],
            ).iloc[0]
            check(
                "report_contribution_table",
                int(row[2]) == evidence.n
                and close(
                    row[3:5],
                    np.round(
                        [
                            evidence.local_relative_reduction * 100,
                            evidence.contribution_to_total_relative_reduction * 100,
                        ],
                        3,
                    ),
                    atol=1e-12,
                ),
            )
        control = {
            "Исходные числа": "primary",
            "Перестановка 1": "permuted_91013",
            "Перестановка 2": "permuted_91029",
            "Перестановка 3": "permuted_91047",
        }
        for row in blocks[6][1:]:
            ctx = dict(
                year=int(row[0]),
                sample="core",
                regime="heldout_regions",
                seed=13,
                target="log_total",
            )
            verify_effect(row, ctx, 2, 3, control[row[1]])
        investments = pd.read_csv(REPORT / "context/investment_paths.csv")
        entities = pd.read_csv(REPORT / "context/sme_entity_eligibility.csv")
        for row in blocks[7][1:]:
            for position, identifier in [(1, 318), (2, 331)]:
                value = subset(
                    investments, year=int(row[0]), territory_id_context_candidate=identifier
                ).value_million_rubles.item()
                check("report_investment_table", close(row[position], value, atol=1e-12))
        for row in blocks[8][1:]:
            source = subset(entities, year=int(row[0]), is_sole_trader=int(row[1] == "ИП")).iloc[0]
            check(
                "report_sme_table",
                close(
                    row[2:6],
                    [
                        source.tax_ids,
                        source.tax_ids_valid_known_workers,
                        round(source.tax_ids_valid_known_workers_pct_of_tax_ids, 2),
                        source.record_candidate_known_workers,
                    ],
                    atol=1e-12,
                ),
            )
        primary = subset(comparisons, seed=13, comparison="primary", target="profile_clr")
        check(
            "all_16_primary_intervals_cross_zero",
            len(primary) == 16 and primary.ci_low.le(0).all() and primary.ci_high.ge(0).all(),
        )
        check("sample_count_prose_corrected", "подвыборках все 16" not in prose)
        recovered = pd.read_csv(REPORT / "audit/new_numeric_cells.csv")
        check(
            "coverage_narrative_counts",
            len(recovered) == 779
            and len(recovered[KEY].drop_duplicates()) == 445
            and recovered.territory_id.nunique() == 223
            and recovered.region_code.nunique() == 11,
        )
        gain = panel[NEW].notna().sum(axis=1) - panel[["old_" + c for c in NEW]].notna().sum(axis=1)
        unscalable = recovered.loc[
            recovered.population_total.isna() | recovered.population_total.le(0)
        ]
        check(
            "nine_unscalable_outside_core",
            int(gain.sum()) == 770 and len(unscalable) == 9 and not unscalable.core_eligible.any(),
        )
        concentrated = recovered.region_name.isin(["Тверская область", "Самарская область"]).sum()
        check(
            "recovery_concentration",
            concentrated == 432 and round(concentrated / len(recovered) * 100, 2) == 55.46,
        )
        lower = subset(investments, territory_id_context_candidate=331).set_index("year")
        vals = lower.value_million_rubles
        check(
            "lower_kolyma_narrative",
            round(vals[2023] / vals[2022], 2) == 41.81
            and round((1 - vals[2024] / vals[2021]) * 100, 2) == 4.25
            and round((1 - vals[2024] / vals[2022]) * 100, 2) == 22.01,
        )
        upper = subset(investments, territory_id_context_candidate=318).sort_values("year")
        check(
            "upper_kolyma_monotonic_published_nominal",
            upper.value_million_rubles.diff().dropna().gt(0).all(),
        )
        observed = pd.read_csv(
            REPORT / "context/sme_worker_observation_by_primary_nace2.csv",
            dtype={"primary_nace2_code": str},
        )
        for code, percent in [("52", 51.40), ("86", 94.96)]:
            part = subset(observed, year=2024, primary_nace2_code=code).iloc[0]
            check(
                "sme_sector_narrative",
                round(part.tax_ids_valid_known_workers / part.legal_entity_tax_ids * 100, 2)
                == percent,
            )
        for year, inflation, coverage in [(2023, 7.80, 79.28), (2024, 6.27, 80.54)]:
            part = subset(entities, year=year, is_sole_trader=0).iloc[0]
            check(
                "sme_unique_id_narrative",
                round((part.source_rows / part.tax_ids - 1) * 100, 2) == inflation
                and round(part.tax_ids_valid_known_workers / part.tax_ids * 100, 2) == coverage,
            )
        controls = subset(
            comparisons, sample="core", regime="heldout_regions", seed=13, target="log_total"
        )
        for year, limits in [(2023, [2.696, 3.896]), (2024, [1.995, 2.894])]:
            part = controls.loc[
                controls.year.eq(year) & controls.comparison.str.startswith("permuted_")
            ]
            check(
                "secondary_permutation_range",
                len(part) == 3
                and close(
                    (part.relative_mse_reduction.agg(["min", "max"]) * 100).round(3),
                    limits,
                    atol=1e-12,
                ),
            )
        ablations = subset(comparisons, sample="core", seed=13, target="log_total")
        for year in [2023, 2024]:
            for regime in ["within_region", "heldout_regions"]:
                part = subset(ablations, year=year, regime=regime)
                part = part.loc[part.comparison.str.startswith("without_")]
                check(
                    "retail_largest_conditional_level_signal",
                    part.loc[part.relative_mse_reduction.idxmax(), "comparison"]
                    == "without_retail",
                )
        record.update(
            contribution_rows=len(contributions),
            partitions=24,
            maximum_contribution_absolute_error=worst,
            report_tables_checked=9,
            report_sha256=sha(REPORT / "REPORT.md"),
            headline_sha256=sha(REPORT / "headline_metrics.csv"),
            contribution_sha256=sha(REPORT / "readout/state_update_contributions.csv"),
            verifier_sha256=sha(Path(__file__)),
            scope="Independent OOF loss arithmetic and report-table reconciliation; no refits",
            status="PASS",
        )
    except Exception as exc:
        record.update(status="FAIL", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (REPORT / "report_claims_QA.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in record.items() if k != "checks"}))


if __name__ == "__main__":
    main()
