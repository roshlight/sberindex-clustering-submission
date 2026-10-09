"""Offline descriptive context for the six previously selected residual cases.

No fitting, new geographical joins, imputation, deflation or source replacement.
Run with --repo and --out; --self-test exercises missing/zero/attribution guards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd

IDS = {"318", "331", "883", "925", "1180", "2527"}
INDICATORS = {
    "8109001": "Инвестиции организаций в основной капитал",
    "8401011": "Отгрузка собственной продукции, работ и услуг",
    "8401003": "Оборот розничной торговли",
}
INPUTS = {
    "economy": "reports/data_round3/economy/annual_candidate_panel.csv",
    "source_cells": "reports/data_round3/economy/source_cells.csv",
    "economy_qa": "reports/data_round3/economy/independent_cell_QA.json",
    "core": "reports/enriched_analysis/core_v2/annual_panel.csv",
    "examples": "reports/enriched_analysis/examples/six_oof_examples_2024.csv",
    "sectors": "reports/round2_analysis/sector/examples.csv",
    "evidence": "reports/data_round3/cases/factual_evidence.csv",
    "source_roles": "reports/data_round3/cases/source_roles.csv",
    "geography": "reports/data_round3/geography/QA.json",
    "sme": "reports/data_round3/sme/vologda_small_organizations_unlinked.csv",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(repo: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(repo / INPUTS[key], dtype=str, keep_default_na=False)


def decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value.quantize(Decimal("0.000000000001")), "f").rstrip("0").rstrip(".") or "0"


def nominal_change(old: str, new: str, ready: bool) -> tuple[str, str, str]:
    if not ready or not old or not new:
        return "", "", "not_computed_missing_or_not_comparable"
    a, b = Decimal(old), Decimal(new)
    if not a.is_finite() or not b.is_finite() or a < 0 or b < 0:
        raise ValueError("Invalid nonnegative published nominal values")
    delta = b - a
    if a == 0:
        return decimal_text(delta), "", "zero_baseline_percentage_undefined"
    with localcontext() as ctx:
        ctx.prec = 50
        growth = (b / a - 1) * 100
    return decimal_text(delta), decimal_text(growth), "conditional_nominal_same_source_key"


def source_pair_ready(a: dict, b: dict) -> bool:
    identity = [
        "territory_id",
        "indicator_id",
        "source_region_url",
        "source_oktmo",
        "official_oktmo_8",
    ]
    measurement = ["unit", "period_code", "activity_code", "source_section"]
    return (
        a["year"] == "2023"
        and b["year"] == "2024"
        and all(a[c] and a[c] == b[c] for c in identity + measurement)
        and a["value_status"] == b["value_status"] == "observed"
        and a["temporal_review_required"] == b["temporal_review_required"] == "False"
    )


def self_test() -> None:
    assert nominal_change("", "25", True)[0:2] == ("", "")
    assert nominal_change("0", "25", True) == ("25", "", "zero_baseline_percentage_undefined")
    assert nominal_change("100", "0", True)[1] == "-100"
    assert nominal_change("100", "200", False)[1] == ""
    a = dict.fromkeys(
        [
            "territory_id",
            "indicator_id",
            "source_region_url",
            "source_oktmo",
            "official_oktmo_8",
            "unit",
            "period_code",
            "activity_code",
            "source_section",
        ],
        "x",
    )
    a.update(year="2023", value_status="observed", temporal_review_required="False")
    b = {**a, "year": "2024"}
    assert source_pair_ready(a, b)
    assert not source_pair_ready(a, {**b, "source_oktmo": "other_code"})
    assert not source_pair_ready(a, {**b, "period_code": "quarter"})
    assert not source_pair_ready(a, {**b, "temporal_review_required": "True"})


def build(repo: Path, out: Path) -> dict:
    # The output can only be the dedicated context tree or a new external scratch path.
    frozen = [
        repo / "reports/data_round3",
        repo / "reports/enriched_analysis",
        repo / "reports/round2_analysis",
    ]
    if any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents for p in frozen):
        raise ValueError("Refusing to write into frozen input trees")
    out.mkdir(parents=True, exist_ok=True)
    examples = read(repo, "examples")
    assert len(examples) == 6 and set(examples.territory_id) == IDS
    assert set(examples.year) == {"2024"}
    core = read(repo, "core")
    assert not core.duplicated(["territory_id", "year"]).any()
    core = core[core.territory_id.isin(IDS)].set_index(["territory_id", "year"])
    assert len(core) == 12
    panel = read(repo, "economy")
    assert (
        len(panel) == 12096 and not panel.duplicated(["territory_id", "year", "indicator_id"]).any()
    )
    qa = json.loads((repo / INPUTS["economy_qa"]).read_text(encoding="utf-8"))
    assert qa["status"] == "PASS" and qa["csv_sha256"] == digest(repo / INPUTS["economy"])
    panel = panel[panel.territory_id.isin(IDS)].copy()
    assert len(panel) == 36
    cells = read(repo, "source_cells")
    keys = ["source_region_url", "source_oktmo", "year", "indicator_id"]
    assert not cells.duplicated(keys).any()
    sources = cells.set_index(keys)
    for row in panel.to_dict("records"):
        c = core.loc[(row["territory_id"], row["year"])]
        assert row["official_oktmo_8"] == c.official_oktmo_8
        assert row["code_reference_date"] == row["year"] + "-01-01"
        assert row["geographic_equivalence_proven"] == ""
        if row["value_status"] == "observed":
            src = sources.loc[tuple(row[k] for k in keys)]
            assert Decimal(src.value) == Decimal(row["value"])
            assert Decimal(src.source_csv_row) == Decimal(row["source_csv_row"])
            for field in [
                "source_sha256",
                "request_key",
                "source_oktmo_name",
                "source_section",
                "period_code",
                "activity_code",
                "unit",
                "value_raw",
                "value_status",
            ]:
                assert row[field] == src[field]
            assert row["unit"] == "thousand_rubles" and row["activity_code"] == "101"
            assert row["period_code"] == ("17" if row["indicator_id"] == "8109001" else "79")
            assert "без субъектов малого предпринимательства" in row["source_section"]
        else:
            assert not row["value"]

    pairs = []
    for (tid, indicator), group in panel.groupby(["territory_id", "indicator_id"], sort=True):
        by_year = group.set_index("year").to_dict("index")
        assert set(by_year) == {"2023", "2024"}
        a, b = ({**by_year[y], "year": y} for y in ["2023", "2024"])
        old, new = (core.loc[(tid, y)] for y in ["2023", "2024"])
        flags_clear = all(
            x[f] == "False" for x in (old, new) for f in ["calendar_review", "code_type_changed"]
        )
        eligible = source_pair_ready(a, b) and flags_clear
        delta, growth, status = nominal_change(a["value"], b["value"], eligible)
        row = {
            "territory_id": tid,
            "municipality": new.municipal_district_name,
            "region": new.region_name,
            "indicator_id": indicator,
            "indicator": INDICATORS[indicator],
            "value_2023": a["value"],
            "value_2024": b["value"],
            "unit": "thousand_rubles",
            "nominal_change_thousand_rubles": delta,
            "nominal_growth_pct": growth,
            "comparison_status": status,
            "same_annual_official_code": a["official_oktmo_8"] == b["official_oktmo_8"],
            "same_source_measurement_definition": all(
                a[k] and a[k] == b[k]
                for k in ["unit", "period_code", "activity_code", "source_section"]
            ),
            "frozen_calendar_and_code_flags_clear": flags_clear,
            "core_eligible_both_years": all(x.core_eligible == "True" for x in (old, new)),
            "conservative_eligible_both_years": all(
                x.conservative_eligible == "True" for x in (old, new)
            ),
            "boundary_equivalence": "unknown",
            "respondent_circle_stable_independently_verified": False,
            "small_business_excluded": True,
            "price_basis": "nominal_not_deflated",
            "availability": "retrospective_2026_snapshot_first_release_and_revisions_unknown",
            "automatic_mask_change": False,
        }
        for year, source in [("2023", a), ("2024", b)]:
            for field in [
                "value_raw",
                "value_status",
                "source_oktmo",
                "official_oktmo_8",
                "source_oktmo_name",
                "source_region_url",
                "source_section",
                "source_sha256",
                "source_csv_row",
                "source_url",
                "request_key",
                "retrieved_at",
            ]:
                row[f"{field}_{year}"] = source[field]
        pairs.append(row)
    pair_frame = pd.DataFrame(pairs)
    assert len(pair_frame) == 18
    pair_frame.to_csv(out / "six_cases_economy_pairs.csv", index=False, encoding="utf-8-sig")

    sectors = read(repo, "sectors")
    assert len(sectors) == 12 and set(sectors.territory_id) == IDS
    sector_rows = []
    for tid, group in sectors.groupby("territory_id"):
        g = group.set_index("year")
        a, b = g.loc["2023"], g.loc["2024"]
        assert a.max_among_observed_sections == b.max_among_observed_sections
        for y in ["2023", "2024"]:
            assert Decimal(g.loc[y].workers_total_source) == Decimal(core.loc[(tid, y)].workers)
        sector_rows.append(
            {
                "territory_id": tid,
                "largest_disclosed_section_both_years": a.max_among_observed_sections,
                "workers_in_section_2023": a.max_observed_workers,
                "workers_in_section_2024": b.max_observed_workers,
                "workers_total_2023": a.workers_total_source,
                "workers_total_2024": b.workers_total_source,
                "section_share_2023": a.max_observed_share_of_official_total,
                "section_share_2024": b.max_observed_share_of_official_total,
                "scope": "largest_among_disclosed_not_all_employment; without_small_business",
                "source_sha256_2023": a.max_source_sha256,
                "source_sha256_2024": b.max_source_sha256,
                "source_csv_row_2023": a.max_source_csv_row,
                "source_csv_row_2024": b.max_source_csv_row,
            }
        )
    pd.DataFrame(sector_rows).to_csv(
        out / "prior_sector_context.csv", index=False, encoding="utf-8-sig"
    )

    evidence = read(repo, "evidence")
    assert len(evidence) == 53 and not evidence.evidence_id.duplicated().any()
    assert set(evidence.territory_id) == {"883", "1180", "2527"}
    # All 53 records keep their existing admissibility, including the two quarantined claims.
    evidence.to_csv(out / "document_evidence_53.csv", index=False, encoding="utf-8-sig")
    ev = evidence.set_index("evidence_id")
    checks = []
    for year in ["2023", "2024"]:
        e = ev.loc["s" + year[-2:] + "_wage"]
        old_value = core.loc[("1180", year)].wage
        difference = Decimal(e.value) - Decimal(old_value)
        checks.append(
            {
                "territory_id": "1180",
                "topic": "wage_" + year,
                "evidence_ids": e.name,
                "document_value": e.value,
                "frozen_value": old_value,
                "unit": "rubles_per_month",
                "difference": decimal_text(difference),
                "status": "exact_published_agreement_not_independent_measurement"
                if difference == 0
                else "unresolved_discrepancy_no_replacement",
                "explanation": "Revision or respondent-circle cause is not established.",
            }
        )
        e = ev.loc["sstat_workers_" + year]
        old_value = core.loc[("1180", year)].workers
        assert Decimal(e.value) == Decimal(old_value)
        checks.append(
            {
                "territory_id": "1180",
                "topic": "workers_" + year,
                "evidence_ids": e.name,
                "document_value": e.value,
                "frozen_value": old_value,
                "unit": "persons",
                "difference": "0",
                "status": "exact_published_agreement_not_independent_measurement",
                "explanation": "Local 5.4/6.4 thousand are retained as rounded published claims.",
            }
        )
    for topic, ids, status, explanation in [
        (
            "amur_december_vs_annual",
            "amur_dec_2023;amur_dec_2024",
            "different_time_grain_no_growth_series_splice",
            "December 6286/11479 do not replace annual means 6285/8927; "
            "revisions across releases are unresolved.",
        ),
        (
            "amur_pdf_suspected_shift",
            "amur_annual_quarantined_workers;amur_annual_quarantined_wage",
            "published_claim_quarantined",
            "1008 and 60964.0 are printed on the source row; "
            "no adjacent-row reassignment or repair.",
        ),
        (
            "svobodnensky_2023_wage",
            "sv23_wage",
            "exact_published_agreement_not_independent_measurement",
            "186125.9 matches the frozen value; retrospective strategy published in 2025, "
            "sector employer coverage unresolved.",
        ),
        (
            "kurkino_institution_counts",
            "k24_admin_staff;k24_zhil_total;k24_zhil_main;k24_zhil_aux;k24_zhil_aup",
            "institution_context_not_municipal_employment",
            "8 staffing positions at 2024-12-31 and 336 institution employees at an "
            "unspecified snapshot cannot explain total annual workers 3812 to 3349.",
        ),
        (
            "svetlovsky_manufacturing_shipments",
            "s23_ship;s24_ship",
            "different_activity_coverage_not_total101_replacement",
            "159344.5/169393 million rubles refer to manufacturing; "
            "new BDPMO total101 is broader and expressed in thousand rubles.",
        ),
        (
            "svetlovsky_sme_denominators",
            "s23_sme_worker_share;s23_sme_trade;s23_sme_manufacturing",
            "different_denominators_no_worker_share_join",
            "39% concerns workers; 36% and 11% concern subjects/organizations, "
            "not sector employment.",
        ),
    ]:
        selected = ev.loc[ids.split(";")]
        assert selected.territory_id.nunique() == 1
        checks.append(
            {
                "territory_id": selected.territory_id.iloc[0],
                "topic": topic,
                "evidence_ids": ids,
                "document_value": "",
                "frozen_value": "",
                "unit": "see_evidence",
                "difference": "",
                "status": status,
                "explanation": explanation,
            }
        )
    assert Decimal(ev.loc["sv23_wage"].value) == Decimal(core.loc[("883", "2023")].wage)
    assert sum(
        Decimal(ev.loc[k].value) for k in ["k24_zhil_main", "k24_zhil_aux", "k24_zhil_aup"]
    ) == Decimal(ev.loc["k24_zhil_total"].value)
    for row in checks:
        selected = ev.loc[row["evidence_ids"].split(";")]
        row["source_urls"] = json.dumps(sorted(set(selected.source_url)), ensure_ascii=False)
        row["source_sha256s"] = json.dumps(sorted(set(selected.source_sha256)))
        row["evidence_status_not_upgraded"] = True
    pd.DataFrame(checks).to_csv(
        out / "document_comparability_checks.csv", index=False, encoding="utf-8-sig"
    )
    roles = read(repo, "source_roles")
    assert set(roles.territory_id) == {"883", "1180", "2527"}
    roles.to_csv(out / "document_source_roles.csv", index=False, encoding="utf-8-sig")
    geo = json.loads((repo / INPUTS["geography"]).read_text(encoding="utf-8"))
    assert geo["exact_recoding_pairs"] == 131 and geo["boundary_equivalence_proven"] == 0
    assert geo["mask_changes"] == 0 and geo["city_2023_temporal_review_retained"]
    sme = read(repo, "sme")
    assert len(sme) == 769 and sme.territory_id.eq("").all() and sme.official_oktmo_8.eq("").all()
    assert sme.boundary_equivalence_confirmed.eq("").all()
    summary = {
        "status": "PASS_WITH_CONDITIONS",
        "selected_prior_cases": 6,
        "case_years": 12,
        "nominal_indicator_cells": len(panel),
        "observed_cells": int(panel.value_status.eq("observed").sum()),
        "unavailable_cells": int(panel.value_status.ne("observed").sum()),
        "nominal_pairs": len(pair_frame),
        "growth_pairs_computed": int(pair_frame.nominal_growth_pct.ne("").sum()),
        "growth_pairs_missing": int(pair_frame.nominal_growth_pct.eq("").sum()),
        "exact_source_cells_reconciled": int(panel.value_status.eq("observed").sum()),
        "document_records_status_preserved": len(evidence),
        "quarantined_records": int(evidence.admissibility.eq("published_claim_quarantined").sum()),
        "undated_institution_records": int(
            evidence.admissibility.eq("institution_context_undated_snapshot").sum()
        ),
        "document_comparability_checks": len(checks),
        "sme_name_joins": 0,
        "boundary_equivalence_claims": 0,
        "models_fit": 0,
        "new_searches": 0,
        "source_replacements": 0,
        "self_test_guards": 8,
        "limitations": [
            "Nominal retrospective organization activity, without small business; no deflation.",
            "Six old residual extremes are not an independent test or causal explanation.",
            "Same codes/definitions do not establish immutable borders or respondent populations.",
            "Document evidence does not replace frozen values or upgrade quarantined candidates.",
        ],
    }
    (out / "QA.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "script_sha256": digest(Path(__file__)),
        "inputs": [{"path": p, "sha256": digest(repo / p)} for p in INPUTS.values()],
        "outputs": [{"path": p.name, "sha256": digest(p)} for p in sorted(out.glob("*.csv"))]
        + [{"path": "QA.json", "sha256": digest(out / "QA.json")}],
        "no_frozen_sources_or_masks_changed": True,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test and args.out is None:
        print("PASS: eight missing/zero/code/period/calendar guards")
        return
    if args.out is None:
        parser.error("--out is required for a report build")
    print(json.dumps(build(args.repo, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
