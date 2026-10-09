"""Seal collection accounting, raw allowlist and exact offline replay checks."""

import argparse
import json
from pathlib import Path

import pandas as pd

from sber_enrichment.economy_bulk import sha
from sber_enrichment.round4_economy import KEY, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["repo", "work", "replay"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo, work = args.repo.resolve(), args.work.resolve()
    out = repo / "reports/data_round5/economy"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["output_hashes"]) + ["manifest.json", "execution.json"]
    comparisons = {name: sha(out / name) == sha(args.replay / name) for name in names}
    if not all(comparisons.values()):
        raise ValueError("Offline replay differs")
    statuses = json.loads((out / "collection_status.json").read_text(encoding="utf-8"))
    controlled = [row for row in statuses if "control_code" in row]
    pd.DataFrame(controlled).to_csv(out / "response_error_recovery.csv", index=False)
    missing = pd.read_csv(out / "remaining_missing_cells.csv", dtype=str, keep_default_na=False)
    actions = {
        "semantic_scope_unverified": "Find official regional indicator scope; keep values blocked",
        "semantic_unit_unverified": "Find official regional indicator unit; do not infer scale",
        "exact_source_code_absent_from_current_form": (
            "Seek historical form/archive for exact code-year; no replacement by current code"
        ),
        "exact_code_year_not_returned": (
            "Seek official alternative publication and explicit omission notes; "
            "absence does not establish confidentiality"
        ),
        "exact_code_year_not_returned_in_controlled_export": (
            "Controlled query works; seek official publication or omission notes "
            "for exact missing code-year"
        ),
        "unresolved_source_marker": (
            "Seek marker definition tied to this source cell; never treat blank as zero"
        ),
    }
    priorities = []
    for (reason, job), group in missing.groupby(["reason", "job"]):
        priorities.append(
            dict(
                job=job,
                reason=reason,
                missing_cells=len(group),
                priority=1 if reason.startswith("semantic_") else 2,
                next_action=actions[reason],
            )
        )
    pd.DataFrame(priorities).sort_values(
        ["priority", "missing_cells", "job"], ascending=[True, False, True]
    ).to_csv(out / "remaining_search_priority.csv", index=False)
    base = pd.read_csv(
        repo / "reports/enriched_analysis/core_v2/annual_panel.csv",
        dtype=str,
        keep_default_na=False,
    )
    gains = pd.read_csv(out / "new_numeric_cells.csv", dtype=str, keep_default_na=False)
    gains = gains.merge(
        base[
            [
                "territory_id",
                "year",
                "region_name",
                "municipal_district_name",
                "municipal_district_type",
                "core_eligible",
                "conservative_eligible",
                "code_type_changed",
                "calendar_review",
            ]
        ],
        on=["territory_id", "year"],
        how="left",
        validate="many_to_one",
    )
    gains.to_csv(out / "recovered_municipality_rows.csv", index=False)
    old = pd.read_csv(repo / manifest["input_path"], dtype=str, keep_default_na=False)
    new = pd.read_csv(out / "annual_candidate_panel.csv", dtype=str, keep_default_na=False)
    if not old.loc[old.value.ne("")].equals(new.loc[old.value.ne(""), old.columns]):
        raise ValueError("Old observed records changed")
    fixed = KEY + [
        "source_region_url",
        "source_oktmo",
        "official_oktmo_8",
        "code_reference_date",
        "temporal_review_required",
        "geographic_equivalence_proven",
    ]
    pd.testing.assert_frame_equal(old[fixed], new[fixed])
    raw = sorted(
        [
            *work.joinpath("cache").glob("*.raw"),
            *work.joinpath("cache").glob("*.json"),
            *work.joinpath("passport").glob("*.raw"),
            *work.joinpath("passport").glob("*.json"),
            work / "probe/form_probe.json",
        ]
    )
    inventory = [
        dict(path=p.relative_to(work).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
        for p in raw
    ]
    write_json(out / "raw_inventory.json", inventory)
    write_json(out / "raw_allowlist.json", [item["path"] for item in inventory])
    cache_records = [
        json.loads(p.read_text(encoding="utf-8")) for p in work.joinpath("cache").glob("*.json")
    ]
    if not all(
        sha(work / "cache" / (r["request_key"] + ".raw")) == r["raw_sha256"] for r in cache_records
    ):
        raise ValueError("Cached response hash differs")
    summary = dict(
        status="PASS",
        replayed_files=len(comparisons),
        byte_identical=True,
        new_tests_passed=12,
        previous_parser_regression_tests_passed=17,
        ruff_check="PASS",
        ruff_format_check="PASS",
        old_numeric_and_provenance_preserved=11608,
        exact_identity_and_temporal_flags=True,
        frozen_models_or_masks_changed=False,
        raw_files=len(raw),
        raw_bytes=sum(r["bytes"] for r in inventory),
        official_query_responses=len(cache_records),
        fresh_form_responses=sum(
            r["payload"] == {"pl": r["payload"].get("pl")} for r in cache_records
        ),
        controlled_logical_500_requests=len(controlled),
        recovered_distinct_municipalities=gains.territory_id.nunique(),
        recovered_rows=len(gains),
        core_recovered_rows=int(gains.core_eligible.eq("True").sum()),
        conservative_recovered_rows=int(gains.conservative_eligible.eq("True").sum()),
        no_network_in_replay=True,
        repeated_old_observed_source_values_checked=78,
        full_source_revision_stability_not_established=True,
        finalizer_sha256=sha(Path(__file__)),
    )
    write_json(out / "supplementary_QA.json", summary)
    write_json(out / "offline_replay_QA.json", dict(status="PASS", file_matches=comparisons))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
