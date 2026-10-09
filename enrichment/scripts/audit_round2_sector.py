"""Reconcile the sector pilot and optionally completed bulk outputs with frozen v3 totals."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
from run_round2_sector import sha, write_json

BASE_ACTIVITIES = (
    "10000",
    "50000",
    "20000",
    "30000",
    "40000",
    "41000",
    "45000",
    "49000",
    "55000",
    "63000",
    "60000",
    "68000",
    "70000",
    "80000",
    "84000",
    "85000",
    "86000",
    "90000",
    "95000",
)


def canonical_frame(path, bulk=False):
    frame = pd.read_csv(
        path,
        dtype={
            "territory_id": str,
            "year": str,
            "indicator_id": str,
            "source_oktmo": str,
            "official_oktmo_8": str,
            "source_activity_code": str,
            "okved2_code": str,
            "value_raw": str,
            "raw_value": str,
        },
    )
    if bulk:
        frame = frame.rename(
            columns={
                "okved2_code": "source_activity_code",
                "okved2_label": "source_activity_label",
                "value_status": "status",
            }
        )
    return frame


def coverage(frame):
    records = []
    for (tid, year, indicator), group in frame.groupby(["territory_id", "year", "indicator_id"]):
        sectors = group[group.source_activity_code.ne("101")]
        total = group[group.source_activity_code.eq("101")]
        if len(total) != 1:
            raise ValueError("Expected exactly one aggregate cell per entity/year/indicator")
        total = total.iloc[0]
        numeric = sectors[sectors.status.eq("observed")]
        missing = sectors[sectors.status.ne("observed")]
        worker = indicator == "8423005"
        records.append(
            {
                "territory_id": tid,
                "year": year,
                "indicator_id": indicator,
                "source_oktmo_name": total.source_oktmo_name,
                "source_oktmo": total.source_oktmo,
                "offered_sectors": len(sectors),
                "observed_sector_cells": len(numeric),
                "marker_cells": int(
                    sectors.status.isin(
                        ["source_marker_unresolved", "unresolved_source_marker"]
                    ).sum()
                ),
                "not_returned_cells": int(sectors.status.eq("not_returned").sum()),
                "aggregate_status": total.status,
                "aggregate_value": total.value,
                "sum_observed_sector_workers": numeric.value.sum() if worker else None,
                "arithmetic_gap_not_an_industry": total.value - numeric.value.sum()
                if worker
                else None,
                "missing_source_activity_codes": "|".join(missing.source_activity_code),
                "standard_19_not_offered": "|".join(
                    sorted(set(BASE_ACTIVITIES) - set(sectors.source_activity_code))
                ),
                "additional_offered_codes": "|".join(
                    sorted(set(sectors.source_activity_code) - set(BASE_ACTIVITIES))
                ),
            }
        )
    return pd.DataFrame(records)


def reconcile(frame, old):
    total = frame[frame.source_activity_code.eq("101")]
    keys = ["territory_id", "year", "indicator_id"]
    joined = total.merge(
        old[keys + ["value"]],
        on=keys,
        suffixes=("_new", "_old"),
        validate="one_to_one",
        how="left",
        indicator=True,
    )
    equality = joined.value_new.eq(joined.value_old) | (
        joined.value_new.isna() & joined.value_old.isna()
    )
    equality &= joined._merge.eq("both")
    mismatches = joined[~equality]
    return {
        "aggregate_cells_checked": len(joined),
        "exact_value_or_null_matches": int(equality.sum()),
        "both_numeric_matches": int(
            (joined.value_new.eq(joined.value_old) & joined.value_new.notna()).sum()
        ),
        "both_null": int((joined.value_new.isna() & joined.value_old.isna()).sum()),
        "unmatched_reference_rows": int(joined._merge.ne("both").sum()),
        "mismatches": json.loads(
            mismatches[keys + ["value_new", "value_old"]].to_json(orient="records")
        ),
    }


def audit(repo, source, out):
    out.mkdir(parents=True, exist_ok=True)
    old_path = repo / "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv"
    old = pd.read_csv(old_path, dtype={"territory_id": str, "year": str, "indicator_id": str})
    pilot = canonical_frame(source / "sector_source.csv")
    summary = {"pilot_reconciliation": reconcile(pilot, old), "frozen_v3_sha256": sha(old_path)}
    coverage(pilot).to_csv(out / "pilot_coverage.csv", index=False)
    for name in ("sector_source.csv", "pilot_targets.csv", "source_inventory.json", "qa.json"):
        shutil.copyfile(
            source / name, out / ("pilot_" + name if name != "pilot_targets.csv" else name)
        )
    pilot_2024 = pilot[
        (pilot.year.eq("2024"))
        & pilot.indicator_id.eq("8423005")
        & pilot.source_activity_code.ne("101")
        & pilot.status.eq("observed")
    ]
    top = (
        pilot_2024.sort_values(["territory_id", "value"], ascending=[True, False])
        .groupby("territory_id")
        .head(3)
    )
    top.to_csv(out / "pilot_top_observed_worker_sectors_2024.csv", index=False)
    bulk_path = source / "bulk/annual_candidate_sector.csv"
    if bulk_path.exists():
        bulk = canonical_frame(bulk_path, bulk=True)
        source_keys = [
            "source_region_url",
            "source_oktmo",
            "indicator_id",
            "year",
            "source_activity_code",
        ]
        original = canonical_frame(source / "bulk/sector_source.csv", bulk=True)
        current_labels = original[source_keys + ["source_oktmo_name"]].rename(
            columns={"source_oktmo_name": "actual_source_oktmo_label"}
        )
        bulk = bulk.merge(current_labels, on=source_keys, how="left", validate="many_to_one")
        bulk["caption_decision"] = "exact_complete_source_form_and_csv"
        supplement_path = source / "caption_2710/sector_source.csv"
        recovered_hashes = set()
        if supplement_path.exists():
            supplement = canonical_frame(supplement_path).rename(
                columns={
                    "raw_value": "value_raw",
                    "csv_record_number": "source_csv_row",
                    "options_source_sha256": "options_sha256",
                }
            )
            decisions = json.loads(
                (source / "caption_2710/accepted_caption_decisions.json").read_text(
                    encoding="utf-8"
                )
            )
            recovered_hashes = {d["source_csv_sha256"] for d in decisions}
            if set(supplement.source_sha256) != recovered_hashes or set(
                supplement.territory_id
            ) != {"2710"}:
                raise ValueError("Caption supplement is not bound to the reviewed captures and ID")
            supplement["period_start"] = supplement.year + "-01-01"
            supplement["period_end"] = supplement.year + "-12-31"
            supplement["status"] = supplement.status.replace(
                {"source_marker_unresolved": "unresolved_source_marker"}
            )
            supplement["source_oktmo_name"] = supplement.actual_source_oktmo_label
            bulk = pd.concat([bulk, supplement], ignore_index=True, sort=False)
            caption_dir = out / "caption_2710"
            caption_dir.mkdir(exist_ok=True)
            for name in (
                "accepted_caption_decisions.json",
                "qa.json",
                "sector_source.csv",
                "source_inventory.json",
            ):
                shutil.copyfile(source / "caption_2710" / name, caption_dir / name)
        full_keys = ["territory_id", "year", "indicator_id", "source_activity_code"]
        if bulk.duplicated(full_keys).any():
            raise ValueError("Duplicate candidate key after explicit caption supplement")
        bulk["geographic_equivalence_confirmed"] = None
        bulk = bulk.sort_values(full_keys).reset_index(drop=True)
        summary["bulk_reconciliation"] = reconcile(bulk, old)
        differences = summary["bulk_reconciliation"]["mismatches"]
        recovered = [
            r
            for r in differences
            if r["territory_id"] == "2710"
            and r["value_old"] is None
            and r["value_new"] is not None
            and recovered_hashes
        ]
        summary["recovered_prior_caption_gap_aggregate_cells"] = recovered
        summary["unexplained_aggregate_differences"] = [
            r for r in differences if r not in recovered
        ]
        summary["combined_candidate"] = {
            "rows": len(bulk),
            "territories": int(bulk.territory_id.nunique()),
            "municipality_years": len(bulk[["territory_id", "year"]].drop_duplicates()),
            "status_counts": bulk.status.value_counts().to_dict(),
            "source_csv_captures": int(bulk.source_sha256.nunique()),
            "duplicate_keys": 0,
            "geographic_equivalence_confirmed": None,
        }
        summary["bulk_run_status"] = json.loads(
            (source / "bulk/qa.json").read_text(encoding="utf-8")
        )
        coverage(bulk).to_csv(out / "bulk_coverage.csv", index=False)
        keys = ["territory_id", "year", "indicator_id", "source_activity_code"]
        compared = pilot.merge(
            bulk[keys + ["value", "status"]],
            on=keys,
            suffixes=("_pilot", "_bulk"),
            validate="one_to_one",
        )
        equal = compared.value_pilot.eq(compared.value_bulk) | (
            compared.value_pilot.isna() & compared.value_bulk.isna()
        )
        status_equal = compared.status_pilot.replace(
            {"source_marker_unresolved": "unresolved_source_marker"}
        ).eq(compared.status_bulk)
        summary["single_vs_batch_pilot"] = {
            "overlapping_cells": len(compared),
            "value_or_null_matches": int(equal.sum()),
            "status_matches_after_literal_alias": int(status_equal.sum()),
            "mismatches": json.loads(
                compared.loc[
                    ~equal | ~status_equal,
                    keys + ["value_pilot", "value_bulk", "status_pilot", "status_bulk"],
                ].to_json(orient="records")
            ),
        }
        for name in ("activity_options.csv", "deferred_targets.json", "input_manifest.json"):
            shutil.copyfile(source / "bulk" / name, out / name)
        parquet_dir = out / "bulk"
        parquet_dir.mkdir(exist_ok=True)
        parquet = parquet_dir / "annual_candidate_sector.parquet"
        with duckdb.connect() as connection:
            connection.register("sector_candidate", bulk)
            connection.execute(
                "COPY sector_candidate TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(parquet)]
            )
            count = connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(parquet)]
            ).fetchone()[0]
        if count != len(bulk) or parquet.stat().st_size >= 40_000_000:
            raise ValueError("Parquet row check failed or artifact exceeds 40 MB")
        summary["candidate_parquet"] = {
            "path": parquet.relative_to(out).as_posix(),
            "rows": count,
            "bytes": parquet.stat().st_size,
            "sha256": sha(parquet),
        }
        shutil.copyfile(
            source / "bulk/reference_links.csv", parquet_dir / "year_specific_links.csv"
        )
    methodology_inventory = source / "docs/source_inventory.json"
    if methodology_inventory.exists():
        shutil.copyfile(methodology_inventory, out / "methodology_source_inventory.json")
    checked = []
    for inventory_path in [
        source / "source_inventory.json",
        source / "bulk/source_inventory.json",
        source / "caption_2710/source_inventory.json",
    ]:
        if not inventory_path.exists():
            continue
        for item in json.loads(inventory_path.read_text(encoding="utf-8")):
            raw = inventory_path.parent / "cache" / (item["request_key"] + ".raw")
            if sha(raw) != item["raw_sha256"] or raw.stat().st_size != item["bytes"]:
                raise ValueError(f"Raw source hash or size mismatch: {raw.name}")
            checked.append(
                {
                    "path": raw.relative_to(source).as_posix(),
                    "bytes": item["bytes"],
                    "sha256": item["raw_sha256"],
                    "url": item["url"],
                    "retrieved_at": item["retrieved_at"],
                    "purpose": item["purpose"],
                }
            )
    write_json(out / "raw_source_inventory.json", checked)
    summary["checked_raw_source_files"] = len(checked)
    summary["unique_raw_source_hashes"] = len({r["sha256"] for r in checked})
    any_mismatch = bool(summary["pilot_reconciliation"]["mismatches"])
    if "bulk_reconciliation" in summary:
        any_mismatch |= bool(summary["unexplained_aggregate_differences"])
        any_mismatch |= bool(summary["single_vs_batch_pilot"]["mismatches"])
    summary["status"] = (
        "WARN"
        if any_mismatch
        else (
            "PARTIAL" if summary.get("bulk_run_status", {}).get("status") == "PARTIAL" else "PASS"
        )
    )
    summary["scope_note"] = (
        "Coverage describes the last assembled CSV snapshot; live checkpoints may be newer. "
        "No claim of full economic sector coverage follows from successful parsing."
    )
    summary["script_sha256"] = sha(__file__)
    write_json(out / "reconciliation_qa.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo.resolve(), args.source_dir.resolve(), args.out.resolve())
