"""Summarize accepted coverage and expose named gaps without changing any source cells."""

import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-root", required=True, type=Path)
parser.add_argument("--report-dir", required=True, type=Path)
args = parser.parse_args()
repo = args.repo_root
report = args.report_dir
panel = pd.read_csv(report / "population/municipal_population.csv", dtype=str)
wide = pd.read_csv(report / "population/municipal_population_wide.csv", dtype=str)
manifest = json.loads((report / "population/manifest.json").read_text(encoding="utf-8"))
source_manifest = json.loads(
    (report / "population_source/manifest.json").read_text(encoding="utf-8")
)
names = []
for date in ["2023-01-01", "2024-01-01"]:
    frame = pd.read_csv(repo / f"data/derived/enrichment/municipal_crosswalk_{date}.csv", dtype=str)
    frame["reference_date"] = date
    names.append(
        frame[
            [
                "territory_id",
                "reference_date",
                "oktmo_digits",
                "region_name",
                "municipal_district_name",
            ]
        ]
    )
labels = pd.concat(names, ignore_index=True)
labelled = panel.merge(
    labels, on=["territory_id", "reference_date"], how="left", validate="many_to_one"
)
assert labelled.region_name.notna().all()
assert labelled.official_oktmo_8.eq(labelled.oktmo_digits.str[:8]).all()
columns = [
    "territory_id",
    "reference_date",
    "region_name",
    "municipal_district_name",
    "official_oktmo_8",
    "source_oktmo",
    "population_type",
    "availability_status",
    "analysis_status",
    "fetch_status",
    "fetch_error",
    "source_row_present",
]
missing = labelled.loc[labelled.analysis_value.isna(), columns]
missing.to_csv(report / "population_gaps.csv", index=False)
region_rows = []
for (date, kind, region), group in labelled.groupby(
    ["reference_date", "population_type", "region_name"]
):
    region_rows.append(
        {
            "reference_date": date,
            "population_type": kind,
            "region_name": region,
            "expected": len(group),
            "observed": int(group.analysis_value.notna().sum()),
            "missing": int(group.analysis_value.isna().sum()),
        }
    )
pd.DataFrame(region_rows).to_csv(report / "population_coverage_by_region.csv", index=False)
summary = {
    "primary_municipalities": int(panel.territory_id.nunique()),
    "reference_links": len(wide),
    "expected_cells": len(panel),
    "source_rows_present": int(panel.source_row_present.str.lower().eq("true").sum()),
    "observed_source_values": int(panel.value.notna().sum()),
    "analysis_values_available": int(panel.analysis_value.notna().sum()),
    "missing_analysis_cells": len(missing),
    "complete_annual_total_and_urban_pairs": int(wide.urban_share.notna().sum()),
    "initial_calendar_links": int(panel.initial_temporal_review.str.lower().eq("true").sum()) // 2,
    "calendar_links_cleared": int(panel.calendar_review_cleared.str.lower().eq("true").sum()) // 2,
    "unresolved_calendar_links": int(panel.temporal_review.str.lower().eq("true").sum()) // 2,
    "source_target_count": source_manifest["target_count"],
    "source_target_status_counts": source_manifest["status_counts"],
    "coverage": manifest["coverage"],
    "imputed_values": 0,
}
assert summary["analysis_values_available"] + summary["missing_analysis_cells"] == 8064
(report / "population_delivery_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
