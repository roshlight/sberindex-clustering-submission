import hashlib
import json

import pandas as pd
from audit_cli import parse_paths

args = parse_paths("Summarize audited coverage by region and type", with_audit=True)
OUT = args.out
m = pd.read_csv(args.audit / "annual_masks_and_diagnostics.csv", dtype=str, keep_default_na=False)
cols = [
    "population_ok",
    "urban_share_ok",
    "employees_ok",
    "salary_ok",
    "mask_core_expanded",
    "mask_core_strict",
    "mask_core_urban_expanded",
    "mask_core_urban_strict",
]
rows = []
for (year, region, type_), x in m.groupby(["year", "region_name", "municipal_district_type"]):
    r = {"year": year, "region_name": region, "municipality_type": type_, "cohort_rows": len(x)}
    r.update({col: int(x[col].eq("True").sum()) for col in cols})
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "coverage_by_region_and_type.csv", index=False)
print(
    "CALENDAR",
    m[m.initial_calendar_flag == "True"][
        [
            "territory_id",
            "year",
            "municipal_district_name",
            "changed_code_or_type",
            "mask_core_expanded",
        ]
    ].to_dict("records"),
)
for year, x in m.groupby("year"):
    numeric = pd.to_numeric(x.employees_per_resident, errors="coerce")
    top = x.assign(ratio=numeric).sort_values("ratio", ascending=False).head(5)
    print(year, "ratio >1", int(numeric.gt(1).sum()), "max", numeric.max())
    print(top[["territory_id", "municipal_district_name", "ratio"]].to_dict("records"))
    print(
        "urban-missing",
        x[x.urban_share_ok != "True"].groupby("municipal_district_type").size().to_dict(),
    )
f = OUT / "audit_manifest.json"
q = json.loads((args.audit / "audit_manifest.json").read_text(encoding="utf-8"))
q["outputs"]["coverage_by_region_and_type.csv"] = hashlib.sha256(
    (OUT / "coverage_by_region_and_type.csv").read_bytes()
).hexdigest()
q["employees_resident_ratio_over_one_by_year"] = {
    year: int(pd.to_numeric(x.employees_per_resident, errors="coerce").gt(1).sum())
    for year, x in m.groupby("year")
}
q["employees_resident_ratio_max_by_year"] = {
    year: float(pd.to_numeric(x.employees_per_resident, errors="coerce").max())
    for year, x in m.groupby("year")
}
f.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
