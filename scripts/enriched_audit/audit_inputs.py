import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
from audit_cli import parse_paths

args = parse_paths("Audit admissible municipal joins and coverage")
R, OUT = (args.repo, args.out)
paths = {
    "p": R / "reports/enrichment_v2/population/municipal_population.csv",
    "e": R / "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv",
    "g": R / "reports/enrichment_v3/spatial/sber_yearly_spatial_attributes.csv",
    "s": R / "reports/enrichment_v3/spatial/bdpmo_spatial_source_cells.csv",
    "static": R / "reports/enrichment_v3/spatial/sber_static_context_2024.csv",
    "pl": R / "reports/enrichment_v3/prices/municipal_region_price_links.csv",
    "cpi": R / "reports/enrichment_v3/prices/cpi_monthly_2023_2024.csv",
    "basket": R / "reports/enrichment_v3/prices/fixed_basket_regional_2024.csv",
    "city": R / "reports/enrichment_v3/prices/city_cost_of_living_2023_2024.csv",
    "area_changes": R / "reports/enrichment_v3/spatial/bdpmo_area_large_changes_review.csv",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def n(x):
    return pd.to_numeric(x, errors="coerce")


def yes(x):
    return x.eq("True")


def code(x):
    valid = x.str.fullmatch("\\d{1,8}")
    return x.where(valid, "").str.zfill(8).where(valid, "")


d = {k: read(v) for k, v in paths.items()}
contracts = {
    k: {
        "path": str(paths[k].relative_to(R)),
        "sha256": sha(paths[k]),
        "rows": len(x),
        "columns": x.columns.tolist(),
    }
    for k, x in d.items()
}
p, e, g, s = (d["p"], d["e"], d["g"], d["s"])
K = ["territory_id", "year"]
p["year"] = p.reference_date.str[:4]
g["year"] = g.snapshot_date.str[:4]
assert len(p) == len(e) == 8064 and len(g) == 4032
assert (
    not p.duplicated(K + ["population_type"]).any()
    and (not e.duplicated(K + ["indicator_id"]).any())
    and (not g.duplicated(K).any())
)
m = g[
    K
    + [
        "official_oktmo_8",
        "region_code",
        "region_name",
        "municipal_district_name",
        "municipal_district_type",
    ]
].copy()
change = {}
for field in [
    "official_oktmo_8",
    "municipal_district_type",
    "region_code",
    "geometry_year_from",
    "geometry_year_to_exclusive",
]:
    table = g.pivot(index="territory_id", columns="year", values=field)
    change[field] = table["2023"].ne(table["2024"])
m["changed_code_or_type"] = m.territory_id.map(
    change["official_oktmo_8"] | change["municipal_district_type"]
)
p["source_code8"] = code(p.source_oktmo)
e["source_code8"] = code(e.source_oktmo)
for kind, label in [("11", "population"), ("12", "urban")]:
    x = p[p.population_type == kind].copy()
    x[label] = n(x.analysis_value)
    x[label + "_ok"] = (
        x[label].notna()
        & x.source_code8.eq(x.official_oktmo_8)
        & yes(x.join_eligible)
        & x.analysis_status.eq("observed_conditional_geography")
        & x.unit.eq("persons")
    )
    cols = [
        "official_oktmo_8",
        "sber_oktmo_11",
        "initial_temporal_review",
        "temporal_review",
        "calendar_review_cleared",
        "calendar_source_sha256",
        "calendar_sber_oktmo_11",
    ]
    x = x[K + [label, label + "_ok"] + cols].rename(columns={v: label + "_" + v for v in cols})
    m = m.merge(x, on=K, validate="one_to_one")
for ind, label, unit in [("8423005", "employees", "человек"), ("8423007", "salary", "рубль")]:
    x = e[e.indicator_id == ind].copy()
    x[label] = n(x.value)
    x[label + "_ok"] = (
        x[label].notna()
        & x.source_code8.eq(x.official_oktmo_8)
        & x.cell_status.eq("observed")
        & x.unit.eq(unit)
        & x.period_code.eq("79")
        & x.okved2_code.eq("101")
        & x.period_start.eq(x.year + "-01-01")
        & x.period_end.eq(x.year + "-12-31")
        & x.code_reference_date.eq(x.year + "-01-01")
        & x.extraction_status.isin(["complete", "partial_source"])
    )
    cols = ["official_oktmo_8", "sber_oktmo_11", "temporal_review_required"]
    x = x[K + [label, label + "_ok"] + cols].rename(columns={v: label + "_" + v for v in cols})
    m = m.merge(x, on=K, validate="one_to_one")
for label in ["population", "urban", "employees", "salary"]:
    m[label + "_ok"] &= m[label + "_official_oktmo_8"].eq(m.official_oktmo_8)
m["population_ok"] &= m.population.gt(0)
m["employees_ok"] &= m.employees.ge(0)
m["salary_ok"] &= m.salary.gt(0)
m["urban_share_ok"] = m.urban_ok & m.population_ok & m.urban.ge(0) & m.urban.le(m.population)
m["initial_calendar_flag"] = (
    yes(m.population_initial_temporal_review)
    | yes(m.employees_temporal_review_required)
    | yes(m.salary_temporal_review_required)
)
m["calendar_clearance_exact_reuse"] = (
    yes(m.population_calendar_review_cleared)
    & m.population_calendar_source_sha256.str.fullmatch("[0-9a-f]{64}")
    & m.population_calendar_sber_oktmo_11.eq(m.population_sber_oktmo_11)
    & m.population_sber_oktmo_11.eq(m.employees_sber_oktmo_11)
    & m.employees_sber_oktmo_11.eq(m.salary_sber_oktmo_11)
    & m.population_official_oktmo_8.eq(m.employees_official_oktmo_8)
    & m.employees_official_oktmo_8.eq(m.salary_official_oktmo_8)
)
m["calendar_date_eligible"] = ~m.initial_calendar_flag | m.calendar_clearance_exact_reuse
m["mask_core_expanded"] = m.population_ok & m.employees_ok & m.salary_ok & m.calendar_date_eligible
m["mask_core_strict"] = m.mask_core_expanded & ~m.changed_code_or_type & ~m.initial_calendar_flag
m["mask_core_urban_expanded"] = m.mask_core_expanded & m.urban_share_ok
m["mask_core_urban_strict"] = m.mask_core_strict & m.urban_share_ok
m["employees_per_resident"] = m.employees.div(m.population).where(m.employees_ok & m.population_ok)
m["employees_per_resident_gt1"] = m.employees_per_resident.gt(1)
s["year"] = s.source_year
s["source_code8"] = code(s.source_oktmo)
s["numeric"] = n(s.value)
s = s.rename(columns={"territory_id_candidate": "territory_id"})
SK = ["source_code8", "year", "indicator_id", "surface_code"]
s["source_key"] = s[SK].agg("|".join, axis=1)
rep = s.groupby("source_key").size()
nu = s.groupby("source_key").numeric.nunique(dropna=False)
assert not ((rep > 1) & (nu > 1)).any()
match = s.merge(
    m[K + ["official_oktmo_8"]], on=K, how="left", validate="many_to_one", indicator=True
)
match["annual_code_matches"] = match.source_code8.eq(match.official_oktmo_8) & match._merge.eq(
    "both"
)
match[~match.annual_code_matches].to_csv(
    OUT / "spatial_source_year_code_mismatches.csv", index=False
)
uq = match[match.annual_code_matches].drop_duplicates("source_key")
assert not uq.duplicated(K + ["indicator_id", "surface_code"]).any()
for ind, surface, label in [
    ("8006001", "total_area", "area_ha"),
    ("8006005", "9", "road_total_km"),
    ("8006005", "1", "road_hard_km"),
    ("8006005", "2", "road_improved_km"),
]:
    x = uq[(uq.indicator_id == ind) & (uq.surface_code == surface)].copy()
    x[label] = x.numeric
    x[label + "_repeated"] = x.source_key.map(rep).gt(1)
    m = m.merge(x[K + [label, label + "_repeated"]], on=K, how="left", validate="one_to_one")
    m[label + "_ok"] = m[label].notna()
m["mask_core_area_same_label_year_expanded"] = m.mask_core_expanded & m.area_ha_ok
m["mask_core_roads_same_label_year_expanded"] = m.mask_core_expanded & m.road_total_km_ok
m["area_large_year_change_review"] = m.official_oktmo_8.isin(set(d["area_changes"].source_oktmo))
m["mask_area_stability_sensitivity"] = (
    m.mask_core_area_same_label_year_expanded
    & ~m.area_large_year_change_review
    & ~m.changed_code_or_type
    & m.area_ha_repeated.eq(False)
)
a = (
    s[(s.indicator_id == "8006001") & (s.source_year == "2023")]
    .drop_duplicates("source_key")
    .copy()
)
a["year"] = "2024"
a = a.merge(m[K + ["official_oktmo_8"]], on=K, validate="many_to_one")
a = a[a.source_code8.eq(a.official_oktmo_8)]
assert not a.duplicated(K).any()
a = a.rename(columns={"numeric": "area_end2023_ha"})
m = m.merge(a[K + ["area_end2023_ha"]], on=K, how="left", validate="one_to_one")
m["mask_density_2024_expanded"] = (
    m.year.eq("2024") & m.population_ok & m.area_end2023_ha.gt(0) & m.calendar_date_eligible
)
m["mask_density_2024_strict"] = (
    m.mask_density_2024_expanded & ~m.changed_code_or_type & ~m.initial_calendar_flag
)
m["density_end2023_jan2024_people_per_km2"] = m.population.div(m.area_end2023_ha / 100).where(
    m.mask_density_2024_expanded
)
x = d["static"].copy()
assert len(x) == 2016 and (not x.territory_id.duplicated().any())
x["year"] = "2024"
x["market_access"] = n(x.market_access)
x["road_distance_to_sber_flagged_center_km"] = n(x.road_distance_to_sber_flagged_center_km)
x = x.rename(columns={"official_oktmo_8": "static2024_code"})
cols = [
    "static2024_code",
    "market_access",
    "market_access_source_year",
    "road_distance_to_sber_flagged_center_km",
    "road_distance_status",
    "road_source_reference_date",
]
m = m.merge(x[K + cols], on=K, how="left", validate="one_to_one")
m["market2024_ok"] = (
    m.year.eq("2024")
    & m.market_access.notna()
    & m.static2024_code.eq(m.official_oktmo_8)
    & m.market_access_source_year.eq("2024")
)
m["distance2024_pair_ok"] = (
    m.year.eq("2024")
    & m.road_distance_status.eq("observed_pair")
    & m.road_distance_to_sber_flagged_center_km.notna()
)
m["distance2024_self_zero_ok"] = (
    m.year.eq("2024")
    & m.road_distance_status.eq("same_territory_zero_by_definition_not_source_cell")
    & m.road_distance_to_sber_flagged_center_km.eq(0)
)
pl = d["pl"].copy()
pl["year"] = pl.reference_date.str[:4]
assert len(pl) == 4032 and (not pl.duplicated(K).any())
m = m.merge(
    pl[K + ["region_code", "source_territory_code"]].rename(
        columns={"region_code": "price_region_code"}
    ),
    on=K,
    how="left",
    validate="one_to_one",
)
m["price_link_ok"] = m.region_code.eq(m.price_region_code) & m.source_territory_code.ne("")
assert m.price_link_ok.all()
for scope in ["expanded", "strict"]:
    m["mask_market2024_" + scope] = m["mask_core_" + scope] & m.market2024_ok & m.price_link_ok
    m["mask_market2024_urban_" + scope] = (
        m["mask_core_urban_" + scope] & m.market2024_ok & m.price_link_ok
    )
metrics = [
    "population_ok",
    "urban_ok",
    "urban_share_ok",
    "employees_ok",
    "salary_ok",
    "changed_code_or_type",
    "initial_calendar_flag",
    "calendar_clearance_exact_reuse",
    "employees_per_resident_gt1",
    "area_ha_ok",
    "road_total_km_ok",
    "road_hard_km_ok",
    "road_improved_km_ok",
    "market2024_ok",
    "price_link_ok",
    "distance2024_pair_ok",
    "distance2024_self_zero_ok",
] + [x for x in m if x.startswith("mask_")]
coverage = pd.DataFrame(
    [
        {
            "mask": col,
            "year": year,
            "eligible_rows": int(x[col].eq(True).sum()),
            "denominator": 2016,
        }
        for col in metrics
        for year, x in m.groupby("year")
    ]
)
coverage.to_csv(OUT / "coverage_masks.csv", index=False)
paired = []
for col in [c for c in metrics if c.startswith("mask_") and "2024" not in c]:
    counts = m[m[col].eq(True)].groupby("territory_id").year.nunique()
    paired.append({"mask": col, "ids_eligible_both_years": int(counts.eq(2).sum())})
pd.DataFrame(paired).to_csv(OUT / "paired_coverage.csv", index=False)
for name, sub in [
    ("annual_masks_and_diagnostics", m),
    ("code_type_changes", m[m.changed_code_or_type]),
    ("calendar_flag_cases", m[m.initial_calendar_flag]),
    ("employees_resident_ratio_over_one", m[m.employees_per_resident_gt1]),
]:
    sub.to_csv(OUT / f"{name}.csv", index=False)
qa = {
    "status": "PASS_conditional_join_audit",
    "generated_at_utc": datetime.now(UTC).isoformat(),
    "annual_rows": len(m),
    "unique_ids": m.territory_id.nunique(),
    "code_change_ids": int(change["official_oktmo_8"].sum()),
    "type_change_ids": int(change["municipal_district_type"].sum()),
    "region_change_ids": int(change["region_code"].sum()),
    "geometry_interval_change_ids": int(
        (change["geometry_year_from"] | change["geometry_year_to_exclusive"]).sum()
    ),
    "population_code_mismatches": int(p.source_code8.ne(p.official_oktmo_8).sum()),
    "economy_code_mismatches": int(e.source_code8.ne(e.official_oktmo_8).sum()),
    "population_current_temporal_flags": int(yes(p.temporal_review).sum()),
    "economy_initial_temporal_long_rows": int(yes(e.temporal_review_required).sum()),
    "independently_boundary_confirmed_economy_rows": int(
        yes(e.independent_boundary_equivalence_confirmed).sum()
    ),
    "spatial_source_rows": len(s),
    "spatial_same_label_year_code_matches": int(match.annual_code_matches.sum()),
    "spatial_same_label_year_code_mismatches": int((~match.annual_code_matches).sum()),
    "spatial_duplicate_keys": int(rep.gt(1).sum()),
    "spatial_duplicate_extra_rows": int((rep - 1).clip(lower=0).sum()),
    "spatial_duplicate_conflicting_keys": int(((rep > 1) & (nu > 1)).sum()),
    "imputations": 0,
    "new_demography_candidates_included": 0,
    "secondary_candidates_included": 0,
    "inputs": contracts,
    "outputs": {f.name: sha(f) for f in OUT.glob("*.csv")},
}
(OUT / "audit_manifest.json").write_text(
    json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps({k: v for k, v in qa.items() if k not in ["inputs", "outputs"]}, ensure_ascii=False)
)
print(coverage[coverage["mask"].str.startswith("mask_")].to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
