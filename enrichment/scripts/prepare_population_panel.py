"""Join observed population to a complete annual cohort grid with explicit review flags."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

SOURCE_KEY = ["source_region_url", "source_oktmo", "reference_date", "population_type"]
PANEL_KEY = ["territory_id", "reference_date", "population_type"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_bool(series, label):
    normalized = series.astype("string").str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Missing or invalid boolean: {label}")
    return normalized.isin(["true", "1"])


def apply_date_decisions(links, decisions):
    """Clear only documented code-calendar conflicts; retain the initial warning."""
    links = links.copy()
    links["initial_temporal_review"] = links.temporal_review
    links["calendar_review_cleared"] = False
    if decisions is None or decisions.empty:
        return links
    decisions = decisions.copy()
    key = ["territory_id", "reference_date", "official_oktmo_8"]
    decisions["territory_id"] = decisions.territory_id.astype(str)
    if decisions[key].isna().any().any() or decisions.duplicated(key).any():
        raise ValueError("Missing or duplicate calendar decision key")
    if not decisions.decision.eq("clear_calendar_code_conflict_only").all():
        raise ValueError("Unknown calendar decision")
    if not strict_bool(decisions.code_date_use_confirmed, "code_date_use_confirmed").all():
        raise ValueError("Unconfirmed code/date decision")
    allowed = pd.to_datetime(decisions.allowed_from, format="%Y-%m-%d", errors="raise")
    reference = pd.to_datetime(decisions.reference_date, format="%Y-%m-%d", errors="raise")
    if allowed.isna().any() or (allowed > reference).any():
        raise ValueError("Code permission begins after the reference date")
    if not decisions.source_sha256.str.fullmatch(r"[a-f0-9]{64}", na=False).all():
        raise ValueError("Missing calendar evidence digest")
    if not decisions.source_capture_kind.isin(["raw_html", "web_tool_text"]).all():
        raise ValueError("Unknown calendar evidence capture type")
    if decisions.source_url.isna().any() or decisions.locator.isna().any():
        raise ValueError("Missing calendar evidence location")
    check = decisions.merge(
        links[key + ["temporal_review"]], on=key, how="left", validate="one_to_one", indicator=True
    )
    if not check["_merge"].eq("both").all() or not check.temporal_review.eq(True).all():
        raise ValueError("Calendar decision does not match an initial flagged exact code")
    decisions = decisions.rename(columns={c: "calendar_" + c for c in decisions if c not in key})
    links = links.merge(decisions, on=key, how="left", validate="one_to_one")
    links["calendar_review_cleared"] = links.calendar_decision.notna()
    links["temporal_review"] = links.initial_temporal_review & ~links.calendar_review_cleared
    return links


def assemble(source, links, target_status, cohort_ids, date_decisions=None):
    source, links, target_status = source.copy(), links.copy(), target_status.copy()
    expected_ids = {str(value) for value in cohort_ids}
    links["territory_id"] = links.territory_id.astype(str)
    expected_dates = {"2023-01-01", "2024-01-01"}
    if set(links.reference_date) != expected_dates:
        raise ValueError("Expected exactly the two annual reference dates")
    if links.duplicated(["territory_id", "reference_date"]).any():
        raise ValueError("Duplicate territory/date reference link")
    if any(set(group.territory_id) != expected_ids for _, group in links.groupby("reference_date")):
        raise ValueError("Each annual link grid must equal the full primary cohort")
    links["temporal_review"] = strict_bool(
        links.temporal_review_required, "temporal_review_required"
    )
    if not links.source_oktmo.str.fullmatch(r"[0-9]{1,8}", na=False).all():
        raise ValueError("Invalid numeric source code")
    if not links.official_oktmo_8.str.fullmatch(r"[0-9]{8}", na=False).all():
        raise ValueError("Invalid eight-digit official code")
    # The source HTML option value is numerical; the official identifier preserves zeroes.
    if not links.source_oktmo.str.zfill(8).eq(links.official_oktmo_8).all():
        raise ValueError("Acquisition code differs from exact annual official code")
    links = apply_date_decisions(links, date_decisions)
    links["join_eligible"] = (
        strict_bool(links.official_code_present, "official_code_present")
        & strict_bool(links.sber_annual_code_present, "sber_annual_code_present")
        & ~links.temporal_review
    )
    for frame in (source, links):
        keys = SOURCE_KEY if frame is source else SOURCE_KEY[:3]
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Missing or duplicate source join key")
    if not source.population_type.isin(["11", "12"]).all():
        raise ValueError("Unknown population type")
    if not source.reference_date.isin(expected_dates).all():
        raise ValueError("Unexpected source date")
    if not source.unit.eq("persons").all() or not source.indicator_id.eq("8112027").all():
        raise ValueError("Unexpected indicator or unit")
    source["value"] = pd.to_numeric(source.value, errors="raise").astype("Int64")
    observed = source.value_status.eq("observed")
    if (observed & source.value.isna()).any() or (~observed & source.value.notna()).any():
        raise ValueError("Observed/missing state conflicts with population value")
    if source.value.dropna().lt(0).any():
        raise ValueError("Negative population")
    source = source.drop(columns=["territory_id", "join_status"], errors="ignore")
    # Keep source provenance names; link metadata receives an explicit prefix on collision.
    collisions = (set(source.columns) & set(links.columns)) - set(SOURCE_KEY)
    links = links.rename(columns={name: "link_" + name for name in collisions})
    grid = links.merge(pd.DataFrame({"population_type": ["11", "12"]}), how="cross")
    panel = grid.merge(source, on=SOURCE_KEY, how="left", validate="one_to_one", indicator=True)
    target_status = target_status.rename(
        columns={
            "region_url": "source_region_url",
            "status": "fetch_status",
            "error": "fetch_error",
        }
    )
    panel = panel.merge(
        target_status[["source_region_url", "source_oktmo", "fetch_status", "fetch_error"]],
        on=["source_region_url", "source_oktmo"],
        how="left",
        validate="many_to_one",
    )
    if panel.fetch_status.isna().any():
        raise ValueError("Reference link has no acquisition status")
    panel["source_row_present"] = panel.pop("_merge").eq("both")
    if (panel.source_row_present & ~panel.fetch_status.isin(["complete", "partial_source"])).any():
        raise ValueError("Source rows conflict with rejected or unfinished acquisition status")
    panel["value"] = panel.value.astype("Int64")
    panel["availability_status"] = panel.value_status.fillna("not_returned")
    panel["analysis_value"] = panel.value.where(
        panel.join_eligible & panel.availability_status.eq("observed")
    ).astype("Int64")
    panel["analysis_status"] = panel.availability_status
    panel.loc[panel.join_eligible & panel.availability_status.eq("observed"), "analysis_status"] = (
        "observed_conditional_geography"
    )
    panel.loc[~panel.join_eligible, "analysis_status"] = "code_or_boundary_review"
    panel.loc[panel.temporal_review, "analysis_status"] = "temporal_review"
    if panel.duplicated(PANEL_KEY).any() or len(panel) != len(expected_ids) * 4:
        raise ValueError("Cohort cardinality changed")
    index = ["territory_id", "reference_date"]
    wide = (
        panel.pivot(index=index, columns="population_type", values="analysis_value")
        .rename(columns={"11": "population_total", "12": "population_urban"})
        .reset_index()
    )
    values = wide[["population_total", "population_urban"]].notna().all(axis=1)
    if (values & wide.population_urban.gt(wide.population_total)).any():
        raise ValueError("Urban population exceeds total")
    valid_ratio = values & wide.population_total.gt(0)
    wide["urban_share"] = (wide.population_urban / wide.population_total).where(valid_ratio)
    wide["urban_share_status"] = valid_ratio.map(
        {True: "observed_components", False: "unavailable"}
    )
    coverage = []
    for (date, kind), group in panel.groupby(["reference_date", "population_type"]):
        coverage.append(
            {
                "reference_date": date,
                "population_type": kind,
                "expected_municipalities": len(group),
                "source_rows_returned": int(group.source_row_present.sum()),
                "source_values_observed": int(group.value.notna().sum()),
                "analysis_values_available": int(group.analysis_value.notna().sum()),
                "temporal_review_links": int(group.temporal_review.sum()),
                "calendar_review_cleared_links": int(group.calendar_review_cleared.sum()),
                "status_counts": group.analysis_status.value_counts().to_dict(),
            }
        )
    return panel.sort_values(PANEL_KEY), wide.sort_values(index), coverage


def prepare(population, reference_links, target_status, cohort, output, date_decisions=None):
    paths = {
        name: Path(value)
        for name, value in {
            "population": population,
            "reference_links": reference_links,
            "target_status": target_status,
            "cohort": cohort,
        }.items()
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    decisions = None
    if date_decisions is not None:
        paths["date_decisions"] = Path(date_decisions)
        decisions = pd.read_csv(date_decisions, dtype=str)
        base = Path(date_decisions).resolve().parent
        for row in decisions.itertuples():
            capture = (base / row.source_capture_path).resolve()
            if not capture.is_relative_to(base) or sha(capture) != row.source_sha256:
                raise ValueError("Calendar evidence path or digest mismatch")
    with duckdb.connect() as db:
        ids = db.execute("SELECT DISTINCT territory_id FROM read_parquet(?)", [str(cohort)]).df()
    panel, wide, coverage = assemble(
        pd.read_csv(population, dtype=str),
        pd.read_csv(reference_links, dtype=str),
        pd.read_csv(target_status, dtype=str),
        ids.territory_id,
        decisions,
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [("municipal_population", panel), ("municipal_population_wide", wide)]:
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8")
        with duckdb.connect() as db:
            db.register("frame", frame)
            db.execute(
                "COPY frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(output / f"{name}.parquet")],
            )
    manifest = {
        "status": "ready_with_conditions",
        "panel_rows": len(panel),
        "annual_rows": len(wide),
        "coverage": coverage,
        "imputed_values": 0,
        "conditions": [
            "Exact regional source code and annual Sber reference link; no fuzzy names.",
            "Source values are retained. analysis_value is null for ineligible or "
            "temporal-review links.",
            "Missing urban population is not inferred to be zero; ratios require both "
            "observed components.",
            "Retrieval in 2026 does not establish historical publication availability "
            "or legal boundary equivalence.",
            "Optional documentary calendar decisions clear only the exact code/date conflict; "
            "initial warnings and independent boundary caveats remain in the long panel.",
        ],
        "inputs": {name: {"path": str(p), "sha256": sha(p)} for name, p in paths.items()},
        "processor_sha256": sha(__file__),
        "output_hashes": {p.name: sha(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["population", "reference-links", "target-status", "cohort", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--date-decisions", type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
