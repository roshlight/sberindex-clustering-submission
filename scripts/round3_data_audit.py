"""Audit and left-join round-three activity measures to the frozen annual cohort.

Monetary values are thousand rubles. Their logarithms per population on January 1
describe local activity intensity, not residents' income or consumption. No model,
imputation, geographic repair, or historical publication-date claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

KEY = ["territory_id", "year"]
SPECS = {
    "8109001": ("investment", "17"),
    "8401011": ("shipments", "79"),
    "8401003": ("retail", "79"),
}
SELECTED = [f"sector_{x}_share" for x in "ABCDHOPQ"]
STATUSES = {"observed", "unresolved_source_marker", "not_returned_or_unaccepted"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}: missing or duplicate keys")


def log_intensity(value: pd.Series, population: pd.Series) -> pd.Series:
    valid = np.isfinite(value) & (value > 0) & np.isfinite(population) & (population > 0)
    result = pd.Series(np.nan, index=value.index, dtype=float)
    # Equivalent to log(value*1000/population), avoiding intermediate overflow.
    result.loc[valid] = np.log(value[valid]) + np.log(1000.0) - np.log(population[valid])
    return result


def build_features(base: pd.DataFrame, economy: pd.DataFrame) -> pd.DataFrame:
    check_unique(base, KEY, "frozen annual panel")
    check_unique(economy, KEY + ["indicator_id"], "economy panel")
    if set(economy.indicator_id) != set(SPECS):
        raise ValueError("Unexpected indicator set")
    if not set(economy.value_status) <= STATUSES:
        raise ValueError("Unknown source status")
    if not economy.value_status.eq("observed").equals(economy.value.notna()):
        raise ValueError("Numeric values and observed status disagree")
    if not np.isfinite(economy.value.dropna()).all():
        raise ValueError("Nonfinite source value")
    if not economy.loc[economy.value.notna(), "unit"].eq("thousand_rubles").all():
        raise ValueError("Wrong source unit")
    result = base.copy()
    for indicator, (prefix, period) in SPECS.items():
        source = economy.loc[economy.indicator_id.eq(indicator)].copy()
        if set(map(tuple, source[KEY].to_numpy())) != set(map(tuple, base[KEY].to_numpy())):
            raise ValueError("Economy grid differs from frozen municipality-year keys")
        present = source.source_sha256.notna()
        if not source.loc[present, "period_code"].eq(period).all():
            raise ValueError("Wrong annual source period")
        if not source.loc[present, "activity_code"].eq("101").all():
            raise ValueError("Wrong activity total")
        fields = [
            "official_oktmo_8",
            "value",
            "value_status",
            "source_sha256",
            "request_key",
            "source_csv_row",
            "source_oktmo",
            "source_region_url",
            "source_oktmo_name",
            "code_reference_date",
            "temporal_review_required",
            "retrieved_at",
        ]
        renamed = {c: f"{prefix}_{c}" for c in fields}
        renamed["value"] = f"{prefix}_thousand_rubles"
        result = result.merge(
            source[KEY + fields].rename(columns=renamed),
            how="left",
            on=KEY,
            validate="one_to_one",
            sort=False,
        )
        if not result.official_oktmo_8.eq(result[f"{prefix}_official_oktmo_8"]).all():
            raise ValueError("Year-specific source code differs from frozen code")
        expected_date = result.year.astype(str) + "-01-01"
        if not result[f"{prefix}_code_reference_date"].eq(expected_date).all():
            raise ValueError("Source reference date differs from frozen year")
        value = result[f"{prefix}_thousand_rubles"]
        pop = result.population_total
        result[f"{prefix}_missing"] = value.isna()
        result[f"{prefix}_log_rubles_per_person"] = log_intensity(value, pop)
        result[f"{prefix}_feature_status"] = np.select(
            [value.isna(), value.le(0), ~np.isfinite(pop) | pop.le(0)],
            [
                "source_value_missing",
                "source_value_nonpositive",
                "population_missing_or_nonpositive",
            ],
            default="eligible_positive_value_and_population",
        )
    pd.testing.assert_frame_equal(result[base.columns], base.reset_index(drop=True))
    return result


def coverage_rows(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for sample, mask in sample_masks(frame).items():
        for values, group in frame.loc[mask].groupby(dimensions, dropna=False, observed=True):
            values = values if isinstance(values, tuple) else (values,)
            key = dict(zip(dimensions, values, strict=True))
            for indicator, (prefix, _) in SPECS.items():
                source = group[f"{prefix}_thousand_rubles"]
                feature = group[f"{prefix}_log_rubles_per_person"]
                row = dict(
                    sample=sample,
                    **key,
                    indicator_id=indicator,
                    prefix=prefix,
                    n_municipality_years=len(group),
                    n_distinct_territories=group.territory_id.nunique(),
                    numeric=int(source.notna().sum()),
                    positive=int(source.gt(0).sum()),
                    zero=int(source.eq(0).sum()),
                    negative=int(source.lt(0).sum()),
                    feature_available=int(feature.notna().sum()),
                    source_missing=int(source.isna().sum()),
                    feature_available_fraction=float(feature.notna().mean()),
                    population_missing_or_nonpositive=int(
                        (~np.isfinite(group.population_total) | group.population_total.le(0)).sum()
                    ),
                )
                for status in sorted(STATUSES):
                    row[status] = int(group[f"{prefix}_value_status"].eq(status).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def sample_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "whole": pd.Series(True, index=frame.index),
        "core": frame.core_eligible,
        "conservative": frame.conservative_eligible,
    }


def contract_tests() -> dict:
    try:
        check_unique(pd.DataFrame({"territory_id": [1, 1], "year": [2023, 2023]}), KEY, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate key accepted")
    values = pd.Series([1.0, 0.0, -1.0, np.nan, 2.0, 2.0])
    populations = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 0.0, np.nan])
    actual = log_intensity(values, populations)
    assert actual.iloc[0] == 0 and actual.iloc[1:].isna().all()
    np.testing.assert_allclose(log_intensity(pd.Series([2.0]), pd.Series([100.0])), [np.log(20)])
    return {
        "duplicate_key_rejected": True,
        "nonpositive_and_missing_stay_null": True,
        "thousand_ruble_conversion_verified": True,
    }


def run(repo: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    inputs = {
        "frozen_panel": repo / "reports/enriched_analysis/core_v2/annual_panel.csv",
        "economy_csv": repo / "reports/data_round3/economy/annual_candidate_panel.csv",
        "economy_parquet": repo / "reports/data_round3/economy/annual_candidate_panel.parquet",
        "source_cells": repo / "reports/data_round3/economy/source_cells.csv",
        "source_only": repo / "reports/data_round3/economy/source_cells_outside_year_link.csv",
        "collection_status": repo / "reports/data_round3/economy/collection_status.json",
        "sector": repo / "reports/round2_analysis/sector/feature_panel.csv",
    }
    input_hashes = {k: sha(v) for k, v in inputs.items()}
    string_types = {
        "official_oktmo_8": str,
        "source_oktmo": str,
        "indicator_id": str,
        "period_code": str,
        "activity_code": str,
        "region_code": str,
    }
    base = pd.read_csv(inputs["frozen_panel"], dtype=string_types)
    economy = pd.read_csv(inputs["economy_csv"], dtype=string_types)
    sector = pd.read_csv(inputs["sector"], dtype=string_types)
    if len(base) != 4032 or base.territory_id.nunique() != 2016 or set(base.year) != {2023, 2024}:
        raise ValueError("Frozen cohort shape changed")
    for flag in ["core_eligible", "conservative_eligible", "calendar_review", "code_type_changed"]:
        if base[flag].dtype != bool or base[flag].isna().any():
            raise ValueError("Frozen mask must be complete booleans")
    with duckdb.connect() as db:
        parquet = db.execute("SELECT * FROM read_parquet(?)", [str(inputs["economy_parquet"])]).df()
    parquet = parquet.astype({k: str for k in string_types if k in parquet})
    # Full CSV/Parquet comparison, allowing serialization dtype differences only.
    for column in economy:
        left, right = economy[column], parquet[column]
        if column in string_types:
            right = right.mask(right.isin(["None", "nan", "<NA>"]))
        if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
            np.testing.assert_allclose(
                pd.to_numeric(left).astype(float),
                pd.to_numeric(right).astype(float),
                equal_nan=True,
                rtol=1e-12,
                atol=0,
            )
        else:
            pd.testing.assert_series_equal(
                left.fillna("").astype(str), right.fillna("").astype(str), check_names=False
            )
    features = build_features(base, economy)
    check_unique(sector, KEY, "sector feature panel")
    sectors = features[KEY].merge(
        sector[KEY + SELECTED], on=KEY, how="left", validate="one_to_one", indicator=True
    )
    if not sectors._merge.eq("both").all():
        raise ValueError("Sector source lacks frozen keys")
    features["sector_selected8_available_count"] = sectors[SELECTED].notna().sum(axis=1)
    features["sector_selected8_complete"] = features.sector_selected8_available_count.eq(8)
    features["activity_all3_features_available"] = (
        features[[f"{x[0]}_log_rubles_per_person" for x in SPECS.values()]].notna().all(axis=1)
    )
    features["activity_vintage"] = "retrieved 2026-09-16; first publication/revisions unknown"
    features["activity_boundary_equivalence_proven"] = pd.Series(
        pd.NA, index=features.index, dtype="boolean"
    )
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "annual_features.csv", index=False)
    with duckdb.connect() as db:
        db.register("features", features)
        db.execute(
            "COPY features TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(out / "annual_features.parquet")],
        )
    print(
        json.dumps(
            {
                "features_ready": str(out / "annual_features.csv"),
                "rows": len(features),
                "sha256": sha(out / "annual_features.csv"),
            }
        ),
        flush=True,
    )
    coverage_rows(features, ["year"]).to_csv(out / "coverage.csv", index=False)
    sliced = features.copy()
    sliced["population_bin"] = (
        pd.cut(
            sliced.population_total,
            bins=[0, 10000, 25000, 50000, 100000, 250000, np.inf],
            right=False,
            labels=[
                "0_to_9999",
                "10000_to_24999",
                "25000_to_49999",
                "50000_to_99999",
                "100000_to_249999",
                "250000_plus",
            ],
        )
        .astype("string")
        .fillna("missing_or_invalid")
    )
    for name, dimensions in {
        "population": ["year", "population_bin"],
        "type": ["year", "municipal_district_type"],
        "region": ["year", "region_code", "region_name"],
    }.items():
        coverage_rows(sliced, dimensions).to_csv(out / f"coverage_by_{name}.csv", index=False)
    patterns, joint = [], []
    for sample, mask in sample_masks(features).items():
        group = features.loc[mask].copy()
        for mode, suffix in [("raw", "thousand_rubles"), ("feature", "log_rubles_per_person")]:
            pattern = pd.Series("", index=group.index)
            for prefix, _ in SPECS.values():
                pattern += group[f"{prefix}_{suffix}"].notna().astype(int).astype(str)
            group["availability_pattern"] = pattern
            counts = group.groupby(["year", "availability_pattern"]).size()
            for (year, pattern), count in counts.items():
                patterns.append(
                    dict(
                        sample=sample,
                        mode=mode,
                        year=year,
                        pattern=pattern,
                        count=int(count),
                        denominator=int(group.year.eq(year).sum()),
                    )
                )
        for year, part in group.groupby("year"):
            joint.append(
                dict(
                    sample=sample,
                    year=year,
                    denominator=len(part),
                    activity_all3=int(part.activity_all3_features_available.sum()),
                    sector_all8=int(part.sector_selected8_complete.sum()),
                    activity3_and_sector8=int(
                        (
                            part.activity_all3_features_available & part.sector_selected8_complete
                        ).sum()
                    ),
                )
            )
    pd.DataFrame(patterns).to_csv(out / "missingness_patterns.csv", index=False)
    pd.DataFrame(joint).to_csv(out / "joint_coverage.csv", index=False)
    missing = economy.loc[economy.value.isna()].copy()
    missing = missing.merge(
        base[
            KEY
            + ["region_name", "municipal_district_name", "core_eligible", "conservative_eligible"]
        ],
        on=KEY,
        validate="many_to_one",
    )
    missing.to_csv(out / "unavailable_indicator_rows.csv", index=False)
    source = pd.read_csv(inputs["source_cells"], dtype=string_types)
    outside = pd.read_csv(inputs["source_only"], dtype=string_types)
    source_key = ["source_region_url", "source_oktmo", "year", "indicator_id"]
    check_unique(source, source_key, "raw parsed source cells")
    computed_outside = source.merge(
        economy[source_key], on=source_key, how="left", validate="one_to_one", indicator=True
    )
    computed_outside = computed_outside.loc[computed_outside._merge.eq("left_only")].drop(
        columns="_merge"
    )
    pd.testing.assert_frame_equal(
        outside.sort_values(source_key).reset_index(drop=True),
        computed_outside[outside.columns].sort_values(source_key).reset_index(drop=True),
        check_dtype=False,
    )
    outside["refusal_reason"] = "no_exact_source_code_link_for_this_observation_year"
    other_year_links = economy[["source_region_url", "source_oktmo", "territory_id", "year"]]
    other_year_links = other_year_links.drop_duplicates()
    refs = (
        other_year_links.groupby(["source_region_url", "source_oktmo"])
        .apply(
            lambda x: json.dumps(x[["territory_id", "year"]].to_dict("records")),
            include_groups=False,
        )
        .rename("existing_links_other_year_for_review_only")
        .reset_index()
    )
    outside = outside.merge(
        refs, on=["source_region_url", "source_oktmo"], how="left", validate="many_to_one"
    )
    outside.to_csv(out / "historical_source_only_29.csv", index=False)
    statuses = json.loads(inputs["collection_status"].read_text("utf-8"))
    rejected = [x for x in statuses if x["status"] != "parsed"]
    pd.DataFrame(rejected, columns=["key", "status", "codes", "error"]).to_csv(
        out / "collection_refusals.csv", index=False
    )
    tests = contract_tests()
    qa = dict(
        status="PASS",
        rows=len(features),
        municipalities=features.territory_id.nunique(),
        input_economy_rows=len(economy),
        parsed_source_cells=len(source),
        linked_source_cells=int(economy.source_sha256.notna().sum()),
        historical_source_only_rows=len(outside),
        historical_source_only_numeric=int(outside.value.notna().sum()),
        numeric_values=int(economy.value.notna().sum()),
        source_missing=int(economy.value.isna().sum()),
        zero_values=int(economy.value.eq(0).sum()),
        negative_values=int(economy.value.lt(0).sum()),
        unavailable_statuses=missing.value_status.value_counts().to_dict(),
        collection_status_counts=pd.Series([x["status"] for x in statuses])
        .value_counts()
        .to_dict(),
        frozen_core_rows=int(base.core_eligible.sum()),
        frozen_conservative_rows=int(base.conservative_eligible.sum()),
        frozen_columns_and_order_unchanged=True,
        csv_parquet_equal=True,
        exact_year_code_agreement=True,
        joins_validate_one_to_one=True,
        named_contract_tests=tests,
        missing_imputed=0,
        geographic_equivalence_proven=0,
        log_features_are_activity_intensity_not_income=True,
        point_in_time_claim=False,
        independent_2024_holdout=False,
    )
    if len(outside) != 29 or len(economy) != 12096:
        raise ValueError("Frozen source accounting changed")
    for name, path in inputs.items():
        if sha(path) != input_hashes[name]:
            raise ValueError("Input changed during audit")
    write_json(out / "QA.json", qa)
    dictionary = dict(
        key=KEY,
        row_grain="frozen municipality-year",
        currency="nominal thousand rubles",
        denominator="frozen population_total at January 1 of observation year",
        indicators={
            code: dict(
                prefix=p,
                annual_period=period,
                activity_total="101",
                scope="organizations excluding small entrepreneurship",
            )
            for code, (p, period) in SPECS.items()
        },
        feature_formula="log(value*1000/population_total), only finite positive operands",
        missing_flag="raw source value absent; not a revised cohort eligibility flag",
        feature_missing="includes absent/nonpositive values and absent/nonpositive population",
        patterns="order: investment, shipments, retail; 1=available, 0=missing",
        sector_columns_used_only_for_joint_coverage=SELECTED,
        unit_conversion="1000 rubles per source unit",
        inflation_adjustment="none",
        boundary_equivalence="unknown; inherited frozen flags not changed",
        publication_vintage="retrieved 2026-09-16; first publication/revisions unknown",
    )
    write_json(out / "feature_dictionary.json", dictionary)
    write_json(
        out / "manifest.json",
        dict(
            script_sha256=sha(Path(__file__)),
            lock_sha256=sha(repo / "uv.lock"),
            input_files={
                name: {"repo_path": path.relative_to(repo).as_posix(), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            output_hashes={
                path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()
            },
        ),
    )
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(contract_tests()))
    else:
        out = args.out or args.repo / "reports/round3_analysis/audit"
        print(json.dumps(run(args.repo.resolve(), out), ensure_ascii=True))


if __name__ == "__main__":
    main()
