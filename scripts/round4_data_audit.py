"""Compare recovered activity data with round three on immutable annual cohorts.

This is retrospective economic description, not a historically available forecast
dataset. All joins are exact municipality/year links from the accepted candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from round3_data_audit import (
    KEY,
    SELECTED,
    SPECS,
    build_features,
    check_unique,
    coverage_rows,
    sample_masks,
    sha,
    write_json,
)

CELL_KEY = KEY + ["indicator_id"]
SOURCE_KEY = ["source_region_url", "source_oktmo", "year", "indicator_id"]
STRINGS = {
    key: str
    for key in [
        "official_oktmo_8",
        "source_oktmo",
        "indicator_id",
        "period_code",
        "activity_code",
        "region_code",
    ]
}
PREFIXES = [item[0] for item in SPECS.values()]
STRINGS.update(
    {
        f"{prefix}_{field}": str
        for prefix in PREFIXES
        for field in ["official_oktmo_8", "source_oktmo"]
    }
)
FROZEN_FLAGS = ["core_eligible", "conservative_eligible", "code_type_changed", "calendar_review"]


def sorted_cells(frame: pd.DataFrame) -> pd.DataFrame:
    check_unique(frame, CELL_KEY, "indicator cells")
    return frame.sort_values(CELL_KEY).reset_index(drop=True)


def compare_candidates(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old, new = sorted_cells(old), sorted_cells(new)
    identity = CELL_KEY + [
        "source_region_url",
        "source_oktmo",
        "official_oktmo_8",
        "code_reference_date",
        "temporal_review_required",
    ]
    pd.testing.assert_frame_equal(old[identity], new[identity], check_dtype=False)
    old_numeric = old.value.notna()
    if not new.loc[old_numeric, "value"].equals(old.loc[old_numeric, "value"]):
        raise ValueError("An old numeric cell was changed or removed")
    pd.testing.assert_frame_equal(
        old.loc[old_numeric, old.columns], new.loc[old_numeric, old.columns], check_dtype=False
    )
    result = old[identity].copy()
    result["old_value"] = old.value
    result["new_value"] = new.value
    result["old_status"] = old.value_status
    result["new_status"] = new.value_status
    result["old_source_sha256"] = old.source_sha256
    result["new_source_sha256"] = new.source_sha256
    result["numeric_gain"] = old.value.isna() & new.value.notna()
    result["old_numeric_preserved"] = old_numeric
    result["old_missing_still_missing"] = old.value.isna() & new.value.isna()
    result["status_changed"] = old.value_status.ne(new.value_status)
    return result


def compare_csv_parquet(csv: pd.DataFrame, path: Path) -> None:
    with duckdb.connect() as db:
        parquet = db.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
    if list(csv.columns) != list(parquet.columns):
        raise ValueError("CSV/Parquet schema mismatch")
    for column in csv:
        left, right = csv[column], parquet[column]
        if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
            np.testing.assert_allclose(
                pd.to_numeric(left).astype(float),
                pd.to_numeric(right).astype(float),
                equal_nan=True,
                rtol=0,
                atol=0,
            )
        else:
            if column in STRINGS:
                right = right.astype("string")
            pd.testing.assert_series_equal(
                left.fillna("").astype(str), right.fillna("").astype(str), check_names=False
            )


def add_prior_compatible_columns(features: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    check_unique(sector, KEY, "sector panel")
    matched = features[KEY].merge(
        sector[KEY + SELECTED], on=KEY, how="left", validate="one_to_one", indicator=True
    )
    if not matched._merge.eq("both").all():
        raise ValueError("Sector panel lacks frozen annual keys")
    result = features.copy()
    result["sector_selected8_available_count"] = matched[SELECTED].notna().sum(axis=1)
    result["sector_selected8_complete"] = result.sector_selected8_available_count.eq(8)
    result["activity_all3_features_available"] = (
        result[[f"{prefix}_log_rubles_per_person" for prefix in PREFIXES]].notna().all(axis=1)
    )
    result["activity_vintage"] = "retrieved 2026-09-16; first publication/revisions unknown"
    result["activity_boundary_equivalence_proven"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    return result


def add_comparison_flags(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    pd.testing.assert_frame_equal(old[KEY + FROZEN_FLAGS], new[KEY + FROZEN_FLAGS])
    result = new.copy()
    for prefix in PREFIXES:
        for suffix in ["thousand_rubles", "log_rubles_per_person", "missing", "feature_status"]:
            result[f"old_{prefix}_{suffix}"] = old[f"{prefix}_{suffix}"]
        result[f"{prefix}_new_numeric"] = (
            old[f"{prefix}_thousand_rubles"].isna() & new[f"{prefix}_thousand_rubles"].notna()
        )
        result[f"{prefix}_new_feature"] = (
            old[f"{prefix}_log_rubles_per_person"].isna()
            & new[f"{prefix}_log_rubles_per_person"].notna()
        )
    result["activity_new_numeric_count"] = result[
        [f"{prefix}_new_numeric" for prefix in PREFIXES]
    ].sum(axis=1)
    result["activity_new_numeric_any"] = result.activity_new_numeric_count.gt(0)
    result["activity_new_feature_count"] = result[
        [f"{prefix}_new_feature" for prefix in PREFIXES]
    ].sum(axis=1)
    for version, frame in [("old", old), ("new", new)]:
        result[f"activity_numeric_count_{version}"] = (
            frame[[f"{prefix}_thousand_rubles" for prefix in PREFIXES]].notna().sum(axis=1)
        )
        result[f"activity_feature_count_{version}"] = (
            frame[[f"{prefix}_log_rubles_per_person" for prefix in PREFIXES]].notna().sum(axis=1)
        )
    result["activity_all3_features_available_old"] = old.activity_all3_features_available
    result["activity_all3_features_common"] = (
        old.activity_all3_features_available & new.activity_all3_features_available
    )
    result["activity_all3_features_newly_complete"] = (
        ~old.activity_all3_features_available & new.activity_all3_features_available
    )
    # Common complete rows are a coverage sensitivity analysis, not a revised core cohort.
    result["core_common_activity_complete"] = (
        result.core_eligible & result.activity_all3_features_common
    )
    result["conservative_common_activity_complete"] = (
        result.conservative_eligible & result.activity_all3_features_common
    )
    result["population_bin"] = (
        pd.cut(
            result.population_total,
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
    return result


def gain_rows(features: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows = []
    for sample, mask in sample_masks(features).items():
        for values, group in features.loc[mask].groupby(dimensions, dropna=False, observed=True):
            values = values if isinstance(values, tuple) else (values,)
            key = dict(zip(dimensions, values, strict=True))
            for prefix in PREFIXES:
                old_value = group[f"old_{prefix}_thousand_rubles"]
                new_value = group[f"{prefix}_thousand_rubles"]
                old_feature = group[f"old_{prefix}_log_rubles_per_person"]
                new_feature = group[f"{prefix}_log_rubles_per_person"]
                rows.append(
                    dict(
                        sample=sample,
                        **key,
                        prefix=prefix,
                        denominator=len(group),
                        distinct_territories=group.territory_id.nunique(),
                        old_numeric=int(old_value.notna().sum()),
                        new_numeric=int(new_value.notna().sum()),
                        numeric_gain=int(group[f"{prefix}_new_numeric"].sum()),
                        old_feature_available=int(old_feature.notna().sum()),
                        new_feature_available=int(new_feature.notna().sum()),
                        feature_gain=int(group[f"{prefix}_new_feature"].sum()),
                        old_missing=int(old_value.isna().sum()),
                        new_missing=int(new_value.isna().sum()),
                        numeric_gain_fraction=float(group[f"{prefix}_new_numeric"].mean()),
                    )
                )
    return pd.DataFrame(rows)


def run(repo: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    inputs = {
        "frozen_panel": repo / "reports/enriched_analysis/core_v2/annual_panel.csv",
        "old_candidate": repo / "reports/data_round3/economy/annual_candidate_panel.csv",
        "new_candidate": repo / "reports/data_round4/economy/annual_candidate_panel.csv",
        "new_parquet": repo / "reports/data_round4/economy/annual_candidate_panel.parquet",
        "prior_features": repo / "reports/round3_analysis/audit/annual_features.csv",
        "sector": repo / "reports/round2_analysis/sector/feature_panel.csv",
        "recovered_source": repo / "reports/data_round4/economy/recovered_source_cells.csv",
        "source_only": repo / "reports/data_round4/economy/source_only_outside_year_link.csv",
        "collector_delta": repo / "reports/data_round4/economy/delta_all_cells.csv",
        "independent_collection_qa": repo / "reports/data_round4/economy/independent_QA.json",
    }
    hashes = {name: sha(path) for name, path in inputs.items()}
    tables = {
        name: pd.read_csv(path, dtype=STRINGS)
        for name, path in inputs.items()
        if path.suffix == ".csv"
    }
    base, old, new = (tables[k] for k in ["frozen_panel", "old_candidate", "new_candidate"])
    if len(base) != 4032 or base.territory_id.nunique() != 2016 or set(base.year) != {2023, 2024}:
        raise ValueError("Frozen cohort shape changed")
    for flag in FROZEN_FLAGS:
        if base[flag].dtype != bool or base[flag].isna().any():
            raise ValueError("Frozen mask is not a complete boolean")
    compare_csv_parquet(new, inputs["new_parquet"])
    delta = compare_candidates(old, new)
    recovered = tables["recovered_source"]
    check_unique(recovered, SOURCE_KEY, "recovered source")
    gains = delta.loc[delta.numeric_gain].merge(
        recovered[SOURCE_KEY + ["value", "source_sha256"]],
        on=SOURCE_KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not gains._merge.eq("both").all():
        raise ValueError("Gain lacks recovered source")
    np.testing.assert_array_equal(gains.new_value, gains.value)
    if not gains.new_source_sha256.eq(gains.source_sha256).all():
        raise ValueError("Gain provenance differs from recovered source")
    collector_delta = sorted_cells(tables["collector_delta"])
    if not delta.numeric_gain.equals(collector_delta.numeric_gain):
        raise ValueError("Computed gains disagree with collector")
    pd.testing.assert_series_equal(delta.old_value, collector_delta.old_value, check_names=False)
    pd.testing.assert_series_equal(
        delta.new_value, collector_delta.candidate_value, check_names=False
    )
    linked = recovered.merge(
        new[SOURCE_KEY], on=SOURCE_KEY, how="left", validate="one_to_one", indicator=True
    )
    outside = linked.loc[linked._merge.eq("left_only"), recovered.columns]
    pd.testing.assert_frame_equal(
        outside.sort_values(SOURCE_KEY).reset_index(drop=True),
        tables["source_only"].sort_values(SOURCE_KEY).reset_index(drop=True),
        check_dtype=False,
    )
    old_features = add_prior_compatible_columns(build_features(base, old), tables["sector"])
    prior = tables["prior_features"]
    if list(prior.columns) != list(old_features.columns):
        raise ValueError("Prior feature schema changed")
    # CSV null extension dtypes differ from in-memory pd.NA only.
    pd.testing.assert_frame_equal(
        old_features.astype(object).where(old_features.notna(), None),
        prior.astype(object).where(prior.notna(), None),
        check_dtype=False,
        rtol=1e-14,
        atol=0,
    )
    new_features = add_prior_compatible_columns(build_features(base, new), tables["sector"])
    features = add_comparison_flags(old_features, new_features)
    pd.testing.assert_frame_equal(features[base.columns], base)
    old_available = old_features[[f"{prefix}_log_rubles_per_person" for prefix in PREFIXES]]
    new_available = features[old_available.columns]
    np.testing.assert_allclose(
        new_available.where(old_available.notna()), old_available, equal_nan=True, rtol=0, atol=0
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
        json.dumps({"features_ready": str(out / "annual_features.csv"), "rows": len(features)}),
        flush=True,
    )
    delta = delta.merge(
        base[
            KEY
            + [
                "region_code",
                "region_name",
                "municipal_district_name",
                "municipal_district_type",
                "population_total",
                *FROZEN_FLAGS,
            ]
        ],
        on=KEY,
        how="left",
        validate="many_to_one",
    )
    delta.to_csv(out / "cell_comparison.csv", index=False)
    delta.loc[delta.numeric_gain].to_csv(out / "new_numeric_cells.csv", index=False)
    delta.loc[delta.new_value.isna()].to_csv(out / "remaining_missing_cells.csv", index=False)
    features.loc[features.activity_new_numeric_any].to_csv(
        out / "affected_municipality_years.csv", index=False
    )
    for name, dimensions in {
        "overall": ["year"],
        "region": ["year", "region_code", "region_name"],
        "population": ["year", "population_bin"],
        "type": ["year", "municipal_district_type"],
        "restrictions": ["year", "code_type_changed", "calendar_review"],
    }.items():
        gain_rows(features, dimensions).to_csv(out / f"gain_by_{name}.csv", index=False)
        coverage_rows(features, dimensions).to_csv(out / f"coverage_by_{name}.csv", index=False)
    joint, transitions = [], []
    for sample, mask in sample_masks(features).items():
        for year, group in features.loc[mask].groupby("year"):
            joint.append(
                dict(
                    sample=sample,
                    year=year,
                    denominator=len(group),
                    old_all3=int(group.activity_all3_features_available_old.sum()),
                    new_all3=int(group.activity_all3_features_available.sum()),
                    common_all3=int(group.activity_all3_features_common.sum()),
                    newly_complete=int(group.activity_all3_features_newly_complete.sum()),
                    affected_rows=int(group.activity_new_numeric_any.sum()),
                    gained_numeric_cells=int(group.activity_new_numeric_count.sum()),
                    gained_features=int(group.activity_new_feature_count.sum()),
                    all8_sector_and_all3_new=int(
                        (
                            group.sector_selected8_complete & group.activity_all3_features_available
                        ).sum()
                    ),
                )
            )
            for mode in ["numeric", "feature"]:
                counts = group.groupby(
                    [f"activity_{mode}_count_old", f"activity_{mode}_count_new"]
                ).size()
                for (old_count, new_count), count in counts.items():
                    transitions.append(
                        dict(
                            sample=sample,
                            year=year,
                            mode=mode,
                            old_available_count=old_count,
                            new_available_count=new_count,
                            count=int(count),
                            denominator=len(group),
                        )
                    )
    pd.DataFrame(joint).to_csv(out / "joint_coverage.csv", index=False)
    pd.DataFrame(transitions).to_csv(out / "availability_transitions.csv", index=False)
    masks = features[
        KEY
        + FROZEN_FLAGS
        + [
            "activity_all3_features_available_old",
            "activity_all3_features_available",
            "activity_all3_features_common",
            "activity_all3_features_newly_complete",
            "activity_new_numeric_any",
            "core_common_activity_complete",
            "conservative_common_activity_complete",
        ]
    ]
    masks.to_csv(out / "sample_membership.csv", index=False)
    qa = dict(
        status="PASS",
        readiness="ready_with_conditions",
        rows=len(features),
        municipalities=features.territory_id.nunique(),
        cells=len(delta),
        old_numeric=int(old.value.notna().sum()),
        new_numeric=int(new.value.notna().sum()),
        numeric_gain=int(delta.numeric_gain.sum()),
        unchanged_old_numeric=int(old.value.notna().sum()),
        numeric_revisions=0,
        numeric_losses=0,
        remaining_missing=int(new.value.isna().sum()),
        missing_statuses=new.loc[new.value.isna(), "value_status"].value_counts().to_dict(),
        recovered_source_rows=len(recovered),
        source_only_rows=len(outside),
        source_only_numeric=int(outside.value.notna().sum()),
        zero_values=int(new.value.eq(0).sum()),
        negative_values=int(new.value.lt(0).sum()),
        affected_municipality_years=int(features.activity_new_numeric_any.sum()),
        affected_distinct_municipalities=int(
            features.loc[features.activity_new_numeric_any, "territory_id"].nunique()
        ),
        gained_log_features=int(features.activity_new_feature_count.sum()),
        gained_numeric_without_valid_log_feature=int(
            (features.activity_new_numeric_count - features.activity_new_feature_count).sum()
        ),
        frozen_core_rows=int(base.core_eligible.sum()),
        frozen_conservative_rows=int(base.conservative_eligible.sum()),
        frozen_code_changed_territories=int(
            base.loc[base.code_type_changed, "territory_id"].nunique()
        ),
        frozen_calendar_territories=int(base.loc[base.calendar_review, "territory_id"].nunique()),
        frozen_columns_order_and_masks_unchanged=True,
        prior_feature_reconstruction_equal=True,
        all_old_numeric_values_and_provenance_preserved=True,
        csv_parquet_equal=True,
        exact_year_code_agreement=True,
        joins_one_to_one=True,
        missing_imputed=0,
        numeric_source_revision_stability="not_estimable_no_recaptured_old_numeric_overlap",
        geographic_equivalence_proven=0,
        point_in_time_claim=False,
        model_fits=0,
        independent_2024_holdout=False,
    )
    if (qa["old_numeric"], qa["numeric_gain"], qa["new_numeric"], qa["cells"]) != (
        10829,
        779,
        11608,
        12096,
    ):
        raise ValueError("Source snapshot accounting changed")
    for name, path in inputs.items():
        if sha(path) != hashes[name]:
            raise ValueError("Input changed during audit")
    write_json(out / "QA.json", qa)
    write_json(
        out / "feature_dictionary.json",
        dict(
            key=KEY,
            row_grain="frozen municipality-year",
            prefix_order=PREFIXES,
            monetary_unit="nominal thousand rubles; excludes small entrepreneurship",
            denominator="frozen population_total at January 1 of observation year",
            formula="log(value*1000/population_total) for finite positive operands only",
            old_prefix="round-three values and availability; same accepted year code",
            new_numeric="previously missing raw value now numeric; zero is not missing",
            new_feature="previously unavailable log intensity now valid",
            common_complete=(
                "old and new three log features all available; coverage sensitivity only"
            ),
            core_and_conservative="unchanged frozen cohort masks, independent of new coverage",
            point_in_time="retrieved 2026-09-16; original publication/revisions unknown",
            geometry="code and year link only; legal boundary equivalence not proven",
            inference="collection failures concentrated by region; gains are not random",
        ),
    )
    write_json(
        out / "manifest.json",
        dict(
            script_sha256=sha(Path(__file__)),
            helper_sha256=sha(repo / "scripts/round3_data_audit.py"),
            lock_sha256=sha(repo / "uv.lock"),
            input_files={
                name: dict(repo_path=path.relative_to(repo).as_posix(), sha256=hashes[name])
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
    args = parser.parse_args()
    repo = args.repo.resolve()
    print(json.dumps(run(repo, args.out or repo / "reports/round4_analysis/audit")))


if __name__ == "__main__":
    main()
