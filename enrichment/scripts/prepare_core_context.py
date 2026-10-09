"""Build one 2024 context row per exact study-cohort territory; no imputation."""

import argparse
import hashlib
import json
import math
import platform
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import duckdb
import pandas as pd

SNAPSHOT_DATE = "2024-01-01"
MEASUREMENT_YEAR = 2024
CROSSWALK_FIELDS = {
    "region_name": "region",
    "oktmo": "oktmo",
    "municipal_district_type": "type",
    "snapshot_date": "snapshot_date",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(frame, source, *, unique):
    if "territory_id" not in frame:
        raise ValueError(f"{source}: missing territory_id")
    values = frame.territory_id.astype("string")
    if values.isna().any() or not values.str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError(f"{source}: territory_id must be a nonnull positive integer")
    try:
        frame["territory_id"] = values.astype("int64")
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{source}: territory_id exceeds integer range") from None
    if unique and frame.territory_id.duplicated().any():
        raise ValueError(f"{source}: duplicate territory_id; one-to-one join refused")
    return frame


def prepare(assignments, crosswalk, market_access, output, expected_count=2016):
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing outputs are not overwritten")
    sources = {
        "assignments": Path(assignments).resolve(),
        "crosswalk": Path(crosswalk).resolve(),
        "market_access": Path(market_access).resolve(),
    }
    source_info = {
        name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in sources.items()
    }
    with duckdb.connect() as connection:
        assignment_ids = connection.execute(
            "SELECT territory_id FROM read_parquet(?)", [str(sources["assignments"])]
        ).df()
        market = connection.execute(
            "SELECT territory_id, market_access FROM read_parquet(?)",
            [str(sources["market_access"])],
        ).df()
    assignment_ids = _ids(assignment_ids, "assignments", unique=False)
    cohort = assignment_ids.drop_duplicates().sort_values("territory_id").reset_index(drop=True)
    if len(cohort) != expected_count:
        raise ValueError(f"cohort count {len(cohort)} differs from expected_count {expected_count}")
    mapping = pd.read_csv(sources["crosswalk"], dtype="string", encoding="utf-8")
    mapping = _ids(mapping, "crosswalk", unique=True)
    missing_columns = set(CROSSWALK_FIELDS) - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"crosswalk: missing required columns {sorted(missing_columns)}")
    if mapping.snapshot_date.isna().any() or not mapping.snapshot_date.eq(SNAPSHOT_DATE).all():
        raise ValueError(f"crosswalk: every row must have snapshot_date {SNAPSHOT_DATE}")
    market = _ids(market, "market_access", unique=True)
    try:
        market["market_access"] = pd.to_numeric(market.market_access, errors="raise").astype(
            "Float64"
        )
    except (ValueError, TypeError):
        raise ValueError("market_access: values must be numeric or null") from None
    if not market.market_access.dropna().map(math.isfinite).all():
        raise ValueError("market_access: infinite values are invalid")
    renamed = mapping[["territory_id", *CROSSWALK_FIELDS]].rename(columns=CROSSWALK_FIELDS)
    context = cohort.merge(
        renamed, on="territory_id", how="left", validate="one_to_one", indicator=True
    )
    if not context._merge.eq("both").all():
        raise ValueError("crosswalk: incomplete coverage of the study cohort")
    context = context.drop(columns="_merge")
    for field in CROSSWALK_FIELDS.values():
        if context[field].isna().any() or context[field].str.strip().eq("").any():
            raise ValueError(f"crosswalk: null or empty {field} in the study cohort")
    context = context.merge(
        market, on="territory_id", how="left", validate="one_to_one", indicator=True
    )
    unmatched = context._merge.eq("left_only")
    missing = context.market_access.isna()
    missing_ids = context.loc[missing, ["territory_id", "region", "oktmo", "type"]].copy()
    missing_ids["missing_reason"] = "source_value_null"
    missing_ids.loc[unmatched.loc[missing_ids.index], "missing_reason"] = "source_row_absent"
    context = context.drop(columns="_merge")
    context["snapshot_date"] = date.fromisoformat(SNAPSHOT_DATE)
    context["measurement_year"] = pd.Series(MEASUREMENT_YEAR, index=context.index, dtype="int16")
    if len(context) != expected_count or not context.territory_id.equals(cohort.territory_id):
        raise ValueError("join changed the exact cohort or multiplied rows")
    # Do not attest a hash for a different file version than the one read above.
    for name, path in sources.items():
        if sha256(path) != source_info[name]["sha256"]:
            raise ValueError(f"{name}: input changed during preparation")

    counts = {
        "assignment_rows": len(assignment_ids),
        "cohort_ids": len(cohort),
        "crosswalk_source_rows": len(mapping),
        "crosswalk_matched_ids": len(context),
        "crosswalk_outside_cohort_ids": len(mapping) - len(context),
        "market_access_source_rows": len(market),
        "market_access_matched_ids": int((~unmatched).sum()),
        "market_access_outside_cohort_ids": len(market) - int((~unmatched).sum()),
        "market_access_observed_ids": int((~missing).sum()),
        "market_access_missing_ids": int(missing.sum()),
        "market_access_absent_source_rows": int(unmatched.sum()),
        "market_access_matched_null_values": int((missing & ~unmatched).sum()),
        "output_rows": len(context),
    }
    output.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as connection:
        connection.register("context_frame", context)
        connection.execute(
            "COPY context_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(output / "municipal_context_2024.parquet")],
        )
    missing_ids.to_csv(output / "market_access_missing_ids.csv", index=False, encoding="utf-8")
    project = Path(__file__).resolve().parent.parent
    manifest = {
        "status": "ready_with_conditions" if missing.any() else "ready",
        "created_at": datetime.now(UTC).isoformat(),
        "operation": "exact_cohort_left_one_to_one_2024_context_no_imputation",
        "expected_count": expected_count,
        "snapshot_date": SNAPSHOT_DATE,
        "measurement_year": MEASUREMENT_YEAR,
        "conditions": [
            "Annual 2024 contextual indicator only; no broadcast to 2023 or monthly observations.",
            "Snapshot and measurement dates do not establish historical publication availability.",
            "Missing market_access values remain null; not zero and not inferred from neighbours.",
            "This table adds context; it does not change baseline features or establish causality.",
        ],
        "counts": counts,
        "coverage": {
            "crosswalk_fraction": counts["crosswalk_matched_ids"] / expected_count,
            "market_access_observed_fraction": counts["market_access_observed_ids"]
            / expected_count,
        },
        "qa": {
            "cohort_identity_preserved": True,
            "duplicate_right_keys": 0,
            "null_join_keys": 0,
            "crosswalk_snapshot_validated": True,
            "left_join_cardinality": "one_to_one",
            "imputed_values": 0,
        },
        "sources": source_info,
        "source_field_mapping": CROSSWALK_FIELDS,
        "processor_sha256": sha256(__file__),
        "environment": {
            "python": platform.python_version(),
            "duckdb": version("duckdb"),
            "pandas": version("pandas"),
            "pyproject_sha256": sha256(project / "pyproject.toml"),
            "uv_lock_sha256": sha256(project / "uv.lock"),
        },
        "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "core_context_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--market-access", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2016)
    args = parser.parse_args(argv)
    manifest = prepare(
        args.assignments, args.crosswalk, args.market_access, args.output, args.expected_count
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
