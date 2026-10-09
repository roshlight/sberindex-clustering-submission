import json
import runpy
from pathlib import Path

import duckdb
import pandas as pd
import pytest

prepare = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_core_context.py"))[
    "prepare"
]


def _parquet(path, frame):
    with duckdb.connect() as connection:
        connection.register("input_frame", frame)
        connection.execute("COPY input_frame TO ? (FORMAT PARQUET)", [str(path)])
    return path


def _inputs(tmp_path, *, market_rows=None, crosswalk_rows=None):
    assignments = _parquet(
        tmp_path / "assignments.parquet",
        pd.DataFrame({"territory_id": [2, 1, 2, 3], "month": ["2023-01"] * 4}),
    )
    crosswalk = tmp_path / "crosswalk.csv"
    rows = crosswalk_rows or [
        {
            "territory_id": number,
            "region_name": "Тестовый регион",
            "oktmo": f"01-000-000-00{number}",
            "municipal_district_type": "муниципальный округ",
            "snapshot_date": "2024-01-01",
        }
        for number in [1, 2, 3, 4]
    ]
    pd.DataFrame(rows).to_csv(crosswalk, index=False, encoding="utf-8")
    market = _parquet(
        tmp_path / "market.parquet",
        pd.DataFrame(
            market_rows
            if market_rows is not None
            else {"territory_id": [1, 2, 4], "market_access": [0.0, None, 50.0]}
        ),
    )
    return assignments, crosswalk, market


def test_exact_cohort_left_join_preserves_zero_null_and_absent_source_row(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "result"
    manifest = prepare(*paths, output, expected_count=3)
    with duckdb.connect() as connection:
        frame = connection.read_parquet(str(output / "municipal_context_2024.parquet")).df()
    assert frame.territory_id.tolist() == [1, 2, 3]
    assert len(frame) == 3
    assert frame.market_access.iloc[0] == 0.0
    assert frame.market_access.iloc[1:].isna().all()
    assert frame.region.tolist() == ["Тестовый регион"] * 3
    assert frame.oktmo.iloc[0] == "01-000-000-001"
    assert frame.snapshot_date.dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-01-01"]
    assert frame.measurement_year.unique().tolist() == [2024]
    assert "month" not in frame
    missing = pd.read_csv(output / "market_access_missing_ids.csv")
    assert missing.territory_id.tolist() == [2, 3]
    assert missing.missing_reason.tolist() == ["source_value_null", "source_row_absent"]
    counts = manifest["counts"]
    assert counts["assignment_rows"] == 4
    assert counts["crosswalk_outside_cohort_ids"] == 1
    assert counts["market_access_outside_cohort_ids"] == 1
    assert counts["market_access_matched_ids"] == 2
    assert counts["market_access_observed_ids"] == 1
    assert counts["market_access_absent_source_rows"] == 1
    assert counts["market_access_matched_null_values"] == 1
    saved = json.loads((output / "core_context_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert all(len(source["sha256"]) == 64 for source in saved["sources"].values())
    assert saved["qa"]["imputed_values"] == 0


@pytest.mark.parametrize("source", ["crosswalk", "market_access"])
def test_duplicate_right_keys_outside_cohort_refuse_join(tmp_path, source):
    paths = _inputs(tmp_path)
    if source == "crosswalk":
        frame = pd.read_csv(paths[1], dtype="string")
        pd.concat([frame, frame.tail(1)]).to_csv(paths[1], index=False)
    else:
        _parquet(
            paths[2],
            pd.DataFrame({"territory_id": [1, 4, 4], "market_access": [10.0, 20.0, 20.0]}),
        )
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=f"{source}: duplicate territory_id"):
        prepare(*paths, output, expected_count=3)
    assert not output.exists()


def test_crosswalk_must_cover_all_cohort_ids(tmp_path):
    paths = _inputs(tmp_path)
    frame = pd.read_csv(paths[1])
    frame.loc[frame.territory_id.ne(3)].to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="incomplete coverage"):
        prepare(*paths, tmp_path / "result", expected_count=3)


def test_mixed_snapshot_dates_refuse_implicit_time_selection(tmp_path):
    paths = _inputs(tmp_path)
    frame = pd.read_csv(paths[1])
    frame.loc[0, "snapshot_date"] = "2023-01-01"
    frame.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="snapshot_date 2024-01-01"):
        prepare(*paths, tmp_path / "result", expected_count=3)


def test_unexpected_cohort_size_and_existing_output_are_rejected(tmp_path):
    paths = _inputs(tmp_path)
    with pytest.raises(ValueError, match="expected_count 4"):
        prepare(*paths, tmp_path / "result", expected_count=4)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="not overwritten"):
        prepare(*paths, output, expected_count=3)


def test_null_join_key_is_not_used_as_a_matching_value(tmp_path):
    paths = _inputs(tmp_path)
    frame = pd.read_csv(paths[1], dtype="string")
    frame.loc[0, "territory_id"] = pd.NA
    frame.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="nonnull positive integer"):
        prepare(*paths, tmp_path / "result", expected_count=3)
