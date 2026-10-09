"""Data-contract tests that target identity, missingness and profile semantics."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from municipal_profiles.data import (
    ARITHMETIC_RESIDUAL,
    EXPECTED_PERIODS,
    PUBLISHED_CATEGORIES,
    TOTAL_CATEGORY,
    load_panel,
    make_profiles,
)

PERIODS = ("2023-01", "2023-02")


def _panel(periods=PERIODS):
    rows = []
    for month in reversed(periods):
        for territory_id in (10, 2):
            for category, value in zip(PUBLISHED_CATEGORIES, (1, 2, 3, 4, 5), strict=True):
                rows.append((month, territory_id, category, value))
            rows.append((month, territory_id, TOTAL_CATEGORY, 30))
    return pd.DataFrame(rows, columns=["date", "territory_id", "category", "value"])


def _save(tmp_path, frame):
    path = tmp_path / "consumption.parquet"
    duckdb.from_df(frame).write_parquet(str(path))
    return path


def test_original_default_calendar_and_sorted_typed_output(tmp_path):
    result = load_panel(_save(tmp_path, _panel(EXPECTED_PERIODS)))
    assert result.shape == (2 * 24 * 6, 4)
    assert list(result.columns) == ["date", "territory_id", "category", "value"]
    assert result["value"].dtype == np.dtype("float64")
    assert result["territory_id"].dtype == np.dtype("int64")
    assert result["date"].iloc[0] == "2023-01"
    assert result["territory_id"].iloc[0] == 2


def test_duplicate_observation_cannot_be_silently_deduplicated(tmp_path):
    frame = _panel()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        load_panel(_save(tmp_path, frame), expected_periods=PERIODS)


def test_missing_category_in_one_month_is_not_a_zero(tmp_path):
    frame = _panel().iloc[1:].copy()
    with pytest.raises(ValueError, match="all 6 categories"):
        load_panel(_save(tmp_path, frame), complete_only=False, expected_periods=PERIODS)


@pytest.mark.parametrize("invalid_value", [np.nan, 0, -1, np.inf])
def test_invalid_expenditure_values_are_rejected(tmp_path, invalid_value):
    frame = _panel().astype({"value": float})
    frame.loc[0, "value"] = invalid_value
    with pytest.raises(ValueError, match="Null|positive"):
        load_panel(_save(tmp_path, frame), expected_periods=PERIODS)


def test_incomplete_history_preserved_or_explicitly_filtered(tmp_path):
    frame = _panel()
    frame = frame.loc[~((frame.territory_id == 10) & (frame.date == "2023-01"))]
    path = _save(tmp_path, frame)
    full = load_panel(path, complete_only=False, expected_periods=PERIODS)
    balanced = load_panel(path, complete_only=True, expected_periods=PERIODS)
    assert len(full) == 18
    assert balanced.territory_id.unique().tolist() == [2]
    assert len(balanced) == 12
    profiles = make_profiles(full)
    assert profiles["2023-01"][0].tolist() == [2]
    assert profiles["2023-02"][0].tolist() == [2, 10]


def test_unexpected_calendar_cannot_silently_change_balanced_population(tmp_path):
    with pytest.raises(ValueError, match="Source calendar differs"):
        load_panel(_save(tmp_path, _panel()))


def test_main_profile_normalizes_five_means_and_excludes_overall_mean():
    frame = _panel()
    profiles = make_profiles(frame)
    assert list(profiles) == list(PERIODS)
    ids, values, categories = profiles["2023-01"]
    assert ids.tolist() == [2, 10]
    assert categories == list(PUBLISHED_CATEGORIES)
    assert TOTAL_CATEGORY not in categories
    np.testing.assert_allclose(values[0], np.arange(1, 6) / 15)
    np.testing.assert_allclose(values.sum(axis=1), 1)
    # Changing the overall mean must not change a main five-component profile.
    frame.loc[frame.category == TOTAL_CATEGORY, "value"] = 300
    np.testing.assert_allclose(make_profiles(frame)["2023-01"][1], values)


def test_leave_one_category_out_is_renormalized():
    _, values, categories = make_profiles(_panel(), drop_category="Транспорт")["2023-01"]
    assert categories == list(PUBLISHED_CATEGORIES[:-1])
    np.testing.assert_allclose(values[0], np.arange(1, 5) / 10)
    with pytest.raises(ValueError, match="drop_category"):
        make_profiles(_panel(), drop_category=TOTAL_CATEGORY)


def test_residual_is_explicit_arithmetic_sensitivity_not_a_published_category():
    _, values, categories = make_profiles(_panel(), include_residual=True)["2023-01"]
    assert categories == [*PUBLISHED_CATEGORIES, ARITHMETIC_RESIDUAL]
    np.testing.assert_allclose(values[0], np.array([1, 2, 3, 4, 5, 15]) / 30)
    _, dropped, _ = make_profiles(_panel(), include_residual=True, drop_category="Транспорт")[
        "2023-01"
    ]
    # Residual still equals 15, not 20, when a published feature is omitted.
    np.testing.assert_allclose(dropped[0], np.array([1, 2, 3, 4, 15]) / 25)


@pytest.mark.parametrize("overall", [14, 15])
def test_negative_or_zero_arithmetic_residual_is_rejected(overall):
    frame = _panel()
    frame.loc[frame.category == TOTAL_CATEGORY, "value"] = overall
    with pytest.raises(ValueError, match="residual.*strictly positive"):
        make_profiles(frame, include_residual=True)


@pytest.mark.parametrize("expanded", [False, True])
def test_prepare_cli_copies_verified_inputs_repeatably(tmp_path, expanded):
    source = tmp_path / "source"
    source.mkdir()
    content = b"synthetic-source-bytes-for-copy-contract"
    if expanded:
        data_dir = source / "hackathon-bundle" / "hackathonlicence"
        data_dir.mkdir(parents=True)
        (data_dir / "consumption.parquet").write_bytes(content)
    else:
        with zipfile.ZipFile(source / "hackathonlicence.zip", "w") as archive:
            archive.writestr("hackathonlicence/consumption.parquet", content)
    (source / "municipality-codebook.csv").write_text("code,name\n2,Example\n", encoding="utf-8")
    (source / "municipal-spending-metadata-decoded.json").write_text("{}", encoding="utf-8")
    audit = {
        "retrieval_date": "2026-09-14",
        "file_hashes": [
            {
                "file": "hackathon-bundle/hackathonlicence/consumption.parquet",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    (source / "bundle-data-audit.json").write_text(json.dumps(audit), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(script),
        "--source-dir",
        str(source),
        "--output-dir",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(before) == {
        "consumption.parquet",
        "municipality-codebook.csv",
        "dataset_metadata.json",
        "source_audit.json",
        "manifest.json",
    }
    manifest = json.loads(before["manifest.json"])
    assert all(record["source_urls"] for record in manifest["files"])
    assert before["consumption.parquet"] == content
    # Existing raw input is never replaced with different bytes.
    (output / "consumption.parquet").write_bytes(b"different existing source")
    failure = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failure.returncode == 1
    assert "Refusing to overwrite" in failure.stderr
    assert (output / "consumption.parquet").read_bytes() == b"different existing source"
