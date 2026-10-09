"""Meaningful interval, geometry and provenance guards for the narrow replay."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round4_geography", Path(__file__).parents[1] / "scripts/collect_round4_geography.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_version_end_is_exclusive_and_ambiguity_rejected():
    rows = [
        {"territory_id": 208, "year_from": 2018, "year_to": 2024, "value": "old"},
        {"territory_id": 208, "year_from": 2024, "year_to": 9999, "value": "new"},
    ]
    assert MODULE.select_version(rows, 208, 2023)["value"] == "old"
    assert MODULE.select_version(rows, 208, 2024)["value"] == "new"
    with pytest.raises(ValueError, match="Expected one version"):
        MODULE.select_version(rows + [rows[1]], 208, 2024)
    with pytest.raises(ValueError, match="Expected one version"):
        MODULE.select_version(rows, 217, 2024)


def test_equal_bytes_never_resolve_legal_boundary_or_calendar_review(tmp_path):
    gpkg, xlsx = tmp_path / "fixture.gpkg", tmp_path / "versions.xlsx"
    with sqlite3.connect(gpkg) as con:
        con.execute(
            "CREATE TABLE t_dict_municipal_districts_poly "
            "(fid,territory_id,year_from,year_to,osm_ref,osm_vers,geom BLOB)"
        )
        for i in MODULE.IDS:
            con.execute(
                "INSERT INTO t_dict_municipal_districts_poly VALUES (?,?,?,?,?,?,?)",
                (i, str(i), 2018, 9999, str(i), "timestamp-literal", b"GP-test-blob"),
            )
        con.execute("CREATE TABLE gpkg_contents (table_name,last_change,srs_id)")
        con.execute(
            "INSERT INTO gpkg_contents VALUES (?,?,?)",
            ("t_dict_municipal_districts_poly", "2024", 4326),
        )
    pd.DataFrame(
        [
            dict(
                territory_id=i,
                year_from=2018,
                year_to=9999,
                municipal_district_name_short=str(i),
                oktmo="00-001-000-000",
                municipal_district_type="fixture",
            )
            for i in MODULE.IDS
        ]
    ).to_excel(xlsx, index=False)
    annual, identities, _ = MODULE.inspect_versions(gpkg, xlsx)
    assert len(annual) == 8
    assert all(row["publisher_geometry_bytes_identical"] for row in identities)
    assert all(row["independent_legal_boundary_equivalence"] == "unknown" for row in identities)
    assert not any(row["automatic_mask_change"] for row in identities)
    assert identities[-1]["city_1051_2023_calendar_review_retained"]
    with sqlite3.connect(gpkg) as con:
        con.execute(
            "UPDATE t_dict_municipal_districts_poly SET year_to=2024 WHERE territory_id='208'"
        )
        con.execute(
            "INSERT INTO t_dict_municipal_districts_poly VALUES (?,?,?,?,?,?,?)",
            (500, "208", 2024, 9999, "208", "new-literal", b"GP-different"),
        )
    _, changed, _ = MODULE.inspect_versions(gpkg, xlsx)
    assert not changed[0]["publisher_geometry_bytes_identical"]


def test_changed_source_bytes_are_rejected(tmp_path):
    (tmp_path / "inputs").mkdir()
    source = tmp_path / "source.pdf"
    source.write_bytes(b"first")
    (tmp_path / "inputs/source_ledger.json").write_text(
        json.dumps(
            [
                dict(
                    source_id="fixture",
                    download_status="downloaded",
                    path="source.pdf",
                    bytes=5,
                    sha256=MODULE.sha256(source),
                )
            ]
        )
    )
    MODULE.verified_inputs(tmp_path)
    source.write_bytes(b"other")
    with pytest.raises(ValueError, match="fingerprint changed"):
        MODULE.verified_inputs(tmp_path)
