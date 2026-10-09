"""Replay must preserve source directories, identities and material numerical differences."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/replay_round5_analysis.py"
SPEC = importlib.util.spec_from_file_location("replay_round5_analysis", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_output_never_overwrites_inputs(tmp_path):
    repo, raw = tmp_path / "repo", tmp_path / "raw"
    repo.mkdir()
    raw.mkdir()
    module.validate_output(repo, raw, tmp_path / "fresh")
    for target in [repo, raw, repo / "fresh", raw / "fresh", tmp_path]:
        with pytest.raises(ValueError):
            module.validate_output(repo, raw, target)


@pytest.mark.parametrize("relative", ["../x", "/x", "C:/x", "C:x", "\\\\host\\x", "x:stream"])
def test_branch_paths_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        module.inside(tmp_path, relative)


def test_numeric_tolerance_does_not_permit_identity_change():
    old = pd.DataFrame({"territory_id": [1, 2], "prediction": [1.0, 2.0]})
    close = old.copy()
    close.loc[0, "prediction"] += 1e-10
    assert module.compare_frame(old, close) < 1e-9
    close.loc[0, "territory_id"] = 3
    with pytest.raises(AssertionError):
        module.compare_frame(old, close)


def test_material_prediction_change_is_rejected():
    old = pd.DataFrame({"prediction": [1.0, 2.0]})
    changed = pd.DataFrame({"prediction": [1.0, 2.01]})
    with pytest.raises(AssertionError):
        module.compare_frame(old, changed)


def test_preprocessing_comparison_preserves_boolean_and_vocabulary():
    expected = {"median": 2.0, "columns": ["region25"], "eligible": True}
    module.compare_json(
        expected, {"median": 2.0 + 1e-10, "columns": ["region25"], "eligible": True}
    )
    for actual in [
        {"median": 2.0, "columns": ["region59"], "eligible": True},
        {"median": 2.0, "columns": ["region25"], "eligible": 1},
        {"median": 2.0, "columns": ["region25"]},
    ]:
        with pytest.raises(ValueError):
            module.compare_json(expected, actual)
