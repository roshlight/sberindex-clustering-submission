"""Replay protects immutable inputs, output identities and material numerical differences."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "replay_round6", Path(__file__).parents[1] / "scripts/replay_round6_analysis.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_replay_output_cannot_overlap_sources(tmp_path):
    repo, raw = tmp_path / "repo", tmp_path / "raw"
    repo.mkdir()
    raw.mkdir()
    module.validate_output(repo, raw, tmp_path / "fresh")
    for path in [repo, raw, repo / "new", raw / "new", tmp_path]:
        with pytest.raises(ValueError):
            module.validate_output(repo, raw, path)


@pytest.mark.parametrize("name", ["../x", "C:/x", "C:x", "/x", "\\\\host\\x", "x:stream"])
def test_manifest_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        module.inside(tmp_path, name)


def test_tolerance_does_not_mask_identity_or_missingness_change():
    expected = pd.DataFrame({"territory_id": [1, 2], "rho": [0.2, float("nan")]})
    actual = expected.copy()
    actual.loc[0, "rho"] += 1e-10
    assert module.compare_frame(expected, actual) < 1e-9
    actual.loc[0, "territory_id"] = 3
    with pytest.raises(AssertionError):
        module.compare_frame(expected, actual)
    actual = expected.fillna(0)
    with pytest.raises(AssertionError):
        module.compare_frame(expected, actual)


def test_material_numerical_difference_fails():
    with pytest.raises(AssertionError):
        module.compare_frame(pd.DataFrame({"rho": [0.1]}), pd.DataFrame({"rho": [0.101]}))


def test_float_identifier_is_still_exact():
    with pytest.raises(AssertionError):
        module.compare_frame(
            pd.DataFrame({"territory_id": [1.0]}),
            pd.DataFrame({"territory_id": [1.0 + 1e-10]}),
        )
