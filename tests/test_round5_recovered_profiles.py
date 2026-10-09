"""Historical source codes must survive reading and report serialization."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def test_leading_zero_source_codes_are_not_coerced(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "round5_profiles", scripts / "round5_recovered_profiles.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch("sys.path", [str(scripts), *sys.path]):
        spec.loader.exec_module(module)
    path = tmp_path / "features.csv"
    path.write_text(
        "region_code,official_oktmo_8,retail_source_oktmo,shipments_source_oktmo,value\n"
        "25,05701000,05701000,05701000,12.5\n",
        encoding="utf-8",
    )
    frame = module.load_features(path)
    assert frame.retail_source_oktmo.iloc[0] == "05701000"
    assert frame.shipments_source_oktmo.iloc[0] == "05701000"
    assert frame.value.iloc[0] == 12.5
    assert ",05701000,05701000,05701000," in frame.to_csv(index=False)
