import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_results.py"
SPEC = importlib.util.spec_from_file_location("verify_results", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _save_manifest(run_dir, manifest):
    manifest["output_hashes"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run_dir.glob("*.csv")
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _valid_run(run_dir):
    assignments, membership, transitions, flows, summary, metrics = [], [], [], [], [], []
    for month, label in (("2023-01", 0), ("2023-02", 9)):
        for territory_id in (1, 2, 3):
            membership.append(
                {"month": month, "territory_id": territory_id, "agreement": 1, "n_comparisons": 21}
            )
            for method in sorted(VERIFY.METHODS):
                assignments.append(
                    {
                        "month": month,
                        "territory_id": territory_id,
                        "method": method,
                        "label": label,
                        **dict(zip(VERIFY.PROFILE, [0.4, 0.2, 0.15, 0.15, 0.1], strict=True)),
                    }
                )
        for method in sorted(VERIFY.METHODS):
            metrics.append(
                {
                    "month": month,
                    "method": method,
                    "n_entities": 3,
                    **dict.fromkeys(VERIFY.METRICS, 0.2),
                    "metric_status": json.dumps(dict.fromkeys(VERIFY.METRICS, "ok")),
                }
            )
    for method in sorted(VERIFY.METHODS):
        pair = {"from_month": "2023-01", "to_month": "2023-02", "lag": 1, "method": method}
        primary = method == "leiden_hellinger"
        for territory_id in (1, 2, 3):
            transitions.append(
                {
                    **pair,
                    "territory_id": territory_id,
                    "from_label": 0,
                    "to_label": 9,
                    "aligned_to_label": 0,
                    "switched": False,
                    "hellinger_profile_change": 0,
                    "transition_reproduction_fraction": 1 if primary else np.nan,
                    "robust_switch": False if primary else None,
                }
            )
        flows.append(
            {**pair, "from_label": 0, "to_label": 9, "count": 3, "row_share": 1, "col_share": 1}
        )
        summary.append(
            {
                **pair,
                "ncommon": 3,
                "ari": 1,
                "switch_share": 0,
                "robust_switch_share_of_all": 0 if primary else np.nan,
                "median_profile_change": 0,
            }
        )
    tables = {
        "assignments": assignments,
        "membership_agreement": membership,
        "transitions": transitions,
        "flows": flows,
        "temporal_summary": summary,
        "metrics": metrics,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(run_dir / f"{name}.csv", index=False)
    manifest = {
        "run_id": "synthetic-validation",
        "status": "complete",
        "config_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "code_sha256": "c" * 64,
        "table_rows": {name: len(rows) for name, rows in tables.items()},
    }
    _save_manifest(run_dir, manifest)
    return manifest


def test_valid_run_is_read_only_and_tampered_payload_fails_hash_check(tmp_path, capsys):
    _valid_run(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert VERIFY.main(["--run-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS", result
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with (tmp_path / "assignments.csv").open("ab") as handle:
        handle.write(b"\n")
    assert VERIFY.main(["--run-dir", str(tmp_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert result["checks"]["output_hashes"] == "FAIL"
    assert any("SHA256 mismatch assignments.csv" in error for error in result["errors"])


def test_invalid_robust_switch_fails_even_with_updated_payload_hash(tmp_path, capsys):
    manifest = _valid_run(tmp_path)
    frame = pd.read_csv(tmp_path / "transitions.csv")
    index = frame.index[frame["method"].eq("leiden_hellinger")][0]
    frame.loc[index, "robust_switch"] = True
    frame.to_csv(tmp_path / "transitions.csv", index=False)
    _save_manifest(tmp_path, manifest)
    assert VERIFY.main(["--run-dir", str(tmp_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["checks"]["output_hashes"] == "PASS"
    assert result["checks"]["transitions"] == "FAIL"
    assert any("robust_switch requires switched" in error for error in result["errors"])


def test_nan_metric_requires_an_explicit_undefined_status(tmp_path):
    manifest = _valid_run(tmp_path)
    frame = pd.read_csv(tmp_path / "metrics.csv")
    frame.loc[0, "silhouette"] = np.nan
    frame.to_csv(tmp_path / "metrics.csv", index=False)
    _save_manifest(tmp_path, manifest)
    result = VERIFY.verify_results(tmp_path)
    assert result["checks"]["metrics"] == "FAIL"
    statuses = json.loads(frame.loc[0, "metric_status"])
    statuses["silhouette"] = "undefined_trivial_partition"
    frame.loc[0, "metric_status"] = json.dumps(statuses)
    frame.to_csv(tmp_path / "metrics.csv", index=False)
    _save_manifest(tmp_path, manifest)
    assert VERIFY.verify_results(tmp_path)["status"] == "PASS"
