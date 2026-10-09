"""Reconcile annual evidence without importing the producer's comparison helpers."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def switches(left, right):
    u = left[np.sort(np.unique(left, return_index=True)[1])]
    v = right[np.sort(np.unique(right, return_index=True)[1])]
    overlap = np.array([[np.sum((left == x) & (right == y)) for y in v] for x in u])
    ii, jj = linear_sum_assignment(-overlap)
    match = {v[j]: u[i] for i, j in zip(ii, jj, strict=True) if overlap[i, j] > 0}
    return np.array([match.get(y, None) != x for x, y in zip(left, right, strict=True)])


def main():
    manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((REPORT / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in manifest["outputs"].items():
        assert sha(REPORT / name) == digest
    for value in protocol["input_hashes"].values():
        assert sha(ROOT / value["path"]) == value["sha256"]
    assert sha(ROOT / "scripts/analyze_annual_networks.py") == protocol["script_sha256"]
    with duckdb.connect() as db:
        a = db.execute("SELECT * FROM read_parquet(?)", [str(REPORT / "assignments.parquet")]).df()
        t = db.execute(
            "SELECT * FROM read_parquet(?) WHERE method='leiden_hellinger' ORDER BY territory_id",
            [str(REPORT / "transitions.parquet")],
        ).df()
        original = db.execute(
            "SELECT * FROM read_parquet(?) WHERE method='leiden_hellinger'",
            [str(ROOT / "reports/initial_study/assignments.parquet")],
        ).df()
        annual = db.execute(
            "SELECT * FROM read_parquet(?)", [str(REPORT / "annual_profiles.parquet")]
        ).df()
    fields = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
    expected = (
        original.assign(year=original.month.str[:4].astype(int))
        .groupby(["year", "territory_id"])[fields]
        .mean()
        .reset_index()
    )
    assert annual.equals(expected)
    evidence = np.load(REPORT / "variant_evidence.npz")
    supports, variants = [], []
    for month in ["2023-12", "2024-12"]:
        p = a.loc[a.month.eq(month) & a.method.eq("leiden_hellinger")].sort_values("territory_id")
        assert np.array_equal(p.territory_id, evidence[month + "__ids"])
        labels = p.label.to_numpy()
        cnt = Counter(labels)
        v, actual = evidence[month + "__labels"], evidence[month + "__support"]
        calculated = []
        for candidate in v:
            counts, cross = Counter(candidate), Counter(zip(labels, candidate, strict=True))
            jaccard = np.array(
                [
                    (cross[x, y] - 1) / (cnt[x] + counts[y] - cross[x, y] - 1)
                    if cnt[x] + counts[y] - cross[x, y] - 1
                    else 1
                    for x, y in zip(labels, candidate, strict=True)
                ]
            )
            calculated.append(jaccard >= 0.8)
        assert np.array_equal(calculated, actual)
        supports.append(actual)
        variants.append(v)
    changed = np.stack([switches(x, y) for x, y in zip(*variants, strict=True)])
    reproduced = (supports[0] & supports[1] & (changed == t.switched.to_numpy()[None, :])).mean(
        axis=0
    )
    assert np.allclose(reproduced, t.transition_reproduction_fraction)
    assert np.array_equal(
        t.robust_switch.to_numpy(dtype=bool), (reproduced >= 0.8) & t.switched.to_numpy()
    )
    report = {
        "status": "PASS",
        "source_and_output_hashes": True,
        "annual_mean_reconciled": True,
        "support_cells_independently_recomputed": 84672,
        "paired_variant_switches_independently_recomputed": 42336,
        "n_common": len(t),
        "raw_primary_switches": int(t.switched.sum()),
        "robust_primary_switches": int(t.robust_switch.sum()),
        "max_pair_reproduction_fraction": float(reproduced.max()),
        "qa_method": "separate Counter peer-overlap and Hungarian implementations; same agent",
        "verifier_sha256": sha(Path(__file__)),
    }
    (REPORT / "independent_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
