"""Read-only verification of result integrity and table consistency.

PASS verifies the listed structural and numerical constraints, not the existence
of economic classes. Manifest hashes detect accidental changes relative to the
manifest; an unsigned manifest is not proof of provenance or authenticity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {"leiden_hellinger", "leiden_cosine", "kmeans", "kmeans_matched_k"}
PROFILE = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
PAIR = ["from_month", "to_month", "lag", "method"]
METRICS = ["silhouette", "calinski_harabasz", "s_dbw", "avi", "avu", "modularity"]
REQUIRED = [
    "assignments",
    "membership_agreement",
    "transitions",
    "flows",
    "temporal_summary",
    "metrics",
]
MONTH = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def columns(frame, names, table):
    missing = set(names) - set(frame.columns)
    require(not missing, f"{table}: missing columns {sorted(missing)}")


def unique(frame, key, table):
    columns(frame, key, table)
    require(not frame[key].isna().any().any(), f"{table}: null key")
    require(not frame.duplicated(key).any(), f"{table}: duplicate key {key}")


def numeric(series, name, *, lower=None, upper=None, nullable=False, integer=False):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    present = values[~np.isnan(values)] if nullable else values
    require(np.isfinite(present).all(), f"{name}: nonfinite or missing values")
    if lower is not None:
        require((present >= lower).all(), f"{name}: values below {lower}")
    if upper is not None:
        require((present <= upper).all(), f"{name}: values above {upper}")
    if integer:
        require((present == np.floor(present)).all(), f"{name}: noninteger values")
    return values


def booleans(series, name, *, nullable=False):
    # CSV round trips may produce bool, object strings, or all-null float columns.
    values = series.map(
        lambda value: (
            pd.NA
            if pd.isna(value)
            else {"true": True, "false": False}.get(str(value).lower(), "invalid")
        )
    )
    require(not values.eq("invalid").any(), f"{name}: invalid Boolean encoding")
    require(nullable or not values.isna().any(), f"{name}: missing Boolean values")
    return values.astype("boolean")


def month_ordinals(series, name):
    require(
        series.map(
            lambda value: isinstance(value, str) and MONTH.fullmatch(value) is not None
        ).all(),
        f"{name}: expected YYYY-MM",
    )
    return series.map(lambda value: int(value[:4]) * 12 + int(value[5:]) - 1)


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _check_assignments(frame):
    columns(frame, ["month", "territory_id", "method", "label", *PROFILE], "assignments")
    require(not frame.empty, "assignments: empty")
    unique(frame, ["month", "territory_id", "method"], "assignments")
    month_ordinals(frame["month"], "assignments.month")
    numeric(frame["territory_id"], "assignments.territory_id", lower=1, integer=True)
    numeric(frame["label"], "assignments.label", integer=True)
    p = frame[PROFILE].to_numpy(dtype=float)
    require(
        np.isfinite(p).all() and (p > 0).all(), "assignments: profiles must be finite and positive"
    )
    require(np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-9), "assignments: P5 must sum to one")
    for month, group in frame.groupby("month"):
        require(set(group["method"]) == METHODS, f"assignments: expected four methods in {month}")
        ids = [set(part["territory_id"]) for _, part in group.groupby("method")]
        require(
            all(values == ids[0] for values in ids[1:]), f"assignments: method ID mismatch {month}"
        )


def _check_membership(frame, assignments):
    columns(frame, ["month", "territory_id", "agreement", "n_comparisons"], "membership")
    unique(frame, ["month", "territory_id"], "membership")
    expected = assignments.loc[
        assignments["method"].eq("leiden_hellinger"), ["month", "territory_id"]
    ]
    actual_keys = set(frame[["month", "territory_id"]].itertuples(index=False, name=None))
    expected_keys = set(expected.itertuples(index=False, name=None))
    require(actual_keys == expected_keys, "membership: keys differ from primary assignments")
    numeric(frame["agreement"], "membership.agreement", lower=0, upper=1)
    counts = numeric(frame["n_comparisons"], "membership.n_comparisons", integer=True)
    require((counts == 21).all(), "membership: n_comparisons must equal 21")


def _check_transitions(frame, assignments):
    columns(
        frame,
        [
            *PAIR,
            "territory_id",
            "from_label",
            "to_label",
            "aligned_to_label",
            "switched",
            "hellinger_profile_change",
            "transition_reproduction_fraction",
            "robust_switch",
        ],
        "transitions",
    )
    unique(frame, [*PAIR, "territory_id"], "transitions")
    difference = month_ordinals(frame["to_month"], "transitions.to_month") - month_ordinals(
        frame["from_month"], "transitions.from_month"
    )
    lag = numeric(frame["lag"], "transitions.lag", integer=True)
    require(
        np.isin(lag, [1, 12]).all() and np.array_equal(lag, difference), "transitions: invalid lag"
    )
    require(frame["method"].isin(METHODS).all(), "transitions: unknown method")
    change = numeric(
        frame["hellinger_profile_change"], "transitions.profile_change", lower=0, upper=1
    )
    fraction = numeric(
        frame["transition_reproduction_fraction"],
        "transitions.reproduction_fraction",
        lower=0,
        upper=1,
        nullable=True,
    )
    switched = booleans(frame["switched"], "transitions.switched")
    robust = booleans(frame["robust_switch"], "transitions.robust_switch", nullable=True)
    require(
        np.array_equal(switched, frame["from_label"].ne(frame["aligned_to_label"])),
        "transitions: switched disagrees with aligned labels",
    )
    reported_robust = robust.fillna(False).to_numpy(dtype=bool)
    require(
        not (reported_robust & (~switched.to_numpy(dtype=bool) | ~(fraction >= 0.8))).any(),
        "transitions: robust_switch requires switched and reproduction_fraction >= 0.8",
    )
    primary = frame["method"].eq("leiden_hellinger").to_numpy()
    require(
        np.isnan(fraction[~primary]).all() and robust.loc[~primary].isna().all(),
        "transitions: alternative-method robustness must be unassessed",
    )
    require(
        np.array_equal(robust.isna(), np.isnan(fraction)),
        "transitions: robust_switch nullness must match reproduction assessment",
    )
    source = assignments.set_index(["month", "territory_id", "method"])
    profiles = []
    for endpoint in ("from", "to"):
        keys = pd.MultiIndex.from_frame(
            frame[[f"{endpoint}_month", "territory_id", "method"]].rename(
                columns={f"{endpoint}_month": "month"}
            )
        )
        selected = source.reindex(keys)
        require(not selected["label"].isna().any(), f"transitions: missing {endpoint} assignment")
        require(
            np.array_equal(selected["label"].to_numpy(), frame[f"{endpoint}_label"]),
            f"transitions: {endpoint} labels disagree with assignments",
        )
        profiles.append(selected[PROFILE].to_numpy(dtype=float))
    expected_change = np.linalg.norm(np.sqrt(profiles[0]) - np.sqrt(profiles[1]), axis=1) / np.sqrt(
        2
    )
    require(
        np.allclose(change, expected_change, atol=1e-10), "transitions: profile change mismatch"
    )


def _check_flows(flows, transitions):
    key = [*PAIR, "from_label", "to_label"]
    columns(flows, [*key, "count", "row_share", "col_share"], "flows")
    unique(flows, key, "flows")
    numeric(flows["count"], "flows.count", lower=1, integer=True)
    for name in ("row_share", "col_share"):
        numeric(flows[name], f"flows.{name}", lower=0, upper=1)
    expected = transitions.groupby(key).size().rename("expected")
    actual = flows.set_index(key)["count"]
    require(
        set(actual.index) == set(expected.index), "flows: raw cluster pairs differ from transitions"
    )
    require(np.array_equal(actual.reindex(expected.index), expected), "flows: count mismatch")
    for endpoint, name in (("from", "row_share"), ("to", "col_share")):
        totals = flows.groupby([*PAIR, f"{endpoint}_label"])["count"].transform("sum")
        require(np.allclose(flows[name], flows["count"] / totals), f"flows: incorrect {name}")
        sums = flows.groupby([*PAIR, f"{endpoint}_label"])[name].sum()
        require(np.allclose(sums, 1), f"flows: {name} does not sum to one")


def _check_summary(summary, transitions, flows, assignments):
    columns(
        summary,
        [
            *PAIR,
            "ncommon",
            "ari",
            "switch_share",
            "robust_switch_share_of_all",
            "median_profile_change",
        ],
        "temporal_summary",
    )
    unique(summary, PAIR, "temporal_summary")
    numeric(summary["ncommon"], "summary.ncommon", lower=0, integer=True)
    expected = {}
    for method, method_rows in assignments.groupby("method"):
        months = {month: set(part["territory_id"]) for month, part in method_rows.groupby("month")}
        ordinals = {int(month[:4]) * 12 + int(month[5:]) - 1: month for month in months}
        for ordinal, before in ordinals.items():
            for lag in (1, 12):
                after = ordinals.get(ordinal + lag)
                if after is not None:
                    expected[(before, after, lag, method)] = months[before] & months[after]
    actual = summary.set_index(PAIR)
    require(set(actual.index) == set(expected), "summary: missing or extra calendar pairs")
    transition_groups = {key: group for key, group in transitions.groupby(PAIR)}
    flow_counts = flows.groupby(PAIR)["count"].sum()
    require(set(transition_groups).issubset(expected), "transitions: unexpected calendar pair")
    for key, ids in expected.items():
        row = actual.loc[key]
        group = transition_groups.get(key, transitions.iloc[:0])
        require(set(group["territory_id"]) == ids, f"summary: common IDs mismatch {key}")
        require(row["ncommon"] == len(ids), f"summary: ncommon mismatch {key}")
        require(
            flow_counts.get(key, 0) == len(ids), f"summary: flow count differs from ncommon {key}"
        )
        if not ids:
            require(
                row[METRIC_SUMMARIES].isna().all(), f"summary: empty pair metrics must be NaN {key}"
            )
            continue
        numeric(pd.Series([row["ari"]]), "summary.ari", lower=-1, upper=1)
        switch_share = booleans(group["switched"], "transitions.switched").mean()
        require(
            np.isclose(row["switch_share"], switch_share), f"summary: switch share mismatch {key}"
        )
        require(
            np.isclose(row["median_profile_change"], group["hellinger_profile_change"].median()),
            f"summary: median profile change mismatch {key}",
        )
        robust = booleans(group["robust_switch"], "transitions.robust_switch", nullable=True)
        if robust.isna().any():
            require(
                pd.isna(row["robust_switch_share_of_all"]), f"summary: unassessed robustness {key}"
            )
        else:
            require(
                np.isclose(row["robust_switch_share_of_all"], robust.mean()),
                f"summary: robust switch share mismatch {key}",
            )


METRIC_SUMMARIES = ["ari", "switch_share", "robust_switch_share_of_all", "median_profile_change"]


def _check_metrics(frame, assignments, name="metrics"):
    columns(frame, [*METRICS, "metric_status"], name)
    values = frame[METRICS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    require(not np.isinf(values).any(), f"{name}: infinite metric")
    for index, (_, row) in enumerate(frame.iterrows()):
        status = json.loads(row["metric_status"])
        require(
            isinstance(status, dict) and set(METRICS).issubset(status), f"{name}: incomplete status"
        )
        for position, metric in enumerate(METRICS):
            reason = status[metric]
            require(
                isinstance(reason, str) and bool(reason.strip()), f"{name}: missing metric status"
            )
            if np.isnan(values[index, position]):
                require(
                    reason.startswith("undefined_"), f"{name}: NaN {metric} requires non-ok status"
                )
            else:
                require(
                    reason.startswith(("ok", "defined_")),
                    f"{name}: finite {metric} has undefined status",
                )
    if name == "metrics":
        columns(frame, ["month", "method", "n_entities"], name)
        unique(frame, ["month", "method"], name)
        expected = assignments.groupby(["month", "method"]).size()
        actual = frame.set_index(["month", "method"])["n_entities"]
        require(set(actual.index) == set(expected.index), "metrics: month/method mismatch")
        require(
            np.array_equal(actual.reindex(expected.index), expected), "metrics: n_entities mismatch"
        )


def verify_results(run_dir: str | Path) -> dict:
    """Return JSON-serializable checks; do not write files or import study code."""
    run_dir = Path(run_dir).resolve()
    checks, errors, warnings = {}, [], []
    tables = {}

    def check(name, action):
        try:
            action()
        except Exception as error:
            checks[name] = "FAIL"
            errors.append(f"{name}: {type(error).__name__}: {error}")
        else:
            checks[name] = "PASS"

    manifest = {}

    def read_manifest():
        manifest.update(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
        require(manifest.get("status") == "complete", "run_manifest status must be complete")
        for field in ("config_sha256", "data_sha256", "code_sha256"):
            require(bool(SHA256.fullmatch(str(manifest.get(field, "")))), f"invalid {field}")

    check("manifest", read_manifest)

    def check_hashes():
        hashes = manifest.get("output_hashes")
        require(isinstance(hashes, dict) and bool(hashes), "missing output_hashes")
        require(
            {f"{name}.csv" for name in REQUIRED}.issubset(hashes), "required tables not all hashed"
        )
        failures = []
        for filename, expected in hashes.items():
            path = (run_dir / filename).resolve()
            if not path.is_relative_to(run_dir) or path == run_dir:
                failures.append(f"unsafe payload path {filename}")
                continue
            if not SHA256.fullmatch(str(expected)) or not path.is_file():
                failures.append(f"invalid hash or missing payload {filename}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != expected.lower():
                failures.append(f"SHA256 mismatch {filename}")
        payload = {
            path.name
            for path in run_dir.iterdir()
            if path.is_file() and path.suffix in {".csv", ".json", ".npz"}
        } - {"run_manifest.json"}
        unlisted = sorted(payload - set(hashes))
        if unlisted:
            warnings.append(f"Unhashed payload files: {', '.join(unlisted)}")
        require(not failures, "; ".join(failures))

    check("output_hashes", check_hashes)
    for name in REQUIRED:
        check(
            f"read_{name}",
            lambda name=name: tables.update({name: _read_csv(run_dir / f"{name}.csv")}),
        )

    def check_rows():
        declared = manifest.get("table_rows")
        require(isinstance(declared, dict), "missing table_rows")
        require(set(REQUIRED).issubset(declared), "required tables missing from table_rows")
        for name, count in declared.items():
            path = (run_dir / f"{name}.csv").resolve()
            require(path.is_relative_to(run_dir), f"unsafe table path {name}")
            frame = tables.get(name)
            if frame is None:
                frame = _read_csv(path)
            require(
                isinstance(count, int) and count >= 0 and len(frame) == count, f"row count {name}"
            )

    check("table_rows", check_rows)
    if all(name in tables for name in REQUIRED):
        a, m, t, f, s, metrics = [tables[name] for name in REQUIRED]
        check("assignments", lambda: _check_assignments(a))
        check("membership_agreement", lambda: _check_membership(m, a))
        check("transitions", lambda: _check_transitions(t, a))
        check("flows", lambda: _check_flows(f, t))
        check("temporal_summary", lambda: _check_summary(s, t, f, a))
        check("metrics", lambda: _check_metrics(metrics, a))
        hashes = manifest.get("output_hashes", {})
        for filename in hashes if isinstance(hashes, dict) else {}:
            if filename == "null_metrics.csv" or filename.startswith("checkpoint_"):
                path = (run_dir / filename).resolve()
                if path.is_relative_to(run_dir) and path.is_file():

                    def check_extra_metrics(path=path, filename=filename):
                        frame = _read_csv(path)
                        if not frame.empty:
                            _check_metrics(frame, a, filename)

                    check(filename, check_extra_metrics)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "run_id": manifest.get("run_id", run_dir.name),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "rows": {name: len(frame) for name, frame in tables.items()},
        "scope": "Artifact integrity and consistency; no claim of true economic classes",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_results(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
