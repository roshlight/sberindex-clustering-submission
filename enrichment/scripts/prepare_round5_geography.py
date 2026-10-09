"""Audit exact SME code presence in official registry versions, without territorial joins."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd

COLS = [
    "TER",
    "KOD1",
    "KOD2",
    "KOD3",
    "KC",
    "RAZDEL",
    "NAME1",
    "Centrum",
    "NomDescr",
    "NomAkt",
    "Status",
    "DateUtv",
    "DateVved",
]
HISTORICAL = ["2023-02-08", "2023-12-19", "2024-01-25", "2024-12-27"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def registry_entries(frame):
    for col, width in [("TER", 2), ("KOD1", 3), ("KOD2", 3), ("KOD3", 3)]:
        if not frame[col].str.fullmatch(rf"[0-9]{{{width}}}").all():
            raise ValueError(f"Invalid registry field: {col}")
    if not frame.RAZDEL.isin(["1", "2"]).all():
        raise ValueError("Unknown registry section")
    if not frame.loc[frame.RAZDEL.eq("1"), "KOD3"].eq("000").all():
        raise ValueError("Section 1 has unexpected final digits")
    # Section 2 includes headings with KOD3=000, not settlement entries.
    selected = frame.loc[frame.TER.ne("00") & (frame.RAZDEL.eq("1") | frame.KOD3.ne("000"))].copy()
    selected["code"] = selected.TER + selected.KOD1 + selected.KOD2
    selected.loc[selected.RAZDEL.eq("2"), "code"] += selected.KOD3
    selected["code_kind"] = "section1_submunicipal_or_group"
    selected.loc[selected.KOD2.eq("000"), "code_kind"] = "section1_top_level_or_group"
    selected.loc[selected.KOD1.eq("000"), "code_kind"] = "region_group"
    selected.loc[selected.RAZDEL.eq("2"), "code_kind"] = "section2_settlement"
    return selected


def attach_presence(codes, entries, label):
    duplicates = entries.code.duplicated(keep=False)
    unique = entries.loc[~duplicates].set_index("code")
    result = codes.copy()
    result[label + "_present"] = result.source_code.isin(entries.code)
    result[label + "_ambiguous"] = result.source_code.isin(entries.loc[duplicates, "code"])
    for source, name in [("NAME1", "name"), ("code_kind", "kind"), ("DateVved", "date_vved")]:
        result[label + "_" + name] = result.source_code.map(unique[source])
    return result


def build(repo, raw, prior_raw, sme_raw, out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    out.mkdir(parents=True)
    inputs = {}
    versions = []
    prior_manifest = repo / "reports/enrichment_v2/code_bridge/registry_vintage_audit.csv"
    inputs[prior_manifest.relative_to(repo).as_posix()] = sha(prior_manifest)
    prior = pd.read_csv(prior_manifest, dtype=str)
    reused = raw / "reused"
    reused.mkdir(exist_ok=True)
    for label in HISTORICAL:
        row = prior.loc[prior.file_version_label.eq(label)].iloc[0]
        source = prior_raw / row.filename
        if sha(source) != row.sha256:
            raise ValueError("Historical source changed")
        target = reused / row.filename
        if not target.exists():
            shutil.copyfile(source, target)
        if sha(target) != row.sha256:
            raise ValueError("Reused source changed")
        metadata_path = prior_raw / (row.filename + ".download.json")
        metadata = read_json(metadata_path)
        if metadata["sha256"] != row.sha256:
            raise ValueError("Historical retrieval metadata disagrees")
        metadata_copy = reused / metadata_path.name
        if metadata_copy.resolve() != metadata_path.resolve():
            shutil.copyfile(metadata_path, metadata_copy)
        versions.append(
            {
                "label": label,
                "path": target.relative_to(raw).as_posix(),
                "url": row.url,
                "sha256": row.sha256,
                "origin": "reused_prior_verified_snapshot",
                "retrieved_at": metadata["retrieved_utc"],
            }
        )
    retrievals = read_json(raw / "downloads/retrievals.json")
    for item in retrievals:
        if item["status"] != 200 or item["error"]:
            raise ValueError("Incomplete new registry source")
        p = raw / "downloads" / item["path"]
        if sha(p) != item["sha256"]:
            raise ValueError("New registry hash mismatch")
        if p.suffix == ".csv":
            versions.append(
                {
                    "label": p.stem.removeprefix("registry_"),
                    "path": p.relative_to(raw).as_posix(),
                    "url": item["url"],
                    "sha256": item["sha256"],
                    "origin": item["origin"],
                    "retrieved_at": item["retrieved_at"],
                }
            )
    if len(versions) != 6:
        raise ValueError("Six declared registry versions required")
    annual = []
    sme_inputs = {}
    source_manifest = repo / "reports/data_round4/sme/manifest.json"
    expected = read_json(source_manifest)["inputs"]
    inputs[source_manifest.relative_to(repo).as_posix()] = sha(source_manifest)
    for year in [2023, 2024]:
        relative = f"extracted/{year}/entire/data_sme_153_entire_year_{year}_v20260605.parquet"
        p = sme_raw / relative
        sme_inputs[relative] = sha(p)
        if sme_inputs[relative] != expected[relative]:
            raise ValueError("SME source hash mismatch")
        with duckdb.connect(config={"threads": 2, "memory_limit": "2GB"}) as con:
            con.read_parquet(str(p)).create_view("sme")
            if con.execute("SELECT count(*) FROM sme WHERE year != ?", [year]).fetchone()[0]:
                raise ValueError("Unexpected SME year")
            frame = con.execute("""
                SELECT coalesce(municipality_code, '') AS source_code,
                  municipality_code IS NULL AS source_code_is_null,
                  count(*) AS source_rows, count(DISTINCT tax_number) AS tax_ids_with_this_code
                FROM sme GROUP BY municipality_code ORDER BY municipality_code NULLS FIRST
            """).df()
            count = con.execute("SELECT count(*) FROM sme").fetchone()[0]
            if int(frame.source_rows.sum()) != count:
                raise ValueError("Code aggregation changed source row count")
        frame.insert(0, "year", year)
        annual.append(frame)
    codes = pd.concat(annual, ignore_index=True)
    codes["code_length"] = codes.source_code.str.len()
    codes["valid_syntax"] = codes.source_code.str.fullmatch(r"(?:[0-9]{8}|[0-9]{11})")
    catalog = []
    comparison = codes.copy()
    duplicate_rows = []
    previous_entries = {}
    for version in versions:
        frame = pd.read_csv(raw / version["path"], sep=";", names=COLS, dtype=str)
        entries = registry_entries(frame)
        duplicate = entries.loc[entries.code.duplicated(keep=False)].copy()
        duplicate.insert(0, "version", version["label"])
        duplicate_rows.append(duplicate)
        dates = pd.to_datetime(frame.DateVved, format="%d.%m.%Y", errors="coerce")
        catalog.append(
            {
                **version,
                "rows": len(frame),
                "code_entries": len(entries),
                "distinct_codes": entries.code.nunique(),
                "duplicate_code_rows": len(duplicate),
                "latest_date_vved": dates.max().date().isoformat(),
                "rows_effective_after_filename_date": int(
                    dates.gt(pd.Timestamp(version["label"])).sum()
                ),
                "filename_is_asof_population": False,
            }
        )
        comparison = attach_presence(comparison, entries, "v" + version["label"].replace("-", ""))
        previous_entries[version["label"]] = entries
    summary = []
    for year, rows in comparison.groupby("year"):
        for version in versions:
            label = "v" + version["label"].replace("-", "")
            for present, group in rows.groupby(label + "_present", sort=True):
                summary.append(
                    {
                        "source_year": int(year),
                        "registry_version_label": version["label"],
                        "exact_code_present": bool(present),
                        "source_rows": int(group.source_rows.sum()),
                        "source_code_groups": len(group),
                        "denominator_source_rows": int(rows.source_rows.sum()),
                        "share_source_rows_pct": 100
                        * group.source_rows.sum()
                        / rows.source_rows.sum(),
                        "historical_address_or_boundary_proven": False,
                    }
                )
    a, b = (previous_entries[x].set_index("code") for x in ["2026-06-01", "2026-07-01"])
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError("June/July drift requires unique registry codes")
    drift = []
    for code in sorted(set(a.index) | set(b.index)):
        status = "removed" if code not in b.index else "added" if code not in a.index else "shared"
        old = a.loc[code, "NAME1"] if code in a.index else ""
        new = b.loc[code, "NAME1"] if code in b.index else ""
        if status != "shared" or old != new:
            drift.append(
                {
                    "code": code,
                    "status": "name_changed" if status == "shared" else status,
                    "june_name": old,
                    "july_name": new,
                }
            )
    pd.DataFrame(catalog).to_csv(out / "registry_versions.csv", index=False)
    codes.to_csv(out / "sme_code_inventory.csv", index=False)
    with duckdb.connect() as con:
        con.from_df(comparison).write_parquet(str(out / "sme_code_presence.parquet"))
    pd.DataFrame(summary).to_csv(out / "code_presence_summary.csv", index=False)
    kinds = (
        comparison.assign(code_kind_june2026=comparison.v20260601_kind.fillna("no_exact_code"))
        .groupby(["year", "code_kind_june2026"], as_index=False)
        .agg(source_rows=("source_rows", "sum"), source_code_groups=("source_code", "size"))
    )
    kinds["denominator_source_rows"] = kinds.groupby("year").source_rows.transform("sum")
    kinds["share_source_rows_pct"] = 100 * kinds.source_rows / kinds.denominator_source_rows
    kinds["municipal_panel_join_accepted"] = False
    kinds.to_csv(out / "code_kind_summary.csv", index=False)
    pd.concat(duplicate_rows).to_csv(out / "registry_duplicate_codes.csv", index=False)
    pd.DataFrame(drift, columns=["code", "status", "june_name", "july_name"]).to_csv(
        out / "june_july_code_changes.csv", index=False
    )
    qa = {
        "status": "ready_with_conditions",
        "scope": "Exact code presence only",
        "registry_versions": len(versions),
        "new_registry_snapshots": 2,
        "source_period_rows": int(codes.source_rows.sum()),
        "year_code_groups": len(codes),
        "municipal_panel_joins": 0,
        "masks_changed": 0,
        "workers_aggregated": False,
        "historical_address_validated": False,
        "historical_boundary_validated": False,
        "tax_ids_with_this_code_additive_across_codes": False,
    }
    write_json(out / "QA.json", qa)
    write_json(
        out / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "repo_inputs": inputs,
            "sme_inputs": sme_inputs,
            "registry_inputs": versions,
            "raw_inputs": {
                p.relative_to(raw).as_posix(): sha(p) for p in sorted(raw.rglob("*")) if p.is_file()
            },
            "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps(qa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(".."))
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--sme-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo, args.raw, args.prior_raw, args.sme_raw, args.out)
