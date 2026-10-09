"""Mechanical extremes of fixed OOF log residuals, with local source-cell checks."""

import argparse
import hashlib
import html
import json
import re
import sys
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from municipal_profiles.data import PUBLISHED_CATEGORIES, TOTAL_CATEGORY, load_panel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return " ".join(html.unescape(str(value)).split())


def same(a, b, label):
    if not np.isclose(float(a), float(b), atol=1e-9, rtol=1e-12):
        raise ValueError(f"Numeric mismatch: {label}: {a} vs {b}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Repository with frozen core_v2, price and enrichment tables",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="Directory containing both named release ZIP assets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New or empty output directory; canonical examples are protected",
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not __debug__:
        parser.error("Run without Python -O: source checks use assertions")
    repo, archive_dir, out = args.repo.resolve(), args.archive_dir.resolve(), args.out.resolve()
    canonical = repo / "reports/enriched_analysis/examples"
    if out.is_relative_to(canonical):
        parser.error("The frozen canonical examples directory cannot be used as output")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out must be a new or empty directory")
    frozen_manifest_path = canonical / "manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    for expected in frozen_manifest["inputs"]:
        relative = Path(expected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            parser.error("Frozen input path must be relative to repository")
        source_path = repo / relative
        if sha(source_path) != expected["sha256"]:
            raise ValueError(f"Frozen input lineage mismatch: {relative}")
    archive_records = []
    for asset in ("server-economy-snapshots.zip", "server-population-snapshots.zip"):
        archive_path = archive_dir / asset
        with zipfile.ZipFile(archive_path) as archive:
            archive_manifest_bytes = archive.read("ARCHIVE_MANIFEST.json")
        archive_records.append(
            {
                "asset": asset,
                "bytes": archive_path.stat().st_size,
                "sha256": sha(archive_path),
                "archive_manifest_sha256": hashlib.sha256(archive_manifest_bytes).hexdigest(),
            }
        )
    out.mkdir(parents=True, exist_ok=True)

    core = repo / "reports/enriched_analysis/core_v2"
    manifest = json.loads((core / "manifest.json").read_text(encoding="utf-8"))
    for name in ("annual_panel.csv", "oof_predictions.parquet"):
        assert sha(core / name) == manifest["outputs"][name]
    db = duckdb.connect()
    pred = db.execute(
        """
        SELECT *, actual_log_total-predicted_log_total AS oof_log_total_residual
        FROM read_parquet(?)
        WHERE year=2024 AND sample='core' AND regime='within_region' AND model='economic'
    """,
        [str(core / "oof_predictions.parquet")],
    ).df()
    assert not pred.territory_id.duplicated().any()
    assert np.isfinite(pred[["actual_log_total", "predicted_log_total"]]).all().all()
    hi = (
        pred.sort_values(["oof_log_total_residual", "territory_id"], ascending=[False, True])
        .head(3)
        .copy()
    )
    lo = (
        pred.sort_values(["oof_log_total_residual", "territory_id"], ascending=[True, True])
        .head(3)
        .copy()
    )
    assert hi.oof_log_total_residual.gt(0).all() and lo.oof_log_total_residual.lt(0).all()
    hi["selection_tail"], lo["selection_tail"] = "largest_positive", "most_negative"
    hi["rank_in_tail"], lo["rank_in_tail"] = [1, 2, 3], [1, 2, 3]
    selected = pd.concat([hi, lo], ignore_index=True)
    annual = pd.read_csv(core / "annual_panel.csv", dtype={"official_oktmo_8": str})
    examples = selected.drop(columns="region_code").merge(
        annual, on=["territory_id", "year"], validate="one_to_one"
    )
    assert len(examples) == 6 and examples.core_eligible.all()
    assert np.allclose(examples.actual_log_total, np.log(examples.total_mean), atol=1e-13, rtol=0)
    examples["fitted_geometric_total_2024_no_smearing"] = np.exp(examples.predicted_log_total)
    examples["actual_to_fitted_ratio"] = (
        examples.total_mean / examples.fitted_geometric_total_2024_no_smearing
    )
    assert np.allclose(examples.actual_to_fitted_ratio, np.exp(examples.oof_log_total_residual))
    price_path = repo / "reports/enriched_analysis/prices/municipal_price_growth.csv"
    prices = pd.read_csv(price_path)
    examples = examples.merge(
        prices[
            [
                "territory_id",
                "nominal_mean_2023",
                "nominal_mean_2024",
                "real_mean_2023_dec2022",
                "real_mean_2024_dec2022",
                "nominal_growth_pct",
                "real_growth_pct",
            ]
        ],
        on="territory_id",
        validate="one_to_one",
    )
    assert np.allclose(examples.total_mean, examples.nominal_mean_2024, atol=1e-9, rtol=0)
    raw_path = repo / "data/raw/consumption.parquet"
    raw = load_panel(raw_path)
    raw = raw[raw.territory_id.isin(examples.territory_id)]
    wide = raw.pivot(index=["territory_id", "date"], columns="category", values="value")
    five = wide[list(PUBLISHED_CATEGORIES)]
    profile = five.div(five.sum(axis=1), axis=0).reset_index()
    profile = profile[profile.date.str.startswith("2024-")]
    medians = profile.groupby("territory_id")[["Продовольствие", "Общественное питание"]].median()
    medians.columns = ["median_monthly_p_food_2024", "median_monthly_p_foodservice_2024"]
    examples = examples.merge(medians, on="territory_id", validate="one_to_one")
    raw.to_csv(out / "selected_consumption_source_rows.csv", index=False)

    economy_path = (
        repo / "reports/enrichment_v3/economy/annual_candidates/annual_candidate_panel.csv"
    )
    population_path = repo / "reports/enrichment_v2/population/municipal_population.csv"
    economy = pd.read_csv(economy_path, dtype={"source_oktmo": str, "indicator_id": str})
    population = pd.read_csv(population_path, dtype={"source_oktmo": str, "population_type": str})
    archives = []
    for file in (
        "enrichment-v3-release/server-economy-snapshots.zip",
        "enrichment-v2-release/server-population-snapshots.zip",
    ):
        path = Path("work") / file
        physical_path = archive_dir / Path(file).name
        archive = zipfile.ZipFile(physical_path)
        archive_manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        archives.append((path, archive, archive_manifest["entries"]))

    def verify_raw_cell(record, source_year):
        source_sha = record.source_sha256
        matches = []
        for path, archive, entries in archives:
            for entry in entries:
                if entry["sha256"] == source_sha or entry.get("original_sha256") == source_sha:
                    matches.append((entry["sha256"] != source_sha, path, archive, entry))
        if not matches:
            raise ValueError(f"Source raw SHA absent from local archives: {source_sha}")
        transformed, path, archive, entry = sorted(matches, key=lambda x: x[0])[0]
        body = archive.read(entry["archive_path"])
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        lines = body.decode("cp1251").splitlines()
        number = int(record.source_csv_row)
        raw_line = lines[number - 1]
        years, header_line = None, None
        for index in range(number - 1, -1, -1):
            line = lines[index]
            if re.fullmatch(r";(?:20\d{2};)+", line.strip()):
                years = line.strip().split(";")[1:-1]
                header_line = index
                break
        if years is None or str(source_year) not in years:
            raise ValueError(f"No explicit requested source year at raw line {number}: {raw_line}")
        pieces = raw_line.rsplit(";", len(years) + 1)
        assert len(pieces) == len(years) + 2 and pieces[-1] == ""
        binding = "exact_municipality_row_caption"
        if norm(pieces[0]) != norm(record.source_oktmo_name):
            # Population exports can put MO in a section or in a parent row.
            assert "population_type_label" in record and norm(pieces[0]) == norm(
                record.population_type_label
            )
            prior = lines[:header_line]
            while prior and not prior[-1].strip():
                prior = prior[:-1]
            block = []
            for line in reversed(prior):
                if not line.strip():
                    break
                block.append(line)
            section = " ".join(reversed(block))
            assert norm(section) == norm(record.source_section), (section, record.source_section)
            # Confirm code selection in the original POST and the exact local MO label.
            metadata_path = str(Path(entry["archive_path"]).with_suffix(".json")).replace("\\", "/")
            request_metadata = json.loads(archive.read(metadata_path))
            query = request_metadata["payload"]["Qry"]
            codes = re.search(r"(?:^|;)oktmo:([^;]+)", query).group(1).split(",")
            assert str(record.source_oktmo) in codes
            if norm(record.source_oktmo_name) not in norm(section):
                parent_caption = None
                for line in reversed(lines[header_line + 1 : number - 1]):
                    cells = line.rsplit(";", len(years) + 1)
                    if (
                        len(cells) == len(years) + 2
                        and cells[0]
                        and all(not cell.strip() for cell in cells[1:])
                    ):
                        parent_caption = cells[0]
                        break
                assert norm(parent_caption) == norm(record.source_oktmo_name)
                binding = "exact_population_parent_row_caption_and_original_POST_code"
            else:
                binding = "exact_population_section_caption_and_original_POST_code"
        token = pieces[1 + years.index(str(source_year))].strip()
        number_value = float(token.replace(" ", "").replace("\xa0", "").replace(",", "."))
        same(number_value, record.value, "raw source cell")
        return {
            "raw_archive": path.as_posix(),
            "archive_member": entry["archive_path"],
            "archive_member_sha256": entry["sha256"],
            "source_sha256": source_sha,
            "archive_normalization": entry.get("transformation", "none"),
            "source_sha_matches_archived_bytes": not transformed,
            "source_csv_row": number,
            "explicit_year_header": ";".join(years),
            "source_value_token": token,
            "raw_caption_binding": binding,
            "raw_value_verified": True,
        }

    checks = []
    for row in examples.itertuples():
        for indicator, field, label in [
            ("8423005", "workers", "workers_2024"),
            ("8423007", "wage", "wage_2024"),
        ]:
            source = economy[
                economy.territory_id.eq(row.territory_id)
                & economy.year.eq(2024)
                & economy.indicator_id.eq(indicator)
            ]
            assert len(source) == 1
            record = source.iloc[0]
            assert record.cell_status == "observed" and int(record.period_code) == 79
            same(getattr(row, field), record.value, field)
            check = verify_raw_cell(record, 2024)
            checks.append(
                {
                    "territory_id": row.territory_id,
                    "metric": label,
                    "year": 2024,
                    "value": float(record.value),
                    "unit": record.unit,
                    "source_oktmo": record.source_oktmo,
                    "source_name": record.source_oktmo_name,
                    "source_url": record.source_url,
                    "period": record.period_label,
                    "provenance_table": economy_path.relative_to(repo).as_posix(),
                    **check,
                }
            )
        source = population[
            population.territory_id.eq(row.territory_id)
            & population.reference_date.eq("2024-01-01")
            & population.population_type.eq("11")
        ]
        assert len(source) == 1
        record = source.iloc[0]
        same(row.population_total, record.analysis_value, "population analysis value")
        same(row.population_total, record.value, "population source value")
        check = verify_raw_cell(record, 2024)
        checks.append(
            {
                "territory_id": row.territory_id,
                "metric": "population_2024_01_01",
                "year": 2024,
                "value": float(record.value),
                "unit": record.unit,
                "source_oktmo": record.source_oktmo,
                "source_name": record.source_oktmo_name,
                "source_url": record.source_url,
                "period": "2024-01-01",
                "provenance_table": population_path.relative_to(repo).as_posix(),
                **check,
            }
        )
        actual_raw = raw[raw.territory_id.eq(row.territory_id)]
        for year in (2023, 2024):
            values = actual_raw[
                actual_raw.category.eq(TOTAL_CATEGORY) & actual_raw.date.str.startswith(str(year))
            ]
            assert len(values) == 12
            same(values.value.mean(), getattr(row, f"nominal_mean_{year}"), f"nominal mean {year}")
        monthly_profile = profile[profile.territory_id.eq(row.territory_id)]
        same(monthly_profile["Продовольствие"].mean(), row.p_food, "annual mean food profile")
        same(
            monthly_profile["Общественное питание"].mean(),
            row.p_foodservice,
            "annual mean foodservice profile",
        )

    provenance = pd.DataFrame(checks)
    provenance.to_csv(out / "source_cell_checks.csv", index=False)
    for metric, prefix in (
        ("wage_2024", "wage"),
        ("workers_2024", "workers"),
        ("population_2024_01_01", "population"),
    ):
        records = provenance[provenance.metric.eq(metric)][
            ["territory_id", "source_url", "source_sha256", "source_csv_row"]
        ]
        records = records.rename(
            columns={col: f"{prefix}_{col}" for col in records if col != "territory_id"}
        )
        examples = examples.merge(records, on="territory_id", validate="one_to_one")
    examples["spending_source_sha256"] = sha(raw_path)
    examples["selection_rule"] = (
        "2024/core/within_region/economic; highest3 and lowest3 "
        "actual_log_total-predicted_log_total; ID ascending breaks ties"
    )
    examples["interpretation"] = (
        "descriptive residual extremes; not shocks, data errors, causal effects or forecast claims"
    )
    keep = [
        "selection_tail",
        "rank_in_tail",
        "territory_id",
        "year",
        "municipal_district_name",
        "municipal_district_type",
        "region_code",
        "region_name",
        "official_oktmo_8",
        "sample",
        "regime",
        "model",
        "fold",
        "actual_log_total",
        "predicted_log_total",
        "oof_log_total_residual",
        "fitted_geometric_total_2024_no_smearing",
        "actual_to_fitted_ratio",
        "nominal_mean_2023",
        "nominal_mean_2024",
        "nominal_growth_pct",
        "real_growth_pct",
        "population_total",
        "workers",
        "wage",
        "median_monthly_p_food_2024",
        "median_monthly_p_foodservice_2024",
        "p_food",
        "p_foodservice",
        "calendar_review",
        "code_type_changed",
        "conservative_eligible",
        "geography_status",
        *[
            col
            for col in examples
            if any(
                col.startswith(prefix)
                for prefix in ("wage_source_", "workers_source_", "population_source_")
            )
        ],
        "spending_source_sha256",
        "selection_rule",
        "interpretation",
    ]
    examples[keep].to_csv(out / "six_oof_examples_2024.csv", index=False)
    qa = {
        "status": "PASS",
        "selection_pool": len(pred),
        "examples": len(examples),
        "selection_ids_in_order": examples.territory_id.tolist(),
        "rule": (
            "Three largest positive and three smallest negative residuals; "
            "territory_id ascending tie-break."
        ),
        "source_cells_checked_against_archived_raw_csv": len(checks),
        "all_source_values_equal": True,
        "raw_csv_exact_source_hash_checks": int(provenance.source_sha_matches_archived_bytes.sum()),
        "archived_source_normalization_checks": int(
            (~provenance.source_sha_matches_archived_bytes).sum()
        ),
        "source_consumption_rows_retained": len(raw),
        "monthly_profile_median_not_annual_panel_mean": True,
        "core_manifest_output_hashes_verified": ["annual_panel.csv", "oof_predictions.parquet"],
        "no_refitting_no_network_no_model_calls": True,
        "interpretation": (
            "Examples selected after outcome inspection; exploratory residual descriptions only."
        ),
    }
    (out / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    for _, archive, _ in archives:
        archive.close()
    inputs = [
        core / "annual_panel.csv",
        core / "oof_predictions.parquet",
        price_path,
        raw_path,
        economy_path,
        population_path,
    ]
    manifest = {
        "script_sha256": sha(Path(__file__)),
        "frozen_manifest_sha256": sha(frozen_manifest_path),
        "archive_dir_semantics": (
            "directory containing the named release assets; physical paths are not embedded"
        ),
        "archive_assets": archive_records,
        "inputs": [{"path": p.relative_to(repo).as_posix(), "sha256": sha(p)} for p in inputs],
        "outputs": [
            {"path": p.name, "sha256": sha(p)}
            for p in sorted(out.iterdir())
            if p.suffix in {".csv", ".json"} and p.name != "manifest.json"
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(qa))
    print(
        examples[
            [
                "territory_id",
                "municipal_district_name",
                "oof_log_total_residual",
                "fitted_geometric_total_2024_no_smearing",
                "actual_to_fitted_ratio",
                "nominal_mean_2024",
                "nominal_growth_pct",
                "real_growth_pct",
                "median_monthly_p_food_2024",
                "median_monthly_p_foodservice_2024",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
