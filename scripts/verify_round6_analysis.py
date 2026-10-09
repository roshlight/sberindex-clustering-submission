"""Independent round-six arithmetic and cohort checks; no production analysis imports."""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

KEYS = ["territory_id", "year"]
PROFILES = ["p_food", "p_health", "p_marketplaces", "p_foodservice", "p_transport"]
SAMPLES = [
    "paired_available",
    "paired_stable_code",
    "paired_without_largest",
    "paired_urban_complete",
    "paired_stable_urban_complete",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parquet(path):
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def compare_table(expected, actual, keys, name, *, atol=1e-10, rtol=1e-10, subset=False):
    """Keys and categorical cells are exact; only declared numeric columns have tolerance."""
    require(not expected.duplicated(keys).any(), f"Expected duplicate key: {name}")
    require(not actual.duplicated(keys).any(), f"Output duplicate key: {name}")
    require(len(expected) == len(actual), f"Row count: {name}")
    require(
        set(expected.columns) <= set(actual.columns)
        if subset
        else set(expected.columns) == set(actual.columns),
        f"Columns: {name}",
    )
    left = expected.sort_values(keys).reset_index(drop=True)
    right = actual.sort_values(keys).reset_index(drop=True)
    for column in expected.columns:
        a, b = left[column], right[column]
        require(a.isna().equals(b.isna()), f"Missingness: {name}/{column}")
        valid = a.notna()
        if pd.api.types.is_bool_dtype(a) or column in keys or not pd.api.types.is_numeric_dtype(a):
            require(a[valid].tolist() == b[valid].tolist(), f"Exact cells: {name}/{column}")
        else:
            require(
                np.allclose(
                    a[valid].to_numpy(float), b[valid].to_numpy(float), atol=atol, rtol=rtol
                ),
                f"Numeric cells: {name}/{column}",
            )


def manifest_check(repo, root, branch, economy_raw):
    folder = root / branch
    manifest = read_json(folder / "manifest.json")
    inputs = {}
    for name, digest in manifest["inputs"].items():
        base = economy_raw if name.startswith("raw/") else repo
        relative = (
            name.removeprefix("raw/") if name.startswith("raw/") else name.removeprefix("repo/")
        )
        path = (base / relative).resolve()
        require(path.is_relative_to(base.resolve()), "Input outside declared base")
        require(sha(path) == digest, f"Input changed: {branch}/{name}")
        inputs[name] = digest
    script = repo / f"scripts/analyze_round6_{branch}.py"
    require(sha(script) == manifest["script_sha256"], f"Production script changed: {branch}")
    outputs = {}
    for name, digest in manifest["outputs"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()), "Output outside declared branch")
        require(sha(path) == digest, f"Output changed: {branch}/{name}")
        outputs[f"{branch}/{name}"] = digest
    outputs[f"{branch}/manifest.json"] = sha(folder / "manifest.json")
    return inputs, outputs


def rank_pair(data):
    """Pandas average ranks and lstsq projection, independent of rankdata/pinv production."""
    ranks = pd.DataFrame(np.asarray(data, float)).rank(method="average").to_numpy()
    controls = np.column_stack([np.ones(len(ranks)), ranks[:, 2:]])
    residual = ranks[:, :2] - controls @ np.linalg.lstsq(controls, ranks[:, :2], rcond=None)[0]

    def pearson(values):
        x, y = values[:, 0], values[:, 1]
        denominator = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean())
        return np.nan if denominator <= 1e-12 else float(np.corrcoef(x, y)[0, 1])

    return np.asarray([pearson(ranks[:, :2]), pearson(residual)])


def independent_statistics(data, draws):
    point = rank_pair(data)
    # Re-ranking each resample retains duplicate-ID ties; no full-sample ranks are reused.
    bootstrap = np.array([rank_pair(data[index]) for index in draws])
    loo = np.array([rank_pair(np.delete(data, index, axis=0)) for index in range(len(data))])
    result = {"rho": point[0], "partial_rho": point[1]}
    for index, name in enumerate(["rho", "partial_rho"]):
        finite = bootstrap[np.isfinite(bootstrap[:, index]), index]
        require(len(finite) >= 0.99 * len(draws), "Undefined bootstrap fraction")
        low, high = np.percentile(finite, [2.5, 97.5])
        result.update(
            {
                name + "_ci_low": low,
                name + "_ci_high": high,
                name + "_bootstrap_valid": len(finite),
                name + "_loo_min": np.nanmin(loo[:, index]),
                name + "_loo_max": np.nanmax(loo[:, index]),
                name + "_loo_valid": np.isfinite(loo[:, index]).sum(),
            }
        )
    return result


def independent_eta(ranks, labels):
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=ranks) / counts
    fitted = means[inverse]
    return np.mean((fitted - ranks.mean()) ** 2) / np.var(ranks)


def independent_bh(values):
    # q_i = min_{j: p_(j)>=p_(i)} m*p_(j)/j, clipped at one.
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = np.ones(len(values))
    for position, (index, _) in enumerate(ordered):
        adjusted[index] = min(
            1.0, min(len(values) * ordered[j][1] / (j + 1) for j in range(position, len(values)))
        )
    return adjusted


def regional_expected_panel(repo):
    raw = pd.read_csv(repo / "reports/data_round6/economy/regional_context_candidate.csv")
    annual = pd.read_csv(repo / "reports/round5_analysis/audit/annual_features.csv")
    frozen = parquet(repo / "reports/enriched_analysis/annual_networks/annual_profiles.parquet")
    assignments = parquet(repo / "reports/enriched_analysis/annual_networks/assignments.parquet")
    require(len(raw) == 276 and not raw.duplicated(KEYS + ["variable"]).any(), "Candidate grain")
    require(
        not raw[
            [
                "source_has_oktmo",
                "historical_boundary_equivalence_proven",
                "replace_frozen_bdpmo_value",
            ]
        ]
        .any()
        .any(),
        "Candidate restrictions",
    )
    compare_table(
        frozen, annual[KEYS + PROFILES], KEYS, "Frozen annual profiles", atol=1e-12, rtol=0
    )
    labels = assignments.loc[assignments.method == "leiden_hellinger"].copy()
    require(set(labels.month) == {"2023-12", "2024-12"}, "Annual label windows")
    labels["year"] = labels.month.str.slice(0, 4).astype(int)
    compare_table(frozen, labels[KEYS + PROFILES], KEYS, "Label profiles", atol=1e-12, rtol=0)
    selected = raw[
        ((raw.region_code == 26) & (raw.variable == "shipments_economic_activity"))
        | ((raw.region_code == 64) & (raw.variable == "food_sales_all_visible_activity"))
    ]
    columns = KEYS + [
        "population_total",
        "urban_share",
        "log_total",
        "total_mean",
        "p_food",
        "core_eligible",
        "conservative_eligible",
        "code_type_changed",
        "calendar_review",
    ]
    panel = selected.merge(annual[columns], on=KEYS, validate="one_to_one")
    panel = panel.merge(labels[KEYS + ["label"]], on=KEYS, validate="one_to_one")
    panel = panel.rename(columns={"value": "value_thousand_rubles"})
    panel["rubles_per_person"] = panel.value_thousand_rubles * 1000 / panel.population_total
    panel["log_intensity"] = np.log(panel.rubles_per_person.where(panel.rubles_per_person > 0))
    panel["row_eligible"] = (
        (panel.value_thousand_rubles > 0)
        & (panel.population_total > 0)
        & np.isfinite(panel[["log_intensity", "log_total", "p_food"]]).all(axis=1)
    )
    groups = {}
    for sample in SAMPLES:
        panel[sample] = False
    for region, frame in panel.groupby("region_code"):
        valid = frame[frame.row_eligible]
        ids = set(valid[valid.year == 2023].territory_id) & set(
            valid[valid.year == 2024].territory_id
        )
        available = frame[frame.territory_id.isin(ids)]
        changed = set(available.loc[available.code_type_changed, "territory_id"])
        urban = available[np.isfinite(available.urban_share)]
        urban_ids = set(urban[urban.year == 2023].territory_id) & set(
            urban[urban.year == 2024].territory_id
        )
        largest = (
            available[available.year == 2023]
            .sort_values(["population_total", "territory_id"], ascending=[False, True])
            .iloc[0]
            .territory_id
        )
        member_sets = [ids, ids - changed, ids - {largest}, urban_ids, urban_ids - changed]
        for sample, members in zip(SAMPLES, member_sets, strict=True):
            panel.loc[(panel.region_code == region) & panel.territory_id.isin(members), sample] = (
                True
            )
            groups[(int(region), sample)] = (sorted(members), int(largest))
    require(
        [len(groups[(r, s)][0]) for r in [26, 64] for s in SAMPLES]
        == [33, 23, 32, 19, 9, 38, 38, 37, 32, 32],
        "Pre-result sample contract",
    )
    return panel, groups


def verify_regions(repo, root, checks):
    folder = root / "regions"
    config = read_json(repo / "configs/round6_analysis_v1.json")
    require(
        config["minimum_n"] == 12
        and config["bootstrap_draws"] == 2000
        and config["permutation_draws"] == 4999
        and config["seed"] == 20260916,
        "Frozen protocol settings",
    )
    panel, groups = regional_expected_panel(repo)
    compare_table(
        panel,
        pd.read_csv(folder / "regional_panel.csv"),
        KEYS,
        "Regional panel",
        atol=1e-8,
        rtol=1e-12,
    )
    coverage, statistics, alignment, profiles, summaries = [], [], [], [], []
    calls = 0
    for (region, sample), (members, largest) in groups.items():
        random = np.random.default_rng(config["seed"] + region + len(members))
        indices = random.integers(0, len(members), (2000, len(members)))
        for year in [2023, 2024]:
            part = panel[
                (panel.region_code == region)
                & (panel.year == year)
                & panel.territory_id.isin(members)
            ].sort_values("territory_id")
            n = len(part)
            coverage.append(
                dict(
                    region_code=region,
                    year=year,
                    sample=sample,
                    n=n,
                    candidates=int(((panel.region_code == region) & (panel.year == year)).sum()),
                    largest_excluded_id=largest if sample == "paired_without_largest" else np.nan,
                    conditional_name_join=True,
                )
            )
            for outcome in ["log_total", "p_food"]:
                entry = dict(
                    region_code=region,
                    year=year,
                    sample=sample,
                    variable=part.variable.iloc[0],
                    outcome=outcome,
                    n=n,
                    status="ESTIMATED" if n >= 12 else "NOT_ESTIMATED_BELOW_MINIMUM_N",
                )
                if n >= 12:
                    data = part[["log_intensity", outcome, "population_total"]].to_numpy(float)
                    entry.update(independent_statistics(data, indices))
                    calls += 1
                    if sample in {"paired_urban_complete", "paired_stable_urban_complete"}:
                        data = part[
                            ["log_intensity", outcome, "population_total", "urban_share"]
                        ].to_numpy(float)
                        extra = independent_statistics(data, indices)
                        entry.update(
                            {
                                "urban_" + key: value
                                for key, value in extra.items()
                                if key.startswith("partial_rho")
                            }
                        )
                        calls += 1
                statistics.append(entry)
            if sample != "paired_available":
                continue
            ranks = part.log_intensity.rank(method="average").to_numpy()
            labels = part.label.to_numpy()
            observed = independent_eta(ranks, labels)
            random = np.random.default_rng(config["seed"] + region * 100 + year)
            null = np.array(
                [independent_eta(ranks, random.permutation(labels)) for _ in range(4999)]
            )
            alignment.append(
                dict(
                    region_code=region,
                    year=year,
                    n=n,
                    clusters=len(set(labels)),
                    rank_eta_squared=observed,
                    null_mean=null.mean(),
                    permutation_p=(1 + np.count_nonzero(null >= observed - 1e-15)) / 5000,
                    permutation_draws=4999,
                )
            )
            for label, group in part.groupby("label"):
                profiles.append(
                    dict(
                        region_code=region,
                        year=year,
                        label=label,
                        n=len(group),
                        small_group=len(group) < 5,
                        intensity_median=group.rubles_per_person.median(),
                        food_share_median=group.p_food.median(),
                        total_median=group.total_mean.median(),
                    )
                )
            summaries.append(
                dict(
                    region_code=region,
                    year=year,
                    n=n,
                    variable=part.variable.iloc[0],
                    intensity_median=part.rubles_per_person.median(),
                    intensity_q25=part.rubles_per_person.quantile(0.25),
                    intensity_q75=part.rubles_per_person.quantile(0.75),
                    largest_id=largest,
                    largest_value=part.loc[
                        part.territory_id == largest, "value_thousand_rubles"
                    ].iloc[0],
                )
            )
    aligned = pd.DataFrame(alignment)
    aligned["permutation_q_bh"] = independent_bh(aligned.permutation_p.to_numpy())
    outputs = [
        ("coverage.csv", coverage, ["region_code", "year", "sample"]),
        ("associations.csv", statistics, ["region_code", "year", "sample", "outcome"]),
        ("cluster_alignment.csv", aligned, ["region_code", "year"]),
        ("cluster_profiles.csv", profiles, ["region_code", "year", "label"]),
        ("regional_summary.csv", summaries, ["region_code", "year"]),
    ]
    for name, expected, keys in outputs:
        compare_table(
            pd.DataFrame(expected),
            pd.read_csv(folder / name),
            keys,
            name,
            atol=1e-8 if name in {"cluster_profiles.csv", "regional_summary.csv"} else 1e-10,
            rtol=1e-12 if name in {"cluster_profiles.csv", "regional_summary.csv"} else 1e-10,
        )
    qa = read_json(folder / "QA.json")
    expected_qa = dict(
        candidate_rows=144,
        association_rows=40,
        cluster_checks=4,
        frozen_profiles_matched=True,
        models_trained=0,
        new_clusters=0,
        accepted_historical_joins=0,
        old_panel_changed=False,
        same_municipalities_in_both_years=True,
    )
    require(all(qa.get(key) == value for key, value in expected_qa.items()), "Regional QA claims")
    checks.append(
        dict(
            name="regions",
            status="PASS",
            association_rows=40,
            not_estimated_rows=4,
            independently_reranked_bootstrap_families=calls,
            bootstrap_draws_per_family=2000,
            independent_permutation_draws=4 * 4999,
            numeric_output_tables=6,
            source_join_remains_conditional=True,
        )
    )


def read_strings(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compare_strings(expected, actual, keys, name, subset=False):
    compare_table(
        expected.fillna("").astype(str), actual.fillna("").astype(str), keys, name, subset=subset
    )


def decimal_literal(value):
    return Decimal(str(value).replace("\u00a0", "").replace(" ", "").replace(",", "."))


def printed_half_step(value):
    text = str(value).strip().replace(",", ".")
    precision = len(text.split(".")[1]) if "." in text else 0
    return Decimal(5) * Decimal(10) ** (-precision - 1)


def independent_rounding(old, new, percent):
    a, b, p = map(decimal_literal, [old, new, percent])
    da, db, dp = map(printed_half_step, [old, new, percent])
    lower, upper = (b - db) / (a + da) * 100, (b + db) / (a - da) * 100
    return dict(
        status="different_beyond_rounding"
        if upper < p - dp or lower > p + dp
        else "rounding_compatible",
        ratio_percent=float(100 * b / a),
        published_percent=float(p),
        difference_pp=float(100 * b / a - p),
        ratio_rounding_low=float(lower),
        ratio_rounding_high=float(upper),
        published_rounding_halfwidth=float(dp),
        difference_cause="unknown",
        same_definition_confirmed=False,
    )


def independent_docx(path):
    """Direct XML tables; never import the collecting/admission extractor."""
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in document.iter(namespace + "tbl"):
        rows = []
        for row in table.findall(namespace + "tr"):
            rows.append(
                [
                    "".join(node.text or "" for node in cell.iter(namespace + "t")).strip()
                    for cell in row.findall(namespace + "tc")
                ]
            )
        tables.append(rows)
    matches = [
        i
        for i, rows in enumerate(tables)
        if "По хозяйственным" in str(rows) and "По чистым" in str(rows)
    ]
    require(len(matches) == 1, "DOCX shipment table locator")
    a, b = tables[matches[0]], tables[matches[0] + 1]
    totals = [[row[1:] for row in table if row and row[0] == "Всего"] for table in [a, b]]
    require(list(map(len, totals)) == [1, 1], "DOCX total rows")
    require(len(totals[0][0]) == 4 and len(totals[1][0]) == 2, "DOCX total columns")
    headers = [" ".join(" ".join(row) for row in a[:3]), " ".join(" ".join(row) for row in b[:2])]
    require(
        all(all(token in h for token in ["2024", "2023", "тыс. рублей"]) for h in headers),
        "DOCX period and units",
    )
    return totals[0][0], totals[1][0], headers


def verify_admission(repo, root, raw, checks):
    folder = root / "admission"
    source = repo / "reports/data_round6/economy"
    candidate = read_strings(source / "regional_context_candidate.csv")
    old = read_strings(repo / "reports/data_round5/economy/annual_candidate_panel.csv")
    missing = read_strings(repo / "reports/data_round5/economy/target_cell_decisions.csv")
    audit = read_strings(repo / "reports/round5_analysis/audit/annual_features.csv")
    saved = read_strings(folder / "row_admission.csv")
    compare_strings(
        candidate, saved[candidate.columns], KEYS + ["variable"], "Admission source cells"
    )
    require(
        all(
            decimal_literal(r.value) == decimal_literal(r.source_value_literal)
            for r in candidate.itertuples()
        ),
        "Candidate decimal literals",
    )
    fields = [
        "region_code",
        "municipal_district_name",
        "municipal_district_type",
        "official_oktmo_8",
        "core_eligible",
        "conservative_eligible",
        "calendar_review",
        "code_type_changed",
    ]
    meta = audit[KEYS + fields].rename(columns={"region_code": "frozen_region_code"})
    expected = candidate.merge(meta, on=KEYS, validate="many_to_one")

    def name_key(text):
        text = text.casefold().replace("ё", "е")
        text = re.sub(
            r"\b(?:муниципальный район|муниципальный округ|городской округ|город-курорт|город)\b",
            "",
            text,
        )
        return " ".join(text.removeprefix("г. ").strip(" .").split())

    lookup = audit[audit.region_code.isin(["26", "64"])].copy()
    lookup["normalized"] = lookup.municipal_district_name.map(name_key)
    require(
        not lookup.duplicated(["region_code", "year", "normalized"]).any(), "Name lookup ambiguity"
    )
    mapping = lookup.set_index(["region_code", "year", "normalized"]).territory_id.to_dict()
    require(
        all(
            mapping[(r.region_code, r.year, name_key(r.source_name))] == r.territory_id
            and r.region_code == r.frozen_region_code
            for r in expected.itertuples()
        ),
        "Name candidate join",
    )
    expected["candidate_indicator_id"] = np.where(
        expected.variable.str.startswith("shipments_"), "8401011", "8401003"
    )
    counterpart = old[
        KEYS + ["indicator_id", "value", "value_status", "publication_vintage"]
    ].rename(
        columns={
            "indicator_id": "candidate_indicator_id",
            "value": "frozen_counterpart_value",
            "value_status": "frozen_counterpart_status",
            "publication_vintage": "frozen_counterpart_vintage",
        }
    )
    expected = expected.merge(
        counterpart, on=KEYS + ["candidate_indicator_id"], validate="many_to_one"
    )
    reasons = missing.rename(
        columns={"indicator_id": "candidate_indicator_id", "reason": "frozen_missing_reason"}
    )
    expected = expected.merge(
        reasons[KEYS + ["candidate_indicator_id", "frozen_missing_reason"]],
        on=KEYS + ["candidate_indicator_id"],
        how="left",
        validate="many_to_one",
    )
    dates = {
        ("26", "2023"): "2024-07-31",
        ("26", "2024"): "2025-07-03",
        ("64", "2023"): "2024-04-23",
        ("64", "2024"): "2025-04-22",
    }
    expected["catalog_date_label"] = [dates[(r.region_code, r.year)] for r in expected.itertuples()]
    for column in [
        "availability_by_observation_year_end_proven",
        "first_publication_date_proven",
        "frozen_value_replacement",
        "model_admission",
        "retail_equivalence_confirmed",
    ]:
        expected[column] = False
    expected["direct_numeric_reconciliation_possible"] = expected.frozen_counterpart_value.ne(
        ""
    ) & expected.variable.str.startswith("shipments_")
    expected["descriptive_context_status"] = "ready_with_conditions"
    compare_strings(expected, saved, KEYS + ["variable"], "All admission rows")
    coverage = []
    for (region, year, variable), group in expected.groupby(["region_code", "year", "variable"]):
        denominator = len(audit[(audit.region_code == region) & (audit.year == year)])
        coverage.append(
            dict(
                region_code=int(region),
                year=int(year),
                variable=variable,
                candidate_rows=len(group),
                distinct_municipalities=group.territory_id.nunique(),
                regional_panel_municipalities=denominator,
                regional_panel_coverage_percent=100 * group.territory_id.nunique() / denominator,
                national_panel_municipalities=2016,
                national_coverage_percent=100 * group.territory_id.nunique() / 2016,
                **{
                    c: int(group[c].eq("True").sum())
                    for c in [
                        "core_eligible",
                        "conservative_eligible",
                        "code_type_changed",
                        "calendar_review",
                    ]
                },
            )
        )
    compare_table(
        pd.DataFrame(coverage),
        pd.read_csv(folder / "coverage.csv"),
        ["region_code", "year", "variable"],
        "Admission coverage",
    )
    intersections = (
        expected.fillna("")
        .groupby(
            KEYS + ["candidate_indicator_id", "frozen_counterpart_status", "frozen_missing_reason"],
            dropna=False,
        )
        .agg(
            candidate_variables=("variable", lambda values: ";".join(sorted(values))),
            candidate_numeric_rows=("variable", "size"),
        )
        .reset_index()
    )
    intersections["missing_cell_filled"] = False
    compare_strings(
        intersections,
        read_strings(folder / "missing_cell_intersections.csv"),
        KEYS + ["candidate_indicator_id"],
        "Missing old positions",
    )
    cooccurrence = old.merge(candidate[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one")
    cooccurrence = cooccurrence[KEYS + ["indicator_id", "value", "value_status", "unit"]]
    cooccurrence["same_metric_reconciliation"] = False
    compare_strings(
        cooccurrence,
        read_strings(folder / "frozen_indicator_cooccurrence.csv"),
        KEYS + ["indicator_id"],
        "Other frozen indicators",
    )
    review = read_json(source / "independent_QA.json")
    prior = {(str(r["territory_id"]), str(r["year"])): r for r in review["stavropol_documents"]}
    published = []
    for doc in read_strings(source / "document_inventory.csv").itertuples():
        evidence = prior[(doc.territory_id, doc.year)]
        shipment, resale = (
            evidence["independent_shipment_tokens"],
            evidence["independent_resale_tokens"],
        )
        path, headers = evidence["pdf_path"], ["", ""]
        method = "pinned_prior_independent_pdf_review"
        if doc.year == "2024":
            path = Path(doc.pdf_path).with_suffix(".docx").as_posix()
            ship_new, resale_new, headers = independent_docx(raw / path)
            require(
                list(map(decimal_literal, ship_new + resale_new))
                == list(map(decimal_literal, shipment + resale)),
                "Fresh DOCX versus independent PDF",
            )
            shipment, resale = ship_new, resale_new
            method = "fresh_stdlib_DOCX_XML_plus_pinned_PDF_review"
        for variable, amount, percent, header in [
            ("shipments_economic_activity", *shipment[:2], headers[0]),
            ("shipments_pure_activity", *shipment[2:], headers[0]),
            ("resale_non_own_goods", *resale, headers[1]),
        ]:
            source_row = candidate[
                (candidate.territory_id == doc.territory_id)
                & (candidate.year == doc.year)
                & (candidate.variable == variable)
            ]
            require(
                len(source_row) == 1
                and decimal_literal(source_row.iloc[0].value) == decimal_literal(amount),
                "Fresh source total versus candidate",
            )
            published.append(
                dict(
                    territory_id=int(doc.territory_id),
                    year=int(doc.year),
                    variable=variable,
                    amount_literal=amount,
                    published_percent_literal=percent,
                    published_percent=float(decimal_literal(percent)),
                    comparison_prior_year=int(doc.year) - 1,
                    raw_path=path,
                    source_sha256=sha(raw / path),
                    source_header=header or np.nan,
                    extraction_method=method,
                    real_or_nominal_index_definition="not_established",
                )
            )
    published = pd.DataFrame(published)
    literals = {
        "amount_literal": str,
        "published_percent_literal": str,
        "old_amount_literal": str,
        "new_amount_literal": str,
    }
    compare_table(
        published,
        pd.read_csv(folder / "published_percentages.csv", dtype=literals),
        KEYS + ["variable"],
        "Published source percentages",
    )
    paired = published[published.year == 2023].merge(
        published[published.year == 2024],
        on=["territory_id", "variable"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    ratios = []
    for r in paired.itertuples():
        ratios.append(
            dict(
                territory_id=r.territory_id,
                variable=r.variable,
                old_amount_literal=r.amount_literal_old,
                new_amount_literal=r.amount_literal_new,
                published_percent_literal=r.published_percent_literal_new,
                **independent_rounding(
                    r.amount_literal_old, r.amount_literal_new, r.published_percent_literal_new
                ),
                old_source_sha256=r.source_sha256_old,
                new_source_sha256=r.source_sha256_new,
            )
        )
    ratios = pd.DataFrame(ratios)
    compare_table(
        ratios,
        pd.read_csv(folder / "cross_vintage_ratio_checks.csv", dtype=literals),
        ["territory_id", "variable"],
        "Cross-edition rounding arithmetic",
    )
    summaries = []
    for variable, group in ratios.groupby("variable"):
        summaries.append(
            dict(
                variable=variable,
                pairs=len(group),
                rounding_compatible=int((group.status == "rounding_compatible").sum()),
                different_beyond_rounding=int((group.status == "different_beyond_rounding").sum()),
                median_absolute_difference_pp=group.difference_pp.abs().median(),
                max_absolute_difference_pp=group.difference_pp.abs().max(),
                mean_difference_pp=group.difference_pp.mean(),
                explanation_identified=False,
            )
        )
    compare_table(
        pd.DataFrame(summaries),
        pd.read_csv(folder / "ratio_summary.csv"),
        ["variable"],
        "Rounding summaries",
    )
    econ, pure = "shipments_economic_activity", "shipments_pure_activity"
    amounts = pd.read_csv(source / "regional_context_candidate.csv")
    relation = (
        amounts[amounts.variable.isin([econ, pure])]
        .pivot(index=KEYS + ["panel_name"], columns="variable", values="value")
        .reset_index()
    )
    relation["economic_minus_pure_thousand_rubles"] = relation[econ] - relation[pure]
    relation["pure_div_economic_percent"] = 100 * relation[pure] / relation[econ]
    relation["difference_interpretation"] = (
        "definition_and_population_difference_not_small_firm_contribution"
    )
    relation["nested_population_or_additivity_proven"] = False
    relation["boundary_equivalence_proven"] = False
    compare_table(
        relation,
        pd.read_csv(folder / "shipment_definition_comparison.csv"),
        KEYS,
        "Shipment scope arithmetic",
        atol=1e-8,
        rtol=1e-12,
    )
    food = []
    for r in read_strings(source / "food_reconciliation.csv").itertuples():
        amounts_year = candidate[
            (candidate.variable == "food_sales_all_visible_activity") & (candidate.year == r.year)
        ]
        total = sum(map(decimal_literal, amounts_year.source_value_literal))
        regional = decimal_literal(r.regional_total_thousand_rubles)
        food.append(
            dict(
                year=int(r.year),
                municipal_rows=len(amounts_year),
                municipal_sum_thousand_rubles=float(total),
                regional_total_thousand_rubles=float(regional),
                regional_minus_municipal_thousand_rubles=float(regional - total),
                residual_percent_of_region=float(100 * (regional - total) / regional),
                residual_cause="unknown",
                allocated_to_municipalities=False,
            )
        )
    compare_table(
        pd.DataFrame(food),
        pd.read_csv(folder / "food_reconciliation.csv"),
        ["year"],
        "Food regional residual",
        atol=1e-8,
        rtol=1e-12,
    )
    decisions = read_strings(folder / "admission_decisions.csv")
    require(set(decisions.variable) == set(candidate.variable), "All admission constructs")
    for r in decisions.itertuples():
        require(
            set(candidate[candidate.variable == r.variable].source_scope) == {r.source_scope},
            "Admission source scope",
        )
        require(
            r.counterpart_indicator_id
            == ("8401011" if r.variable.startswith("shipments_") else "8401003"),
            "Counterpart indicator",
        )
        require(
            r.frozen_value_replacement == r.missing_cell_fill == r.model_admission == "False",
            "No automatic admission",
        )
        require(
            r.historical_boundary_equivalence == r.vintage_equivalence == "unknown"
            and r.join_admission == "name_candidate_only",
            "Join remains unproven",
        )
    unique = expected.drop_duplicates(KEYS)
    qa = read_json(folder / "QA.json")
    facts = dict(
        rows=len(expected),
        municipalities=expected.territory_id.nunique(),
        municipality_years=len(unique),
        unique_candidate_keys=True,
        candidate_to_metadata_multiplier=1.0,
        unmatched_candidate_rows=0,
        name_region_year_independent_matches=len(expected),
        distinct_missing_old_positions_with_related_candidate=len(intersections),
        direct_numeric_same_construct_reconciliations=int(
            expected.direct_numeric_reconciliation_possible.sum()
        ),
        other_frozen_numeric_cooccurrences=int(cooccurrence.value.ne("").sum()),
        old_panel_rows=len(old),
        old_numeric_values=int(old.value.ne("").sum()),
        old_missing_values=int(old.value.eq("").sum()),
        missing_cells_filled=0,
        new_panel_values=0,
        models_fit=0,
        fresh_DOCX_rechecks=33,
        published_percent_rows=len(published),
        two_vintage_percent_checks=len(ratios),
        source_input_hashes_verified=len(read_json(source / "manifest.json")["raw_inputs"]),
        network_requests=0,
        core_municipality_years=int(unique.core_eligible.eq("True").sum()),
        conservative_municipality_years=int(unique.conservative_eligible.eq("True").sum()),
        code_changed_municipality_years=int(unique.code_type_changed.eq("True").sum()),
        code_changed_municipalities=unique[
            unique.code_type_changed == "True"
        ].territory_id.nunique(),
    )
    require(all(qa.get(key) == value for key, value in facts.items()), "Admission QA counts")
    checks.append(
        dict(
            name="admission",
            status="PASS",
            numerical_tables=10,
            candidate_rows=276,
            independently_read_docx=33,
            cross_edition_comparisons=99,
            old_missing_cells_filled=0,
            new_pdf_reextraction=False,
        )
    )


def verify_context_sources(repo, folder):
    rows = []

    def add(family, key, measurement, date, kind, retrieved, digest, url):
        rows.append(
            dict(
                source_family=family,
                evidence_key=key,
                measurement_time=measurement,
                source_date_or_version=date,
                date_kind=kind,
                retrieved_at=retrieved,
                sha256=digest,
                source_url=url,
                historical_model_input_approved=False,
            )
        )

    evidence = read_strings(repo / "reports/data_round4/cases/evidence_ledger.csv")
    for key, group in evidence[evidence.territory_id_context_candidate == "331"].groupby(
        "source_id"
    ):
        r = group.iloc[0]
        add(
            "municipal_investment_yearbook",
            key,
            ";".join(sorted(set(group.period))),
            r.catalogue_display_date or r.edition_year,
            "catalogue_label" if r.catalogue_display_date else "edition_year",
            r.retrieved_at_utc,
            r.source_sha256,
            r.source_url,
        )
    for r in read_strings(repo / "reports/data_round6/cases/evidence_ledger.csv").itertuples():
        add(
            "airport_documentary_evidence",
            r.evidence_id,
            r.reported_year,
            r.source_publication_date,
            "publication_or_registry_revision_as_reported",
            r.retrieved_at_utc,
            r.source_sha256,
            r.source_url,
        )
    geo = repo / "reports/data_round6/geography"
    downloads = (
        read_strings(geo / "downloaded_sources.csv")
        .drop_duplicates("sha256", keep="last")
        .set_index("sha256")
    )
    for r in read_strings(geo / "crosswalk_versions.csv").itertuples():
        src = downloads.loc[r.sha256]
        add(
            "oktmo_transition_release",
            r.version,
            "effective_dates_not_in_csv",
            r.version,
            "release_label",
            src.retrieved_at,
            r.sha256,
            src.url,
        )
    versions = read_strings(repo / "reports/data_round5/geography/registry_versions.csv").set_index(
        "sha256"
    )
    parents = read_strings(geo / "priority_settlement_parent_candidates.csv")
    for r in parents[["registry_version_label", "registry_sha256"]].drop_duplicates().itertuples():
        src = versions.loc[r.registry_sha256]
        add(
            "historical_classifier_release",
            r.registry_version_label,
            "records_have_individual_introduction_dates",
            r.registry_version_label,
            "registry_release_label",
            src.retrieved_at,
            r.registry_sha256,
            src.get("source_url", src.get("url", "")),
        )
    sme = repo / "reports/data_round6/sme"
    sample = read_strings(sme / "worker_sample_summary.csv").iloc[0]
    attempts = read_strings(sme / "retrieval_attempts.csv")
    retrieved = ";".join(
        sorted(set(attempts.loc[attempts.url == sample.archive_url, "retrieved_at"]))
    )
    for r in read_strings(sme / "worker_dates.csv").itertuples():
        add(
            "fns_workers_selected_xml",
            r.workers_state,
            r.data_date,
            r.document_date,
            "xml_document_date",
            retrieved,
            sample.sample_sha256,
            sample.archive_url,
        )
    for r in read_strings(sme / "author_source_lineage.csv").itertuples():
        add(
            "author_code_and_assets",
            r.path,
            "asset_vintage_not_independently_dated_per_row",
            r.git_commit,
            "git_commit",
            r.prior_package_retrieved_at,
            r.sha256,
            r.git_url,
        )
    compare_strings(
        pd.DataFrame(rows),
        read_strings(folder / "source_availability.csv"),
        ["source_family", "evidence_key"],
        "Document dates and source lineage",
        subset=True,
    )
    require(len(rows) == 45, "Availability documentary rows")


def verify_context(repo, root, checks):
    folder = root / "context"
    base = repo / "reports/data_round6"
    expected_tables = {}

    def check_table(name, rows, keys, dtype=None):
        expected = pd.DataFrame(rows)
        actual = pd.read_csv(folder / name, dtype=dtype)
        compare_table(expected, actual, keys, name, atol=1e-8, rtol=1e-12)
        expected_tables[name] = len(expected)

    finance = pd.read_csv(base / "cases/airport_annual_finance.csv")
    investment = pd.read_csv(
        repo / "reports/round5_analysis/context/chersky_investment_alignment.csv"
    ).set_index("year")
    cash = finance[finance.financial_kind == "annual_cash_execution"].copy()
    comparisons = []
    for r in cash.itertuples():
        other = investment.loc[r.year]
        a, b, half = (
            r.value_million_rubles,
            other.value_million_rubles,
            other.rounding_half_step_million_rubles,
        )
        require(
            pd.isna(a) or abs(a - r.value_thousand_rubles / 1000) < 1e-12,
            "Cash thousand/million units",
        )
        require(pd.isna(a) == (r.observation_status == "blank_not_zero"), "Cash missing status")
        comparisons.append(
            dict(
                year=r.year,
                municipality=other.municipality,
                airport_cash_million_rubles=a,
                cash_status=r.observation_status,
                municipal_investment_million_rubles=b,
                municipal_rounding_half_step_million_rubles=half,
                cash_divided_by_municipal_investment=a / b,
                ratio_lower_from_municipal_rounding_only=a / (b + half),
                ratio_upper_from_municipal_rounding_only=a / (b - half),
                ratio_is_project_share=False,
                model_join_accepted=False,
                cash_scope=r.scope,
                investment_scope=other.scope,
                cash_publication_date=r.publication_date,
                cash_source_cells=r.source_cells,
                cash_source_sha256=r.source_sha256,
                cash_source_url=r.source_url,
                investment_source_sha256s=other.source_sha256s,
                investment_source_urls=other.source_urls,
            )
        )
    check_table("airport_comparison.csv", comparisons, ["year"])
    by_year = {r["year"]: r for r in comparisons}
    changes = []
    for start, end in [(2022, 2023), (2023, 2024)]:
        for variable in ["airport_cash_million_rubles", "municipal_investment_million_rubles"]:
            x, y = by_year[start][variable], by_year[end][variable]
            changes.append(
                dict(
                    start_year=start,
                    end_year=end,
                    metric=variable,
                    start_million_rubles=x,
                    end_million_rubles=y,
                    nominal_change_million_rubles=y - x,
                    nominal_change_pct=100 * (y - x) / x,
                    cross_metric_decomposition=False,
                )
            )
    check_table("airport_changes.csv", changes, ["start_year", "end_year", "metric"])
    edges = read_strings(base / "geography/candidate_transitions_to_panel.csv")
    groups = []
    for (year, code), part in edges.groupby(["year", "source_code"]):
        require(
            all(
                part[c].nunique() == 1
                for c in ["region", "source_rows", "new_code_predecessors_all_regions"]
            ),
            "Code group input consistency",
        )
        parents, predecessors = part.parent_code.nunique(), part.old_code.nunique()
        total_predecessors = int(part.new_code_predecessors_all_regions.iloc[0])
        state = "single_observed_predecessor_parent"
        if parents > 1:
            state = "multiple_candidate_parents"
        elif total_predecessors > predecessors:
            state = "additional_predecessor_outside_candidate_set"
        elif predecessors > 1:
            state = "multiple_settlements_same_candidate_parent"
        groups.append(
            dict(
                year=int(year),
                code_region=part.region.iloc[0],
                source_code=code,
                source_rows_once=int(part.source_rows.iloc[0]),
                candidate_edge_rows=len(part),
                candidate_old_codes=predecessors,
                all_crosswalk_predecessor_codes=total_predecessors,
                candidate_parent_codes=parents,
                candidate_territories=";".join(sorted(set(part.territory_id))),
                ambiguity_class=state,
                source_rows_are_unique_firms=False,
                historical_assignment_approved=False,
            )
        )
    group_frame = pd.DataFrame(groups)
    check_table(
        "geography_code_groups.csv",
        groups,
        ["year", "source_code"],
        dtype={"source_code": str, "candidate_territories": str},
    )
    inventory = read_strings(repo / "reports/data_round5/geography/sme_code_inventory.csv")
    totals = (
        inventory.assign(source_rows=inventory.source_rows.astype(int))
        .groupby("year")
        .source_rows.sum()
    )
    nonnull = inventory[inventory.source_code_is_null == "False"]
    require(not nonnull.duplicated(["year", "source_code"]).any(), "Inventory code/year uniqueness")
    index = nonnull.set_index(["year", "source_code"]).source_rows
    require(
        all(
            int(index.loc[(str(r.year), r.source_code)]) == r.source_rows_once
            for r in group_frame.itertuples()
        ),
        "Unique source row denominator lineage",
    )
    partitions = [
        (year, region, part)
        for (year, region), part in group_frame.groupby(["year", "code_region"])
    ]
    partitions += [
        (year, "ALL_CANDIDATE_CODE_REGIONS", part) for year, part in group_frame.groupby("year")
    ]
    queue = []
    for year, region, part in partitions:
        source_rows = int(part.source_rows_once.sum())
        naive = int((part.source_rows_once * part.candidate_edge_rows).sum())
        denominator = int(totals.loc[str(year)])
        multiple = part[part.all_crosswalk_predecessor_codes > 1]
        queue.append(
            dict(
                year=year,
                code_region=region,
                source_code_groups=len(part),
                source_rows_once=source_rows,
                denominator_all_sme_year_rows=denominator,
                share_all_sme_year_rows_pct=100 * source_rows / denominator,
                naive_edge_weighted_rows=naive,
                naive_excess_rows=naive - source_rows,
                naive_excess_pct_of_correct_queue=100 * (naive - source_rows) / source_rows,
                multi_predecessor_code_groups=len(multiple),
                multi_predecessor_rows_once=int(multiple.source_rows_once.sum()),
                multiple_candidate_parent_groups=int((part.candidate_parent_codes > 1).sum()),
                additional_predecessor_outside_groups=int(
                    (part.ambiguity_class == "additional_predecessor_outside_candidate_set").sum()
                ),
                historical_assignments_approved=0,
                denominator_is_code_region_population=False,
            )
        )
    check_table("geography_queue_summary.csv", queue, ["year", "code_region"])
    ambiguous = read_strings(base / "geography/ambiguous_transition_edges.csv")
    reverse = ambiguous.predecessor_code_count.astype(int) > 1
    forward = ambiguous.successor_code_count.astype(int) > 1
    require((reverse | forward).all(), "Ambiguity source contains ordinary edge")
    ambiguous["edge_class"] = np.where(
        reverse & forward, "both_directions", np.where(reverse, "reverse_only", "forward_only")
    )
    summaries = [
        dict(
            edge_class=label,
            edges=len(part),
            distinct_old_codes=part.old_code.nunique(),
            distinct_new_codes=part.new_code.nunique(),
            regions=part.region.nunique(),
            denominator_all_ambiguous_edges=len(ambiguous),
            edge_share_pct=100 * len(part) / len(ambiguous),
            unique_code_counts_not_additive_between_classes=True,
        )
        for label, part in ambiguous.groupby("edge_class")
    ]
    check_table("transition_ambiguity_summary.csv", summaries, ["edge_class"])
    parents = read_strings(base / "geography/priority_settlement_parent_candidates.csv")
    parent_summaries = []
    for (year, ter), part in parents.groupby(["year", "ter"]):
        nparents = part.groupby("settlement_code").parent_code.nunique()
        parent_summaries.append(
            dict(
                year=int(year),
                classifier_ter=ter,
                candidate_rows=len(part),
                settlement_codes=part.settlement_code.nunique(),
                multiple_parent_settlement_codes=int((nparents > 1).sum()),
                parent_codes=part.parent_code.nunique(),
                record_date_after_reference_rows=int(
                    (part.record_date_after_reference == "True").sum()
                ),
                unparsed_date_rows=int((part.record_date_unparsed == "True").sum()),
                registry_version_label=part.registry_version_label.iloc[0],
                historical_company_assignments_approved=0,
            )
        )
    check_table(
        "settlement_parent_summary.csv",
        parent_summaries,
        ["year", "classifier_ter"],
        dtype={"classifier_ter": str},
    )
    syntax = pd.read_csv(base / "sme/asset_code_syntax.csv")
    assets = []
    for asset, part in syntax.groupby("asset"):
        counts = part.set_index("code_state")["rows"].to_dict()
        denominator = int(part.denominator_rows.iloc[0])
        exact = int(counts["exact_8_or_11_digits"])
        require(
            sum(counts.values()) == denominator and part.denominator_rows.nunique() == 1,
            "Asset population partition",
        )
        assets.append(
            dict(
                asset=asset,
                source_rows=denominator,
                exact_syntax_rows=exact,
                syntax_review_rows=denominator - exact,
                syntax_review_pct=100 * (denominator - exact) / denominator,
                scientific_notation_rows=counts["scientific_notation"],
                other_digit_length_rows=counts["other_digit_length"],
                empty_rows=counts["empty"],
                other_syntax_rows=counts["other_syntax"],
                exact_syntax_proves_historical_validity=False,
                rows_are_distinct_companies=False,
            )
        )
    check_table("sme_asset_readiness.csv", assets, ["asset"])
    workers = []
    for r in pd.read_csv(base / "sme/worker_dates.csv").itertuples():
        measurement, document = [
            datetime.strptime(t, "%d.%m.%Y").date() for t in [r.data_date, r.document_date]
        ]
        for target in [2023, 2024]:
            workers.append(
                dict(
                    target_year=target,
                    data_date=measurement.isoformat(),
                    document_date=document.isoformat(),
                    workers_state=r.workers_state,
                    records=r.records,
                    denominator_sample_records=r.denominator_records,
                    document_lag_days_from_data_date=(document - measurement).days,
                    measurement_year_matches_target=measurement.year == target,
                    document_after_target_year_end=document.year > target,
                    eligible_historical_input=False,
                    representative_sample=False,
                )
            )
    check_table("worker_temporal_eligibility.csv", workers, ["target_year", "workers_state"])
    verify_context_sources(repo, folder)
    expected_tables["source_availability.csv"] = 45
    qa = read_json(folder / "QA.json")
    require(
        qa["tables"] == expected_tables and qa["model_adoption"] is False,
        "Context QA counts and non-adoption",
    )
    claims = read_strings(folder / "claim_evidence.csv").set_index("claim")
    expected_status = {
        "cash_2022_and_2023_documented": "accepted_documentary",
        "code_queue_double_count_quantified": "accepted_scoped",
        "worker_sample_measurement_and_document_date_differ": "accepted_scoped",
        "cash_is_municipal_investment": "unsupported",
        "airport_contributed_to_spike": "unknown",
        "airport_cash_2024_zero": "unsupported",
        "other_projects_or_statistical_scope_explain_spike": "unknown",
        "queue_gives_extra_municipal_coverage": "unsupported",
        "single_candidate_parent_proves_location": "unsupported",
        "workers_xml_improves_2023_2024_features": "unsupported",
        "asset_syntax_issue_is_firm_join_error": "unsupported",
    }
    require(claims.status.to_dict() == expected_status, "Documentary claim status calibration")
    require(
        all(
            qa.get(key) is True
            for key in [
                "cash_ratio_is_not_share",
                "missing_cash2024_preserved",
                "code_groups_unique_year_code",
                "queue_matches_original_code_inventory",
                "no_employee_totals_computed",
                "no_historical_company_join",
            ]
        ),
        "Context semantic safeguards",
    )
    checks.append(
        dict(
            name="context",
            status="PASS",
            arithmetic_tables=8,
            source_availability_rows=45,
            claim_status_rows=11,
            historical_company_assignments=0,
            fresh_pdf_xlsx_xml_reextraction=False,
        )
    )


def verify(repo, root, economy_raw):
    checks, input_hashes, raw_hashes, artifact_hashes, manifests = [], {}, {}, {}, {}
    for branch in ["regions", "admission", "context"]:
        inputs, outputs = manifest_check(repo, root, branch, economy_raw)
        for name, digest in inputs.items():
            target = raw_hashes if name.startswith("raw/") else input_hashes
            key = (
                name.removeprefix("raw/") if name.startswith("raw/") else name.removeprefix("repo/")
            )
            require(key not in target or target[key] == digest, "Conflicting shared input hashes")
            target[key] = digest
        artifact_hashes.update(outputs)
        manifests[branch] = sha(root / branch / "manifest.json")
        input_hashes[f"scripts/analyze_round6_{branch}.py"] = sha(
            repo / f"scripts/analyze_round6_{branch}.py"
        )
    verify_regions(repo, root, checks)
    verify_admission(repo, root, economy_raw, checks)
    verify_context(repo, root, checks)
    return dict(
        status="PASS",
        checks=checks,
        inputs=input_hashes,
        raw_inputs=raw_hashes,
        artifact_sha256=artifact_hashes,
        branch_manifests_sha256=manifests,
        verifier_sha256=sha(Path(__file__)),
        production_functions_imported=False,
        model_refits=0,
        scope=(
            "Independent arithmetic and stated-population validation; "
            "conditional source joins preserved"
        ),
        tolerances={"rank_statistics_atol_rtol": 1e-10, "money_atol": 1e-8, "money_rtol": 1e-12},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--economy-raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (args.root or repo / "reports/round6_analysis").resolve()
    raw = (args.economy_raw or repo.parents[1] / "work/data-round6/economy").resolve()
    require(not args.out.exists(), "Choose a new QA path; prior evidence is not overwritten")
    try:
        result = verify(repo, root, raw)
    except Exception as error:
        result = dict(status="FAIL", error=str(error), verifier_sha256=sha(Path(__file__)))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))


if __name__ == "__main__":
    main()
