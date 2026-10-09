"""Describe restored municipalities without changing cohorts or assigning new clusters."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from round3_data_audit import sha, write_json


def load_features(path):
    codes = ["region_code", "official_oktmo_8"]
    codes += [f"{prefix}_source_oktmo" for prefix in ["investment", "shipments", "retail"]]
    return pd.read_csv(path, dtype=dict.fromkeys(codes, str))


def run(root, output):
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    path = root / "audit/annual_features.csv"
    source_hash = sha(path)
    data = load_features(path)
    rows, summaries = [], []
    for year in [2023, 2024]:
        for prefix, region in [("shipments", "59"), ("retail", "25")]:
            frame = data.loc[
                data.year.eq(year) & data.region_code.eq(region) & data.core_eligible
            ].copy()
            feature = f"{prefix}_log_rubles_per_person"
            gained = frame[f"{prefix}_new_feature"]
            valid = frame[feature].notna()
            ranks = frame[feature].rank(method="average", pct=True)
            old = frame.loc[~gained & valid, feature]
            new = frame.loc[gained, feature]
            assert len(new) == (21 if prefix == "shipments" else 11)
            for r in frame.loc[gained].itertuples():
                amount = getattr(r, f"{prefix}_thousand_rubles")
                rows.append(
                    dict(
                        territory_id=r.territory_id,
                        year=year,
                        prefix=prefix,
                        region_code=region,
                        region_name=r.region_name,
                        municipality_name=r.municipal_district_name,
                        municipality_type=r.municipal_district_type,
                        source_oktmo=getattr(r, f"{prefix}_source_oktmo"),
                        source_sha256=getattr(r, f"{prefix}_source_sha256"),
                        thousand_rubles=amount,
                        population_jan1=r.population_total,
                        rubles_per_person=amount * 1000 / r.population_total,
                        within_region_available_percentile=float(ranks.loc[r.Index]),
                        rank_denominator=int(valid.sum()),
                    )
                )
            summaries.append(
                dict(
                    year=year,
                    prefix=prefix,
                    region_code=region,
                    region_name=frame.region_name.iloc[0],
                    core_n=len(frame),
                    recovered_n=len(new),
                    old_observed_n=len(old),
                    new_observed_n=int(valid.sum()),
                    remaining_missing_n=int((~valid).sum()),
                    recovered_median_rubles_per_person=float(np.exp(new).median()),
                    previously_observed_median_rubles_per_person=float(np.exp(old).median()),
                    median_ratio=float(np.exp(new).median() / np.exp(old).median()),
                    recovered_median_population=float(
                        frame.loc[gained, "population_total"].median()
                    ),
                    previously_observed_median_population=float(
                        frame.loc[~gained & valid, "population_total"].median()
                    ),
                )
            )
    recovered = pd.DataFrame(rows).sort_values(["prefix", "territory_id", "year"])
    assert len(recovered) == 64 and recovered.territory_id.nunique() == 32
    pair = recovered.loc[recovered.year.eq(2023)].merge(
        recovered.loc[recovered.year.eq(2024)],
        on=["territory_id", "prefix"],
        validate="one_to_one",
        suffixes=("_2023", "_2024"),
    )
    pair["nominal_aggregate_change_fraction"] = (
        pair.thousand_rubles_2024 / pair.thousand_rubles_2023 - 1
    )
    pair["nominal_per_person_change_fraction"] = (
        pair.rubles_per_person_2024 / pair.rubles_per_person_2023 - 1
    )
    # These are two snapshots with the accepted annual codes, not constant-boundary growth.
    output.mkdir(parents=True)
    recovered.to_csv(output / "recovered_municipality_profiles.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "regional_selection_summary.csv", index=False)
    pair.to_csv(output / "recovered_two_year_pairs.csv", index=False)
    write_json(
        output / "QA.json",
        {
            "status": "PASS",
            "recovered_rows": len(recovered),
            "paired_municipalities": len(pair),
            "all_values_positive": bool(recovered.rubles_per_person.gt(0).all()),
            "grain": "municipality-year-indicator, frozen core only",
            "weight": "equal municipality weight; medians are not population weighted",
            "rank": "ascending average rank / nonmissing regional core count including recovered",
            "comparison": "recovered vs previously observed in the same region and indicator",
            "limitations": [
                "Observation recovery is selected by source form and municipality type, not random",
                "Nominal activity intensity is not resident income or full business activity",
                "Two-year ratios are not inflation-adjusted or proven constant-boundary growth",
                "No new cluster assignment or causal effect is inferred",
            ],
        },
    )
    assert sha(path) == source_hash
    write_json(
        output / "manifest.json",
        {
            "input": "audit/annual_features.csv",
            "input_sha256": source_hash,
            "code_sha256": sha(Path(__file__)),
            "outputs": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)
