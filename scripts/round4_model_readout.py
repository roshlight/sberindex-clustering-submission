"""Derive descriptive loss contributions from frozen OOF predictions without refitting."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from round2_model_analysis import KEY, TARGETS, sha


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=False)
    models = root / "models"
    panel = pd.read_csv(models / "model_panel.csv")
    oof = duckdb.read_parquet(str(models / "oof_predictions.parquet")).df()
    comparisons = pd.read_csv(models / "cv_comparisons.csv")
    primary = comparisons.loc[comparisons.comparison.eq("primary")].copy()
    primary.to_csv(output / "primary_all_samples.csv", index=False)
    contributions = []
    for year in [2023, 2024]:
        d = panel.loc[panel.year.eq(year) & panel.core_eligible].copy()
        d["completeness_stratum"] = np.select(
            [d.common_complete_new3, d.expanded_complete_new3],
            ["common_complete", "newly_complete"],
            default="still_incomplete",
        )
        for regime in ["within_region", "heldout_regions"]:
            s = oof.loc[
                oof.year.eq(year)
                & oof["sample"].eq("core")
                & oof.regime.eq(regime)
                & oof.seed.eq(13)
            ]
            for target, coordinates in [("log_total", TARGETS[:1]), ("profile_clr", TARGETS[1:])]:
                losses = d[[*KEY, "region_code", "recovered_any", "completeness_stratum"]].copy()
                for arm in ["legacy_full", "recovered_full"]:
                    model = s.loc[s.model.eq(arm)].copy()
                    model[arm] = np.square(
                        model[["actual_" + t for t in coordinates]].to_numpy()
                        - model[["predicted_" + t for t in coordinates]].to_numpy()
                    ).sum(axis=1)
                    losses = losses.merge(model[[*KEY, arm]], on=KEY, validate="one_to_one")
                denominator = losses.legacy_full.sum()
                expected = comparisons.loc[
                    comparisons.year.eq(year)
                    & comparisons["sample"].eq("core")
                    & comparisons.regime.eq(regime)
                    & comparisons.seed.eq(13)
                    & comparisons.target.eq(target)
                    & comparisons.comparison.eq("state_update"),
                    "relative_mse_reduction",
                ].item()
                for scope in ["region_code", "recovered_any", "completeness_stratum"]:
                    total_contribution = 0.0
                    for group, part in losses.groupby(scope, sort=True):
                        old, new = part.legacy_full.sum(), part.recovered_full.sum()
                        contribution = (old - new) / denominator
                        total_contribution += contribution
                        contributions.append(
                            {
                                "year": year,
                                "regime": regime,
                                "target": target,
                                "slice": scope,
                                "group": str(group),
                                "n": len(part),
                                "regions": part.region_code.nunique(),
                                "legacy_sse": old,
                                "recovered_sse": new,
                                "sse_change": new - old,
                                "legacy_total_sse": denominator,
                                "local_relative_reduction": (old - new) / old,
                                "contribution_to_total_relative_reduction": contribution,
                            }
                        )
                    np.testing.assert_allclose(total_contribution, expected, atol=1e-12)
    pd.DataFrame(contributions).to_csv(output / "state_update_contributions.csv", index=False)
    core = primary.loc[primary["sample"].eq("core") & primary.seed.eq(13)]
    profile = core.loc[core.target.eq("profile_clr")]
    stable = primary.loc[
        primary["sample"].eq("stable_code") & primary.seed.eq(13) & primary.target.eq("profile_clr")
    ]
    adoption = bool(
        profile.relative_mse_reduction.ge(0.01).all()
        & profile.ci_low.gt(0).all()
        & stable.relative_mse_reduction.gt(0).all()
    )
    qa = {
        "status": "PASS",
        "adoption_passed": adoption,
        "main_four_primary_intervals_include_zero": bool(
            (profile.ci_low.le(0) & profile.ci_high.ge(0)).all()
        ),
        "all_primary_seed13_intervals_include_zero": bool(
            (
                primary.loc[primary.seed.eq(13) & primary.target.eq("profile_clr"), "ci_low"].le(0)
                & primary.loc[primary.seed.eq(13) & primary.target.eq("profile_clr"), "ci_high"].ge(
                    0
                )
            ).all()
        ),
        "postfit_contribution_sums_match_state_update": True,
        "slice_analysis": (
            "Descriptive post-fit diagnostic, no subgroup hypothesis test or causal attribution"
        ),
    }
    (output / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    manifest = {
        "inputs": {
            p.name: sha(p)
            for p in [
                models / "model_panel.csv",
                models / "oof_predictions.parquet",
                models / "cv_comparisons.csv",
            ]
        },
        "code_sha256": sha(__file__),
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(qa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.out)
