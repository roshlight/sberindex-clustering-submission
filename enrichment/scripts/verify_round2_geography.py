"""Independently check saved round-2 geography tables and all source/output hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def verify(repo: Path, report: Path) -> dict:
    def read(name):
        return pd.read_csv(report / name, dtype=str, keep_default_na=False)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    cases = read("case_register_78.csv")
    evidence = read("exact_registry_evidence_156.csv")
    sources = read("source_catalog.csv")
    pairs = read("settlement_recoding_pairs.csv")
    review = read("recoding_source_review.csv")
    calendar = read("calendar_priority_6.csv")
    checks = {
        "78_unique_cases": len(cases) == 78 and cases.territory_id.nunique() == 78,
        "156_unique_case_year_rows": len(evidence) == 156
        and not evidence.duplicated(["territory_id", "snapshot_date"]).any(),
        "all_78_exact_full_code_pairs": all(
            cases[f"sber_oktmo_11_{y}"].equals(cases[f"official_oktmo_11_{y}"])
            for y in ["2023", "2024"]
        ),
        "all_new_codes_DateVved_2024_01_01": set(cases.registry_DateVved_2024) == {"01.01.2024"},
        "all_78_second_vintage_confirmed": set(cases.registry_2024_second_vintage_confirmed)
        == {"True"},
        "no_boundary_promotion": set(cases.boundary_equivalence_status) == {"unknown"},
        "22_recoding_docs_10624_pairs": pairs.source_file.nunique() == 22 and len(pairs) == 10624,
        "full11_digit_pairs": bool(
            pairs.old_settlement_oktmo_11.str.fullmatch(r"\d{11}").all()
            and pairs.new_settlement_oktmo_11.str.fullmatch(r"\d{11}").all()
        ),
        "no_automatic_settlement_to_MO_join": set(pairs.municipal_territory_id) == {""}
        and set(pairs.municipal_join_status) == {"not_joined"},
        "20_explicit_law_reference_rows_preserved": len(review) == 20
        and set(review.reason) == {"explicit_law_reference_without_numeric_successor"}
        and review.source_note_verbatim.nunique() == 9,
        "6_calendar_cases_disjoint": len(calendar) == 6
        and not (set(calendar.territory_id) & set(cases.territory_id)),
        "all_37_source_hashes": len(sources) == 37
        and all(sha((repo / r.source_path).resolve()) == r.sha256 for r in sources.itertuples()),
        "all_manifest_output_hashes": all(
            sha(report / name) == value["sha256"] for name, value in manifest["outputs"].items()
        ),
        "source_catalog_has_no_viewer_auth_tokens": not any(
            "token=" in str(v).lower() for v in sources.source_url
        ),
        "all_bundled_input_hashes": all(
            sha((repo / value["path"]).resolve()) == value["sha256"]
            for value in manifest["inputs"].values()
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "semantic_scope": (
            "Code/date and three institutional succession decisions only; "
            "independent boundary equivalence remains unknown."
        ),
        "manifest_sha256": sha(report / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve(), args.report_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
