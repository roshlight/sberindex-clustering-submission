"""Summarize saved fifth-round discovery, preserving failures and model limitations."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(raw, out):
    if out.exists():
        raise FileExistsError("Use a new summary directory")
    out.mkdir(parents=True)
    runs, documents = [], []
    checked = 0
    for path in sorted(raw.glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text("utf-8"))
        for name, expected in manifest["output_hashes"].items():
            p = (path.parent / name).resolve()
            if not p.is_relative_to(path.parent.resolve()) or sha(p) != expected:
                raise ValueError("Server output missing or changed")
            checked += 1
        rows = json.loads((path.parent / "document_inventory.json").read_text("utf-8"))
        review = json.loads((path.parent / "review_status.json").read_text("utf-8"))
        runs.append(
            {
                "run_id": path.parent.name,
                "manifest_sha256": sha(path),
                "status": manifest["status"],
                "started_at": manifest["started_at"],
                "finished_at": manifest["finished_at"],
                "engines": manifest["config"]["engines"],
                "search": manifest["search"],
                "cards": manifest["cards"],
                "model_reviews": review,
                "usage": manifest["usage"],
                "model": manifest["config"]["extract_model"],
                "document_status": dict(
                    Counter("downloaded" if r.get("raw_file") else r.get("error") for r in rows)
                ),
            }
        )
        for row in rows:
            if row.get("raw_file"):
                p = (path.parent / row["raw_file"]).resolve()
                if not p.is_relative_to(path.parent.resolve()) or sha(p) != row["sha256"]:
                    raise ValueError("Raw document hash changed")
            documents.append(
                {
                    "run_id": path.parent.name,
                    **{
                        k: row.get(k)
                        for k in [
                            "source_id",
                            "url",
                            "title",
                            "fetched_at",
                            "status",
                            "error",
                            "sha256",
                            "raw_file",
                            "raw_bytes",
                        ]
                    },
                }
            )
    if not runs:
        raise ValueError("No completed server runs")
    result = {
        "status": "PASS_WITH_DISCOVERY_LIMITATIONS",
        "verified_manifest_output_hashes": checked,
        "runs": runs,
        "cards_are_numeric_evidence": False,
        "critic_succeeded": all(r["model_reviews"]["critic_status"] == "complete" for r in runs),
        "automated_search_result_rows": sum(r["search"]["result_rows"] for r in runs),
        "reported_cost_usd": sum(r["usage"]["reported_cost_usd"] for r in runs),
    }
    (out / "server_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
    )
    with (out / "server_document_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(documents[0]))
        writer.writeheader()
        writer.writerows(documents)
    print(json.dumps({"runs": len(runs), "verified_hashes": checked, "documents": len(documents)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.out)
