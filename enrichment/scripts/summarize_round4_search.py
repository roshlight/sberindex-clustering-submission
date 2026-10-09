"""Verify saved server-run manifests and report discovery coverage without model claims."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    docs = []
    checked = 0
    for path in sorted(raw.glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in manifest["output_hashes"].items():
            file = (path.parent / name).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError("Unsafe manifest path")
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed output: {path.parent.name}/{name}")
            checked += 1
        documents = json.loads((path.parent / "document_inventory.json").read_text("utf-8"))
        status = Counter("downloaded" if x.get("raw_file") else x.get("error") for x in documents)
        runs.append(
            {
                "run_id": path.parent.name,
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": manifest["status"],
                "started_at": manifest["started_at"],
                "finished_at": manifest["finished_at"],
                "engines": manifest["config"]["engines"],
                "search": manifest["search"],
                "document_status": dict(status),
                "cards": manifest["cards"],
                "model_reviews": manifest["model_reviews"],
                "usage": manifest["usage"],
                "model": manifest["config"]["extract_model"],
            }
        )
        for item in documents:
            docs.append(
                {
                    "run_id": path.parent.name,
                    **{
                        k: item.get(k)
                        for k in (
                            "source_id",
                            "url",
                            "title",
                            "fetched_at",
                            "status",
                            "error",
                            "sha256",
                            "raw_file",
                            "raw_bytes",
                        )
                    },
                }
            )
    if not runs:
        raise ValueError("No saved runs")
    summary = {
        "status": "PASS_WITH_DISCOVERY_LIMITATIONS",
        "verified_manifest_output_hashes": checked,
        "runs": runs,
        "cards_are_numeric_evidence": False,
        "critic_succeeded": False,
        "automated_search_result_rows": sum(x["search"]["result_rows"] for x in runs),
        "reported_cost_usd": sum(x["usage"]["reported_cost_usd"] for x in runs),
    }
    (out / "server_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "server_document_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(docs[0]))
        writer.writeheader()
        writer.writerows(docs)
    print(json.dumps({"runs": len(runs), "verified_hashes": checked, "documents": len(docs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.out)
