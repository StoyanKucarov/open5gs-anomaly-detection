#!/usr/bin/env python3
"""
models/merge_per_deploy.py

Merges multi-run calibration results with per-deployment (single-run) results
into a single eval_results.json + eval_per_fault.csv per modality.

Calibration models train globally on all runs and generalise without retraining.
Per-deployment models must be trained on the target cluster's own pre-phase.

Usage
-----
  python models/merge_per_deploy.py

Output
------
  models/{logs,metrics,traces}/out/eval_results_merged.json
  models/{logs,metrics,traces}/out/eval_per_fault_merged.csv
"""

import csv
import json
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

PER_DEPLOY = {
    "logs":    {"DeepLog", "LogBERT", "LogRobust"},
    "metrics": {"OmniAnomaly", "AnomalyTransformer"},
    "traces":  {"TraceAnomaly", "TraceSieve"},
}
CALIBRATION = {
    "logs":    {"LogCluster", "FeatureModel"},
    "metrics": {"MetricPCA", "MetricZScore", "MetricFeature"},
    "traces":  {"TraceRPCA", "TraceRCA", "MicroRank"},
}

def load_results(path: Path) -> tuple[list[dict], list[str], dict]:
    if not path.exists():
        return [], [], {}
    payload = json.loads(path.read_text())
    return payload["results"], payload["model_names"], payload.get("meta", {})


def merge_modality(modality: str) -> None:
    out_dir    = MODELS / modality / "out"
    multi_path = out_dir / "eval_results.json"
    pdep_path  = out_dir / "eval_results_run3.json"

    multi_results, multi_names, multi_meta = load_results(multi_path)
    pdep_results,  pdep_names,  _          = load_results(pdep_path)

    if not multi_results and not pdep_results:
        print(f"[{modality}] No results found — skipping.")
        return

    calib_models = CALIBRATION[modality]
    pdep_models  = PER_DEPLOY[modality]

    calib_rows = [r for r in multi_results if r["model"] in calib_models]
    pdep_rows  = [r for r in pdep_results  if r["model"] in pdep_models]

    if not pdep_rows:
        print(f"[{modality}] No per-deployment results found at {pdep_path} "
              f"— heatmap will only show calibration models.")

    merged     = calib_rows + pdep_rows
    all_models = sorted({r["model"] for r in merged},
                        key=lambda m: (m not in calib_models, m))

    out_json   = out_dir / "eval_results_merged.json"
    out_csv    = out_dir / "eval_per_fault_merged.csv"

    meta = {
        **multi_meta,
        "merged": True,
        "calibration_source": "multi-run",
        "per_deploy_source":  "run3 (C-fault-detection-4-clean)",
        "calibration_models": sorted(calib_models),
        "per_deploy_models":  sorted(pdep_models),
    }
    payload = {"meta": meta, "model_names": all_models, "results": merged}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[{modality}] Saved merged JSON  -> {out_json}")

    if merged:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            w.writeheader()
            w.writerows(merged)
        print(f"[{modality}] Saved merged CSV   -> {out_csv}")

    by_slug: dict[str, dict] = {}
    for r in merged:
        by_slug.setdefault(r["slug"], {})[r["model"]] = r["auroc"]

    cols  = all_models
    width = max(len(m) for m in cols) + 2
    hdr   = f"  {'Fault':<46}" + "".join(f"{m:>{width}}" for m in cols)
    print()
    print(f"  [{modality.upper()}] Merged AUROC summary")
    print("  " + "=" * len(hdr))
    print(hdr)
    print("  " + "-" * len(hdr))
    for slug in sorted(by_slug):
        row = by_slug[slug]
        vals = "".join(f"{row.get(m, 0.0):>{width}.3f}" for m in cols)
        print(f"  {slug:<46}{vals}")
    print("  " + "=" * len(hdr))
    print()


if __name__ == "__main__":
    for mod in ("logs", "metrics", "traces"):
        merge_modality(mod)
    print("Done.  Run plot_results.py --results out/eval_results_merged.json "
          "in each modality directory to regenerate heatmaps.")
