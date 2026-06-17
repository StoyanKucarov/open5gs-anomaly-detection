#!/usr/bin/env python3
"""
models/metrics/plot_results.py

Generates all visualizations from a saved eval_results.json produced by
models/metrics/evaluate.py.  Re-run freely without re-training any models.

Outputs (models/metrics/out/)
------------------------------
  eval_f1_heatmap.png      — Average Precision (PR-AUC) per fault × model
  eval_auroc_heatmap.png   — AUROC per fault × model
  eval_per_fault_summary.txt

Usage
-----
  python plot_results.py [--results PATH]

  --results PATH   path to eval_results.json
                   (default: models/metrics/out/eval_results.json)
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_delay":       "#4f8fd9",
    "network_partition":   "#16a085",
    "protocol_attack":     "#9b59b6",
}

_CANONICAL_CLASS = {
    "01-cpu-stress-amf":                        "resource_exhaustion",
    "02-memory-pressure-upf":                   "resource_exhaustion",
    "03-pod-crash-amf":                         "component_failure",
    "04-network-delay-gnb-amf":                 "network_delay",
    "05-network-partition-amf-scp":             "network_partition",
    "06-packet-loss-upf":                       "network_partition",
    "07-pod-crash-smf":                         "component_failure",
    "08-cpu-stress-scp":                        "resource_exhaustion",
    "09-network-delay-nrf":                     "network_delay",
    "10-pfcp-session-establishment-flood-upf":  "protocol_attack",
    "11-pfcp-session-deletion-upf":             "protocol_attack",
    "12-pfcp-session-modification-drop-upf":    "protocol_attack",
    "13-pfcp-session-modification-dupl-upf":    "protocol_attack",
    "14-upf-infrastructure-packet-loss":        "network_partition",
    "15-nrf-cascade":                           "component_failure",
    "16-cpu-stress-ausf":                       "resource_exhaustion",
    "17-network-delay-scp":                     "network_delay",
    "18-cpu-stress-nrf":                        "resource_exhaustion",
    "19-udm-pod-crash":                         "component_failure",
    "20-mongodb-pod-kill":                      "component_failure",
    "21-n2-partition-amf-gnb":                  "network_partition",
    "22-memory-pressure-amf":                   "resource_exhaustion",
}


def load_results(path: Path) -> tuple[list[dict], list[str], dict]:
    payload = json.loads(path.read_text())
    return payload["results"], payload["model_names"], payload.get("meta", {})


def plot_heatmap(results: list[dict], metric: str,
                 model_names: list[str], out_path: Path) -> None:
    by_slug: dict[str, dict[str, float]] = {}
    slug_to_class: dict[str, str] = {}
    for r in results:
        by_slug.setdefault(r["slug"], {})[r["model"]] = r.get(metric, 0.0)
        slug_to_class[r["slug"]] = _CANONICAL_CLASS.get(r["slug"], r["fault_class"])

    slugs  = sorted(by_slug, key=lambda s: (slug_to_class[s], s))
    matrix = np.array([[by_slug[s].get(m, 0.0) for m in model_names]
                       for s in slugs])

    vmin      = 0.0 if metric in ("f1", "best_f1", "avg_precision", "best_recall") else 0.4
    label_str = {
        "best_f1":       "Best-threshold F1",
        "avg_precision": "Average Precision (PR-AUC)",
        "best_recall":   "Recall @ optimal F1 threshold",
        "auroc":         "AUROC",
        "f1":            "F1",
    }.get(metric, metric.upper())

    fig, ax = plt.subplots(
        figsize=(len(model_names) * 2.2 + 2, len(slugs) * 0.42 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=1.0)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticks(range(len(slugs)))
    ax.set_yticklabels(slugs, fontsize=8)

    for tick, slug in zip(ax.get_yticklabels(), slugs):
        tick.set_color(FAULT_CLASS_COLORS.get(slug_to_class[slug], "black"))

    for i, slug in enumerate(slugs):
        for j, m in enumerate(model_names):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="black" if 0.3 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, label=label_str)
    ax.set_title(f"Per-Fault {label_str} — Metrics Anomaly Detection\n"
                 "(y-label colour = fault class)", fontsize=11)

    present = sorted(set(slug_to_class.values()),
                     key=list(FAULT_CLASS_COLORS).index)
    patches = [mpatches.Patch(color=FAULT_CLASS_COLORS[fc],
                               label=fc.replace("_", " ").title())
               for fc in present]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.35, 1.0), fontsize=8, title="Fault class")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


def write_summary(results: list[dict], model_names: list[str],
                  path: Path) -> None:
    by_slug: dict[str, dict[str, dict]] = {}
    for r in results:
        by_slug.setdefault(r["slug"], {})[r["model"]] = r

    col_w  = 16
    header = f"{'Fault':<38} {'Class':<22}"
    for m in model_names:
        header += f"  {m+' AP':>{col_w}} {m+' AUC':>{col_w}}"

    lines = ["=" * len(header), header, "-" * len(header)]
    prev_class = None

    anchor = model_names[0]
    for slug in sorted(by_slug, key=lambda s: (
            by_slug[s].get(anchor, {}).get("fault_class", ""), s)):
        row = by_slug[slug]
        fc  = next((v["fault_class"] for v in row.values()), "")
        if fc != prev_class:
            if prev_class is not None:
                lines.append("")
            prev_class = fc

        line = f"{slug:<38} {fc:<22}"
        for m in model_names:
            mr    = row.get(m, {})
            ap    = mr.get("avg_precision", 0.0)
            auroc = mr.get("auroc",        0.5)
            line += f"  {ap:>{col_w}.3f} {auroc:>{col_w}.3f}"
        lines.append(line)

    lines += ["=" * len(header), ""]
    text = "\n".join(lines)
    print(text)
    path.write_text(text)
    print(f"Saved -> {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=OUT / "eval_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(
            f"{args.results} not found — run evaluate.py first.")

    results, model_names, meta = load_results(args.results)
    print(f"Loaded {len(results)} result rows "
          f"({len(model_names)} models: {', '.join(model_names)})")
    if meta:
        print(f"  Run at {meta.get('timestamp', '?')}  "
              f"data={meta.get('data', '?')}")

    plot_heatmap(results, "avg_precision", model_names, OUT / "eval_f1_heatmap.png")
    plot_heatmap(results, "auroc",         model_names, OUT / "eval_auroc_heatmap.png")
    plot_heatmap(results, "best_recall",   model_names, OUT / "eval_recall_heatmap.png")
    write_summary(results, model_names, OUT / "eval_per_fault_summary.txt")

    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
