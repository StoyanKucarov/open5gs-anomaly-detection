#!/usr/bin/env python3
"""
visualizations/logs/03_error_template_distribution.py

Error template distribution across faults: which templates dominate which
faults, helping identify shared vs. fault-specific signatures.

Usage:
  python 03_error_template_distribution.py [--data PATH] [--out PATH]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).parent))
from _log_features import (  # type: ignore
    load_loki, simple_template, strip_ansi,
    available_experiments, DEFAULT_DATA,
)

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_fault":       "#4f8fd9",
    "protocol_attack":     "#8a4fd9",
    "dependency_failure":  "#4fd980",
}
TOP_N = 30


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out",  type=Path, default=Path(__file__).parent / "out")
    return ap.parse_args()


def collect_template_counts(experiments, data_dir):
    faults, classes, per_fault_counts = [], [], []
    for slug, _ft, _nf, fc in experiments:
        exp_dir = data_dir / slug
        rows    = load_loki(exp_dir, "during", "errors.csv")
        counts: Counter = Counter()
        for r in rows:
            counts[simple_template(r.get("line", ""))[:80]] += 1
        faults.append(slug)
        classes.append(fc)
        per_fault_counts.append(counts)

    global_counts: Counter = Counter()
    for c in per_fault_counts:
        global_counts.update(c)
    top_templates = [t for t, _ in global_counts.most_common(TOP_N)]

    matrix = np.zeros((len(faults), len(top_templates)), dtype=float)
    for i, counts in enumerate(per_fault_counts):
        total = max(sum(counts.values()), 1)
        for j, tmpl in enumerate(top_templates):
            matrix[i, j] = counts.get(tmpl, 0) / total

    return faults, classes, top_templates, matrix, per_fault_counts


def plot_template_heatmap(faults, classes, templates, matrix, out_dir, data_name):
    order     = sorted(range(len(faults)), key=lambda i: (classes[i], faults[i]))
    matrix_s  = matrix[order]
    faults_s  = [faults[i]  for i in order]
    classes_s = [classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(20, 12))
    im = ax.imshow(matrix_s.T, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(faults_s)))
    ax.set_xticklabels(faults_s, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(templates)))
    ax.set_yticklabels(templates, fontsize=6)

    for tick, cls in zip(ax.get_xticklabels(), classes_s):
        tick.set_color(FAULT_CLASS_COLORS.get(cls, "black"))

    plt.colorbar(im, ax=ax, label="Fraction of during-phase error lines")
    ax.set_title(
        f"Top-{TOP_N} Error Templates x Fault (normalised per fault)\n"
        f"Data: {data_name}",
        fontsize=10,
    )
    legend_patches = [mpatches.Patch(color=c, label=fc)
                      for fc, c in FAULT_CLASS_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              title="Fault class (x-label colour)")
    plt.tight_layout()
    out_path = out_dir / "03a_template_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.close()


def plot_error_volume_stack(faults, classes, templates, matrix,
                            per_fault_counts, out_dir, data_name):
    totals    = [sum(c.values()) for c in per_fault_counts]
    order     = sorted(range(len(faults)), key=lambda i: (classes[i], faults[i]))
    faults_s  = [faults[i]  for i in order]
    classes_s = [classes[i] for i in order]
    totals_s  = [totals[i]  for i in order]
    matrix_s  = matrix[order]

    cmap   = plt.colormaps["tab20"].resampled(min(len(templates), 20))
    fig, ax = plt.subplots(figsize=(18, 7))
    bottom  = np.zeros(len(faults_s))

    for j, tmpl in enumerate(templates[:20]):
        heights = matrix_s[:, j] * np.array(totals_s, dtype=float)
        ax.bar(range(len(faults_s)), heights, bottom=bottom,
               color=cmap(j), label=tmpl[:40] if j < 10 else None)
        bottom += heights

    remainder = np.clip(np.array(totals_s, dtype=float) - bottom, 0, None)
    ax.bar(range(len(faults_s)), remainder, bottom=bottom,
           color="#cccccc", label="other templates")

    ax.set_xticks(range(len(faults_s)))
    ax.set_xticklabels(faults_s, rotation=45, ha="right", fontsize=6.5)
    for tick, cls in zip(ax.get_xticklabels(), classes_s):
        tick.set_color(FAULT_CLASS_COLORS.get(cls, "black"))

    ax.set_ylabel("Error log line count")
    ax.set_title(
        f"During-Phase Error Volume per Fault, Split by Top Template Buckets\n"
        f"Data: {data_name}",
        fontsize=10,
    )
    ax.legend(fontsize=5, loc="upper right", ncol=2, title="Error template")
    plt.tight_layout()
    out_path = out_dir / "03b_error_volume_stack.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.close()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    experiments = available_experiments(args.data)
    if not experiments:
        print(f"No experiments found in {args.data}")
        return

    faults, classes, templates, matrix, per_fault_counts = \
        collect_template_counts(experiments, args.data)
    plot_template_heatmap(faults, classes, templates, matrix,
                          args.out, args.data.name)
    plot_error_volume_stack(faults, classes, templates, matrix,
                            per_fault_counts, args.out, args.data.name)


if __name__ == "__main__":
    main()
