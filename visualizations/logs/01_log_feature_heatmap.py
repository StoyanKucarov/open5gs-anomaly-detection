#!/usr/bin/env python3
"""
visualizations/logs/01_log_feature_heatmap.py

Per-fault log feature matrix heatmap.

Usage:
  python 01_log_feature_heatmap.py [--data PATH] [--out PATH]

  --data   path to an experiment run directory  (default: C-fault-detection)
  --out    output directory for PNG files        (default: ./out)
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).parent))
from _log_features import extract_features, available_experiments, DEFAULT_DATA  # type: ignore

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_fault":       "#4f8fd9",
    "protocol_attack":     "#8a4fd9",
    "dependency_failure":  "#4fd980",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="Experiment run directory")
    ap.add_argument("--out",  type=Path, default=Path(__file__).parent / "out",
                    help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    experiments = available_experiments(args.data)
    if not experiments:
        print(f"No experiments found in {args.data}")
        return

    rows, labels, classes = [], [], []
    for slug, _ft, _nf, fc in experiments:
        feat = extract_features(slug, fc, args.data)
        rows.append(feat)
        labels.append(slug)
        classes.append(fc)

    feature_names = list(rows[0].keys())
    matrix = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)

    col_mean = matrix.mean(axis=0)
    col_std  = matrix.std(axis=0)
    col_std[col_std == 0] = 1.0
    matrix_z = np.clip((matrix - col_mean) / col_std, -3, 3)

    order     = sorted(range(len(labels)), key=lambda i: (classes[i], labels[i]))
    matrix_z  = matrix_z[order]
    labels_s  = [labels[i]  for i in order]
    classes_s = [classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(22, 11))
    im = ax.imshow(matrix_z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

    ax.set_xticks(range(len(labels_s)))
    ax.set_xticklabels(labels_s, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)

    for tick, cls in zip(ax.get_xticklabels(), classes_s):
        tick.set_color(FAULT_CLASS_COLORS.get(cls, "black"))

    plt.colorbar(im, ax=ax, label="z-score (clipped +/-3)")
    ax.set_title(
        f"Log Feature Heatmap — {len(labels_s)} Faults x Log Features (z-scored)\n"
        f"Data: {args.data.name}",
        fontsize=11,
    )
    legend_patches = [mpatches.Patch(color=c, label=fc)
                      for fc, c in FAULT_CLASS_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              title="Fault class (x-label colour)")

    plt.tight_layout()
    out_path = args.out / "01_log_feature_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
