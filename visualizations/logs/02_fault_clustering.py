#!/usr/bin/env python3
"""
visualizations/logs/02_fault_clustering.py

Hierarchical clustering dendrogram + t-SNE scatter of faults using log feature
vectors.  Works on any experiment run directory that shares the EXPERIMENTS
slug registry.

Usage:
  python 02_fault_clustering.py [--data PATH] [--out PATH]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

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
MARKERS = {
    "resource_exhaustion": "o",
    "component_failure":   "s",
    "network_fault":       "^",
    "protocol_attack":     "D",
    "dependency_failure":  "P",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out",  type=Path, default=Path(__file__).parent / "out")
    return ap.parse_args()


def build_matrix(experiments, data_dir):
    rows, labels, fault_types, classes = [], [], [], []
    for slug, ft, _nf, fc in experiments:
        feat = extract_features(slug, fc, data_dir)
        rows.append(list(feat.values()))
        labels.append(slug)
        fault_types.append(ft)
        classes.append(fc)
    matrix = np.array(rows, dtype=float)
    col_std = matrix.std(axis=0)
    col_std[col_std == 0] = 1.0
    matrix_z = (matrix - matrix.mean(axis=0)) / col_std
    return matrix_z, labels, fault_types, classes


def plot_dendrogram(matrix_z, labels, classes, out_dir, data_name):
    Z = linkage(matrix_z, method="ward", metric="euclidean")
    fig, ax = plt.subplots(figsize=(18, 8))
    ddata = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45,
                       color_threshold=0, above_threshold_color="#999999")

    label_to_class = dict(zip(labels, classes))
    for lbl, tick in zip(ddata["ivl"], ax.get_xticklabels()):
        tick.set_color(FAULT_CLASS_COLORS.get(label_to_class.get(lbl, ""), "black"))
        tick.set_fontsize(7)

    ax.set_title(
        f"Hierarchical Clustering — {len(labels)} Faults by Log Features\n"
        f"Ward linkage, Euclidean on z-scored features  |  Data: {data_name}",
        fontsize=11,
    )
    ax.set_ylabel("Distance")
    legend_patches = [mpatches.Patch(color=c, label=fc)
                      for fc, c in FAULT_CLASS_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              title="Fault class (leaf colour)")
    plt.tight_layout()
    out_path = out_dir / "02a_dendrogram.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.close()


def plot_tsne(matrix_z, labels, fault_types, classes, out_dir, data_name):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed — skipping t-SNE plot")
        return

    perplexity = min(5, len(labels) - 1)
    if perplexity < 1:
        print("Too few samples for t-SNE — skipping")
        return

    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                  max_iter=2000, init="pca")
    coords = tsne.fit_transform(matrix_z)

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, (x, y) in enumerate(coords):
        fc     = classes[i]
        color  = FAULT_CLASS_COLORS.get(fc, "grey")
        marker = MARKERS.get(fc, "o")
        ax.scatter(x, y, c=color, marker=marker, s=120, zorder=3,
                   edgecolors="white", linewidths=0.5)
        short = re.sub(r"^\d+-", "", labels[i])
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.5, color="#333333")

    legend_patches = [mpatches.Patch(color=FAULT_CLASS_COLORS[fc], label=fc)
                      for fc in FAULT_CLASS_COLORS]
    ax.legend(handles=legend_patches, fontsize=8, title="Fault class")
    ax.set_title(
        f"t-SNE of {len(labels)} Faults from Log Feature Vectors\n"
        f"perplexity={perplexity}  |  Data: {data_name}",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "02b_tsne.png"
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

    matrix_z, labels, fault_types, classes = build_matrix(experiments, args.data)
    plot_dendrogram(matrix_z, labels, classes, args.out, args.data.name)
    plot_tsne(matrix_z, labels, fault_types, classes, args.out, args.data.name)


if __name__ == "__main__":
    main()
