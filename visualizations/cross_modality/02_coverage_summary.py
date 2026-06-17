#!/usr/bin/env python3
"""
visualizations/cross_modality/02_coverage_summary.py

Three-panel cross-modality coverage summary figure.

Panel A — Fault-class × modality coverage rate heatmap
    For each of the 5 fault classes and 3 modalities: fraction of faults in that
    class where the modality's best model achieves AUROC >= threshold.
    Shows WHERE each modality succeeds and fails (the structural "why").

Panel B — Coverage-pattern stacked bars per fault class
    Each fault class is one bar, stacked into segments representing how many
    faults fall in each modality-overlap region (L-only, M-only, T-only,
    L+M, L+T, M+T, all-three, none).  Reveals per-class complementarity.

Panel C — Three-circle Venn diagram of fault coverage
    22 circles (one per fault), laid out in 6 occupied regions of a Venn.
    Each circle coloured by fault class.  Numbers and region labels added.
    Makes the minimum-pair argument visually immediate.

Usage
-----
    python 02_coverage_summary.py [--threshold FLOAT] [--out DIR]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_delay":       "#4f8fd9",
    "network_partition":   "#16a085",
    "protocol_attack":     "#9b59b6",
}
FAULT_CLASS_LABELS = {
    "resource_exhaustion": "Resource\nExhaustion",
    "component_failure":   "Component\nFailure",
    "network_delay":       "Network\nDelay",
    "network_partition":   "Network\nPartition",
    "protocol_attack":     "Protocol\nAttack",
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
MODALITY_COLORS = {
    "Logs":    "#3498db",
    "Metrics": "#e74c3c",
    "Traces":  "#2ecc71",
}
MODALITY_SHORT = {"logs": "L", "metrics": "M", "traces": "T"}

# Colour for each modality-overlap combination
COMBO_COLORS = {
    frozenset(["logs"]):                     "#3498db",   # L only
    frozenset(["metrics"]):                  "#e74c3c",   # M only
    frozenset(["traces"]):                   "#2ecc71",   # T only
    frozenset(["logs", "metrics"]):          "#8e44ad",   # L+M
    frozenset(["logs", "traces"]):           "#16a085",   # L+T  ← key pair
    frozenset(["metrics", "traces"]):        "#c0392b",   # M+T
    frozenset(["logs", "metrics", "traces"]):"#2c3e50",   # all
    frozenset():                             "#aaaaaa",   # none
}
COMBO_LABELS = {
    frozenset(["logs"]):                     "L only",
    frozenset(["metrics"]):                  "M only",
    frozenset(["traces"]):                   "T only",
    frozenset(["logs", "metrics"]):          "L + M",
    frozenset(["logs", "traces"]):           "L + T",
    frozenset(["metrics", "traces"]):        "M + T",
    frozenset(["logs", "metrics", "traces"]):"L + M + T",
    frozenset():                             "None",
}

SHORT_FAULT = {
    "01-cpu-stress-amf":                        "01",
    "02-memory-pressure-upf":                   "02",
    "03-pod-crash-amf":                         "03",
    "04-network-delay-gnb-amf":                 "04",
    "05-network-partition-amf-scp":             "05",
    "06-packet-loss-upf":                       "06",
    "07-pod-crash-smf":                         "07",
    "08-cpu-stress-scp":                        "08",
    "09-network-delay-nrf":                     "09",
    "10-pfcp-session-establishment-flood-upf":  "10",
    "11-pfcp-session-deletion-upf":             "11",
    "12-pfcp-session-modification-drop-upf":    "12",
    "13-pfcp-session-modification-dupl-upf":    "13",
    "14-upf-infrastructure-packet-loss":        "14",
    "15-nrf-cascade":                           "15",
    "16-cpu-stress-ausf":                       "16",
    "17-network-delay-scp":                     "17",
    "18-cpu-stress-nrf":                        "18",
    "19-udm-pod-crash":                         "19",
    "20-mongodb-pod-kill":                      "20",
    "21-n2-partition-amf-gnb":                  "21",
    "22-memory-pressure-amf":                   "22",
}

def load_data(models_dir: Path, threshold: float):
    """
    Returns:
        best[sl][mod] = best AUROC for that fault/modality
        fault_meta[sl] = (fault_type, fault_class)
        coverage_pattern[sl] = frozenset of modalities >= threshold
    """
    best: dict[str, dict[str, float]] = defaultdict(dict)
    fault_meta: dict[str, tuple] = {}

    for mod in ["logs", "metrics", "traces"]:
        d = json.loads((models_dir / mod / "out" / "eval_results.json").read_text())
        for r in d["results"]:
            sl = r["slug"]
            fault_meta[sl] = (r["fault_type"],
                              _CANONICAL_CLASS.get(sl, r["fault_class"]))
            prev = best[sl].get(mod, 0.0)
            if r["auroc"] > prev:
                best[sl][mod] = r["auroc"]

    coverage_pattern = {
        sl: frozenset(m for m in ["logs", "metrics", "traces"]
                      if best[sl].get(m, 0.0) >= threshold)
        for sl in best
    }
    return best, fault_meta, coverage_pattern


def panel_a(ax, best, fault_meta, threshold):
    modalities = ["logs", "metrics", "traces"]
    classes    = ["component_failure", "network_delay",
                  "network_partition", "protocol_attack", "resource_exhaustion"]

    # coverage_rate[fc][mod] = (n_covered, n_total)
    counts: dict = {fc: {m: [0, 0] for m in modalities} for fc in classes}
    aurocs: dict = {fc: {m: []     for m in modalities} for fc in classes}
    for sl, (_, fc) in fault_meta.items():
        if fc not in counts:
            continue
        for m in modalities:
            v = best[sl].get(m, 0.0)
            counts[fc][m][1] += 1
            if v >= threshold:
                counts[fc][m][0] += 1
            aurocs[fc][m].append(v)

    matrix    = np.zeros((len(classes), len(modalities)))
    n_covered = np.zeros((len(classes), len(modalities)), dtype=int)
    n_total   = np.zeros((len(classes), len(modalities)), dtype=int)
    for i, fc in enumerate(classes):
        for j, m in enumerate(modalities):
            n_c = counts[fc][m][0]
            n_t = counts[fc][m][1]
            matrix[i, j]    = n_c / n_t if n_t > 0 else 0.0
            n_covered[i, j] = n_c
            n_total[i, j]   = n_t

    cmap = LinearSegmentedColormap.from_list(
        "cover", ["#d73027", "#fee08b", "#1a9850"], N=256)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="none")

    for i in range(len(classes)):
        for j in range(len(modalities)):
            rate = matrix[i, j]
            txt  = f"{n_covered[i,j]}/{n_total[i,j]}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if rate < 0.45 or rate > 0.88 else "black")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Logs", "Metrics", "Traces"],
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(classes)))
    ylabels = ax.set_yticklabels(
        [FAULT_CLASS_LABELS[c] for c in classes], fontsize=10)
    for lbl, fc in zip(ylabels, classes):
        lbl.set_color(FAULT_CLASS_COLORS[fc])
        lbl.set_fontweight("bold")
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_tick_params(pad=8)
    ax.set_title(f"Coverage rate per fault class\n(AUROC ≥ {threshold:.0%})",
                 fontsize=11, pad=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.set_label("Fraction covered", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])


def panel_b(ax, fault_meta, coverage_pattern):
    classes = ["component_failure", "network_delay",
               "network_partition", "protocol_attack", "resource_exhaustion"]
    # Segment order (roughly L→M→T→pairs→all)
    seg_order = [
        frozenset(["logs", "metrics", "traces"]),
        frozenset(["logs", "traces"]),
        frozenset(["logs", "metrics"]),
        frozenset(["metrics", "traces"]),
        frozenset(["logs"]),
        frozenset(["traces"]),
        frozenset(["metrics"]),
        frozenset(),
    ]

    counts: dict = {fc: defaultdict(int) for fc in classes}
    for sl, (_, fc) in fault_meta.items():
        if fc in counts:
            counts[fc][coverage_pattern[sl]] += 1

    y        = np.arange(len(classes))
    bar_h    = 0.62
    bottoms  = np.zeros(len(classes))
    legend_handles = []

    for seg in seg_order:
        vals   = np.array([counts[fc][seg] for fc in classes], dtype=float)
        col    = COMBO_COLORS[seg]
        label  = COMBO_LABELS[seg]
        if vals.sum() == 0:
            continue
        bars = ax.barh(y, vals, left=bottoms, height=bar_h,
                       color=col, edgecolor="white", linewidth=1.0, label=label)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 1:
                ax.text(b + v / 2, i, str(int(v)), ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color="white",
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                    foreground="black")])
        bottoms += vals
        legend_handles.append(mpatches.Patch(color=col, label=label))

    ax.set_yticks(y)
    ylabels = ax.set_yticklabels(
        [FAULT_CLASS_LABELS[c] for c in classes], fontsize=10)
    for lbl, fc in zip(ylabels, classes):
        lbl.set_color(FAULT_CLASS_COLORS[fc])
        lbl.set_fontweight("bold")
    ax.set_xlabel("Number of faults", fontsize=10)
    ax.set_title("Coverage pattern per fault class", fontsize=11, pad=8)
    ax.set_xlim(0, max(bottoms) + 0.8)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25)

    ax.legend(handles=legend_handles[::-1], loc="lower right",
              fontsize=8.5, title="Covered by", frameon=True,
              handlelength=1.2, handleheight=0.9)


def panel_c(ax, fault_meta, coverage_pattern, threshold):
    """
    Three overlapping circles for L, M, T.
    Each of the 22 faults is drawn as a small dot in its coverage region,
    coloured by fault class.  Region counts are annotated.
    """
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.5, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Circle centres and radius — spread enough to keep pairwise intersections readable
    r    = 1.85
    cx_L = -1.05
    cy_L =  0.72
    cx_M =  1.05
    cy_M =  0.72
    cx_T =  0.0
    cy_T = -1.05

    for (cx, cy), col in [
        ((cx_L, cy_L), MODALITY_COLORS["Logs"]),
        ((cx_M, cy_M), MODALITY_COLORS["Metrics"]),
        ((cx_T, cy_T), MODALITY_COLORS["Traces"]),
    ]:
        ax.add_patch(plt.Circle((cx, cy), r, color=col, alpha=0.22,
                                linewidth=0, zorder=1))
        ax.add_patch(plt.Circle((cx, cy), r, color=col, alpha=0.80,
                                linewidth=2.4, fill=False, zorder=3))

    ax.text(cx_L - 1.55, cy_L + 1.25, "Logs",    fontsize=14, fontweight="bold",
            color=MODALITY_COLORS["Logs"],    ha="center", va="bottom")
    ax.text(cx_M + 1.55, cy_M + 1.25, "Metrics", fontsize=14, fontweight="bold",
            color=MODALITY_COLORS["Metrics"], ha="center", va="bottom")
    ax.text(cx_T,        cy_T - 1.35, "Traces",  fontsize=14, fontweight="bold",
            color=MODALITY_COLORS["Traces"],  ha="center", va="top")

    region_centres = {
        frozenset(["logs"]):                     (-2.35,  1.35),
        frozenset(["metrics"]):                  ( 2.35,  1.35),
        frozenset(["traces"]):                   ( 0.00, -2.60),
        frozenset(["logs", "metrics"]):          ( 0.00,  1.80),
        frozenset(["logs", "traces"]):           (-1.08, -0.42),
        frozenset(["metrics", "traces"]):        ( 1.08, -0.42),
        frozenset(["logs", "metrics", "traces"]):(  0.0,   0.45),
        frozenset():                             (-3.50, -2.80),
    }

    region_faults: dict[frozenset, list[str]] = defaultdict(list)
    for sl in fault_meta:
        region_faults[coverage_pattern[sl]].append(sl)

    dot_radius = 0.155
    dot_spacing = 0.40

    for region, faults_in in region_faults.items():
        if not faults_in:
            continue
        cx, cy = region_centres[region]
        n     = len(faults_in)
        cols  = min(n, 4)
        rows  = (n + cols - 1) // cols
        # Centre the grid on (cx, cy)
        x_off = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) * dot_spacing
        y_off = np.linspace( (rows - 1) / 2,-(rows - 1) / 2, rows) * dot_spacing

        for idx, sl in enumerate(sorted(faults_in)):
            xi  = idx % cols
            yi  = idx // cols
            dx  = x_off[xi]
            dy  = y_off[yi]
            col = FAULT_CLASS_COLORS[fault_meta[sl][1]]
            ax.add_patch(plt.Circle(
                (cx + dx, cy + dy), dot_radius, color=col,
                linewidth=1.2, zorder=6))
            ax.add_patch(plt.Circle(
                (cx + dx, cy + dy), dot_radius, color="white",
                linewidth=1.2, fill=False, zorder=7))
            lbl = SHORT_FAULT.get(sl, sl[:2])
            ax.text(cx + dx, cy + dy, lbl,
                    ha="center", va="center", fontsize=5.2,
                    color="white", fontweight="bold", zorder=8)

    for region, faults_in in region_faults.items():
        if not faults_in or region == frozenset():
            continue
        cx, cy = region_centres[region]
        rows   = (len(faults_in) + 3) // 4
        lbl_y  = cy - rows * dot_spacing / 2 - 0.42
        ax.text(cx, lbl_y, f"n={len(faults_in)}",
                ha="center", va="top", fontsize=9,
                color="#222222", fontweight="bold", zorder=8)

    # Annotate L+T as minimum-sufficient pair
    lx, ly = region_centres[frozenset(["logs", "traces"])]
    ax.annotate(
        "L+T\nminimum\ncomplete pair\n(22/22)",
        xy=(lx - 0.15, ly + 0.65),
        xytext=(lx - 2.55, ly + 2.0),
        fontsize=8.5,
        color=COMBO_COLORS[frozenset(["logs", "traces"])],
        fontweight="bold",
        ha="center",
        arrowprops=dict(
            arrowstyle="->,head_width=0.28,head_length=0.18",
            color=COMBO_COLORS[frozenset(["logs", "traces"])],
            lw=1.8,
        ),
    )

    ax.set_title("Fault distribution across modality coverage regions  "
                 "(each dot = one fault, coloured by fault class)",
                 fontsize=12, pad=8)

    legend_handles = [
        mpatches.Patch(color=c, label=k.replace("_", " ").title())
        for k, c in FAULT_CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, title="Fault class", frameon=True,
              title_fontsize=9)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out",    type=Path, default=OUT)
    ap.add_argument("--threshold", type=float, default=0.70)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {args.models} ...")
    best, fault_meta, coverage_pattern = load_data(args.models, args.threshold)
    print(f"  Faults: {len(fault_meta)}  threshold: {args.threshold:.0%}")

    fig = plt.figure(figsize=(18, 13))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.0, 1.5],
        hspace=0.38, wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])   # Venn spans both columns

    print("Panel A: fault-class x modality heatmap ...")
    panel_a(ax_a, best, fault_meta, args.threshold)

    print("Panel B: stacked bars per fault class ...")
    panel_b(ax_b, fault_meta, coverage_pattern)

    print("Panel C: Venn diagram with fault dots ...")
    panel_c(ax_c, fault_meta, coverage_pattern, args.threshold)

    fig.suptitle(
        "Cross-modality coverage summary  "
        f"(AUROC ≥ {args.threshold:.0%} = covered)",
        fontsize=15, y=0.99, fontweight="bold",
    )

    path = args.out / "fig4_coverage_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
