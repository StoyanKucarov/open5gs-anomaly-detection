#!/usr/bin/env python3
"""
visualizations/cross_modality/01_cross_modality_coverage.py

Three-figure cross-modality coverage analysis across all 15 models × 22 faults.

Figure 1  Full AUROC heatmap
    15 models (grouped Logs | Metrics | Traces) × 22 faults (grouped by class).
    Cells: AUROC, RdYlGn, 0→1.  Modality bands and fault-class row bars added.

Figure 2  Best-per-modality + coverage gap
    Left: 3-column heatmap — best model AUROC within each modality per fault.
    Right: stacked bar — how many faults each modality alone covers vs. needs
    another modality to reach the coverage threshold.
    Bottom strip: which modality combination covers each fault.

Figure 3  Optimal model-pair / trio rankings
    For every pair of individual models (105 pairs) and every cross-modality
    trio (one model per modality, 125 trios), compute the number of the 22
    faults where at least one member of the set achieves AUROC >= threshold.
    Show: top-12 pairs and top-12 trios as horizontal bar charts, coloured
    by modality mix.

Usage
-----
    python 01_cross_modality_coverage.py [--out DIR] [--threshold FLOAT]
    default threshold = 0.70
"""

import argparse
import json
import sys
from itertools import combinations, product
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"

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
MODALITY_COLORS = {
    "Logs":    "#3498db",
    "Metrics": "#e74c3c",
    "Traces":  "#2ecc71",
}
MODALITY_HATCHES = {"Logs": "//", "Metrics": "xx", "Traces": ".."}

SHORT_FAULT = {
    "01-cpu-stress-amf":                        "01 cpu-stress-amf",
    "02-memory-pressure-upf":                   "02 mem-pressure-upf",
    "03-pod-crash-amf":                         "03 pod-crash-amf",
    "04-network-delay-gnb-amf":                 "04 net-delay-gnb-amf",
    "05-network-partition-amf-scp":             "05 net-partition-amf-scp",
    "06-packet-loss-upf":                       "06 pkt-loss-upf",
    "07-pod-crash-smf":                         "07 pod-crash-smf",
    "08-cpu-stress-scp":                        "08 cpu-stress-scp",
    "09-network-delay-nrf":                     "09 net-delay-nrf",
    "10-pfcp-session-establishment-flood-upf":  "10 pfcp-estab-flood",
    "11-pfcp-session-deletion-upf":             "11 pfcp-deletion",
    "12-pfcp-session-modification-drop-upf":    "12 pfcp-mod-drop",
    "13-pfcp-session-modification-dupl-upf":    "13 pfcp-mod-dupl",
    "14-upf-infrastructure-packet-loss":        "14 infra-pkt-loss",
    "15-nrf-cascade":                           "15 nrf-cascade",
    "16-cpu-stress-ausf":                       "16 cpu-stress-ausf",
    "17-network-delay-scp":                     "17 net-delay-scp",
    "18-cpu-stress-nrf":                        "18 cpu-stress-nrf",
    "19-udm-pod-crash":                         "19 udm-pod-crash",
    "20-mongodb-pod-kill":                      "20 mongodb-kill",
    "21-n2-partition-amf-gnb":                  "21 n2-partition",
    "22-memory-pressure-amf":                   "22 mem-pressure-amf",
}

def load_results(results_dir: Path) -> dict:
    """
    Returns:
        auroc[modality][model_name][slug] = float
        fault_meta[slug] = (fault_type, fault_class)
        model_order = {"Logs": [...], "Metrics": [...], "Traces": [...]}
    """
    modality_map = {"logs": "Logs", "metrics": "Metrics", "traces": "Traces"}
    auroc:      dict[str, dict[str, dict[str, float]]] = {}
    fault_meta: dict[str, tuple[str, str]] = {}
    model_order: dict[str, list[str]] = {}

    for mod_dir, mod_label in modality_map.items():
        path = results_dir / mod_dir / "out" / "eval_results.json"
        d    = json.loads(path.read_text())
        model_order[mod_label] = d["model_names"]
        auroc[mod_label] = {m: {} for m in d["model_names"]}
        for r in d["results"]:
            slug  = r["slug"]
            model = r["model"]
            auroc[mod_label][model][slug] = r["auroc"]
            fault_meta[slug] = (r["fault_type"],
                                _CANONICAL_CLASS.get(slug, r["fault_class"]))

    return auroc, fault_meta, model_order


def sorted_faults(fault_meta: dict) -> list[str]:
    return sorted(fault_meta, key=lambda s: (fault_meta[s][1], s))


def fig1_full_heatmap(auroc, fault_meta, model_order, slugs, threshold, out_dir):
    modalities   = ["Logs", "Metrics", "Traces"]
    all_models   = [(mod, m) for mod in modalities for m in model_order[mod]]
    n_models     = len(all_models)
    n_faults     = len(slugs)

    matrix = np.zeros((n_faults, n_models))
    for j, (mod, m) in enumerate(all_models):
        for i, sl in enumerate(slugs):
            matrix[i, j] = auroc[mod][m].get(sl, 0.0)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.4, vmax=1.0, interpolation="none")

    mod_starts = [0, 5, 10]
    band_colors = {"Logs": "#3498db22", "Metrics": "#e74c3c22", "Traces": "#2ecc7122"}
    for mi, (mod, col) in enumerate(zip(modalities, ["#3498db22", "#e74c3c22", "#2ecc7122"])):
        start = mod_starts[mi]
        ax.axvspan(start - 0.5, start + 4.5, color=col, zorder=0)
        ax.text(start + 2, -1.5, mod, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=list(MODALITY_COLORS.values())[mi])

    for x in [4.5, 9.5]:
        ax.axvline(x, color="white", linewidth=2.5, zorder=3)

    # Threshold dashed contour
    mask_below = matrix < threshold
    for i in range(n_faults):
        for j in range(n_models):
            if matrix[i, j] < threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#555555", linewidth=0.4, zorder=4))

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([m for _, m in all_models], rotation=45, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(n_faults))
    ax.set_yticklabels([SHORT_FAULT.get(s, s) for s in slugs], fontsize=8.5)
    for tick, sl in zip(ax.get_yticklabels(), slugs):
        tick.set_color(FAULT_CLASS_COLORS[fault_meta[sl][1]])
    ax.tick_params(left=False, bottom=False)

    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.02, pad=0.01)

    patches = [mpatches.Patch(color=FAULT_CLASS_COLORS[k],
                               label=k.replace("_", " ").title())
               for k in FAULT_CLASS_COLORS]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.12, 1.0), fontsize=8, title="Fault class",
              frameon=True)

    ax.set_title(
        f"AUROC — all 15 models × 22 faults  (grey outline = below {threshold:.0%})",
        fontsize=12, pad=18)

    plt.tight_layout()
    path = out_dir / "fig1_full_auroc_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig2_modality_coverage(auroc, fault_meta, model_order, slugs, threshold, out_dir):
    modalities = ["Logs", "Metrics", "Traces"]

    best: dict[str, dict[str, float]] = {}
    for mod in modalities:
        best[mod] = {}
        for sl in slugs:
            best[mod][sl] = max(
                auroc[mod][m].get(sl, 0.0) for m in model_order[mod]
            )

    n_faults = len(slugs)
    matrix   = np.array([[best[mod][sl] for mod in modalities] for sl in slugs])

    fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.4, vmax=1.0, interpolation="none")

    for i in range(n_faults):
        for j, mod in enumerate(modalities):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if v < 0.58 else "black",
                    fontweight="bold" if v >= threshold else "normal")

    for i in range(n_faults):
        for j in range(3):
            if matrix[i, j] < threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#222222", linewidth=1.2, zorder=4))

    # Fault-class colour strip (right edge of heatmap)
    ax2 = ax.inset_axes([1.01, 0, 0.04, 1], transform=ax.transAxes)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, n_faults)
    ax2.axis("off")
    for i, sl in enumerate(slugs):
        fc = fault_meta[sl][1]
        ax2.add_patch(plt.Rectangle(
            (0, n_faults - 1 - i), 1, 1,
            color=FAULT_CLASS_COLORS[fc], linewidth=0))

    ax.set_xticks(range(3))
    ax.set_xticklabels(modalities, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_faults))
    ax.set_yticklabels([SHORT_FAULT.get(s, s) for s in slugs], fontsize=8.5)
    for tick, sl in zip(ax.get_yticklabels(), slugs):
        tick.set_color(FAULT_CLASS_COLORS[fault_meta[sl][1]])
    ax.tick_params(left=False, bottom=False)
    ax.set_title("Best-model-per-modality AUROC", fontsize=11, pad=10)
    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.04, pad=0.12)

    ax = axes[1]

    def count_covered(mod_set):
        return sum(
            1 for sl in slugs
            if any(best[m][sl] >= threshold for m in mod_set)
        )

    combos = [
        ("Logs",             ["Logs"]),
        ("Metrics",          ["Metrics"]),
        ("Traces",           ["Traces"]),
        ("Logs + Metrics",   ["Logs", "Metrics"]),
        ("Logs + Traces",    ["Logs", "Traces"]),
        ("Metrics + Traces", ["Metrics", "Traces"]),
        ("All three",        ["Logs", "Metrics", "Traces"]),
    ]
    labels   = [c[0] for c in combos]
    counts   = [count_covered(c[1]) for c in combos]

    combo_colors = [
        MODALITY_COLORS["Logs"],
        MODALITY_COLORS["Metrics"],
        MODALITY_COLORS["Traces"],
        "#8e44ad",   # L+M
        "#16a085",   # L+T
        "#c0392b",   # M+T
        "#2c3e50",   # All
    ]

    bars = ax.barh(range(len(combos)), counts, color=combo_colors,
                   height=0.65, edgecolor="white", linewidth=1.2)
    ax.axvline(22, color="#aaaaaa", linestyle="--", linewidth=1.2, zorder=0)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(cnt + 0.2, i, f"{cnt}/22", va="center", fontsize=9.5,
                fontweight="bold")

    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Faults covered (AUROC ≥ {:.0%})".format(threshold), fontsize=10)
    ax.set_title("Coverage by modality combination", fontsize=11, pad=10)
    ax.tick_params(left=False, bottom=True)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", alpha=0.3)

    patches = [mpatches.Patch(color=FAULT_CLASS_COLORS[k],
                               label=k.replace("_", " ").title())
               for k in FAULT_CLASS_COLORS]
    axes[0].legend(handles=patches, loc="lower left",
                   bbox_to_anchor=(0, -0.18), ncol=3, fontsize=8,
                   title="Fault class", frameon=True)

    fig.suptitle(
        "Cross-modality coverage analysis  "
        f"(coverage threshold = {threshold:.0%} AUROC)",
        fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig2_modality_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig3_optimal_pairs(auroc, fault_meta, model_order, slugs, threshold, out_dir):
    modalities  = ["Logs", "Metrics", "Traces"]

    all_mods: list[str] = []
    all_names: list[str] = []
    all_vecs:  list[np.ndarray] = []

    for mod in modalities:
        for m in model_order[mod]:
            all_mods.append(mod)
            all_names.append(m)
            all_vecs.append(
                np.array([auroc[mod][m].get(sl, 0.0) for sl in slugs])
            )

    n_models = len(all_names)
    n_faults = len(slugs)

    def coverage(indices):
        combined = np.max(np.stack([all_vecs[i] for i in indices]), axis=0)
        return int((combined >= threshold).sum())

    pair_results = []
    for i, j in combinations(range(n_models), 2):
        cov = coverage([i, j])
        pair_results.append((cov, all_names[i], all_mods[i],
                              all_names[j], all_mods[j]))
    pair_results.sort(reverse=True)

    logs_idx    = [i for i, m in enumerate(all_mods) if m == "Logs"]
    metrics_idx = [i for i, m in enumerate(all_mods) if m == "Metrics"]
    traces_idx  = [i for i, m in enumerate(all_mods) if m == "Traces"]

    trio_results = []
    for i, j, k in product(logs_idx, metrics_idx, traces_idx):
        cov = coverage([i, j, k])
        trio_results.append((cov, all_names[i], all_names[j], all_names[k]))
    trio_results.sort(reverse=True)

    single_results = []
    for i in range(n_models):
        cov = int((all_vecs[i] >= threshold).sum())
        single_results.append((cov, all_names[i], all_mods[i]))
    single_results.sort(reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    ax = axes[0]
    top_single = single_results[:10]
    labels = [r[1] for r in top_single]
    counts = [r[0] for r in top_single]
    colors = [MODALITY_COLORS[r[2]] for r in top_single]
    bars = ax.barh(range(len(top_single)), counts, color=colors,
                   height=0.65, edgecolor="white")
    ax.axvline(22, color="#aaaaaa", linestyle="--", linewidth=1.2, zorder=0)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(cnt + 0.15, i, f"{cnt}/22", va="center", fontsize=9)
    ax.set_yticks(range(len(top_single)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Faults covered", fontsize=10)
    ax.set_title("Top single models", fontsize=11, pad=8)
    ax.invert_yaxis()
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.3)
    modality_patches = [
        mpatches.Patch(color=c, label=m)
        for m, c in MODALITY_COLORS.items()
    ]
    ax.legend(handles=modality_patches, loc="lower right", fontsize=8,
              frameon=True)

    ax = axes[1]
    top_pairs = pair_results[:12]
    labels = [f"{r[1]}\n+ {r[3]}" for r in top_pairs]
    counts = [r[0] for r in top_pairs]

    def mix_color(mA, mB):
        # Blend: same modality = solid, cross = half-half (use avg)
        if mA == mB:
            return MODALITY_COLORS[mA]
        c1 = np.array(plt.matplotlib.colors.to_rgb(MODALITY_COLORS[mA]))
        c2 = np.array(plt.matplotlib.colors.to_rgb(MODALITY_COLORS[mB]))
        return tuple(0.5 * (c1 + c2))

    colors = [mix_color(r[2], r[4]) for r in top_pairs]
    # Draw stacked two-tone bars for cross-modality pairs
    for idx, (r, cnt) in enumerate(zip(top_pairs, counts)):
        c1 = MODALITY_COLORS[r[2]]
        c2 = MODALITY_COLORS[r[4]]
        ax.barh(idx, cnt / 2, color=c1, height=0.65, left=0,
                edgecolor="none")
        ax.barh(idx, cnt - cnt / 2, color=c2, height=0.65, left=cnt / 2,
                edgecolor="none")
        ax.text(cnt + 0.15, idx, f"{cnt}/22", va="center", fontsize=9)

    ax.axvline(22, color="#aaaaaa", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Faults covered", fontsize=10)
    ax.set_title("Top model pairs", fontsize=11, pad=8)
    ax.invert_yaxis()
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=modality_patches, loc="lower right", fontsize=8,
              frameon=True)

    ax = axes[2]
    top_trios = trio_results[:12]
    labels = [f"{r[1]}\n+ {r[2]}\n+ {r[3]}" for r in top_trios]
    counts = [r[0] for r in top_trios]
    for idx, (r, cnt) in enumerate(zip(top_trios, counts)):
        cl = MODALITY_COLORS["Logs"]
        cm = MODALITY_COLORS["Metrics"]
        ct = MODALITY_COLORS["Traces"]
        seg = cnt / 3
        ax.barh(idx, seg,       color=cl, height=0.65, left=0,       edgecolor="none")
        ax.barh(idx, seg,       color=cm, height=0.65, left=seg,      edgecolor="none")
        ax.barh(idx, cnt - 2*seg, color=ct, height=0.65, left=2*seg, edgecolor="none")
        ax.text(cnt + 0.15, idx, f"{cnt}/22", va="center", fontsize=9)

    ax.axvline(22, color="#aaaaaa", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_yticks(range(len(top_trios)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Faults covered", fontsize=10)
    ax.set_title("Top cross-modality trios\n(1 model per modality)", fontsize=11, pad=8)
    ax.invert_yaxis()
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=modality_patches, loc="lower right", fontsize=8,
              frameon=True)

    fig.suptitle(
        f"Optimal model combinations for fault coverage  (threshold = {threshold:.0%} AUROC)",
        fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / "fig3_optimal_pairs_trios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    print("\n  == Top 5 single models ==")
    for r in single_results[:5]:
        print(f"    {r[1]:25s} ({r[2]:7s})  {r[0]}/22 faults")

    print("\n  == Top 5 pairs ==")
    for r in pair_results[:5]:
        print(f"    {r[1]:25s} ({r[2]:7s})  +  {r[3]:25s} ({r[4]:7s})  "
              f"-> {r[0]}/22 faults")

    print("\n  == Top 5 cross-modality trios ==")
    for r in trio_results[:5]:
        print(f"    {r[1]:25s} + {r[2]:25s} + {r[3]:25s}  -> {r[0]}/22 faults")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=ROOT / "models",
                    help="Root models/ dir containing logs/metrics/traces subdirs")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="AUROC threshold for 'covered' (default 0.70)")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results} ...")
    auroc, fault_meta, model_order = load_results(args.results)
    slugs = sorted_faults(fault_meta)
    print(f"  Faults: {len(slugs)}  |  threshold: {args.threshold:.0%}")

    print("\nFigure 1: Full AUROC heatmap ...")
    fig1_full_heatmap(auroc, fault_meta, model_order, slugs,
                      args.threshold, args.out)

    print("Figure 2: Modality coverage gaps ...")
    fig2_modality_coverage(auroc, fault_meta, model_order, slugs,
                           args.threshold, args.out)

    print("Figure 3: Optimal pairs / trios ...")
    fig3_optimal_pairs(auroc, fault_meta, model_order, slugs,
                       args.threshold, args.out)

    print(f"\nAll figures saved to {args.out}/")


if __name__ == "__main__":
    main()
