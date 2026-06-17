#!/usr/bin/env python3
"""
visualizations/robustness/01_cross_run_variance.py

Approach 1 — Cross-run variance as a natural robustness measure.

Each fault was run 2–4 times under identical conditions.  Real deployment
variability (log volume jitter, Prometheus scrape timing, Beyla span counts)
causes AUROC to vary between runs.  std(AUROC) across runs for the same fault
is a hardware-free robustness estimate: high std → model is sensitive to
measurement noise; low std → stable under natural variability.

Reads: models/{logs,metrics,traces}/out/eval_per_fault.csv
       (produced by evaluate.py --multi-run)

Outputs:
  out/fig1_std_heatmap.png     — per-modality heatmaps of std(AUROC)
  out/fig2_robustness_rank.png — mean std per model (robustness ranking)
  out/fig3_auroc_spread.png    — box plots of per-run AUROC spread per model

Usage
-----
  python 01_cross_run_variance.py [--out DIR]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"

MODALITIES = {
    "Logs":    ROOT / "models" / "logs"    / "out" / "eval_per_fault.csv",
    "Metrics": ROOT / "models" / "metrics" / "out" / "eval_per_fault.csv",
    "Traces":  ROOT / "models" / "traces"  / "out" / "eval_per_fault.csv",
}

MODALITY_COLORS = {"Logs": "#3498db", "Metrics": "#e74c3c", "Traces": "#2ecc71"}

FAULT_CLASS = {
    "01-cpu-stress-amf":                       "resource_exhaustion",
    "02-memory-pressure-upf":                  "resource_exhaustion",
    "03-pod-crash-amf":                        "component_failure",
    "04-network-delay-gnb-amf":               "network_delay",
    "05-network-partition-amf-scp":           "network_partition",
    "06-packet-loss-upf":                     "network_partition",
    "07-pod-crash-smf":                       "component_failure",
    "08-cpu-stress-scp":                      "resource_exhaustion",
    "09-network-delay-nrf":                   "network_delay",
    "10-pfcp-session-establishment-flood-upf":"protocol_attack",
    "11-pfcp-session-deletion-upf":           "protocol_attack",
    "12-pfcp-session-modification-drop-upf":  "protocol_attack",
    "13-pfcp-session-modification-dupl-upf":  "protocol_attack",
    "14-upf-infrastructure-packet-loss":      "network_partition",
    "15-nrf-cascade":                         "component_failure",
    "16-cpu-stress-ausf":                     "resource_exhaustion",
    "17-network-delay-scp":                   "network_delay",
    "18-cpu-stress-nrf":                      "resource_exhaustion",
    "19-udm-pod-crash":                       "component_failure",
    "20-mongodb-pod-kill":                    "component_failure",
    "21-n2-partition-amf-gnb":               "network_partition",
    "22-memory-pressure-amf":                "resource_exhaustion",
}
CLASS_ORDER  = ["resource_exhaustion","component_failure","network_delay",
                "network_partition","protocol_attack"]
CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_delay":       "#4f8fd9",
    "network_partition":   "#16a085",
    "protocol_attack":     "#9b59b6",
}
FAULT_ORDER = sorted(FAULT_CLASS, key=lambda s: (CLASS_ORDER.index(FAULT_CLASS[s]), s))


def base_slug(slug: str) -> str:
    return slug.split("__r")[0]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_variance(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Returns {base_slug: {model: [auroc_run1, auroc_run2, ...]}}"""
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        slug  = base_slug(r["slug"])
        model = r["model"]
        try:
            auroc = float(r["auroc"])
        except (KeyError, ValueError):
            continue
        data[slug][model].append(auroc)
    return data


def build_matrices(variance: dict) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Returns (faults, models, mean_matrix, std_matrix)."""
    faults = [f for f in FAULT_ORDER if f in variance]
    models = sorted({m for slug_data in variance.values() for m in slug_data})
    n_f, n_m = len(faults), len(models)
    mean_mat = np.full((n_f, n_m), np.nan)
    std_mat  = np.full((n_f, n_m), np.nan)
    for i, fault in enumerate(faults):
        for j, model in enumerate(models):
            vals = variance.get(fault, {}).get(model, [])
            if len(vals) >= 2:
                mean_mat[i, j] = np.mean(vals)
                std_mat [i, j] = np.std(vals, ddof=1)
            elif len(vals) == 1:
                mean_mat[i, j] = vals[0]
    return faults, models, mean_mat, std_mat


def fig1_std_heatmap(all_data: dict, out: Path) -> None:
    available = {k: v for k, v in all_data.items() if v["rows"]}
    n_mod = len(available)
    if n_mod == 0:
        print("[fig1] No data found — run evaluate.py --multi-run first.")
        return

    fig, axes = plt.subplots(1, n_mod, figsize=(6 * n_mod, 9))
    if n_mod == 1:
        axes = [axes]
    fig.suptitle("Cross-Run AUROC std — Robustness to Deployment Variability",
                 fontsize=13, fontweight="bold", y=1.01)

    max_std = 0.0
    for mod_data in available.values():
        s = mod_data["std_mat"]
        if not np.all(np.isnan(s)):
            max_std = max(max_std, np.nanmax(s))
    vmax = max(max_std, 0.15)

    for ax, (modality, mod_data) in zip(axes, available.items()):
        faults  = mod_data["faults"]
        models  = mod_data["models"]
        std_mat = mod_data["std_mat"]
        n_f, n_m = std_mat.shape

        im = ax.imshow(std_mat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(n_m))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_f))
        ax.set_yticklabels(faults, fontsize=7)
        ax.set_title(modality, fontsize=11, fontweight="bold",
                     color=MODALITY_COLORS[modality], pad=6)

        for i in range(n_f):
            for j in range(n_m):
                v = std_mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black" if v < vmax * 0.6 else "white")

        # Fault-class row bands (left strip)
        for i, fault in enumerate(faults):
            fc = FAULT_CLASS.get(fault, "")
            ax.add_patch(plt.Rectangle((-0.75, i - 0.5), 0.5, 1.0,
                                       color=CLASS_COLORS.get(fc, "#aaa"),
                                       transform=ax.transData, clip_on=False))

        ax.set_xlim(-0.5, n_m - 0.5)
        ax.set_ylim(-0.5, n_f - 0.5)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="std(AUROC) across runs")

    legend_patches = [mpatches.Patch(color=c, label=cl.replace("_", " "))
                      for cl, c in CLASS_COLORS.items()]
    axes[0].legend(handles=legend_patches, loc="lower left",
                   bbox_to_anchor=(0, -0.25), ncol=3, fontsize=7,
                   title="Fault class", title_fontsize=7)

    path = out / "fig1_std_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig2_robustness_rank(all_data: dict, out: Path) -> None:
    model_stds: dict[str, tuple[list[float], str]] = {}
    for modality, mod_data in all_data.items():
        if not mod_data["rows"]:
            continue
        std_mat = mod_data["std_mat"]
        for j, model in enumerate(mod_data["models"]):
            col = std_mat[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) >= 1:
                model_stds[model] = (valid.tolist(), modality)

    if not model_stds:
        print("[fig2] No data.")
        return

    models_sorted = sorted(model_stds, key=lambda m: np.mean(model_stds[m][0]))
    means = [np.mean(model_stds[m][0]) for m in models_sorted]
    sems  = [np.std(model_stds[m][0], ddof=1) / max(len(model_stds[m][0]) ** 0.5, 1)
             for m in models_sorted]
    colors = [MODALITY_COLORS[model_stds[m][1]] for m in models_sorted]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(models_sorted))))
    y = np.arange(len(models_sorted))
    bars = ax.barh(y, means, xerr=sems, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(models_sorted, fontsize=9)
    ax.set_xlabel("Mean std(AUROC) across runs  ←  more stable", fontsize=10)
    ax.set_title("Model Robustness Ranking\n(lower = more stable under deployment variability)",
                 fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

    for bar, mean in zip(bars, means):
        ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=8)

    legend_patches = [mpatches.Patch(color=c, label=m)
                      for m, c in MODALITY_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    ax.set_xlim(left=0)

    path = out / "fig2_robustness_rank.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig3_auroc_spread(all_data: dict, out: Path) -> None:
    from collections import OrderedDict

    box_data: dict[str, dict[str, list[float]]] = {}
    for modality, mod_data in all_data.items():
        if not mod_data["rows"]:
            continue
        variance = mod_data["variance"]
        model_vals: dict[str, list[float]] = defaultdict(list)
        for slug_data in variance.values():
            for model, vals in slug_data.items():
                if len(vals) >= 2:
                    model_vals[model].extend(vals)
        if model_vals:
            box_data[modality] = dict(model_vals)

    if not box_data:
        print("[fig3] No multi-run data (need ≥2 runs per fault).")
        return

    n_mod = len(box_data)
    fig, axes = plt.subplots(1, n_mod, figsize=(5 * n_mod, 5), sharey=True)
    if n_mod == 1:
        axes = [axes]
    fig.suptitle("AUROC Distribution Across Runs per Model",
                 fontsize=12, fontweight="bold")

    for ax, (modality, model_vals) in zip(axes, box_data.items()):
        models = sorted(model_vals, key=lambda m: np.median(model_vals[m]))
        data   = [model_vals[m] for m in models]
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        color = MODALITY_COLORS[modality]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUROC" if ax == axes[0] else "")
        ax.set_title(modality, fontsize=10, fontweight="bold", color=color)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.7, color="gray", lw=0.8, ls="--", alpha=0.6, label="AUROC=0.70")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].legend(fontsize=8, loc="lower right")
    path = out / "fig3_auroc_spread.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}
    for modality, csv_path in MODALITIES.items():
        rows     = load_csv(csv_path)
        variance = compute_variance(rows)
        if variance:
            faults, models, mean_mat, std_mat = build_matrices(variance)
        else:
            faults, models, mean_mat, std_mat = [], [], np.empty((0,0)), np.empty((0,0))
        all_data[modality] = dict(rows=rows, variance=variance,
                                   faults=faults, models=models,
                                   mean_mat=mean_mat, std_mat=std_mat)
        n_runs = max((len(v) for slug_data in variance.values()
                      for v in slug_data.values()), default=0)
        print(f"[{modality}] {len(rows)} rows, {len(variance)} faults, "
              f"max {n_runs} runs per fault×model")

    fig1_std_heatmap(all_data, args.out)
    fig2_robustness_rank(all_data, args.out)
    fig3_auroc_spread(all_data, args.out)
    print(f"\nDone — outputs in {args.out}/")


if __name__ == "__main__":
    main()
