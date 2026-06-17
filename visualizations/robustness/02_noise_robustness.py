#!/usr/bin/env python3
"""
visualizations/robustness/02_noise_robustness.py

Approach 3 — AUROC vs noise level curves.

Reads the noise-sweep results produced by run_sweep.py and plots how each
model's mean AUROC degrades as noise increases.  Also reads the baseline
(no-noise) eval_results.json for each modality.

Noise semantics
---------------
  Logs    : --noise-frac F  = F fraction of test template IDs replaced randomly
  Metrics : --noise-std  σ  = N(0, σ) added to normalised feature vectors
  Traces  : --noise-std  σ  = N(0, σ) added to 48-dim feature vectors

Outputs
-------
  out/fig4_noise_auroc_curves.png    — AUROC vs noise per modality (3 panels)
  out/fig5_noise_sensitivity_rank.png — ranked bar of AUROC drop at max noise
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT   = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT    = Path(__file__).parent / "out"

NOISE_LEVELS = [0.05, 0.10, 0.25, 0.50, 1.00]
MODALITY_COLORS = {"Logs": "#3498db", "Metrics": "#e74c3c", "Traces": "#2ecc71"}
MODALITY_DIRS   = {"Logs": "logs", "Metrics": "metrics", "Traces": "traces"}


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("results", [])


def mean_auroc_per_model(results: list[dict]) -> dict[str, float]:
    from collections import defaultdict
    sums: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sums[r["model"]].append(r["auroc"])
    return {m: np.mean(v) for m, v in sums.items()}


def load_modality(modality: str, mod_dir: str) -> dict:
    out_dir  = MODELS / mod_dir / "out"
    baseline = load_results(out_dir / "eval_results.json")
    baseline_auroc = mean_auroc_per_model(baseline)

    noise_auroc: dict[float, dict[str, float]] = {}
    for level in NOISE_LEVELS:
        stem = f"eval_results_noise_{str(level).replace('.', 'p')}.json"
        res  = load_results(out_dir / stem)
        if res:
            noise_auroc[level] = mean_auroc_per_model(res)

    return {"baseline": baseline_auroc, "noise": noise_auroc}


def fig4_noise_curves(all_mod: dict, out: Path) -> None:
    available = {k: v for k, v in all_mod.items() if v["noise"]}
    n_mod = len(available)
    if n_mod == 0:
        print("[fig4] No noise-sweep results found. Run run_sweep.py first.")
        return

    fig, axes = plt.subplots(1, n_mod, figsize=(5 * n_mod, 5), sharey=False)
    if n_mod == 1:
        axes = [axes]
    fig.suptitle("AUROC Degradation Under Noise  (model trained on clean data, tested on noisy)",
                 fontsize=12, fontweight="bold")

    for ax, (modality, data) in zip(axes, available.items()):
        baseline  = data["baseline"]
        noise_map = data["noise"]
        models    = sorted(baseline, key=lambda m: -baseline[m])
        levels    = sorted(noise_map)
        color     = MODALITY_COLORS[modality]

        cmap   = plt.cm.get_cmap("tab10")
        n_m    = len(models)

        for idx, model in enumerate(models):
            y = [baseline.get(model, np.nan)] + [
                noise_map[lv].get(model, np.nan) for lv in levels
            ]
            x = [0.0] + levels
            mc = cmap(idx % 10)
            ax.plot(x, y, marker="o", markersize=4, label=model, color=mc,
                    linewidth=1.4, alpha=0.85)

        ax.set_xlabel(
            "Noise level (frac of template IDs replaced)" if modality == "Logs"
            else "Noise std σ added to feature vectors",
            fontsize=9
        )
        ax.set_ylabel("Mean AUROC (across 22 faults)", fontsize=9)
        ax.set_title(modality, fontsize=10, fontweight="bold",
                     color=MODALITY_COLORS[modality])
        ax.axhline(0.7, color="gray", lw=0.8, ls="--", alpha=0.5, label="0.70 threshold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)

    path = out / "fig4_noise_auroc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig5_noise_sensitivity(all_mod: dict, out: Path) -> None:
    drops: list[tuple[str, float, str]] = []  # (model, drop, modality)
    for modality, data in all_mod.items():
        baseline  = data["baseline"]
        noise_map = data["noise"]
        if not noise_map:
            continue
        max_level = max(noise_map)
        noisy     = noise_map[max_level]
        for model, base_auroc in baseline.items():
            noisy_auroc = noisy.get(model, np.nan)
            if not np.isnan(noisy_auroc):
                drops.append((model, base_auroc - noisy_auroc, modality))

    if not drops:
        print("[fig5] No data.")
        return

    drops.sort(key=lambda x: x[1])  # ascending = most robust first
    models     = [d[0] for d in drops]
    drop_vals  = [d[1] for d in drops]
    colors     = [MODALITY_COLORS[d[2]] for d in drops]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(models))))
    y = np.arange(len(models))
    ax.barh(y, drop_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("AUROC drop at maximum noise  (baseline − noisy)  ← smaller = more robust",
                  fontsize=9)
    ax.set_title(f"Model Sensitivity to Noise (at max noise level {NOISE_LEVELS[-1]})",
                 fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

    for bar, val in zip(ax.patches, drop_vals):
        ax.text(max(val + 0.003, 0.003), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=8)

    legend_patches = [mpatches.Patch(color=c, label=m)
                      for m, c in MODALITY_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    ax.set_xlim(left=min(0, min(drop_vals) - 0.02))

    path = out / "fig5_noise_sensitivity_rank.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_mod = {}
    for modality, mod_dir in MODALITY_DIRS.items():
        data = load_modality(modality, mod_dir)
        n_levels = len(data["noise"])
        print(f"[{modality}] baseline={len(data['baseline'])} models, "
              f"{n_levels}/{len(NOISE_LEVELS)} noise levels found")
        all_mod[modality] = data

    fig4_noise_curves(all_mod, args.out)
    fig5_noise_sensitivity(all_mod, args.out)
    print(f"\nDone — outputs in {args.out}/")


if __name__ == "__main__":
    main()
