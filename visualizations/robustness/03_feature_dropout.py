#!/usr/bin/env python3
"""
visualizations/robustness/03_feature_dropout.py

Approach 2 — AUROC degradation when a feature group is completely absent.

Reads dropout results produced by run_sweep.py and compares against baseline.
Shows which models are robust to sensor failure and which depend critically on
specific feature groups (http, cpu, memory, network, 5g_control for metrics;
span_count, error_rate, latency, global for traces).

Outputs
-------
  out/fig6_dropout_metrics.png  — grouped bar: metric models × dropped groups
  out/fig7_dropout_traces.png   — grouped bar: trace models × dropped groups
  out/fig8_dropout_heatmap.png  — heatmap: AUROC retention (%) model × group
"""

import json
import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT    = Path(__file__).parent / "out"

MODALITY_COLORS = {"Logs": "#3498db", "Metrics": "#e74c3c", "Traces": "#2ecc71"}

DROPOUT_GROUPS = {
    "metrics": ["http", "cpu", "memory", "network", "5g_control"],
    "traces":  ["span_count", "error_rate", "latency", "global"],
}

GROUP_LABELS = {
    "http":       "HTTP\nclient/server",
    "cpu":        "CPU\nusage",
    "memory":     "Memory",
    "network":    "Network\nbytes",
    "5g_control": "5G control-\nplane KPIs",
    "span_count": "Span\ncounts",
    "error_rate": "Error\nrates",
    "latency":    "Latency\n(mean+p95)",
    "global":     "Global\nwindow stats",
}


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("results", [])


def mean_auroc_per_model(results: list[dict]) -> dict[str, float]:
    from collections import defaultdict
    sums: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sums[r["model"]].append(r["auroc"])
    return {m: float(np.mean(v)) for m, v in sums.items()}


def load_dropout_data(mod_dir: str, groups: list[str]) -> dict:
    out_dir  = MODELS / mod_dir / "out"
    baseline = mean_auroc_per_model(load_results(out_dir / "eval_results.json"))
    group_auroc: dict[str, dict[str, float]] = {}
    for g in groups:
        res = load_results(out_dir / f"eval_results_dropout_{g}.json")
        if res:
            group_auroc[g] = mean_auroc_per_model(res)
    return {"baseline": baseline, "groups": group_auroc}


def _grouped_bar(ax, models: list[str], groups: list[str],
                 baseline: dict, group_auroc: dict,
                 title: str, color: str) -> None:
    n_m  = len(models)
    n_g  = len(groups) + 1  # +1 for baseline
    x    = np.arange(n_m)
    w    = 0.8 / n_g

    ax.bar(x - 0.4 + w / 2, [baseline.get(m, 0) for m in models],
           width=w, label="Baseline (no dropout)", color="steelblue", alpha=0.9)

    cmap = plt.cm.get_cmap("Set2")
    for gi, group in enumerate(groups):
        auroc_vals = [group_auroc.get(group, {}).get(m, np.nan) for m in models]
        offset     = (gi + 1) * w - 0.4 + w / 2
        ax.bar(x + offset, auroc_vals, width=w,
               label=GROUP_LABELS.get(group, group),
               color=cmap(gi / max(len(groups) - 1, 1)), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean AUROC (22 faults)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.7, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)


def fig6_dropout_bars(mod_dir: str, modality: str, groups: list[str],
                      out: Path, fig_num: int) -> None:
    data = load_dropout_data(mod_dir, groups)
    if not data["groups"]:
        print(f"[fig{fig_num}] No dropout results for {modality}. Run run_sweep.py --dropout-only first.")
        return

    baseline  = data["baseline"]
    g_auroc   = data["groups"]
    models    = sorted(baseline, key=lambda m: -baseline[m])

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models)), 5))
    _grouped_bar(ax, models, groups, baseline, g_auroc,
                 f"{modality} — AUROC with feature group removed (Approach 2)",
                 MODALITY_COLORS[modality])
    fig.suptitle("Feature Group Dropout: which groups matter most?",
                 fontsize=12, fontweight="bold", y=1.02)

    path = out / f"fig{fig_num}_dropout_{mod_dir}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def fig8_dropout_heatmap(out: Path) -> None:
    all_rows: list[tuple[str, str, float]] = []  # (model, group_label, retention%)

    for mod_dir, groups in [("metrics", DROPOUT_GROUPS["metrics"]),
                             ("traces",  DROPOUT_GROUPS["traces"])]:
        data = load_dropout_data(mod_dir, groups)
        if not data["groups"]:
            continue
        baseline = data["baseline"]
        for group, g_auroc in data["groups"].items():
            for model, auroc in g_auroc.items():
                base = baseline.get(model, 0)
                retention = auroc / base if base > 1e-6 else np.nan
                label = f"{GROUP_LABELS.get(group, group).replace(chr(10), ' ')}\n({mod_dir})"
                all_rows.append((model, label, retention))

    if not all_rows:
        print("[fig8] No dropout data found.")
        return

    models = sorted({r[0] for r in all_rows})
    groups = list(dict.fromkeys(r[1] for r in all_rows))  # preserve order
    mat    = np.full((len(models), len(groups)), np.nan)
    for model, group, val in all_rows:
        i = models.index(model)
        j = groups.index(group)
        mat[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(groups)), max(4, 0.5 * len(models))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.1,
                   interpolation="nearest")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("AUROC Retention when Feature Group is Zeroed\n"
                 "(1.0 = no degradation, <0.9 = significant drop)",
                 fontsize=11, fontweight="bold")

    for i in range(len(models)):
        for j in range(len(groups)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="black" if 0.7 < v < 1.05 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("AUROC retention  (dropout / baseline)", fontsize=9)

    path = out / "fig8_dropout_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig6_dropout_bars("metrics", "Metrics", DROPOUT_GROUPS["metrics"], args.out, 6)
    fig6_dropout_bars("traces",  "Traces",  DROPOUT_GROUPS["traces"],  args.out, 7)
    fig8_dropout_heatmap(args.out)
    print(f"\nDone — outputs in {args.out}/")


if __name__ == "__main__":
    main()
