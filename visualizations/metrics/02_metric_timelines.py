#!/usr/bin/env python3
"""
visualizations/metrics/02_metric_timelines.py

Per-fault time-series of key metrics spanning pre → fault → post.

Six representative metrics are shown per panel, each normalised to [0, 1]
using its pre-phase min/max so they share the same axis scale:

  CPU usage      (total container CPU rate)
  Memory         (total container working-set bytes)
  Network TX     (total bytes/s sent)
  HTTP req rate  (total server-side request rate)
  Active UEs     (SMF UE count — session-plane health)
  Pod restarts   (crash/restart indicator)

The fault injection window is shaded red, with dashed boundary lines.

Usage:
  python 02_metric_timelines.py [--data PATH] [--out PATH]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "models" / "metrics"))

from lib import EXPERIMENTS          # type: ignore
from data_loader import _load_csv_aggregated, _DEFAULT_DATA, _METRICS  # type: ignore

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_fault":       "#4f8fd9",
    "protocol_attack":     "#8a4fd9",
    "dependency_failure":  "#4fd980",
}

# (display label, csv_stem, aggregation)
KEY_METRICS = [
    ("CPU usage",      "container_cpu_usage_rate",         "sum"),
    ("Memory",         "container_memory_working_set_bytes","sum"),
    ("Net TX",         "network_tx_bytes_rate",             "sum"),
    ("HTTP req/s",     "beyla_http_server_request_rate",    "sum"),
    ("Active UEs",     "open5gs_smf_ues_active",            "sum"),
    ("Pod restarts",   "pod_restarts",                      "sum"),
]

METRIC_COLORS = ["#e05a2b", "#4f8fd9", "#2db84d", "#8a4fd9", "#e0b82b", "#d94f4f"]

import json


def load_timeline(exp_dir: Path) -> dict:
    p = exp_dir / "timeline.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_metric_timeseries(exp_dir: Path, stem: str, agg: str
                           ) -> dict[int, float]:
    """Load all three phases and merge into one timestamp→value dict."""
    merged: dict[int, float] = {}
    for phase in ("pre", "during", "post"):
        path = exp_dir / "prometheus" / phase / f"{stem}.csv"
        merged.update(_load_csv_aggregated(path, agg))
    return merged


def normalise(series: list[float], ref_vals: list[float]) -> list[float]:
    """Scale to [0,1] using reference (pre-phase) min/max; clip to [-0.1, 1.5]."""
    lo, hi = min(ref_vals, default=0.0), max(ref_vals, default=1.0)
    span   = hi - lo if hi != lo else 1.0
    return [max(-0.1, min(1.5, (v - lo) / span)) for v in series]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--out",  type=Path,
                    default=Path(__file__).parent / "out")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    experiments = [
        (slug, ft, fc) for slug, ft, _nf, fc in EXPERIMENTS
        if (args.data / slug / "prometheus" / "pre").is_dir()
    ]
    if not experiments:
        print(f"No experiments found in {args.data}")
        return

    ncols = 4
    nrows = (len(experiments) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 6, nrows * 3.2),
                             squeeze=False)

    for idx, (slug, _ft, fc) in enumerate(experiments):
        ax      = axes[idx // ncols][idx % ncols]
        exp_dir = args.data / slug
        tl      = load_timeline(exp_dir)

        t_pre_start   = tl.get("pre",   {}).get("start", 0)
        t_fault_start = tl.get("fault", {}).get("start", 0)
        t_fault_end   = tl.get("fault", {}).get("end",   0)

        rfs = t_fault_start - t_pre_start   # relative fault start (s)
        rfe = t_fault_end   - t_pre_start

        for (label, stem, agg), color in zip(KEY_METRICS, METRIC_COLORS):
            ts_dict = load_metric_timeseries(exp_dir, stem, agg)
            if not ts_dict:
                continue

            times_sorted = sorted(ts_dict)
            values       = [ts_dict[t] for t in times_sorted]
            rel_times    = [t - t_pre_start for t in times_sorted]

            # Pre-phase values used as normalisation reference
            ref_vals = [ts_dict[t] for t in times_sorted
                        if t < t_fault_start]
            if not ref_vals:
                ref_vals = values

            norm_vals = normalise(values, ref_vals)
            ax.plot(rel_times, norm_vals, color=color, lw=0.9,
                    alpha=0.85, label=label)

        ax.axvspan(rfs, rfe, alpha=0.12, color="red")
        ax.axvline(rfs, color="red",     lw=1.0, ls="--", alpha=0.7)
        ax.axvline(rfe, color="darkred", lw=0.8, ls=":",  alpha=0.7)
        ax.axhline(1.0, color="grey",    lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0.0, color="grey",    lw=0.5, ls="--", alpha=0.5)

        color = FAULT_CLASS_COLORS.get(fc, "black")
        ax.set_title(re.sub(r"^\d+-", "", slug), fontsize=7,
                     color=color, pad=3)
        ax.set_xlabel("s from pre-start", fontsize=6)
        ax.set_ylabel("normalised value", fontsize=6)
        ax.set_ylim(-0.15, 1.6)
        ax.tick_params(labelsize=5)
        ax.grid(True, linestyle="--", alpha=0.25)

    for idx in range(len(experiments), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    metric_handles = [
        mpatches.Patch(color=c, label=lbl)
        for (lbl, _, _), c in zip(KEY_METRICS, METRIC_COLORS)
    ]
    metric_handles += [mpatches.Patch(color="red", alpha=0.3, label="fault window")]
    class_handles = [
        mpatches.Patch(color=c, label=fc)
        for fc, c in FAULT_CLASS_COLORS.items()
    ]
    fig.legend(handles=metric_handles + class_handles,
               loc="lower center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Key Metric Timelines — {len(experiments)} Faults\n"
        "Values normalised to pre-phase [0=pre_min, 1=pre_max]  |  "
        "title colour = fault class",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_path = args.out / "02_metric_timelines.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
