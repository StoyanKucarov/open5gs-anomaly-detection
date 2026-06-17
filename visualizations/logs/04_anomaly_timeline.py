#!/usr/bin/env python3
"""
visualizations/logs/04_anomaly_timeline.py

Per-fault timeline of log activity spanning pre -> fault -> post, with the
fault injection window shaded.  Verifies that timeline.json boundaries align
with actual log signal changes.

Usage:
  python 04_anomaly_timeline.py [--data PATH] [--out PATH]
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
sys.path.insert(0, str(Path(__file__).parent))
from _log_features import (  # type: ignore
    load_loki, load_timeline, available_experiments, DEFAULT_DATA,
)

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_fault":       "#4f8fd9",
    "protocol_attack":     "#8a4fd9",
    "dependency_failure":  "#4fd980",
}
BIN_S = 15


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out",  type=Path, default=Path(__file__).parent / "out")
    return ap.parse_args()


def build_timeseries(exp_dir, tl):
    t_pre_start   = tl.get("pre",   {}).get("start", 0)
    t_fault_start = tl.get("fault", {}).get("start", 0)
    t_fault_end   = tl.get("fault", {}).get("end",   0)

    all_bins: dict[int, int] = defaultdict(int)
    err_bins: dict[int, int] = defaultdict(int)

    for phase in ("pre", "during", "post"):
        for fname, target in (("all.csv", all_bins), ("errors.csv", err_bins)):
            for r in load_loki(exp_dir, phase, fname):
                try:
                    b = int(r["timestamp_ns"]) // 1_000_000_000 // BIN_S
                    target[b] += 1
                except (KeyError, ValueError):
                    pass

    if not all_bins:
        return [], [], [], t_fault_start - t_pre_start, t_fault_end - t_pre_start

    b_start = min(all_bins)
    b_end   = max(all_bins)
    bins    = list(range(b_start, b_end + 1))
    times   = [b * BIN_S - t_pre_start for b in bins]
    all_c   = [all_bins.get(b, 0) for b in bins]
    err_c   = [err_bins.get(b, 0) for b in bins]
    return times, all_c, err_c, t_fault_start - t_pre_start, t_fault_end - t_pre_start


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    experiments = available_experiments(args.data)
    if not experiments:
        print(f"No experiments found in {args.data}")
        return

    n     = len(experiments)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3),
                             squeeze=False)

    for idx, (slug, _ft, _nf, fc) in enumerate(experiments):
        ax      = axes[idx // ncols][idx % ncols]
        exp_dir = args.data / slug
        tl      = load_timeline(exp_dir)
        times, all_c, err_c, rfs, rfe = build_timeseries(exp_dir, tl)

        color = FAULT_CLASS_COLORS.get(fc, "steelblue")
        if times:
            ax.fill_between(times, all_c, alpha=0.25, color="steelblue")
            ax.fill_between(times, err_c, alpha=0.7,  color=color)
            ax.axvspan(rfs, rfe, alpha=0.12, color="red")
            ax.axvline(rfs, color="red",     lw=1.0, ls="--")
            ax.axvline(rfe, color="darkred", lw=0.8, ls=":")

        ax.set_title(re.sub(r"^\d+-", "", slug), fontsize=7, color=color, pad=3)
        ax.set_xlabel("s from pre-start", fontsize=6)
        ax.set_ylabel(f"lines / {BIN_S}s",  fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color="steelblue", alpha=0.4, label="all logs"),
        mpatches.Patch(color="grey",      alpha=0.7, label="errors"),
        mpatches.Patch(color="red",       alpha=0.3, label="fault window"),
    ] + [mpatches.Patch(color=c, label=fc) for fc, c in FAULT_CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Log Activity Timelines — {n} Faults  (bin={BIN_S}s)\n"
        f"Data: {args.data.name}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path = args.out / "04_anomaly_timeline.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
