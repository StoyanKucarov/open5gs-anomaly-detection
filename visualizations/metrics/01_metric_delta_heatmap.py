#!/usr/bin/env python3
"""
visualizations/metrics/01_metric_delta_heatmap.py

Per-fault metric delta heatmap.

For each fault and each metric, computes the normalised shift:
    δ = (during_mean - pre_mean) / max(pre_std, ε)

A positive δ means the metric increased during the fault; negative means it
dropped.  Large |δ| = the metric is "acting out" for that fault.

Metrics are grouped by category (CPU, Memory, Network, 5G, Pod) and rows use
a diverging colormap centred at 0.  Fault columns are coloured by fault class.

Usage:
  python 01_metric_delta_heatmap.py [--data PATH] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "models" / "metrics"))

from lib import EXPERIMENTS          # type: ignore
from data_loader import (            # type: ignore
    _load_csv_aggregated, FEATURE_NAMES, _METRICS, _DEFAULT_DATA,
)

FAULT_CLASS_COLORS = {
    "resource_exhaustion": "#e07b39",
    "component_failure":   "#d94f4f",
    "network_fault":       "#4f8fd9",
    "protocol_attack":     "#8a4fd9",
    "dependency_failure":  "#4fd980",
}

# Metrics to exclude — monitoring/sidecar overhead, not 5G NF signals
_EXCLUDE = {"cpu_beyla", "cpu_monitoring", "mem_beyla", "mem_monitoring"}

# Metric groups for visual grouping (row separators in the heatmap).
# Every non-excluded metric must appear in exactly one group.
METRIC_GROUPS = {
    "Queries / HTTP": [
        "http_server_req_rate", "http_client_req_rate",
        "http_server_duration", "http_client_duration",
        "http_server_err_rate", "http_client_err_rate",
    ],
    "CPU": ["cpu_usage", "cpu_throttled", "cpu_node"],
    "Memory": ["mem_container", "mem_node_available"],
    "Network": ["net_tx", "net_rx"],
    "AMF": [
        "amf_reg_req", "amf_reg_succ", "amf_reg_fail",
        "amf_auth_fail", "amf_auth_reject",
        "amf_sessions", "amf_subscribers",
        "amf_ran_ue_count", "amf_gnb_count", "amf_paging_req",
    ],
    "SMF / PFCP": [
        "smf_pdu_req", "smf_pdu_succ", "smf_sessions",
        "smf_ues", "smf_bearers", "smf_qos_flows",
        "smf_n4_estab", "smf_n4_report", "smf_n4_report_succ",
        "pfcp_sessions", "pfcp_peers",
    ],
    "UPF / GTP": ["upf_sessions", "upf_qos_flows", "upf_n4_estab", "gtp_failed"],
}

# Excluded metrics are dropped entirely; every remaining metric must be in a group.
_ORDERED: list[str] = []
_GROUP_BOUNDS: list[tuple[int, str]] = []   # (first_row_index, group_name)
for _grp, _mets in METRIC_GROUPS.items():
    _GROUP_BOUNDS.append((len(_ORDERED), _grp))
    _ORDERED.extend([m for m in _mets if m in FEATURE_NAMES and m not in _EXCLUDE])

_METRIC_AGG = {name: agg for _, name, agg in _METRICS}


def load_phase_means(exp_dir: Path, phase: str) -> dict[str, float]:
    """Return {metric_name: mean_value} for one phase of one experiment."""
    prom_dir = exp_dir / "prometheus" / phase
    means: dict[str, float] = {}
    for fname, name, agg in _METRICS:
        data = _load_csv_aggregated(prom_dir / f"{fname}.csv", agg)
        if data:
            means[name] = float(np.mean(list(data.values())))
        else:
            means[name] = 0.0
    return means


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--out",  type=Path,
                    default=Path(__file__).parent / "out")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    slugs, fault_classes, deltas = [], [], []

    for slug, _ft, _nf, fc in EXPERIMENTS:
        exp_dir = args.data / slug
        if not (exp_dir / "prometheus" / "pre").is_dir():
            continue

        pre    = load_phase_means(exp_dir, "pre")
        during = load_phase_means(exp_dir, "during")

        pre_std: dict[str, float] = {}
        for fname, name, agg in _METRICS:
            prom_dir = exp_dir / "prometheus" / "pre"
            data = _load_csv_aggregated(prom_dir / f"{fname}.csv", agg)
            pre_std[name] = float(np.std(list(data.values()))) if data else 1.0

        row = {}
        for name in FEATURE_NAMES:
            if name in _EXCLUDE:
                continue
            pre_val = pre.get(name, 0.0)
            std     = pre_std.get(name, 0.0)
            scale   = max(std, abs(pre_val), 0.001)
            row[name] = (during.get(name, 0.0) - pre_val) / scale

        slugs.append(slug)
        fault_classes.append(fc)
        deltas.append(row)

    if not slugs:
        print(f"No experiments found in {args.data}")
        return

    order   = sorted(range(len(slugs)), key=lambda i: (fault_classes[i], slugs[i]))
    slugs   = [slugs[i]         for i in order]
    classes = [fault_classes[i] for i in order]
    matrix  = np.array([[deltas[i].get(m, np.nan) for m in _ORDERED] for i in order],
                       dtype=float).T   # (n_metrics, n_faults)

    # Drop rows with no meaningful signal across any fault.
    # Threshold: max |delta| < 0.15 means the metric never moves visibly.
    MIN_SIGNAL = 0.15
    row_max    = np.nanmax(np.abs(matrix), axis=1)  # (n_metrics,)
    keep_mask  = row_max >= MIN_SIGNAL
    matrix     = matrix[keep_mask]
    kept_names = [name for name, keep in zip(_ORDERED, keep_mask) if keep]

    kept_set   = set(kept_names)
    new_bounds: list[tuple[int, str]] = []
    cursor = 0
    for start, grp in _GROUP_BOUNDS:
        grp_metrics = [m for m in _ORDERED[start:] if m in kept_set]
        if grp_metrics and grp_metrics[0] in kept_names:
            idx = kept_names.index(grp_metrics[0])
            new_bounds.append((idx, grp))

    matrix = np.clip(matrix, -4, 4)

    n_met, n_flt = matrix.shape
    fig, ax = plt.subplots(figsize=(max(14, n_flt * 0.65), max(8, n_met * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)

    ax.set_xticks(range(n_flt))
    ax.set_xticklabels(slugs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_met))
    ax.set_yticklabels(kept_names, fontsize=7)

    for tick, cls in zip(ax.get_xticklabels(), classes):
        tick.set_color(FAULT_CLASS_COLORS.get(cls, "black"))

    for start, grp in new_bounds:
        if start > 0:
            ax.axhline(start - 0.5, color="white", lw=1.5, alpha=0.8)
        ax.text(-0.6, start, grp, fontsize=7, fontweight="bold",
                va="top", ha="right", transform=ax.get_yaxis_transform(),
                color="#333333")

    plt.colorbar(im, ax=ax, label="δ = (during − pre) / pre_std  [clipped ±4]",
                 fraction=0.03, pad=0.02)
    ax.set_title(
        f"Metric Delta Heatmap — {n_flt} Faults × {n_met} Metrics  "
        f"(rows with max|δ| < {MIN_SIGNAL} removed)\n"
        "Red = metric increased during fault   Blue = decreased   "
        "x-label colour = fault class",
        fontsize=10,
    )

    patches = [mpatches.Patch(color=c, label=fc)
               for fc, c in FAULT_CLASS_COLORS.items()]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.18, 1.0), fontsize=8, title="Fault class")

    plt.tight_layout()
    out_path = args.out / "01_metric_delta_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
