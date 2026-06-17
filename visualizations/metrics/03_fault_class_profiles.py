#!/usr/bin/env python3
"""
visualizations/metrics/03_fault_class_profiles.py

Fault-class metric fingerprints — grouped bar chart.

For each fault class, computes the mean absolute delta per metric category:
    |δ| = mean over faults in class of |( during_mean - pre_mean ) / pre_std|

This shows which metric categories are most disturbed for each type of fault,
giving an at-a-glance "fingerprint" of each fault class.

Usage:
  python 03_fault_class_profiles.py [--data PATH] [--out PATH]
"""

import argparse
import json
import sys
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

_EXCLUDE = {"cpu_beyla", "cpu_monitoring", "mem_beyla", "mem_monitoring"}

CATEGORIES = {
    "Queries\n/ HTTP": [
        "http_server_req_rate", "http_client_req_rate",
        "http_server_duration", "http_client_duration",
        "http_server_err_rate", "http_client_err_rate",
    ],
    "CPU": ["cpu_usage", "cpu_throttled", "cpu_node"],
    "Memory": ["mem_container", "mem_node_available"],
    "Network": ["net_tx", "net_rx"],
    "AMF\nControl": [
        "amf_reg_req", "amf_reg_succ", "amf_reg_fail",
        "amf_auth_fail", "amf_auth_reject",
        "amf_sessions", "amf_subscribers",
        "amf_ran_ue_count", "amf_gnb_count", "amf_paging_req",
    ],
    "SMF / PFCP\nSessions": [
        "smf_pdu_req", "smf_pdu_succ", "smf_sessions",
        "smf_ues", "smf_bearers", "smf_qos_flows",
        "smf_n4_estab", "smf_n4_report", "smf_n4_report_succ",
        "pfcp_sessions", "pfcp_peers",
    ],
    "UPF / GTP": ["upf_sessions", "upf_qos_flows", "upf_n4_estab", "gtp_failed"],
}


def load_delta(exp_dir: Path) -> dict[str, float]:
    """Compute per-metric normalised delta for one experiment."""
    deltas: dict[str, float] = {}
    for fname, name, agg in _METRICS:
        if name in _EXCLUDE:
            continue
        pre_path    = exp_dir / "prometheus" / "pre"    / f"{fname}.csv"
        during_path = exp_dir / "prometheus" / "during" / f"{fname}.csv"
        pre_data    = _load_csv_aggregated(pre_path,    agg)
        dur_data    = _load_csv_aggregated(during_path, agg)
        if not pre_data or not dur_data:
            deltas[name] = 0.0
            continue
        pre_vals = list(pre_data.values())
        pre_mean = np.mean(pre_vals)
        pre_std  = np.std(pre_vals)
        dur_mean = np.mean(list(dur_data.values()))
        scale = max(pre_std, abs(pre_mean), 0.001)
        deltas[name] = float((dur_mean - pre_mean) / scale)
    return deltas


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    ap.add_argument("--out",  type=Path,
                    default=Path(__file__).parent / "out")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    class_deltas: dict[str, list[dict[str, float]]] = {}
    for slug, _ft, _nf, fc in EXPERIMENTS:
        exp_dir = args.data / slug
        if not (exp_dir / "prometheus" / "pre").is_dir():
            continue
        class_deltas.setdefault(fc, []).append(load_delta(exp_dir))

    if not class_deltas:
        print(f"No experiments found in {args.data}")
        return

    cat_names  = list(CATEGORIES.keys())
    class_names = sorted(class_deltas.keys())

    # matrix: (n_classes, n_categories)
    profile = np.zeros((len(class_names), len(cat_names)))
    for ci, fc in enumerate(class_names):
        for ki, (cat, metrics) in enumerate(CATEGORIES.items()):
            abs_deltas = []
            for fault_delta in class_deltas[fc]:
                for m in metrics:
                    if m in fault_delta:
                        abs_deltas.append(abs(fault_delta[m]))
            profile[ci, ki] = np.mean(abs_deltas) if abs_deltas else 0.0

    n_classes = len(class_names)
    n_cats    = len(cat_names)
    x         = np.arange(n_cats)
    bar_w     = 0.8 / n_classes

    fig, ax = plt.subplots(figsize=(14, 6))
    for ci, fc in enumerate(class_names):
        offset = (ci - n_classes / 2 + 0.5) * bar_w
        ax.bar(x + offset, profile[ci], bar_w,
               label=fc.replace("_", " "),
               color=FAULT_CLASS_COLORS.get(fc, "grey"),
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=10)
    ax.set_ylabel("Mean |δ|  (|during − pre| / pre_std)", fontsize=10)
    ax.set_title(
        "Fault-Class Metric Fingerprints\n"
        "Height = how much each metric category deviates during that fault type",
        fontsize=11,
    )
    ax.legend(title="Fault class", fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = args.out / "03_fault_class_profiles.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
