#!/usr/bin/env python3
"""
visualizations/feature_importance/02_feature_importance_by_fault_class.py

Top-5 feature importance per modality × fault class.
Same z-score delta metric as 01:  (mean_during - mean_pre) / std_all

Layout: 5 rows (fault classes) × 3 columns (Log / Metric / Trace).
"""

import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

FAULT_CLASSES = [
    "resource_exhaustion",
    "component_failure",
    "network_delay",
    "network_partition",
    "protocol_attack",
]
FC_DISPLAY = {
    "resource_exhaustion": "Resource Exhaustion  (n=6)",
    "component_failure":   "Component Failure  (n=5)",
    "network_delay":       "Network Delay  (n=3)",
    "network_partition":   "Network Partition  (n=4)",
    "protocol_attack":     "Protocol Attack  (n=4)",
}
FC_COUNTS = {
    "resource_exhaustion": 6,
    "component_failure":   5,
    "network_delay":       3,
    "network_partition":   4,
    "protocol_attack":     4,
}
FC_COLORS = {
    "resource_exhaustion": "#f39c12",
    "component_failure":   "#8e44ad",
    "network_delay":       "#2980b9",
    "network_partition":   "#16a085",
    "protocol_attack":     "#c0392b",
}

SURGE_COLOR = "#c0392b"
DROP_COLOR  = "#1a6faf"


# helpers (mirrored from 01_feature_importance.py)

def _load_module(name, path, extra_paths=None):
    for p in (extra_paths or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _zdelta(pre_mat, dur_mat):
    mu_pre  = pre_mat.mean(axis=0)
    mu_dur  = dur_mat.mean(axis=0)
    std_all = np.vstack([pre_mat, dur_mat]).std(axis=0).clip(1e-6)
    return (mu_dur - mu_pre) / std_all


def _clean_template(tmpl):
    s = tmpl
    if s.startswith('{"t":'):
        m_msg = re.search(r'"msg"\s*:\s*"([^"]+)"', s)
        if m_msg and re.search(r'[a-zA-Z]{3}', m_msg.group(1)):
            label = re.sub(r"(<[A-Z0-9_]+>\s*)+", "<*>", m_msg.group(1)).strip()
            s = f"[mongo] {label}"
        else:
            m_c   = re.search(r'"c"\s*:\s*"([^"]+)"', s)
            m_ctx = re.search(r'"ctx"\s*:\s*"([^"]+)"', s)
            comp  = m_c.group(1)   if m_c   else "?"
            ctx   = re.sub(r"<[^>]+>", "<*>", m_ctx.group(1)) if m_ctx else ""
            s     = f"[mongo:{comp}] {ctx}".strip(": ")
    else:
        s = re.sub(r"^\[?<N>-<N>-<N>[^]]*\]?\s*", "", s)
        s = re.sub(r"^\[?<UUID>[^]]*\]?\s*", "", s)
        s = re.sub(r"\[<N>\|[a-z]+\]\s*", "", s)
    s = re.sub(r"(<[A-Z0-9_]+>\s*){2,}", "<*> ", s)
    s = s.strip()
    return (s[:30] + "...") if len(s) > 33 else (s or tmpl[:33])


def _clean_trace_feat(name):
    if name.startswith("g_"):
        return "Global: " + name[2:].replace("_", " ")
    parts = name.split("_", 1)
    svc   = parts[0].upper()
    stat  = parts[1].replace("log_", "").replace("_", " ") if len(parts) > 1 else ""
    return f"{svc}: {stat}"


def log_features_by_class(top_k=5, bucket_s=30):
    logs_dir = str(ROOT / "models" / "logs")
    logs_dl  = _load_module("logs_data_loader",
                    ROOT / "models" / "logs" / "data_loader.py",
                    extra_paths=[logs_dir, str(ROOT / "analysis")])
    print("[Logs] Loading all records...")
    data    = logs_dl.load_all()
    all_rec = data["train"] + data["test"]

    tid_label = {}
    for r in all_rec:
        if r.template_id not in tid_label:
            tid_label[r.template_id] = _clean_template(r.template)

    tid_list = sorted(tid_label.keys())
    tid_idx  = {t: i for i, t in enumerate(tid_list)}
    F        = len(tid_list)

    def _count_matrix(records):
        buckets = defaultdict(list)
        for r in records:
            if r.timestamp_ns > 0:
                bid = int(r.timestamp_ns // 1_000_000_000) // bucket_s
                buckets[bid].append(r)
        rows = []
        for recs in buckets.values():
            row = np.zeros(F, dtype=np.float32)
            for r in recs:
                row[tid_idx[r.template_id]] += 1.0
            rows.append(row)
        return np.stack(rows) if rows else np.zeros((1, F))

    out = {}
    for fc in FAULT_CLASSES:
        pre_recs = [r for r in all_rec if r.phase == "pre"    and r.fault_class == fc]
        dur_recs = [r for r in all_rec if r.phase == "during" and r.fault_class == fc]
        pre_mat  = _count_matrix(pre_recs)
        dur_mat  = _count_matrix(dur_recs)
        # Require template to appear in >= max(2, n_faults) pre-windows
        min_sup  = max(2, FC_COUNTS[fc])
        support  = (pre_mat > 0).sum(axis=0)
        active   = np.where(support >= min_sup)[0]
        if active.size == 0:
            out[fc] = []
            continue
        deltas      = _zdelta(pre_mat[:, active], dur_mat[:, active])
        ranked      = np.argsort(np.abs(deltas))[::-1]
        active_tids = [tid_list[i] for i in active]
        out[fc]     = [(tid_label[active_tids[idx]], float(deltas[idx])) for idx in ranked[:top_k]]
        print(f"  [{fc}] top: {out[fc][0][0][:40]} delta={out[fc][0][1]:+.2f}")
    return out


def metric_features_by_class(top_k=5):
    met_dl = _load_module("metrics_data_loader",
                ROOT / "models" / "metrics" / "data_loader.py",
                extra_paths=[str(ROOT / "models" / "metrics"), str(ROOT / "analysis")])
    print("[Metrics] Loading all records...")
    data = met_dl.load_all()
    FEATURE_NAMES = met_dl.FEATURE_NAMES
    all_rec = data["train"] + data["test"]

    out = {}
    for fc in FAULT_CLASSES:
        pre = [r.values for r in all_rec if r.phase == "pre"    and r.fault_class == fc]
        dur = [r.values for r in all_rec if r.phase == "during" and r.fault_class == fc]
        if not pre or not dur:
            out[fc] = []
            continue
        pre_mat = np.nan_to_num(np.array(pre, dtype=np.float64))
        dur_mat = np.nan_to_num(np.array(dur, dtype=np.float64))
        deltas  = _zdelta(pre_mat, dur_mat)
        ranked  = np.argsort(np.abs(deltas))[::-1]
        out[fc] = [(FEATURE_NAMES[idx], float(deltas[idx])) for idx in ranked[:top_k]]
        print(f"  [{fc}] top: {out[fc][0][0]} delta={out[fc][0][1]:+.2f}")
    return out


def trace_features_by_class(top_k=5):
    tr_dl = _load_module("traces_data_loader",
                ROOT / "models" / "traces" / "data_loader.py",
                extra_paths=[str(ROOT / "models" / "traces"), str(ROOT / "analysis")])
    print("[Traces] Loading all records...")
    data = tr_dl.load_all()
    FEATURE_NAMES = tr_dl.FEATURE_NAMES
    all_rec = data["train"] + data["test"]

    out = {}
    for fc in FAULT_CLASSES:
        pre = [r.values for r in all_rec if r.phase == "pre"    and r.fault_class == fc]
        dur = [r.values for r in all_rec if r.phase == "during" and r.fault_class == fc]
        if not pre or not dur:
            out[fc] = []
            continue
        pre_mat = np.nan_to_num(np.array(pre, dtype=np.float64))
        dur_mat = np.nan_to_num(np.array(dur, dtype=np.float64))
        deltas  = _zdelta(pre_mat, dur_mat)
        ranked  = np.argsort(np.abs(deltas))[::-1]
        out[fc] = [(_clean_trace_feat(FEATURE_NAMES[idx]), float(deltas[idx])) for idx in ranked[:top_k]]
        print(f"  [{fc}] top: {out[fc][0][0]} delta={out[fc][0][1]:+.2f}")
    return out


def _make_bar_panel(ax, results, fontsize=7.5):
    if not results:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.axis("off")
        return
    names  = [r[0] for r in results]
    deltas = [r[1] for r in results]
    colors = [SURGE_COLOR if d > 0 else DROP_COLOR for d in deltas]
    y      = list(range(len(results)))
    ax.barh(y[::-1], [abs(d) for d in deltas], color=colors,
            edgecolor="white", height=0.55)
    ax.set_yticks(y[::-1])
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=fontsize - 1)
    x_max = max(abs(d) for d in deltas) if deltas else 1
    ax.set_xlim(right=x_max * 1.25)
    for i, d in enumerate(deltas):
        rank = len(results) - 1 - i
        c = SURGE_COLOR if d > 0 else DROP_COLOR
        ax.text(abs(d) + x_max * 0.03, rank,
                f"{'+'if d>0 else''}{d:.2f}",
                va="center", fontsize=fontsize - 1, color=c)


COL_HEADERS = [
    "Logs — DeepLog\n(per-template count)",
    "Metrics — TranAD\n(per-KPI value)",
    "Traces — GAL-MAD\n(per-service span stat)",
]


def plot(log_by_class, metric_by_class, trace_by_class):
    n_rows = len(FAULT_CLASSES)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(18, n_rows * 4.0),
        gridspec_kw={"wspace": 0.55, "hspace": 0.45},
    )
    fig.suptitle(
        "Top-5 Feature Importance per Modality × Fault Class\n"
        "z-score delta = (mean_during − mean_pre) / std_all  "
        "|  Logs: DeepLog  ·  Metrics: TranAD  ·  Traces: GAL-MAD",
        fontsize=11, y=1.015,
    )

    # Column headers (top row only)
    for col, hdr in enumerate(COL_HEADERS):
        axes[0, col].set_title(hdr, fontsize=9.5, pad=7, fontweight="bold")

    modality_dicts = [log_by_class, metric_by_class, trace_by_class]

    for row, fc in enumerate(FAULT_CLASSES):
        # Row label strip on the left edge of the leftmost axes
        ax0 = axes[row, 0]
        ax0.annotate(
            FC_DISPLAY[fc],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.85, 0.5), textcoords="axes fraction",
            fontsize=8.5, fontweight="bold",
            va="center", ha="right",
            color=FC_COLORS[fc],
            annotation_clip=False,
        )
        # Thin left border to visually group the row
        for spine_name in ["left"]:
            ax0.spines[spine_name].set_color(FC_COLORS[fc])
            ax0.spines[spine_name].set_linewidth(2.5)

        for col, d in enumerate(modality_dicts):
            ax = axes[row, col]
            _make_bar_panel(ax, d.get(fc, []))
            ax.set_xlabel("|z-delta|", fontsize=7)

    legend_handles = [
        Patch(facecolor=SURGE_COLOR, label="surges during fault"),
        Patch(facecolor=DROP_COLOR,  label="drops during fault"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.015), framealpha=0.8)

    fig.text(
        0.5, -0.04,
        "Each panel is independently scaled.  "
        "z-delta > 2 = feature moved > 2σ from its normal baseline during the fault.",
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4f8", ec="#aab7c4"),
    )

    out_path = OUT / "feature_importance_by_fault_class.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    log_by_class    = log_features_by_class(top_k=5)
    metric_by_class = metric_features_by_class(top_k=5)
    trace_by_class  = trace_features_by_class(top_k=5)
    plot(log_by_class, metric_by_class, trace_by_class)
    print("Done.")
