#!/usr/bin/env python3
"""
visualizations/feature_importance/01_feature_importance.py

Top-5 feature importance per modality using the actual features each model
operates on, ranked by mean normalised deviation between pre-phase (normal)
and during-phase (anomalous) windows.

  Metric used: z-score delta = (mean_during - mean_pre) / std_all

  Logs    -> DeepLog (best AUROC 0.897)
             Features: per-template count in 30-second windows (2300+ templates).
             The templates with the highest |delta| are the ones whose frequency
             changes most — exactly the sequences DeepLog's LSTM learns to flag.

  Metrics -> TranAD (best AUROC 0.597)
             Features: the 42 raw Prometheus KPI values TranAD reconstructs.

  Traces  -> GAL-MAD (best AUROC 0.782)
             Features: the 48-dim vector (11 services x 4 stats + 4 global)
             that GAL-MAD's GAT encoder processes.

All three use the same scale so bars within each panel are directly comparable.
Across panels the scale differs by design (different physical units per modality).
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

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


# module loader helper  (avoids name collision between three data_loader.py)

def _load_module(name: str, path: Path, extra_paths: list[str] = None):
    for p in (extra_paths or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _zdelta(pre_mat: np.ndarray, dur_mat: np.ndarray) -> np.ndarray:
    """
    pre_mat: (N_pre, F)   dur_mat: (N_dur, F)
    Returns (F,) array of (mean_during - mean_pre) / std_all.

    std_all is computed over the combined pre+during matrix so that features
    whose pre-phase variance is near-zero (e.g. error rates that are 0 in
    normal operation) don't produce exploding z-scores.
    """
    mu_pre  = pre_mat.mean(axis=0)
    mu_dur  = dur_mat.mean(axis=0)
    std_all = np.vstack([pre_mat, dur_mat]).std(axis=0).clip(1e-6)
    return (mu_dur - mu_pre) / std_all


def log_feature_importance(top_k: int = 5, bucket_s: int = 30):
    """
    Build per-template count vectors for every 30-second window, then compute
    the z-score delta between pre-phase and during-phase windows.

    Returns list of (template_label, z_delta, direction) sorted by |z_delta|.
    """
    logs_dir = str(ROOT / "models" / "logs")
    logs_dl  = _load_module(
        "logs_data_loader",
        ROOT / "models" / "logs" / "data_loader.py",
        extra_paths=[logs_dir, str(ROOT / "analysis")]
    )

    print("[Logs] Loading records...")
    data    = logs_dl.load_all()
    all_rec = data["train"] + data["test"]

    tid_label: dict[int, str] = {}
    for r in all_rec:
        if r.template_id not in tid_label:
            tid_label[r.template_id] = _clean_template(r.template)

    def _bucket(records):
        buckets: dict[int, list] = defaultdict(list)
        for r in records:
            if r.timestamp_ns > 0:
                bid = int(r.timestamp_ns // 1_000_000_000) // bucket_s
                buckets[bid].append(r)
        return buckets

    pre_recs  = [r for r in all_rec if r.phase == "pre"]
    dur_recs  = [r for r in all_rec if r.phase == "during"]

    pre_buckets = _bucket(pre_recs)
    dur_buckets = _bucket(dur_recs)

    # Only include templates that appear in at least 5 windows (reduce noise)
    tid_list = sorted(tid_label.keys())
    tid_idx  = {t: i for i, t in enumerate(tid_list)}
    F        = len(tid_list)

    def _count_matrix(buckets):
        rows = []
        for recs in buckets.values():
            row = np.zeros(F, dtype=np.float32)
            for r in recs:
                row[tid_idx[r.template_id]] += 1.0
            rows.append(row)
        return np.stack(rows) if rows else np.zeros((1, F))

    print("[Logs] Building count matrices...")
    pre_mat = _count_matrix(pre_buckets)
    dur_mat = _count_matrix(dur_buckets)

    support = (pre_mat > 0).sum(axis=0)
    active  = np.where(support >= 5)[0]

    pre_active = pre_mat[:, active]
    dur_active = dur_mat[:, active]
    active_tids = [tid_list[i] for i in active]

    deltas = _zdelta(pre_active, dur_active)
    ranked = np.argsort(np.abs(deltas))[::-1]

    results = []
    for idx in ranked[:top_k]:
        tid   = active_tids[idx]
        label = tid_label[tid]
        d     = float(deltas[idx])
        results.append((label, d))
    return results


def _clean_template(tmpl: str) -> str:
    """Shorten a Drain template to a readable label."""
    s = tmpl
    # MongoDB structured JSON log: {"t":{"$date":...},"s":...,"c":...,"msg":...}
    if s.startswith('{"t":'):
        # Try "msg" field first; skip if it's only Drain wildcards
        m_msg = re.search(r'"msg"\s*:\s*"([^"]+)"', s)
        if m_msg and re.search(r'[a-zA-Z]{3}', m_msg.group(1)):
            label = m_msg.group(1)
            label = re.sub(r"(<[A-Z0-9_]+>\s*)+", "<*>", label).strip()
            s = f"[mongo] {label}"
        else:
            # Fall back to component "c" + severity "s"
            m_c   = re.search(r'"c"\s*:\s*"([^"]+)"', s)
            m_ctx = re.search(r'"ctx"\s*:\s*"([^"]+)"', s)
            comp  = m_c.group(1)   if m_c   else "?"
            ctx   = m_ctx.group(1) if m_ctx else ""
            ctx   = re.sub(r"<[^>]+>", "<*>", ctx)
            s     = f"[mongo:{comp}] {ctx}".strip(": ")
    else:
        # Remove leading timestamp bracket e.g. [<N>-<N>-<N> <N>:<PORT>:...]
        s = re.sub(r"^\[?<N>-<N>-<N>[^]]*\]?\s*", "", s)
        s = re.sub(r"^\[?<UUID>[^]]*\]?\s*", "", s)
        # Remove log-level tags like [<N>|nas]
        s = re.sub(r"\[<N>\|[a-z]+\]\s*", "", s)
    s = re.sub(r"(<[A-Z0-9_]+>\s*){2,}", "<*> ", s)
    s = s.strip()
    return (s[:55] + "...") if len(s) > 58 else (s or tmpl[:58])


def metric_feature_importance(top_k: int = 5):
    """
    Compute z-score delta per KPI between pre and during phases.
    These are exactly the 42 values TranAD reconstructs at each timestep.
    """
    met_dl = _load_module(
        "metrics_data_loader",
        ROOT / "models" / "metrics" / "data_loader.py",
        extra_paths=[str(ROOT / "models" / "metrics"), str(ROOT / "analysis")]
    )

    print("[Metrics] Loading records...")
    data = met_dl.load_all()

    FEATURE_NAMES = met_dl.FEATURE_NAMES
    all_rec = data["train"] + data["test"]

    pre_mat = np.stack([r.values for r in all_rec if r.phase == "pre"])
    dur_mat = np.stack([r.values for r in all_rec if r.phase == "during"])
    pre_mat = np.nan_to_num(pre_mat.astype(np.float64))
    dur_mat = np.nan_to_num(dur_mat.astype(np.float64))

    deltas  = _zdelta(pre_mat, dur_mat)
    ranked  = np.argsort(np.abs(deltas))[::-1]

    results = []
    for idx in ranked[:top_k]:
        results.append((FEATURE_NAMES[idx], float(deltas[idx])))
    return results


def trace_feature_importance(top_k: int = 5):
    """
    Compute z-score delta per trace feature between pre and during phases.
    These are the same (service, stat) pairs that GAL-MAD's GAT encoder uses.
    """
    tr_dl = _load_module(
        "traces_data_loader",
        ROOT / "models" / "traces" / "data_loader.py",
        extra_paths=[str(ROOT / "models" / "traces"), str(ROOT / "analysis")]
    )

    print("[Traces] Loading records...")
    data = tr_dl.load_all()

    FEATURE_NAMES = tr_dl.FEATURE_NAMES
    all_rec = data["train"] + data["test"]

    pre_mat = np.stack([r.values for r in all_rec if r.phase == "pre"])
    dur_mat = np.stack([r.values for r in all_rec if r.phase == "during"])
    pre_mat = np.nan_to_num(pre_mat.astype(np.float64))
    dur_mat = np.nan_to_num(dur_mat.astype(np.float64))

    deltas  = _zdelta(pre_mat, dur_mat)
    ranked  = np.argsort(np.abs(deltas))[::-1]

    results = []
    for idx in ranked[:top_k]:
        # Clean name: "amf_span_count" -> "AMF: span count"
        raw   = FEATURE_NAMES[idx]
        label = _clean_trace_feat(raw)
        results.append((label, float(deltas[idx])))
    return results


def _clean_trace_feat(name: str) -> str:
    """'amf_log_mean_dur' -> 'AMF: mean dur'  /  'g_trace_count' -> 'Global: trace count'"""
    if name.startswith("g_"):
        return "Global: " + name[2:].replace("_", " ")
    parts = name.split("_", 1)
    svc   = parts[0].upper()
    stat  = parts[1].replace("log_", "").replace("_", " ") if len(parts) > 1 else ""
    return f"{svc}: {stat}"


SURGE_COLOR = "#c0392b"   # red  — feature surges during fault
DROP_COLOR  = "#1a6faf"   # blue — feature drops during fault


def _make_bar_panel(ax, results, xlabel, title):
    names  = [r[0] for r in results]
    deltas = [r[1] for r in results]
    colors = [SURGE_COLOR if d > 0 else DROP_COLOR for d in deltas]
    y      = list(range(len(results)))

    ax.barh(y[::-1], [abs(d) for d in deltas],
            color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(y[::-1])
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    x_max = max(abs(d) for d in deltas) if deltas else 1
    for i, d in enumerate(deltas):
        rank = len(results) - 1 - i
        arrow = "+" if d > 0 else "-"
        c = SURGE_COLOR if d > 0 else DROP_COLOR
        ax.text(abs(d) + x_max * 0.02, rank,
                f"{arrow}{abs(d):.2f}", va="center", fontsize=8, color=c)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=SURGE_COLOR, label="surges during fault"),
                 Patch(facecolor=DROP_COLOR,  label="drops during fault")],
        loc="lower right", fontsize=7.5, framealpha=0.7
    )


def plot(log_feats, metric_feats, trace_feats):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Top-5 Feature Importance per Modality  "
        "(z-score delta: (mean_during - mean_pre) / std_all)\n"
        "Logs: DeepLog  |  Metrics: TranAD  |  Traces: GAL-MAD",
        fontsize=11, y=1.02
    )

    _make_bar_panel(
        axes[0], log_feats,
        xlabel="|z-score delta| (template count per 30s window)",
        title="Logs — DeepLog\nper-template count shift"
    )
    _make_bar_panel(
        axes[1], metric_feats,
        xlabel="|z-score delta| (KPI value)",
        title="Metrics — TranAD\nper-KPI value shift"
    )
    _make_bar_panel(
        axes[2], trace_feats,
        xlabel="|z-score delta| (span stat)",
        title="Traces — GAL-MAD\nper-service span stat shift"
    )

    fig.text(
        0.5, -0.04,
        "z-score delta > 2 means the feature moved more than 2 standard deviations from its normal baseline during the fault.\n"
        "All bars use the same scale within each panel. Across panels the units differ (counts vs KPI values vs span durations).",
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4f8", ec="#aab7c4")
    )

    plt.tight_layout()
    out_path = OUT / "feature_importance_top5.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    print("=== Feature Importance (z-score delta) ===\n")

    log_feats    = log_feature_importance(top_k=5)
    metric_feats = metric_feature_importance(top_k=5)
    trace_feats  = trace_feature_importance(top_k=5)

    print("\n--- Logs (DeepLog) ---")
    for name, d in log_feats:
        print(f"  {'surge' if d>0 else 'drop ':5s}  delta={d:+.2f}  {name}")

    print("\n--- Metrics (TranAD) ---")
    for name, d in metric_feats:
        print(f"  {'surge' if d>0 else 'drop ':5s}  delta={d:+.2f}  {name}")

    print("\n--- Traces (GAL-MAD) ---")
    for name, d in trace_feats:
        print(f"  {'surge' if d>0 else 'drop ':5s}  delta={d:+.2f}  {name}")

    plot(log_feats, metric_feats, trace_feats)
    print("\nDone.")
