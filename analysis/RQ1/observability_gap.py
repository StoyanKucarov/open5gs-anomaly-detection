"""
analysis/observability_gap.py

Characterises the "observability gap": which faults are detectable only at
the eBPF/kernel level, which are detectable by Prometheus alone, which by both,
and which by neither.

Uses the per-fault verdicts from fault_detection_coverage.py and enriches them
with signal-strength scores to produce a heatmap and a narrative classification.

Produces:
  figures/observability_gap_heatmap.pdf
  figures/observability_gap_venn.pdf
  tables/observability_gap_classification.csv
  tables/observability_gap_signal_strength.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FIGURES_DIR, TABLES_DIR, ALL_FAULTS, FAULT_LABELS, FAULT_CATEGORIES,
    SIGMA_THRESHOLD, BEYLA_LATENCY_INCREASE_THRESHOLD, BEYLA_ERROR_RATE_DELTA_THRESHOLD,
    FIGURE_DPI, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    PALETTE,
)
from load_data import load_fault


# ---------------------------------------------------------------------------
# Signal strength scoring
# ---------------------------------------------------------------------------
# Each signal gets a continuous score in [0, 1] representing how strongly
# it fired during the fault relative to the pre-fault baseline.
# 0 = no change, 1 = maximum deviation observed across all faults.
# These raw scores are normalised per-column after computing all faults.

SIGNAL_COLUMNS = [
    "prom_cpu_zscore",
    "prom_mem_zscore",
    "prom_throttle_zscore",
    "prom_net_zscore",
    "prom_restarts",
    "prom_k8s_events",
    "ebpf_latency_increase",
    "ebpf_error_rate_delta",
    "ebpf_span_drop",
    "loki_error_count",
    "loki_ue_failures",
    "nrf_drop",
]

SIGNAL_LABELS = {
    "prom_cpu_zscore":        "Prom: CPU z-score",
    "prom_mem_zscore":        "Prom: Mem z-score",
    "prom_throttle_zscore":   "Prom: CPU throttle z-score",
    "prom_net_zscore":        "Prom: Network I/O z-score",
    "prom_restarts":          "Prom: Pod restarts",
    "prom_k8s_events":        "Prom: K8s Warning events",
    "ebpf_latency_increase":  "eBPF: Latency increase (%)",
    "ebpf_error_rate_delta":  "eBPF: Error rate delta",
    "ebpf_span_drop":         "eBPF: Span count drop (%)",
    "loki_error_count":       "Loki: Error log lines",
    "loki_ue_failures":       "Loki: UE failure lines",
    "nrf_drop":               "NRF: Registration drop",
}

# Group signals by method for the heatmap column grouping
SIGNAL_GROUPS = {
    "Prometheus": ["prom_cpu_zscore", "prom_mem_zscore", "prom_throttle_zscore",
                   "prom_net_zscore", "prom_restarts", "prom_k8s_events"],
    "eBPF/Beyla": ["ebpf_latency_increase", "ebpf_error_rate_delta", "ebpf_span_drop"],
    "Loki": ["loki_error_count", "loki_ue_failures"],
    "NRF API": ["nrf_drop"],
}


def _zscore_deviation(pre_prom: dict, during_prom: dict, key: str) -> float:
    """
    Return the z-score of the during-fault mean relative to the pre-fault
    distribution. Clamped to [0, 10].
    """
    def _vals(prom_dict):
        df = prom_dict.get(key, pd.DataFrame())
        if df.empty or "value" not in df.columns:
            return np.array([])
        return pd.to_numeric(df["value"], errors="coerce").dropna().values

    pre_vals = _vals(pre_prom)
    dur_vals = _vals(during_prom)

    if len(pre_vals) < 2 or len(dur_vals) == 0:
        return 0.0

    pre_mean = float(np.mean(pre_vals))
    pre_std = float(np.std(pre_vals))
    dur_mean = float(np.mean(dur_vals))

    if pre_std == 0:
        return float(abs(dur_mean - pre_mean) / max(pre_mean, 1e-9) * 10)

    return float(min(abs(dur_mean - pre_mean) / pre_std, 10.0))


def _max_zscore(pre_prom: dict, during_prom: dict, keys: list) -> float:
    return max((_zscore_deviation(pre_prom, during_prom, k) for k in keys), default=0.0)


def compute_signal_strengths() -> pd.DataFrame:
    rows = []
    for fault in ALL_FAULTS:
        print(f"  [gap] Scoring {fault}...")
        data = load_fault(fault)
        pre = data["pre"]
        during = data["during"]

        pre_prom = pre["prometheus"]
        dur_prom = during["prometheus"]

        # Prometheus signals
        cpu_z = _max_zscore(pre_prom, dur_prom, [
            "container_cpu_usage_rate", "monitoring_cpu_usage_rate"])
        mem_z = _max_zscore(pre_prom, dur_prom, [
            "container_memory_working_set_bytes", "monitoring_memory_working_set"])
        throttle_z = _zscore_deviation(pre_prom, dur_prom, "container_cpu_throttled_rate")
        net_z = _max_zscore(pre_prom, dur_prom, [
            "network_rx_bytes_rate", "network_tx_bytes_rate"])

        # Pod restarts: count increase
        pre_restarts_df = pre_prom.get("pod_restarts", pd.DataFrame())
        dur_restarts_df = dur_prom.get("pod_restarts", pd.DataFrame())
        pre_max_r = float(pd.to_numeric(
            pre_restarts_df.get("value", pd.Series()), errors="coerce").max()) \
            if not pre_restarts_df.empty else 0.0
        dur_max_r = float(pd.to_numeric(
            dur_restarts_df.get("value", pd.Series()), errors="coerce").max()) \
            if not dur_restarts_df.empty else 0.0
        restart_delta = max(0.0, dur_max_r - pre_max_r)

        # K8s events: count of Warning events
        k8s_warnings = sum(1 for e in during["events"] if e.get("type") == "Warning")

        # eBPF/Beyla signals
        pre_p95 = _beyla_p95(pre["jaeger_summary"])
        dur_p95 = _beyla_p95(during["jaeger_summary"])
        if not (np.isnan(pre_p95) or np.isnan(dur_p95)) and pre_p95 > 0:
            latency_increase = max(0.0, (dur_p95 - pre_p95) / pre_p95)
        else:
            latency_increase = 0.0

        pre_err = _beyla_err_rate(pre["jaeger_summary"])
        dur_err = _beyla_err_rate(during["jaeger_summary"])
        err_delta = max(0.0, dur_err - pre_err) if not (np.isnan(pre_err) or np.isnan(dur_err)) else 0.0

        pre_spans = _span_count(pre["jaeger_summary"])
        dur_spans = _span_count(during["jaeger_summary"])
        span_drop_pct = max(0.0, (pre_spans - dur_spans) / max(pre_spans, 1)) if pre_spans > 0 else 0.0

        # Loki signals
        pre_errors = len(pre["loki"].get("errors", pd.DataFrame()))
        dur_errors = len(during["loki"].get("errors", pd.DataFrame()))
        loki_error_delta = max(0, dur_errors - pre_errors)
        loki_ue = len(during["loki"].get("ue_failures", pd.DataFrame()))

        # NRF drop
        nrf_drop = _nrf_drop_count(pre["nrf"], during["nrf"])

        rows.append({
            "fault": fault,
            "prom_cpu_zscore":       cpu_z,
            "prom_mem_zscore":       mem_z,
            "prom_throttle_zscore":  throttle_z,
            "prom_net_zscore":       net_z,
            "prom_restarts":         restart_delta,
            "prom_k8s_events":       float(k8s_warnings),
            "ebpf_latency_increase": latency_increase,
            "ebpf_error_rate_delta": err_delta,
            "ebpf_span_drop":        span_drop_pct,
            "loki_error_count":      float(loki_error_delta),
            "loki_ue_failures":      float(loki_ue),
            "nrf_drop":              float(nrf_drop),
        })

    df = pd.DataFrame(rows).set_index("fault")
    return df


def _beyla_p95(jaeger_summary: dict) -> float:
    if not jaeger_summary:
        return np.nan
    p95s = [v["duration_us_p95"] for v in jaeger_summary.values()
            if isinstance(v, dict) and "duration_us_p95" in v]
    return float(np.mean(p95s)) if p95s else np.nan


def _beyla_err_rate(jaeger_summary: dict) -> float:
    if not jaeger_summary:
        return np.nan
    rates = [v["error_rate"] for v in jaeger_summary.values()
             if isinstance(v, dict) and "error_rate" in v]
    return float(np.mean(rates)) if rates else np.nan


def _span_count(jaeger_summary: dict) -> int:
    if not jaeger_summary:
        return 0
    return sum(v.get("span_count", 0) for v in jaeger_summary.values()
               if isinstance(v, dict))


def _nrf_drop_count(pre_nrf: dict, dur_nrf: dict) -> int:
    if not pre_nrf or not dur_nrf:
        return 0
    drops = 0
    for nf_type, pre_count in pre_nrf.items():
        if not isinstance(pre_count, int):
            continue
        dur_count = dur_nrf.get(nf_type, pre_count)
        if isinstance(dur_count, int) and dur_count < pre_count:
            drops += pre_count - dur_count
    return drops


# ---------------------------------------------------------------------------
# Normalise signal strengths to [0, 1] for heatmap
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise each column to [0, 1]."""
    result = df.copy()
    for col in df.columns:
        col_max = df[col].max()
        if col_max > 0:
            result[col] = df[col] / col_max
        else:
            result[col] = 0.0
    return result


# ---------------------------------------------------------------------------
# Classify faults into observability gap categories
# ---------------------------------------------------------------------------

def classify_faults(strength_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each fault into one of four categories:
      - prometheus_only: detected by Prometheus signals, not eBPF
      - ebpf_only:       detected by eBPF signals, not Prometheus
      - both:            detected by both
      - neither:         not clearly detected by either
    """
    prom_cols = SIGNAL_GROUPS["Prometheus"]
    ebpf_cols = SIGNAL_GROUPS["eBPF/Beyla"]

    # Thresholds on normalised scores
    PROM_THRESHOLD = 0.20   # normalised score > 20% of max = "detected"
    EBPF_THRESHOLD = 0.20

    norm_df = normalise(strength_df)

    rows = []
    for fault in strength_df.index:
        prom_score = norm_df.loc[fault, prom_cols].max()
        ebpf_score = norm_df.loc[fault, ebpf_cols].max()

        prom_det = bool(prom_score > PROM_THRESHOLD)
        ebpf_det = bool(ebpf_score > EBPF_THRESHOLD)

        if prom_det and ebpf_det:
            category = "both"
        elif prom_det:
            category = "prometheus_only"
        elif ebpf_det:
            category = "ebpf_only"
        else:
            category = "neither"

        rows.append({
            "fault": fault,
            "fault_label": FAULT_LABELS.get(fault, fault),
            "prom_max_score": round(float(prom_score), 3),
            "ebpf_max_score": round(float(ebpf_score), 3),
            "category": category,
        })

    return pd.DataFrame(rows).set_index("fault")


# ---------------------------------------------------------------------------
# Figure: Signal strength heatmap
# ---------------------------------------------------------------------------

def plot_signal_strength_heatmap(strength_df: pd.DataFrame):
    norm_df = normalise(strength_df)

    # Order faults by category
    ordered_faults = []
    for cat_faults in FAULT_CATEGORIES.values():
        for f in cat_faults:
            if f in norm_df.index:
                ordered_faults.append(f)
    for f in ALL_FAULTS:
        if f not in ordered_faults and f in norm_df.index:
            ordered_faults.append(f)

    if not ordered_faults:
        print("  [gap] No data for heatmap, skipping")
        return

    matrix = norm_df.loc[ordered_faults, SIGNAL_COLUMNS].values
    row_labels = [FAULT_LABELS.get(f, f) for f in ordered_faults]
    col_labels = [SIGNAL_LABELS[c] for c in SIGNAL_COLUMNS]

    fig_height = max(7, len(ordered_faults) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=FIGURE_DPI)

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(SIGNAL_COLUMNS)))
    ax.set_xticklabels(col_labels, fontsize=FONT_SIZE_TICK - 1, rotation=45, ha="left")
    ax.set_yticks(range(len(ordered_faults)))
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE_TICK)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Column group separators
    group_boundaries = []
    pos = 0
    for group_name, cols in SIGNAL_GROUPS.items():
        pos += len(cols)
        group_boundaries.append((group_name, pos))

    prev = 0
    for group_name, end in group_boundaries:
        mid = (prev + end - 1) / 2
        ax.text(mid, -1.5, group_name, ha="center", va="bottom",
                fontsize=FONT_SIZE_TICK, fontweight="bold",
                transform=ax.get_xaxis_transform())
        if end < len(SIGNAL_COLUMNS):
            ax.axvline(end - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        prev = end

    plt.colorbar(im, ax=ax, label="Normalised signal strength", shrink=0.6)

    ax.set_title("Observability gap — signal strength per fault per method",
                 fontsize=FONT_SIZE_TITLE, pad=40)

    fig.tight_layout()
    out = FIGURES_DIR / "observability_gap_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gap] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Venn-style bar chart of gap categories
# ---------------------------------------------------------------------------

def plot_gap_categories(classification_df: pd.DataFrame):
    cats = ["both", "prometheus_only", "ebpf_only", "neither"]
    cat_labels = ["Both detect", "Prometheus only", "eBPF only", "Neither"]
    cat_colors = ["#4CAF50", PALETTE["prometheus"], PALETTE["ebpf"], "#9E9E9E"]

    counts = [int((classification_df["category"] == c).sum()) for c in cats]
    total = len(classification_df)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=FIGURE_DPI)
    bars = ax.bar(cat_labels, counts, color=cat_colors, alpha=0.85, width=0.5)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{count}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=FONT_SIZE_TICK, fontweight="bold")

    ax.set_ylabel("Number of faults", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"Observability gap classification ({total} faults)",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_ylim(0, max(counts) + 3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "observability_gap_categories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gap] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Scatter plot — Prometheus score vs. eBPF score per fault
# ---------------------------------------------------------------------------

def plot_prom_vs_ebpf_scatter(classification_df: pd.DataFrame):
    cat_colors = {
        "both":            "#4CAF50",
        "prometheus_only": PALETTE["prometheus"],
        "ebpf_only":       PALETTE["ebpf"],
        "neither":         "#9E9E9E",
    }
    cat_labels = {
        "both":            "Both detect",
        "prometheus_only": "Prometheus only",
        "ebpf_only":       "eBPF only",
        "neither":         "Neither",
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)

    for cat, color in cat_colors.items():
        subset = classification_df[classification_df["category"] == cat]
        if subset.empty:
            continue
        ax.scatter(subset["prom_max_score"], subset["ebpf_max_score"],
                   c=color, label=cat_labels[cat], s=80, alpha=0.85, zorder=3)

    # Threshold lines
    ax.axvline(0.20, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.20, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.21, 0.01, "Prom threshold", fontsize=FONT_SIZE_TICK - 1, color="gray")
    ax.text(0.01, 0.21, "eBPF threshold", fontsize=FONT_SIZE_TICK - 1, color="gray",
            rotation=90, va="bottom")

    # Annotate each point with fault label
    for fault, row in classification_df.iterrows():
        label = FAULT_LABELS.get(str(fault), str(fault))
        # Shorten label for readability
        short = label.split("–")[-1].strip() if "–" in label else label
        ax.annotate(short, (row["prom_max_score"], row["ebpf_max_score"]),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Prometheus max signal strength (normalised)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("eBPF/Beyla max signal strength (normalised)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Observability gap: Prometheus vs. eBPF signal strength per fault",
                 fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "observability_gap_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [gap] Saved {out}")


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

def save_tables(strength_df: pd.DataFrame, classification_df: pd.DataFrame):
    # Signal strength (raw)
    strength_out = strength_df.copy()
    strength_out.index = [FAULT_LABELS.get(f, f) for f in strength_out.index]
    strength_out.to_csv(TABLES_DIR / "observability_gap_signal_strength.csv",
                        float_format="%.4f")
    print(f"  [gap] Saved {TABLES_DIR / 'observability_gap_signal_strength.csv'}")

    # Classification
    class_out = classification_df.copy()
    class_out.to_csv(TABLES_DIR / "observability_gap_classification.csv")
    print(f"  [gap] Saved {TABLES_DIR / 'observability_gap_classification.csv'}")

    # Summary text for paper
    cats = classification_df["category"].value_counts()
    summary_lines = [
        "Observability Gap Summary",
        "=" * 40,
        f"Total faults analysed: {len(classification_df)}",
        "",
        f"Both methods detect:    {cats.get('both', 0)} faults",
        f"Prometheus only:        {cats.get('prometheus_only', 0)} faults",
        f"eBPF only:              {cats.get('ebpf_only', 0)} faults",
        f"Neither detects:        {cats.get('neither', 0)} faults",
        "",
        "eBPF-only faults (observability gap):",
    ]
    ebpf_only = classification_df[classification_df["category"] == "ebpf_only"]
    for _, row in ebpf_only.iterrows():
        summary_lines.append(f"  - {row['fault_label']}")

    summary_path = TABLES_DIR / "observability_gap_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"  [gap] Saved {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("[gap] Computing observability gap analysis...")
    strength_df = compute_signal_strengths()
    classification_df = classify_faults(strength_df)
    save_tables(strength_df, classification_df)
    plot_signal_strength_heatmap(strength_df)
    plot_gap_categories(classification_df)
    plot_prom_vs_ebpf_scatter(classification_df)
    print("[gap] Done.")
    return strength_df, classification_df


if __name__ == "__main__":
    run()
