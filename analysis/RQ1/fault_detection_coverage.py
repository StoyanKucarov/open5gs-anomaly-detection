"""
analysis/fault_detection_coverage.py

Computes per-fault detection verdicts for each signal type, then produces
the fault detection coverage matrix (Table 2 in the paper).

Detection logic:
  Prometheus signals — flag as detected if any metric deviates > SIGMA_THRESHOLD
  standard deviations from the pre-fault window mean, OR if pod restarts > 0,
  OR if K8s Warning events appear during the fault window.

  eBPF/Beyla signals — flag as detected if p95 latency increases > 50% from
  pre-fault baseline, OR error rate increases > 10pp.

  Loki signals — flag as detected if any error log lines appear during fault
  that were absent (or significantly more frequent) in the pre-fault window.

Produces:
  figures/fault_detection_coverage_matrix.pdf
  tables/fault_detection_coverage.csv
  tables/fault_detection_verdicts.csv   (raw per-signal verdicts)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FIGURES_DIR, TABLES_DIR, ALL_FAULTS, FAULT_LABELS, FAULT_CATEGORIES,
    SIGMA_THRESHOLD, BEYLA_LATENCY_INCREASE_THRESHOLD, BEYLA_ERROR_RATE_DELTA_THRESHOLD,
    FIGURE_DPI, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)
from load_data import load_fault


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _prom_series_mean_std(prom_dict: dict, key: str) -> tuple:
    """Return (mean, std) of 'value' column for a Prometheus CSV key."""
    df = prom_dict.get(key, pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan, np.nan
    vals = pd.to_numeric(df["value"], errors="coerce").dropna()
    if vals.empty:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std())


def _prom_series_max(prom_dict: dict, key: str) -> float:
    df = prom_dict.get(key, pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    return float(pd.to_numeric(df["value"], errors="coerce").max())


def _pod_restarts_increased(pre_prom: dict, during_prom: dict) -> bool:
    """True if any pod restart count is higher during fault than pre-fault."""
    pre_df = pre_prom.get("pod_restarts", pd.DataFrame())
    dur_df = during_prom.get("pod_restarts", pd.DataFrame())
    if pre_df.empty or dur_df.empty:
        return False
    pre_max = pd.to_numeric(pre_df["value"], errors="coerce").max()
    dur_max = pd.to_numeric(dur_df["value"], errors="coerce").max()
    if pd.isna(pre_max) or pd.isna(dur_max):
        return False
    return float(dur_max) > float(pre_max)


def _k8s_warning_events(events: list) -> bool:
    """True if any Warning-type K8s events exist."""
    return any(e.get("type") == "Warning" for e in events)


def _loki_error_count(loki_dict: dict) -> int:
    """Count error log lines in the errors CSV."""
    df = loki_dict.get("errors", pd.DataFrame())
    if df.empty:
        return 0
    return len(df)


def _beyla_p95_us(jaeger_summary: dict) -> float:
    """Mean p95 latency (µs) across all services."""
    if not jaeger_summary:
        return np.nan
    p95s = [v["duration_us_p95"] for v in jaeger_summary.values()
            if isinstance(v, dict) and "duration_us_p95" in v]
    return float(np.mean(p95s)) if p95s else np.nan


def _beyla_error_rate(jaeger_summary: dict) -> float:
    """Mean error rate across all services."""
    if not jaeger_summary:
        return np.nan
    rates = [v["error_rate"] for v in jaeger_summary.values()
             if isinstance(v, dict) and "error_rate" in v]
    return float(np.mean(rates)) if rates else np.nan


# ---------------------------------------------------------------------------
# Per-fault detection logic
# ---------------------------------------------------------------------------

PROMETHEUS_METRICS_TO_CHECK = [
    "container_cpu_usage_rate",
    "container_memory_working_set_bytes",
    "container_cpu_throttled_rate",
    "monitoring_cpu_usage_rate",
    "network_rx_bytes_rate",
    "network_tx_bytes_rate",
    "beyla_http_server_request_rate",
    "beyla_http_client_request_rate",
]


def detect_prometheus(pre: dict, during: dict) -> dict:
    """
    Returns dict of signal_name → bool (detected).
    Uses >SIGMA_THRESHOLD σ deviation from pre-fault mean, pod restarts, or K8s events.
    """
    results = {}

    # Metric deviation check
    for metric in PROMETHEUS_METRICS_TO_CHECK:
        pre_mean, pre_std = _prom_series_mean_std(pre["prometheus"], metric)
        dur_mean, _ = _prom_series_mean_std(during["prometheus"], metric)

        if np.isnan(pre_mean) or np.isnan(dur_mean):
            results[f"prom_{metric}"] = False
            continue

        if np.isnan(pre_std) or pre_std == 0:
            # No variance in pre — any change counts
            results[f"prom_{metric}"] = abs(dur_mean - pre_mean) > pre_mean * 0.10
        else:
            z = abs(dur_mean - pre_mean) / pre_std
            results[f"prom_{metric}"] = bool(z > SIGMA_THRESHOLD)

    # Pod restarts
    results["prom_pod_restarts"] = _pod_restarts_increased(
        pre["prometheus"], during["prometheus"]
    )

    # K8s Warning events
    results["prom_k8s_events"] = _k8s_warning_events(during["events"])

    return results


def detect_ebpf(pre: dict, during: dict) -> dict:
    """
    Returns dict of signal_name → bool (detected).
    Uses p95 latency increase > 50% or error rate increase > 10pp.
    """
    pre_p95 = _beyla_p95_us(pre["jaeger_summary"])
    dur_p95 = _beyla_p95_us(during["jaeger_summary"])
    pre_err = _beyla_error_rate(pre["jaeger_summary"])
    dur_err = _beyla_error_rate(during["jaeger_summary"])

    latency_detected = False
    if not (np.isnan(pre_p95) or np.isnan(dur_p95)) and pre_p95 > 0:
        latency_detected = (dur_p95 - pre_p95) / pre_p95 > BEYLA_LATENCY_INCREASE_THRESHOLD

    error_detected = False
    if not (np.isnan(pre_err) or np.isnan(dur_err)):
        error_detected = (dur_err - pre_err) > BEYLA_ERROR_RATE_DELTA_THRESHOLD

    # Span count drop (service went dark — no traces at all during fault)
    pre_spans = sum(v.get("span_count", 0) for v in pre["jaeger_summary"].values()
                    if isinstance(v, dict)) if pre["jaeger_summary"] else 0
    dur_spans = sum(v.get("span_count", 0) for v in during["jaeger_summary"].values()
                    if isinstance(v, dict)) if during["jaeger_summary"] else 0
    span_drop = (pre_spans > 10) and (dur_spans < pre_spans * 0.5)

    return {
        "ebpf_latency_p95": latency_detected,
        "ebpf_error_rate":  error_detected,
        "ebpf_span_drop":   span_drop,
    }


def detect_loki(pre: dict, during: dict) -> dict:
    """
    Returns dict of signal_name → bool (detected).
    Detected if error log count during fault is > pre-fault count + 2σ,
    or if any UE failure / SCP routing error lines appear.
    """
    pre_errors = _loki_error_count(pre["loki"])
    dur_errors = _loki_error_count(during["loki"])

    # Simple threshold: more than 5 new error lines, or 2x increase
    error_increase = dur_errors > max(pre_errors + 5, pre_errors * 2 + 1)

    ue_failures = not during["loki"].get("ue_failures", pd.DataFrame()).empty
    scp_routing = not during["loki"].get("scp_routing", pd.DataFrame()).empty

    return {
        "loki_errors":      error_increase,
        "loki_ue_failures": ue_failures,
        "loki_scp_routing": scp_routing,
    }


def detect_nrf(pre: dict, during: dict) -> dict:
    """
    Detected if any NF type has fewer registered instances during fault
    than pre-fault (NF deregistered or unreachable).
    """
    pre_nrf = pre.get("nrf", {})
    dur_nrf = during.get("nrf", {})

    if not pre_nrf or not dur_nrf:
        return {"nrf_registration_drop": False}

    for nf_type, pre_count in pre_nrf.items():
        if not isinstance(pre_count, int):
            continue
        dur_count = dur_nrf.get(nf_type, pre_count)
        if isinstance(dur_count, int) and dur_count < pre_count:
            return {"nrf_registration_drop": True}

    return {"nrf_registration_drop": False}


# ---------------------------------------------------------------------------
# Aggregate verdict: detected by method
# ---------------------------------------------------------------------------

def aggregate_verdict(prom_signals: dict, ebpf_signals: dict,
                      loki_signals: dict, nrf_signals: dict) -> dict:
    """
    Collapse per-signal booleans into per-method detected flag.
    """
    prom_detected = any(prom_signals.values())
    ebpf_detected = any(ebpf_signals.values())
    loki_detected = any(loki_signals.values())
    nrf_detected = any(nrf_signals.values())

    return {
        "prometheus": prom_detected,
        "ebpf":       ebpf_detected,
        "loki":       loki_detected,
        "nrf":        nrf_detected,
        "any":        prom_detected or ebpf_detected or loki_detected or nrf_detected,
    }


# ---------------------------------------------------------------------------
# Run detection for all faults
# ---------------------------------------------------------------------------

def compute_all_verdicts() -> tuple:
    """
    Returns (verdicts_df, raw_signals_df).

    verdicts_df: fault × method boolean matrix
    raw_signals_df: fault × individual signal boolean matrix
    """
    verdict_rows = []
    signal_rows = []

    for fault in ALL_FAULTS:
        print(f"  [coverage] Processing {fault}...")
        data = load_fault(fault)
        pre = data["pre"]
        during = data["during"]

        prom_sigs = detect_prometheus(pre, during)
        ebpf_sigs = detect_ebpf(pre, during)
        loki_sigs = detect_loki(pre, during)
        nrf_sigs = detect_nrf(pre, during)

        verdict = aggregate_verdict(prom_sigs, ebpf_sigs, loki_sigs, nrf_sigs)
        verdict_rows.append({"fault": fault, **verdict})

        signal_rows.append({
            "fault": fault,
            **prom_sigs,
            **ebpf_sigs,
            **loki_sigs,
            **nrf_sigs,
        })

    verdicts_df = pd.DataFrame(verdict_rows).set_index("fault")
    signals_df = pd.DataFrame(signal_rows).set_index("fault")
    return verdicts_df, signals_df


# ---------------------------------------------------------------------------
# Figure: Coverage matrix heatmap
# ---------------------------------------------------------------------------

def plot_coverage_matrix(verdicts_df: pd.DataFrame):
    """
    Heatmap: faults (rows) × methods (columns), green=detected, red=missed.
    Faults grouped by category.
    """
    methods = ["prometheus", "ebpf", "loki", "nrf"]
    method_labels = ["Prometheus", "eBPF/Beyla", "Loki logs", "NRF API"]

    # Order faults by category
    ordered_faults = []
    category_boundaries = []
    for cat, faults in FAULT_CATEGORIES.items():
        for f in faults:
            if f in verdicts_df.index:
                ordered_faults.append(f)
        category_boundaries.append((cat, len(ordered_faults)))

    # Add any faults not in categories
    for f in ALL_FAULTS:
        if f not in ordered_faults and f in verdicts_df.index:
            ordered_faults.append(f)

    if not ordered_faults:
        print("  [coverage] No fault data available, skipping coverage matrix")
        return

    matrix = verdicts_df.loc[ordered_faults, methods].astype(float).values
    row_labels = [FAULT_LABELS.get(f, f) for f in ordered_faults]

    fig_height = max(6, len(ordered_faults) * 0.45)
    fig, ax = plt.subplots(figsize=(7, fig_height), dpi=FIGURE_DPI)

    # Custom colormap: red=0 (missed), green=1 (detected)
    cmap = matplotlib.colors.ListedColormap(["#FFCDD2", "#C8E6C9"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_labels, fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_yticks(range(len(ordered_faults)))
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE_TICK)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Cell annotations
    for i in range(len(ordered_faults)):
        for j in range(len(methods)):
            val = matrix[i, j]
            text = "✓" if val == 1 else "✗"
            color = "#1B5E20" if val == 1 else "#B71C1C"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=FONT_SIZE_TICK + 1, color=color, fontweight="bold")

    # Category separator lines
    prev_end = 0
    for cat, end in category_boundaries:
        if end > prev_end and end < len(ordered_faults):
            ax.axhline(end - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
        prev_end = end

    # Category labels on left margin
    prev_end = 0
    for cat, end in category_boundaries:
        mid = (prev_end + end - 1) / 2
        ax.text(-0.7, mid, cat, ha="right", va="center",
                fontsize=FONT_SIZE_TICK - 1, color="#555555",
                rotation=0, transform=ax.get_yaxis_transform())
        prev_end = end

    ax.set_title("Fault detection coverage: Prometheus vs. eBPF vs. Loki vs. NRF API",
                 fontsize=FONT_SIZE_TITLE, pad=20)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C8E6C9", edgecolor="gray", label="Detected"),
        Patch(facecolor="#FFCDD2", edgecolor="gray", label="Not detected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              bbox_to_anchor=(1.0, -0.05), fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    out = FIGURES_DIR / "fault_detection_coverage_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [coverage] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Detection rate summary bar chart
# ---------------------------------------------------------------------------

def plot_detection_rates(verdicts_df: pd.DataFrame):
    methods = ["prometheus", "ebpf", "loki", "nrf"]
    method_labels = ["Prometheus", "eBPF/Beyla", "Loki logs", "NRF API"]
    colors = ["#E6522C", "#00ADD8", "#9C27B0", "#607D8B"]

    n = len(verdicts_df)
    rates = [verdicts_df[m].sum() / n * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=FIGURE_DPI)
    bars = ax.bar(method_labels, rates, color=colors, alpha=0.85, width=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE_TICK, fontweight="bold")

    ax.set_ylabel("Detection rate (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"Fault detection rate by method ({n} faults)",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "fault_detection_rates.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [coverage] Saved {out}")


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

def save_tables(verdicts_df: pd.DataFrame, signals_df: pd.DataFrame):
    # Main coverage table with human-readable labels
    out_df = verdicts_df.copy()
    out_df.index = [FAULT_LABELS.get(f, f) for f in out_df.index]
    out_df.to_csv(TABLES_DIR / "fault_detection_coverage.csv")
    print(f"  [coverage] Saved {TABLES_DIR / 'fault_detection_coverage.csv'}")

    # Raw per-signal verdicts
    signals_out = signals_df.copy()
    signals_out.index = [FAULT_LABELS.get(f, f) for f in signals_out.index]
    signals_out.to_csv(TABLES_DIR / "fault_detection_verdicts.csv")
    print(f"  [coverage] Saved {TABLES_DIR / 'fault_detection_verdicts.csv'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("[coverage] Computing fault detection coverage...")
    verdicts_df, signals_df = compute_all_verdicts()
    save_tables(verdicts_df, signals_df)
    plot_coverage_matrix(verdicts_df)
    plot_detection_rates(verdicts_df)
    print("[coverage] Done.")
    return verdicts_df, signals_df


if __name__ == "__main__":
    run()
