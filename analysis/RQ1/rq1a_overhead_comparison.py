"""
analysis/rq1a_overhead_comparison.py

RQ1a: How do Prometheus vs. eBPF compare in CPU and memory overhead
under equivalent workloads (50 UEs, steady-state)?

Produces:
  figures/rq1a_cpu_overhead_comparison.pdf
  figures/rq1a_memory_overhead_comparison.pdf
  tables/rq1a_overhead_summary.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FIGURES_DIR, TABLES_DIR, PALETTE,
    FIGURE_DPI, FIGURE_SIZE_WIDE,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)
from load_data import (
    load_baseline_top,
    load_prometheus_overhead,
    load_ebpf_overhead,
    mean_cpu_millicores, mean_memory_mib,
    beyla_mean_cpu_millicores, beyla_mean_memory_mib,
    monitoring_mean_cpu_millicores, monitoring_mean_memory_mib,
    per_pod_mean_cpu, per_pod_mean_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nf_pods_only(series: pd.Series) -> pd.Series:
    """Keep only Open5GS NF pods (drop ueransim, chaos, beyla, etc.)."""
    nf_prefixes = (
        "open5gs-amf", "open5gs-ausf", "open5gs-bsf", "open5gs-nrf",
        "open5gs-nssf", "open5gs-pcf", "open5gs-scp", "open5gs-smf",
        "open5gs-udm", "open5gs-udr", "open5gs-upf",
    )
    mask = series.index.str.startswith(nf_prefixes)
    return series[mask]


def _shorten_pod(name: str) -> str:
    """open5gs-amf-7d9f8b-xxx → amf"""
    parts = name.split("-")
    if parts[0] == "open5gs" and len(parts) >= 2:
        return parts[1].upper()
    return name


def _aggregate_by_nf(series: pd.Series) -> pd.Series:
    """Sum per-pod values by NF type (e.g. all amf-* pods → AMF)."""
    nf_series = _nf_pods_only(series)
    nf_series.index = nf_series.index.map(_shorten_pod)
    return nf_series.groupby(level=0).sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------------

def build_summary() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per condition:
      condition, nf_cpu_m, nf_mem_mib, monitoring_cpu_m, monitoring_mem_mib,
      beyla_cpu_m, beyla_mem_mib
    """
    rows = []

    # --- Baseline (no telemetry) ---
    bl = load_baseline_top("steady")
    pods_df = bl["pods"]
    if not pods_df.empty:
        nf_cpu = pods_df.groupby("pod")["cpu_m"].mean()
        nf_cpu = _nf_pods_only(nf_cpu).sum()
        nf_mem = pods_df.groupby("pod")["mem_mi"].mean()
        nf_mem = _nf_pods_only(nf_mem).sum()
    else:
        nf_cpu = nf_mem = np.nan
    rows.append({
        "condition": "Baseline\n(no telemetry)",
        "nf_cpu_m": nf_cpu,
        "nf_mem_mib": nf_mem,
        "monitoring_cpu_m": 0.0,
        "monitoring_mem_mib": 0.0,
        "beyla_cpu_m": 0.0,
        "beyla_mem_mib": 0.0,
    })

    # --- Prometheus at default interval (5s) ---
    prom_data = load_prometheus_overhead("5s")
    prom = prom_data["prometheus"]
    rows.append({
        "condition": "Prometheus\n(5s interval)",
        "nf_cpu_m": mean_cpu_millicores(prom),
        "nf_mem_mib": mean_memory_mib(prom),
        "monitoring_cpu_m": monitoring_mean_cpu_millicores(prom),
        "monitoring_mem_mib": monitoring_mean_memory_mib(prom),
        "beyla_cpu_m": 0.0,
        "beyla_mem_mib": 0.0,
    })

    # --- eBPF/Beyla at 100% sampling ---
    ebpf_data = load_ebpf_overhead("100pct")
    ebpf = ebpf_data["prometheus"]
    rows.append({
        "condition": "eBPF/Beyla\n(100% sampling)",
        "nf_cpu_m": mean_cpu_millicores(ebpf),
        "nf_mem_mib": mean_memory_mib(ebpf),
        "monitoring_cpu_m": monitoring_mean_cpu_millicores(ebpf),
        "monitoring_mem_mib": monitoring_mean_memory_mib(ebpf),
        "beyla_cpu_m": beyla_mean_cpu_millicores(ebpf),
        "beyla_mem_mib": beyla_mean_memory_mib(ebpf),
    })

    # --- Both stacks active (eBPF 100% + Prometheus 5s) ---
    # Use the eBPF 100% dataset which has both stacks running
    rows.append({
        "condition": "Both stacks\n(Prom 5s + eBPF 100%)",
        "nf_cpu_m": mean_cpu_millicores(ebpf),
        "nf_mem_mib": mean_memory_mib(ebpf),
        "monitoring_cpu_m": monitoring_mean_cpu_millicores(ebpf),
        "monitoring_mem_mib": monitoring_mean_memory_mib(ebpf),
        "beyla_cpu_m": beyla_mean_cpu_millicores(ebpf),
        "beyla_mem_mib": beyla_mean_memory_mib(ebpf),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: CPU overhead comparison (stacked bar)
# ---------------------------------------------------------------------------

def plot_cpu_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)

    x = np.arange(len(df))
    width = 0.55

    nf_cpu = df["nf_cpu_m"].fillna(0).values
    mon_cpu = df["monitoring_cpu_m"].fillna(0).values
    beyla_cpu = df["beyla_cpu_m"].fillna(0).values

    bars_nf = ax.bar(x, nf_cpu, width, label="5G NFs (open5gs)", color="#4CAF50", alpha=0.85)
    bars_mon = ax.bar(x, mon_cpu, width, bottom=nf_cpu,
                      label="Monitoring stack (Prometheus/Grafana)", color=PALETTE["prometheus"], alpha=0.85)
    bars_beyla = ax.bar(x, beyla_cpu, width, bottom=nf_cpu + mon_cpu,
                        label="Beyla eBPF agent", color=PALETTE["ebpf"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Mean CPU usage (millicores)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1a — CPU overhead: Prometheus vs. eBPF (50 UEs, steady-state)",
                 fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate total on top of each bar
    for i, (nf, mon, bey) in enumerate(zip(nf_cpu, mon_cpu, beyla_cpu)):
        total = nf + mon + bey
        if not np.isnan(total) and total > 0:
            ax.text(i, total + total * 0.02, f"{total:.0f}m",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICK - 1, fontweight="bold")

    fig.tight_layout()
    out = FIGURES_DIR / "rq1a_cpu_overhead_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1a] Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Memory overhead comparison (stacked bar)
# ---------------------------------------------------------------------------

def plot_memory_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)

    x = np.arange(len(df))
    width = 0.55

    nf_mem = df["nf_mem_mib"].fillna(0).values
    mon_mem = df["monitoring_mem_mib"].fillna(0).values
    beyla_mem = df["beyla_mem_mib"].fillna(0).values

    ax.bar(x, nf_mem, width, label="5G NFs (open5gs)", color="#4CAF50", alpha=0.85)
    ax.bar(x, mon_mem, width, bottom=nf_mem,
           label="Monitoring stack", color=PALETTE["prometheus"], alpha=0.85)
    ax.bar(x, beyla_mem, width, bottom=nf_mem + mon_mem,
           label="Beyla eBPF agent", color=PALETTE["ebpf"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Mean memory usage (MiB)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1a — Memory overhead: Prometheus vs. eBPF (50 UEs, steady-state)",
                 fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for i, (nf, mon, bey) in enumerate(zip(nf_mem, mon_mem, beyla_mem)):
        total = nf + mon + bey
        if not np.isnan(total) and total > 0:
            ax.text(i, total + total * 0.02, f"{total:.0f} MiB",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICK - 1, fontweight="bold")

    fig.tight_layout()
    out = FIGURES_DIR / "rq1a_memory_overhead_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1a] Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Per-NF CPU breakdown (grouped bars, Prometheus vs eBPF)
# ---------------------------------------------------------------------------

def plot_per_nf_cpu():
    prom_data = load_prometheus_overhead("5s")
    ebpf_data = load_ebpf_overhead("100pct")

    prom_per_nf = _aggregate_by_nf(per_pod_mean_cpu(prom_data["prometheus"]))
    ebpf_per_nf = _aggregate_by_nf(per_pod_mean_cpu(ebpf_data["prometheus"]))

    all_nfs = sorted(set(prom_per_nf.index) | set(ebpf_per_nf.index))
    if not all_nfs:
        print("  [rq1a] No per-NF CPU data available, skipping per-NF plot")
        return

    prom_vals = [prom_per_nf.get(nf, 0.0) for nf in all_nfs]
    ebpf_vals = [ebpf_per_nf.get(nf, 0.0) for nf in all_nfs]

    x = np.arange(len(all_nfs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_nfs) * 0.9), 4), dpi=FIGURE_DPI)
    ax.bar(x - width / 2, prom_vals, width, label="Prometheus (5s)", color=PALETTE["prometheus"], alpha=0.85)
    ax.bar(x + width / 2, ebpf_vals, width, label="eBPF/Beyla (100%)", color=PALETTE["ebpf"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_nfs, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Mean CPU (millicores)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1a — Per-NF CPU overhead: Prometheus vs. eBPF (50 UEs, steady-state)",
                 fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1a_per_nf_cpu.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1a] Saved {out}")


# ---------------------------------------------------------------------------
# Table: overhead summary CSV
# ---------------------------------------------------------------------------

def save_summary_table(df: pd.DataFrame):
    out = TABLES_DIR / "rq1a_overhead_summary.csv"
    df_out = df.copy()
    df_out["condition"] = df_out["condition"].str.replace("\n", " ")
    df_out.to_csv(out, index=False, float_format="%.2f")
    print(f"  [rq1a] Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("[rq1a] Building overhead comparison...")
    df = build_summary()
    save_summary_table(df)
    plot_cpu_comparison(df)
    plot_memory_comparison(df)
    plot_per_nf_cpu()
    print("[rq1a] Done.")


if __name__ == "__main__":
    run()
