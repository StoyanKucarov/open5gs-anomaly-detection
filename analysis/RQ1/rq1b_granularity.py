"""
analysis/rq1b_granularity.py

RQ1b: What are the trade-offs between collection frequency/granularity,
fault detection signal quality, and monitoring cost?
At what granularity do diminishing returns appear?

Prometheus: 3 scrape intervals (1s, 5s, 15s)
eBPF/Beyla: 3 sampling rates (100%, 50%, 10%)

Produces:
  figures/rq1b_prom_cpu_vs_interval.pdf
  figures/rq1b_prom_mem_vs_interval.pdf
  figures/rq1b_ebpf_cpu_vs_sampling.pdf
  figures/rq1b_ebpf_mem_vs_sampling.pdf
  figures/rq1b_prom_self_metrics.pdf
  figures/rq1b_beyla_latency_vs_sampling.pdf
  tables/rq1b_granularity_summary.csv
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
    FIGURES_DIR, TABLES_DIR, PALETTE,
    FIGURE_DPI, FIGURE_SIZE_SINGLE, FIGURE_SIZE_WIDE,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    PROM_INTERVALS, EBPF_SAMPLING_RATES,
)
from load_data import (
    load_all_prometheus_overhead,
    load_all_ebpf_overhead,
    mean_cpu_millicores, mean_memory_mib,
    monitoring_mean_cpu_millicores, monitoring_mean_memory_mib,
    beyla_mean_cpu_millicores, beyla_mean_memory_mib,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interval_to_seconds(iv: str) -> int:
    return int(iv.rstrip("s"))


def _sampling_to_pct(sr: str) -> int:
    return int(sr.rstrip("pct"))


def _jaeger_p95(jaeger_summary: dict) -> float:
    """Mean p95 latency (µs) across all services in a Jaeger summary."""
    if not jaeger_summary:
        return np.nan
    p95s = [v["duration_us_p95"] for v in jaeger_summary.values() if "duration_us_p95" in v]
    return float(np.mean(p95s)) if p95s else np.nan


def _jaeger_span_count(jaeger_summary: dict) -> int:
    if not jaeger_summary:
        return 0
    return sum(v.get("span_count", 0) for v in jaeger_summary.values())


def _prom_self_mean(self_metrics: dict, key: str) -> float:
    df = self_metrics.get(key, pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    return float(pd.to_numeric(df["value"], errors="coerce").mean())


# ---------------------------------------------------------------------------
# Build summary tables
# ---------------------------------------------------------------------------

def build_prom_table(all_prom: dict) -> pd.DataFrame:
    rows = []
    for iv in PROM_INTERVALS:
        data = all_prom[iv]
        prom = data["prometheus"]
        self_m = data["self"]
        rows.append({
            "interval": iv,
            "interval_s": _interval_to_seconds(iv),
            "nf_cpu_m": mean_cpu_millicores(prom),
            "nf_mem_mib": mean_memory_mib(prom),
            "monitoring_cpu_m": monitoring_mean_cpu_millicores(prom),
            "monitoring_mem_mib": monitoring_mean_memory_mib(prom),
            "prom_head_chunks": _prom_self_mean(self_m, "head_chunks"),
            "prom_active_appenders": _prom_self_mean(self_m, "active_appenders"),
            "prom_wal_writes": _prom_self_mean(self_m, "wal_writes"),
        })
    return pd.DataFrame(rows)


def build_ebpf_table(all_ebpf: dict) -> pd.DataFrame:
    rows = []
    for sr in EBPF_SAMPLING_RATES:
        data = all_ebpf[sr]
        prom = data["prometheus"]
        rows.append({
            "sampling_rate": sr,
            "sampling_pct": _sampling_to_pct(sr),
            "nf_cpu_m": mean_cpu_millicores(prom),
            "nf_mem_mib": mean_memory_mib(prom),
            "beyla_cpu_m": beyla_mean_cpu_millicores(prom),
            "beyla_mem_mib": beyla_mean_memory_mib(prom),
            "jaeger_p95_us": _jaeger_p95(data["jaeger_summary"]),
            "jaeger_span_count": _jaeger_span_count(data["jaeger_summary"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures: Prometheus overhead vs. scrape interval
# ---------------------------------------------------------------------------

def plot_prom_cpu_vs_interval(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    x = df["interval_s"].values
    mon_cpu = df["monitoring_cpu_m"].fillna(0).values

    ax.plot(x, mon_cpu, "o-", color=PALETTE["prometheus"], linewidth=2, markersize=7,
            label="Monitoring stack CPU")
    ax.fill_between(x, 0, mon_cpu, color=PALETTE["prometheus"], alpha=0.15)

    for xi, yi in zip(x, mon_cpu):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.0f}m", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=FONT_SIZE_TICK)

    ax.set_xlabel("Scrape interval (seconds)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean CPU (millicores)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1b — Prometheus CPU overhead vs. scrape interval",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi}s" for xi in x], fontsize=FONT_SIZE_TICK)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_prom_cpu_vs_interval.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


def plot_prom_mem_vs_interval(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    x = df["interval_s"].values
    mon_mem = df["monitoring_mem_mib"].fillna(0).values

    ax.plot(x, mon_mem, "s-", color=PALETTE["prometheus"], linewidth=2, markersize=7,
            label="Monitoring stack memory")
    ax.fill_between(x, 0, mon_mem, color=PALETTE["prometheus"], alpha=0.15)

    for xi, yi in zip(x, mon_mem):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.0f} MiB", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=FONT_SIZE_TICK)

    ax.set_xlabel("Scrape interval (seconds)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean memory (MiB)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1b — Prometheus memory overhead vs. scrape interval",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi}s" for xi in x], fontsize=FONT_SIZE_TICK)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_prom_mem_vs_interval.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


def plot_prom_self_metrics(df: pd.DataFrame):
    """Three-panel plot: head chunks, active appenders, WAL writes vs. interval."""
    metrics = [
        ("prom_head_chunks", "Head chunks", "count"),
        ("prom_active_appenders", "Active appenders", "count"),
        ("prom_wal_writes", "WAL writes/s", "rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=FIGURE_DPI)

    x = df["interval_s"].values
    for ax, (col, label, unit) in zip(axes, metrics):
        y = df[col].fillna(0).values
        ax.bar([f"{xi}s" for xi in x], y, color=PALETTE["prometheus"], alpha=0.8)
        ax.set_title(label, fontsize=FONT_SIZE_LABEL)
        ax.set_xlabel("Scrape interval", fontsize=FONT_SIZE_TICK)
        ax.set_ylabel(unit, fontsize=FONT_SIZE_TICK)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("RQ1b — Prometheus self-metrics vs. scrape interval",
                 fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_prom_self_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


# ---------------------------------------------------------------------------
# Figures: eBPF overhead vs. sampling rate
# ---------------------------------------------------------------------------

def plot_ebpf_cpu_vs_sampling(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    x = df["sampling_pct"].values
    beyla_cpu = df["beyla_cpu_m"].fillna(0).values

    ax.plot(x, beyla_cpu, "o-", color=PALETTE["ebpf"], linewidth=2, markersize=7,
            label="Beyla eBPF agent CPU")
    ax.fill_between(x, 0, beyla_cpu, color=PALETTE["ebpf"], alpha=0.15)

    for xi, yi in zip(x, beyla_cpu):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.0f}m", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=FONT_SIZE_TICK)

    ax.set_xlabel("Trace sampling rate (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean CPU (millicores)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1b — Beyla eBPF CPU overhead vs. sampling rate",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi}%" for xi in x], fontsize=FONT_SIZE_TICK)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_ebpf_cpu_vs_sampling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


def plot_ebpf_mem_vs_sampling(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    x = df["sampling_pct"].values
    beyla_mem = df["beyla_mem_mib"].fillna(0).values

    ax.plot(x, beyla_mem, "s-", color=PALETTE["ebpf"], linewidth=2, markersize=7,
            label="Beyla eBPF agent memory")
    ax.fill_between(x, 0, beyla_mem, color=PALETTE["ebpf"], alpha=0.15)

    for xi, yi in zip(x, beyla_mem):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.0f} MiB", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=FONT_SIZE_TICK)

    ax.set_xlabel("Trace sampling rate (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean memory (MiB)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1b — Beyla eBPF memory overhead vs. sampling rate",
                 fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi}%" for xi in x], fontsize=FONT_SIZE_TICK)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_ebpf_mem_vs_sampling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


def plot_beyla_latency_vs_sampling(df: pd.DataFrame):
    """Span count and p95 latency vs. sampling rate — shows observability value."""
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    x = df["sampling_pct"].values
    spans = df["jaeger_span_count"].fillna(0).values
    p95 = df["jaeger_p95_us"].fillna(0).values / 1000  # µs → ms

    color_spans = PALETTE["ebpf"]
    color_p95 = "#FF5722"

    ax1.bar([f"{xi}%" for xi in x], spans, color=color_spans, alpha=0.7, label="Span count")
    ax1.set_xlabel("Trace sampling rate (%)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Total spans collected", fontsize=FONT_SIZE_LABEL, color=color_spans)
    ax1.tick_params(axis="y", labelcolor=color_spans)

    ax2 = ax1.twinx()
    ax2.plot([f"{xi}%" for xi in x], p95, "D--", color=color_p95, linewidth=2,
             markersize=7, label="p95 latency (ms)")
    ax2.set_ylabel("p95 request latency (ms)", fontsize=FONT_SIZE_LABEL, color=color_p95)
    ax2.tick_params(axis="y", labelcolor=color_p95)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_SIZE_LEGEND, loc="upper left")

    ax1.set_title("RQ1b — Beyla observability value vs. sampling rate",
                  fontsize=FONT_SIZE_TITLE)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_beyla_latency_vs_sampling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


# ---------------------------------------------------------------------------
# Combined cost-vs-value figure (dual axis: overhead + span count)
# ---------------------------------------------------------------------------

def plot_combined_cost_value(prom_df: pd.DataFrame, ebpf_df: pd.DataFrame):
    """
    Side-by-side: Prometheus overhead vs. interval | eBPF overhead vs. sampling rate.
    Shows the 'diminishing returns' point for each method.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=FIGURE_DPI)

    # --- Prometheus panel ---
    x_p = prom_df["interval_s"].values
    cpu_p = prom_df["monitoring_cpu_m"].fillna(0).values
    ax1.plot(x_p, cpu_p, "o-", color=PALETTE["prometheus"], linewidth=2, markersize=8)
    ax1.fill_between(x_p, 0, cpu_p, color=PALETTE["prometheus"], alpha=0.15)
    ax1.set_xticks(x_p)
    ax1.set_xticklabels([f"{xi}s" for xi in x_p], fontsize=FONT_SIZE_TICK)
    ax1.set_xlabel("Scrape interval", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Monitoring CPU (millicores)", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("Prometheus: overhead vs. granularity", fontsize=FONT_SIZE_LABEL)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # Annotate % reduction from 1s to 15s
    if len(cpu_p) >= 2 and cpu_p[0] > 0:
        reduction = (cpu_p[0] - cpu_p[-1]) / cpu_p[0] * 100
        ax1.annotate(f"−{reduction:.0f}% CPU\n(1s→15s)",
                     xy=(x_p[-1], cpu_p[-1]),
                     xytext=(x_p[-1] - 3, cpu_p[-1] + cpu_p[0] * 0.15),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     fontsize=FONT_SIZE_TICK, color="gray")

    # --- eBPF panel ---
    x_e = ebpf_df["sampling_pct"].values
    cpu_e = ebpf_df["beyla_cpu_m"].fillna(0).values
    spans_e = ebpf_df["jaeger_span_count"].fillna(0).values

    ax2_cpu = ax2
    ax2_spans = ax2.twinx()

    ax2_cpu.plot(x_e, cpu_e, "o-", color=PALETTE["ebpf"], linewidth=2, markersize=8,
                 label="Beyla CPU")
    ax2_cpu.fill_between(x_e, 0, cpu_e, color=PALETTE["ebpf"], alpha=0.15)
    ax2_spans.plot(x_e, spans_e, "s--", color="#FF9800", linewidth=1.5, markersize=6,
                   label="Span count")

    ax2_cpu.set_xticks(x_e)
    ax2_cpu.set_xticklabels([f"{xi}%" for xi in x_e], fontsize=FONT_SIZE_TICK)
    ax2_cpu.set_xlabel("Sampling rate", fontsize=FONT_SIZE_LABEL)
    ax2_cpu.set_ylabel("Beyla CPU (millicores)", fontsize=FONT_SIZE_LABEL, color=PALETTE["ebpf"])
    ax2_spans.set_ylabel("Spans collected", fontsize=FONT_SIZE_LABEL, color="#FF9800")
    ax2_cpu.set_title("eBPF/Beyla: overhead vs. sampling rate", fontsize=FONT_SIZE_LABEL)
    ax2_cpu.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax2_cpu.set_axisbelow(True)

    lines1, labels1 = ax2_cpu.get_legend_handles_labels()
    lines2, labels2 = ax2_spans.get_legend_handles_labels()
    ax2_cpu.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_SIZE_LEGEND)

    fig.suptitle("RQ1b — Overhead vs. granularity: cost-value trade-off",
                 fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    out = FIGURES_DIR / "rq1b_cost_value_tradeoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1b] Saved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary_table(prom_df: pd.DataFrame, ebpf_df: pd.DataFrame):
    prom_out = prom_df.copy()
    prom_out.insert(0, "method", "prometheus")
    prom_out = prom_out.rename(columns={"interval": "condition", "interval_s": "condition_numeric"})

    ebpf_out = ebpf_df.copy()
    ebpf_out.insert(0, "method", "ebpf")
    ebpf_out = ebpf_out.rename(columns={"sampling_rate": "condition", "sampling_pct": "condition_numeric"})

    combined = pd.concat([prom_out, ebpf_out], ignore_index=True)
    out = TABLES_DIR / "rq1b_granularity_summary.csv"
    combined.to_csv(out, index=False, float_format="%.2f")
    print(f"  [rq1b] Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("[rq1b] Building granularity analysis...")
    all_prom = load_all_prometheus_overhead()
    all_ebpf = load_all_ebpf_overhead()

    prom_df = build_prom_table(all_prom)
    ebpf_df = build_ebpf_table(all_ebpf)

    save_summary_table(prom_df, ebpf_df)

    plot_prom_cpu_vs_interval(prom_df)
    plot_prom_mem_vs_interval(prom_df)
    plot_prom_self_metrics(prom_df)
    plot_ebpf_cpu_vs_sampling(ebpf_df)
    plot_ebpf_mem_vs_sampling(ebpf_df)
    plot_beyla_latency_vs_sampling(ebpf_df)
    plot_combined_cost_value(prom_df, ebpf_df)

    print("[rq1b] Done.")


if __name__ == "__main__":
    run()
