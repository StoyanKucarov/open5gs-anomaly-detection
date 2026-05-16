"""
analysis/rq1c_scalability.py

RQ1c: How does telemetry method choice affect overhead and observability
coverage under increasing and bursty workloads?

Scenarios: 10, 50, 100, 200 UEs (steady) + 50, 100 UEs (bursty)

Produces:
  figures/rq1c_cpu_vs_ue_count.pdf
  figures/rq1c_mem_vs_ue_count.pdf
  figures/rq1c_bursty_vs_steady.pdf
  figures/rq1c_jaeger_spans_vs_ue_count.pdf
  tables/rq1c_scalability_summary.csv
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
    FIGURES_DIR, TABLES_DIR, PALETTE, SCALABILITY_SCENARIOS, SCALABILITY_NOTE,
    FIGURE_DPI, FIGURE_SIZE_SINGLE, FIGURE_SIZE_WIDE,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
)
from load_data import (
    load_all_scalability,
    mean_cpu_millicores, mean_memory_mib,
    monitoring_mean_cpu_millicores, monitoring_mean_memory_mib,
    beyla_mean_cpu_millicores, beyla_mean_memory_mib,
)


# ---------------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------------

def build_summary(all_data: dict) -> pd.DataFrame:
    rows = []
    for (ue, pat), data in all_data.items():
        prom = data["prometheus"]
        jaeger = data["jaeger_summary"]
        span_count = sum(v.get("span_count", 0) for v in jaeger.values()) if jaeger else 0
        p95_vals = [v["duration_us_p95"] for v in jaeger.values()
                    if "duration_us_p95" in v] if jaeger else []
        p95_mean = float(np.mean(p95_vals)) if p95_vals else np.nan

        rows.append({
            "ue_count": int(ue),
            "pattern": pat,
            "nf_cpu_m": mean_cpu_millicores(prom),
            "nf_mem_mib": mean_memory_mib(prom),
            "monitoring_cpu_m": monitoring_mean_cpu_millicores(prom),
            "monitoring_mem_mib": monitoring_mean_memory_mib(prom),
            "beyla_cpu_m": beyla_mean_cpu_millicores(prom),
            "beyla_mem_mib": beyla_mean_memory_mib(prom),
            "jaeger_span_count": span_count,
            "jaeger_p95_us": p95_mean,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["pattern", "ue_count"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Figure: CPU overhead vs. UE count (steady-state)
# ---------------------------------------------------------------------------

def plot_cpu_vs_ue_count(df: pd.DataFrame):
    steady = df[df["pattern"] == "steady"].sort_values("ue_count")
    if steady.empty:
        print("  [rq1c] No steady-state data, skipping CPU vs UE count plot")
        return

    x = steady["ue_count"].values
    nf_cpu = steady["nf_cpu_m"].fillna(0).values
    mon_cpu = steady["monitoring_cpu_m"].fillna(0).values
    beyla_cpu = steady["beyla_cpu_m"].fillna(0).values

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    ax.plot(x, nf_cpu, "o-", color="#4CAF50", linewidth=2, markersize=7, label="5G NFs")
    ax.plot(x, mon_cpu, "s-", color=PALETTE["prometheus"], linewidth=2, markersize=7,
            label="Prometheus monitoring")
    ax.plot(x, beyla_cpu, "^-", color=PALETTE["ebpf"], linewidth=2, markersize=7,
            label="Beyla eBPF agent")

    ax.set_xlabel("Number of UEs", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean CPU (millicores)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1c — CPU overhead vs. UE count (steady-state)", fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(xi) for xi in x], fontsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Note about 500 UE limit
    ax.text(0.98, 0.02, "Max 200 UEs (kind cluster limit)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_SIZE_TICK - 1, color="gray", style="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "rq1c_cpu_vs_ue_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1c] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Memory overhead vs. UE count (steady-state)
# ---------------------------------------------------------------------------

def plot_mem_vs_ue_count(df: pd.DataFrame):
    steady = df[df["pattern"] == "steady"].sort_values("ue_count")
    if steady.empty:
        print("  [rq1c] No steady-state data, skipping memory vs UE count plot")
        return

    x = steady["ue_count"].values
    nf_mem = steady["nf_mem_mib"].fillna(0).values
    mon_mem = steady["monitoring_mem_mib"].fillna(0).values
    beyla_mem = steady["beyla_mem_mib"].fillna(0).values

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    ax.plot(x, nf_mem, "o-", color="#4CAF50", linewidth=2, markersize=7, label="5G NFs")
    ax.plot(x, mon_mem, "s-", color=PALETTE["prometheus"], linewidth=2, markersize=7,
            label="Prometheus monitoring")
    ax.plot(x, beyla_mem, "^-", color=PALETTE["ebpf"], linewidth=2, markersize=7,
            label="Beyla eBPF agent")

    ax.set_xlabel("Number of UEs", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean memory (MiB)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("RQ1c — Memory overhead vs. UE count (steady-state)", fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(xi) for xi in x], fontsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    ax.text(0.98, 0.02, "Max 200 UEs (kind cluster limit)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_SIZE_TICK - 1, color="gray", style="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "rq1c_mem_vs_ue_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1c] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Bursty vs. steady comparison (grouped bars at 50 and 100 UEs)
# ---------------------------------------------------------------------------

def plot_bursty_vs_steady(df: pd.DataFrame):
    ue_counts = [50, 100]
    metrics = [
        ("nf_cpu_m", "NF CPU (m)"),
        ("monitoring_cpu_m", "Monitoring CPU (m)"),
        ("beyla_cpu_m", "Beyla CPU (m)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), dpi=FIGURE_DPI)

    for ax, (col, label) in zip(axes, metrics):
        x = np.arange(len(ue_counts))
        width = 0.35

        steady_vals = []
        bursty_vals = []
        for ue in ue_counts:
            s_row = df[(df["ue_count"] == ue) & (df["pattern"] == "steady")]
            b_row = df[(df["ue_count"] == ue) & (df["pattern"] == "bursty")]
            steady_vals.append(float(s_row[col].values[0]) if not s_row.empty else 0.0)
            bursty_vals.append(float(b_row[col].values[0]) if not b_row.empty else 0.0)

        ax.bar(x - width / 2, steady_vals, width, label="Steady", color=PALETTE["steady"], alpha=0.85)
        ax.bar(x + width / 2, bursty_vals, width, label="Bursty", color=PALETTE["bursty"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{ue} UEs" for ue in ue_counts], fontsize=FONT_SIZE_TICK)
        ax.set_ylabel(label, fontsize=FONT_SIZE_TICK)
        ax.set_title(label, fontsize=FONT_SIZE_LABEL)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("RQ1c — Bursty vs. steady-state overhead comparison",
                 fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    out = FIGURES_DIR / "rq1c_bursty_vs_steady.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1c] Saved {out}")


# ---------------------------------------------------------------------------
# Figure: Jaeger span count vs. UE count (observability coverage at scale)
# ---------------------------------------------------------------------------

def plot_jaeger_spans_vs_ue_count(df: pd.DataFrame):
    steady = df[df["pattern"] == "steady"].sort_values("ue_count")
    if steady.empty:
        print("  [rq1c] No steady-state data, skipping Jaeger spans plot")
        return

    x = steady["ue_count"].values
    spans = steady["jaeger_span_count"].fillna(0).values
    p95 = steady["jaeger_p95_us"].fillna(0).values / 1000  # µs → ms

    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)

    ax1.bar(x, spans, width=np.diff(np.append(x, x[-1] + 50)) * 0.6,
            color=PALETTE["ebpf"], alpha=0.7, label="Span count", align="center")
    ax1.set_xlabel("Number of UEs", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Total spans collected", fontsize=FONT_SIZE_LABEL, color=PALETTE["ebpf"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["ebpf"])

    ax2 = ax1.twinx()
    ax2.plot(x, p95, "D--", color="#FF5722", linewidth=2, markersize=7, label="p95 latency (ms)")
    ax2.set_ylabel("p95 request latency (ms)", fontsize=FONT_SIZE_LABEL, color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_SIZE_LEGEND, loc="upper left")

    ax1.set_title("RQ1c — eBPF observability coverage vs. UE count",
                  fontsize=FONT_SIZE_TITLE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(xi) for xi in x], fontsize=FONT_SIZE_TICK)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "rq1c_jaeger_spans_vs_ue_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [rq1c] Saved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary_table(df: pd.DataFrame):
    out = TABLES_DIR / "rq1c_scalability_summary.csv"
    df.to_csv(out, index=False, float_format="%.2f")
    # Also write the 200 UE note as a sidecar
    note_out = TABLES_DIR / "rq1c_scalability_note.txt"
    note_out.write_text(SCALABILITY_NOTE)
    print(f"  [rq1c] Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("[rq1c] Building scalability analysis...")
    all_data = load_all_scalability()
    df = build_summary(all_data)
    save_summary_table(df)
    plot_cpu_vs_ue_count(df)
    plot_mem_vs_ue_count(df)
    plot_bursty_vs_steady(df)
    plot_jaeger_spans_vs_ue_count(df)
    print("[rq1c] Done.")


if __name__ == "__main__":
    run()
