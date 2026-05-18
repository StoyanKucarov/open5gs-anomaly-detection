"""06 - Signal-modality necessity (Fu 2025 framing).

Is any single observability modality (metrics / logs / traces / k8s-events /
rtt) sufficient on its own? Per fault, record which modalities detect it, and
count faults that would be MISSED if only one modality were collected.

Outputs:
  data/analysis/modality_necessity.csv
  data/analysis/plots/modality_coverage.png
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c

# group raw modalities into the 5 observability sources
MOD = {"metric": "metrics", "log": "logs", "trace": "traces",
       "event": "k8s_events", "nrf": "metrics", "rtt": "rtt"}
SOURCES = ["metrics", "logs", "traces", "k8s_events", "rtt"]


def run() -> pd.DataFrame:
    c.ensure_dirs()
    rows = []
    for slug in c.fault_slugs():
        _, res = c.get_detection(slug)
        hit = {src: False for src in SOURCES}
        for s in c.SCORED_SIGNALS:
            if res[s.key].manifested:
                hit[MOD[s.modality]] = True
        rows.append({"fault": slug, "family": c.FAULTS[slug].family,
                     **{f"by_{s}": hit[s] for s in SOURCES},
                     "n_modalities": sum(hit.values())})
    df = pd.DataFrame(rows).set_index("fault")
    df.to_csv(c.OUT_DIR / "modality_necessity.csv")

    cov = {s: int(df[f"by_{s}"].sum()) for s in SOURCES}
    only = {s: int(((df[f"by_{s}"]) & (df["n_modalities"] == 1)).sum())
            for s in SOURCES}
    missed = {s: int((~df[f"by_{s}"]).sum()) for s in SOURCES}
    _plot(cov, missed, len(df))
    print(f"[06] modality_necessity.csv  coverage={cov}; "
          f"faults missed if only that modality={missed}")
    return df


def _plot(cov, missed, n):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(SOURCES))
    ax.bar([i - 0.2 for i in x], [cov[s] for s in SOURCES], 0.4,
           label="faults detected", color="tab:blue")
    ax.bar([i + 0.2 for i in x], [missed[s] for s in SOURCES], 0.4,
           label="faults MISSED if only this modality", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(SOURCES)
    ax.set_ylabel(f"# faults (of {n})")
    ax.set_title("Single-modality sufficiency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "modality_coverage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
