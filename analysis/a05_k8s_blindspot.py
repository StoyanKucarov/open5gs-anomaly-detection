"""05 - Kubernetes blind-spot analysis (Flora 2022 framing).

Which faults does the orchestration layer (K8s events + pod restart/ready)
actually catch, and which are completely invisible to it while still being
visible at the application or infrastructure layer?

Outputs:
  data/analysis/k8s_blindspot.csv
  data/analysis/plots/k8s_blindspot.png
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c

ORCH_KEYS = [s.key for s in c.SCORED_SIGNALS if s.layer == "orchestration"]
OTHER_KEYS = [s.key for s in c.SCORED_SIGNALS if s.layer != "orchestration"]


def run() -> pd.DataFrame:
    c.ensure_dirs()
    rows = []
    for slug in c.fault_slugs():
        _, res = c.get_detection(slug)
        orch = {k for k in ORCH_KEYS if res[k].manifested}
        other = {k for k in OTHER_KEYS if res[k].manifested}
        caught = bool(orch)
        rows.append({
            "fault": slug,
            "family": c.FAULTS[slug].family,
            "k8s_caught": caught,
            "k8s_signals": ";".join(sorted(orch)),
            "visible_elsewhere": bool(other),
            "blindspot": (not caught) and bool(other),
            "n_non_orch_signals": len(other),
        })
    df = pd.DataFrame(rows).set_index("fault")
    df.to_csv(c.OUT_DIR / "k8s_blindspot.csv")
    n = len(df)
    caught = int(df["k8s_caught"].sum())
    blind = int(df["blindspot"].sum())
    _plot(df)
    print(f"[05] k8s_blindspot.csv  K8s caught {caught}/{n}; "
          f"{blind} faults invisible to orchestration but visible elsewhere")
    return df


def _plot(df: pd.DataFrame) -> None:
    order = df.sort_values(["k8s_caught", "n_non_orch_signals"])
    colors = ["tab:green" if v else "tab:red" for v in order["k8s_caught"]]
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 4))
    ax.bar(range(len(order)), order["n_non_orch_signals"], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order.index, rotation=90, fontsize=7)
    ax.set_ylabel("# non-orchestration signals")
    ax.set_title("K8s blind spot — green: caught by K8s, red: K8s blind "
                 "(bar height = signal evidence at other layers)")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "k8s_blindspot.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
