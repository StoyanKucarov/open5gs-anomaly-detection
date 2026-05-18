"""02 - Temporal manifestation: when does each layer first show a signal.

For every fault, time-to-first-signal (seconds after injection t0) per layer,
and which layer manifests first ("never" if a layer stays silent).

Outputs:
  data/analysis/temporal_layers.csv       (fault x {infra,orch,app} first-time + which)
  data/analysis/first_signal_per_fault.csv (fault -> earliest signal overall)
  data/analysis/plots/temporal_heatmap.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c

NEVER = np.inf


def run() -> pd.DataFrame:
    c.ensure_dirs()
    layer_of = {s.key: s.layer for s in c.SCORED_SIGNALS}
    rows, first_rows = [], []
    for slug in c.fault_slugs():
        _, res = c.get_detection(slug)
        per_layer = {L: (NEVER, None) for L in c.LAYERS}
        overall = (NEVER, None)
        for k, h in res.items():
            if k not in layer_of or not h.manifested or h.t_detect is None:
                continue
            L = layer_of[k]
            if h.t_detect < per_layer[L][0]:
                per_layer[L] = (h.t_detect, k)
            if h.t_detect < overall[0]:
                overall = (h.t_detect, k)
        row = {"fault": slug}
        for L in c.LAYERS:
            t, k = per_layer[L]
            row[f"{L}_t"] = "" if t is NEVER else round(t, 1)
            row[f"{L}_signal"] = k or ""
        ranked = [L for L in c.LAYERS if per_layer[L][0] is not NEVER]
        ranked.sort(key=lambda L: per_layer[L][0])
        row["first_layer"] = ranked[0] if ranked else "none"
        row["layer_order"] = " -> ".join(ranked) if ranked else "none"
        rows.append(row)
        first_rows.append({"fault": slug,
                           "first_signal": overall[1] or "none",
                           "first_layer": (layer_of.get(overall[1]) if overall[1] else "none"),
                           "t_detect_s": "" if overall[0] is NEVER else round(overall[0], 1),
                           "family": c.FAULTS[slug].family})
    tl = pd.DataFrame(rows).set_index("fault")
    tl.to_csv(c.OUT_DIR / "temporal_layers.csv")
    pd.DataFrame(first_rows).set_index("fault").to_csv(c.OUT_DIR / "first_signal_per_fault.csv")
    _heatmap(tl)
    print(f"[02] temporal_layers.csv  first-layer dist: "
          f"{tl['first_layer'].value_counts().to_dict()}")
    return tl


def _heatmap(tl: pd.DataFrame) -> None:
    M = []
    for L in c.LAYERS:
        M.append([np.nan if tl.loc[i, f"{L}_t"] == "" else float(tl.loc[i, f"{L}_t"])
                  for i in tl.index])
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(max(12, len(tl) * 0.45), 4))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(c.LAYERS)))
    ax.set_yticklabels(c.LAYERS)
    ax.set_xticks(range(len(tl)))
    ax.set_xticklabels(tl.index, rotation=90, fontsize=7)
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            v = M[y, x]
            ax.text(x, y, "—" if np.isnan(v) else f"{v:.0f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if (np.isnan(v) or v > np.nanmax(M) / 2) else "black")
    fig.colorbar(im, ax=ax, label="time to first signal (s after t0)")
    ax.set_title("Temporal manifestation per layer  (— = never)")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "temporal_heatmap.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
