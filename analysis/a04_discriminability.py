"""04 - Signal discriminability & fault fingerprinting (RQ sub-q2 / Fu 2025).

  * per-signal specificity: entropy of the fault-family distribution among
    faults where the signal fires (low entropy = points at few families =
    discriminative); plus prevalence.
  * Jaccard co-occurrence matrix between signals (redundancy).
  * per-fault fingerprint (the scored signal set) and nearest-confusable fault.

Outputs:
  data/analysis/discriminability.csv
  data/analysis/signal_correlation.csv
  data/analysis/fingerprints.csv
  data/analysis/plots/confusion.png
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c


def run() -> None:
    c.ensure_dirs()
    atlas = pd.read_csv(c.OUT_DIR / "fault_atlas.csv", index_col=0)
    fam = {s: c.FAULTS[s].family for s in atlas.index}
    fams = sorted(set(fam.values()))

    # ---- discriminability ----
    rows = []
    for k in atlas.columns:
        firing = atlas.index[atlas[k] == 1]
        n = len(firing)
        if n == 0:
            rows.append({"signal": k, "prevalence": 0, "n_families": 0,
                         "entropy": float("nan"), "specificity": float("nan")})
            continue
        dist = pd.Series([fam[s] for s in firing]).value_counts(normalize=True)
        ent = -sum(p * math.log2(p) for p in dist)
        max_ent = math.log2(len(fams))
        rows.append({
            "signal": k, "prevalence": n,
            "n_families": dist.size,
            "entropy": round(ent, 3),
            "specificity": round(1 - ent / max_ent, 3) if max_ent else 1.0,
        })
    disc = pd.DataFrame(rows).set_index("signal").sort_values(
        "specificity", ascending=False, na_position="last")
    disc.to_csv(c.OUT_DIR / "discriminability.csv")

    # ---- Jaccard co-occurrence ----
    keys = list(atlas.columns)
    J = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            ua = set(atlas.index[atlas[a] == 1])
            ub = set(atlas.index[atlas[b] == 1])
            J[i, j] = len(ua & ub) / len(ua | ub) if (ua | ub) else 0.0
    pd.DataFrame(J, index=keys, columns=keys).to_csv(
        c.OUT_DIR / "signal_correlation.csv")

    # ---- fingerprints + confusability ----
    sigsets = {s: frozenset(atlas.columns[atlas.loc[s] == 1]) for s in atlas.index}
    frows = []
    n = len(atlas.index)
    C = np.zeros((n, n))
    idx = list(atlas.index)
    for i, s in enumerate(idx):
        best, bestj = -1.0, None
        for j, o in enumerate(idx):
            if s == o:
                continue
            u = sigsets[s] | sigsets[o]
            sim = len(sigsets[s] & sigsets[o]) / len(u) if u else 0.0
            C[i, j] = sim
            if sim > best:
                best, bestj = sim, o
        frows.append({"fault": s, "n_signals": len(sigsets[s]),
                      "fingerprint": ";".join(sorted(sigsets[s])),
                      "nearest_fault": bestj,
                      "nearest_jaccard": round(best, 3),
                      "unique": best < 1.0})
    pd.DataFrame(frows).set_index("fault").to_csv(c.OUT_DIR / "fingerprints.csv")

    _plot(C, idx)
    uniq = sum(1 for r in frows if r["unique"])
    print(f"[04] discriminability.csv  {uniq}/{n} faults have a unique fingerprint; "
          f"top signal: {disc.index[0]}")


def _plot(C, idx):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, rotation=90, fontsize=6)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx, fontsize=6)
    fig.colorbar(im, ax=ax, label="Jaccard similarity of signal sets")
    ax.set_title("Fault-pair confusability (1.0 = indistinguishable)")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "confusion.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
