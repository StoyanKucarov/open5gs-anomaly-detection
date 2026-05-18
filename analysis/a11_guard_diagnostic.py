"""11 - Flat-baseline-guard diagnostic.

For every scored z-score signal x every fault: compute the pre-phase baseline
(mean / sigma, pooled across NFs) and the three competing detection gates

    z_gate   = Z * sigma                (noise-relative)
    rel_gate = FLAT_REL * |mean|         (relative-size)
    floor    = sig.floor * FLOOR_MULT    (absolute, hand-tuned)

The binding gate is whichever is largest (that is the actual decision
boundary for that cell). This makes visible *why* Z is non-load-bearing for
quiet signals and the floor is the real lever (see a10 sensitivity).

Outputs:
  data/analysis/guard_diagnostic.csv     (per signal x fault: mean, sigma, gates, binding, ratio)
  data/analysis/plots/guard_diagnostic.png
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import analysis.common as c

GATES = ["z (Z·σ)", "rel (30%·μ)", "floor", "no data"]
GCOL = ["#4c72b0", "#dd8452", "#c44e52", "#dddddd"]


def _baseline(ctx, sig):
    """Pooled pre-phase mean/sigma across all NF rows for one signal."""
    pre = c._read_prom(ctx.root, "pre", sig.source)
    if pre is None:
        return None
    split = c._beyla_series_by_nf if sig.source.startswith("beyla_") else c._series_by_nf
    parts = [d["value"] for d in split(pre).values() if len(d)]
    if not parts:
        return None
    v = pd.concat(parts, ignore_index=True)
    v = pd.to_numeric(v, errors="coerce").dropna()
    if len(v) < 3:
        return None
    return float(v.mean()), float(v.std(ddof=0))


def run() -> pd.DataFrame:
    c.ensure_dirs()
    sigs = [s for s in c.SCORED_SIGNALS if s.kind == "zscore"]
    slugs = c.fault_slugs()
    rows, cat = [], np.full((len(sigs), len(slugs)), 3, dtype=int)  # default no-data

    for i, sig in enumerate(sigs):
        for j, slug in enumerate(slugs):
            ctx = c.load_ctx(slug)
            b = _baseline(ctx, sig)
            if b is None:
                rows.append({"signal": sig.key, "fault": slug, "baseline_mean": "",
                             "baseline_std": "", "z_gate": "", "rel_gate": "",
                             "floor": sig.floor, "ratio": sig.ratio, "binding": "no data"})
                continue
            m, sd = b
            zg = c.Z * sd
            rg = c.FLAT_REL * abs(m)
            fl = sig.floor * c.FLOOR_MULT
            gates = {"z (Z·σ)": zg, "rel (30%·μ)": rg, "floor": fl}
            binding = max(gates, key=gates.get)
            cat[i, j] = GATES.index(binding)
            rows.append({"signal": sig.key, "fault": slug,
                         "baseline_mean": round(m, 6), "baseline_std": round(sd, 6),
                         "z_gate": round(zg, 6), "rel_gate": round(rg, 6),
                         "floor": fl, "ratio": sig.ratio, "binding": binding})

    df = pd.DataFrame(rows)
    df.to_csv(c.OUT_DIR / "guard_diagnostic.csv", index=False)
    _plot(cat, sigs, slugs)
    fl_share = (cat == 2).sum() / max(1, (cat != 3).sum()) * 100
    print(f"[11] guard_diagnostic.csv  floor is the binding gate in "
          f"{fl_share:.0f}% of populated cells")
    return df


def _plot(cat, sigs, slugs):
    fig, ax = plt.subplots(figsize=(max(11, len(slugs) * 0.5),
                                    max(6, len(sigs) * 0.42)))
    cmap = ListedColormap(GCOL)
    im = ax.imshow(cat, aspect="auto", cmap=cmap,
                   norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N))
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels([f"{s.key}  (floor={s.floor:g}"
                        + (f", ratio={s.ratio:g})" if s.ratio else ")")
                        for s in sigs], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.5)
    cbar.ax.set_yticklabels(GATES, fontsize=8)
    ax.set_title("Flat-baseline-guard diagnostic — which gate is the binding "
                 "decision boundary per signal × fault\n"
                 "(orange/red = Z is NOT load-bearing here; "
                 "see guard_diagnostic.csv for the raw mean/σ/gate values)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "guard_diagnostic.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
