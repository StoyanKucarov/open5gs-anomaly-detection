"""09 - NF impact matrix: which network function is broken by which fault.

Reads the per-fault propagation chains (03) and builds an NF x fault matrix.
A cell holds the first-signal onset (s after t0) for that NF under that fault;
the chaos target NF is marked separately so collateral damage is visible.

  cell value: onset seconds  (-1 = NF showed no signal under that fault)
  is_target  : NF is the one the chaos selector hit

Outputs:
  data/analysis/nf_fault_matrix.csv      (onset; T-prefixed string if target)
  data/analysis/nf_fault_onset.csv       (numeric onset only, -1 = unaffected)
  data/analysis/nf_impact_summary.csv    (per-NF rollup)
  data/analysis/plots/nf_impact_heatmap.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c


def run() -> pd.DataFrame:
    c.ensure_dirs()
    slugs = c.fault_slugs()
    chains: dict[str, dict] = {}
    nfs: set[str] = set()
    for slug in slugs:
        data = json.loads((c.OUT_DIR / "propagation" / f"{slug}.json").read_text())
        per_nf = {e["nf"]: e for e in data["chain"]}
        chains[slug] = {"target": c.FAULTS[slug].target_nf, "nf": per_nf}
        nfs.update(per_nf)
        nfs.add(c.FAULTS[slug].target_nf)
    nf_order = sorted(nfs)

    disp, onset = {}, {}
    for slug in slugs:
        tgt = chains[slug]["target"]
        per_nf = chains[slug]["nf"]
        dcol, ocol = [], []
        for nf in nf_order:
            e = per_nf.get(nf)
            t = e["first_t"] if e else None
            ocol.append(-1.0 if t is None else float(t))
            if nf == tgt and e:
                dcol.append(f"T{t:g}")          # target, detected
            elif nf == tgt:
                dcol.append("T-")               # target, but no signal seen
            elif e:
                dcol.append(f"{t:g}")            # collateral
            else:
                dcol.append("")
        disp[slug] = dcol
        onset[slug] = ocol

    disp_df = pd.DataFrame(disp, index=nf_order)
    disp_df.index.name = "nf"
    disp_df.to_csv(c.OUT_DIR / "nf_fault_matrix.csv")

    onset_df = pd.DataFrame(onset, index=nf_order)
    onset_df.index.name = "nf"
    onset_df.to_csv(c.OUT_DIR / "nf_fault_onset.csv")

    # per-NF rollup
    rows = []
    for nf in nf_order:
        affected = [s for s in slugs if onset_df.loc[nf, s] >= 0]
        targeted = [s for s in slugs if c.FAULTS[s].target_nf == nf]
        collateral = [s for s in affected if c.FAULTS[s].target_nf != nf]
        ons = [onset_df.loc[nf, s] for s in affected]
        rows.append({
            "nf": nf,
            "n_faults_affecting": len(affected),
            "n_targeted": len(targeted),
            "n_collateral": len(collateral),
            "median_onset_s": round(float(np.median(ons)), 1) if ons else "",
            "targeted_by": ";".join(targeted),
            "collateral_from": ";".join(collateral),
        })
    summ = (pd.DataFrame(rows).set_index("nf")
            .sort_values("n_faults_affecting", ascending=False))
    summ.to_csv(c.OUT_DIR / "nf_impact_summary.csv")

    _plot(onset_df, chains, slugs, nf_order)
    most = summ.index[0]
    print(f"[09] nf_fault_matrix.csv  {len(nf_order)} NFs x {len(slugs)} faults; "
          f"most-impacted NF: {most} ({summ.loc[most,'n_faults_affecting']} faults)")
    return disp_df


def _plot(onset_df, chains, slugs, nf_order):
    M = onset_df[slugs].to_numpy(dtype=float)
    masked = np.ma.masked_less(M, 0)
    fig, ax = plt.subplots(figsize=(max(12, len(slugs) * 0.5),
                                    max(4, len(nf_order) * 0.45)))
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(nf_order)))
    ax.set_yticklabels(nf_order, fontsize=8)
    # ring the chaos-target cell
    for j, slug in enumerate(slugs):
        tgt = chains[slug]["target"]
        if tgt in nf_order:
            i = nf_order.index(tgt)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="tab:red", lw=1.6))
    fig.colorbar(im, ax=ax, label="first-signal onset (s after t0)")
    ax.set_title("NF impact — which NF breaks under which fault "
                 "(red box = chaos target; white = unaffected)")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "nf_impact_heatmap.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
