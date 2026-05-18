"""01 - Fault atlas: binary fault x signal matrix, grouped by layer.

Outputs:
  data/analysis/fault_atlas.csv          (22 x scored-signals, 0/1)
  data/analysis/fault_atlas_by_layer.csv (per fault: signals fired per layer)
  data/analysis/fault_atlas_summary.txt  (per-family profiles, dead/always-on)
  data/analysis/plots/atlas_heatmap.png
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c


def run() -> pd.DataFrame:
    c.ensure_dirs()
    slugs = c.fault_slugs()
    cols = [s.key for s in c.SCORED_SIGNALS]
    rows = []
    for slug in slugs:
        _, res = c.get_detection(slug)
        rows.append({"fault": slug, **{k: int(res[k].manifested) for k in cols}})
    atlas = pd.DataFrame(rows).set_index("fault")
    atlas.to_csv(c.OUT_DIR / "fault_atlas.csv")

    # per-layer counts
    layer_of = {s.key: s.layer for s in c.SCORED_SIGNALS}
    lr = []
    for slug in slugs:
        row = {"fault": slug}
        for L in c.LAYERS:
            keys = [k for k in cols if layer_of[k] == L]
            fired = [k for k in keys if atlas.loc[slug, k] == 1]
            row[f"{L}_n"] = len(fired)
            row[f"{L}_signals"] = ";".join(fired)
        lr.append(row)
    pd.DataFrame(lr).set_index("fault").to_csv(c.OUT_DIR / "fault_atlas_by_layer.csv")

    # summary text
    lines = ["FAULT ATLAS SUMMARY", "=" * 60, ""]
    fam = {}
    for slug in slugs:
        fam.setdefault(c.FAULTS[slug].family, []).append(slug)
    for f, members in sorted(fam.items()):
        lines.append(f"\n[{f}]  ({len(members)} faults)")
        sub = atlas.loc[members]
        for k in cols:
            n = int(sub[k].sum())
            if n:
                lines.append(f"  {k:<24} {n}/{len(members)}")
    dead = [k for k in cols if atlas[k].sum() == 0]
    always = [k for k in cols if atlas[k].sum() == len(slugs)]
    lines += ["", f"NEVER FIRED ({len(dead)}): " + ", ".join(dead),
              f"ALWAYS FIRED ({len(always)}): " + ", ".join(always),
              "", "Caveat metrics excluded from scoring (EXTENSIONS.md 10.10): "
              + ", ".join(s.key for s in c.SIGNALS if s.caveat)]
    (c.OUT_DIR / "fault_atlas_summary.txt").write_text("\n".join(lines))

    _heatmap(atlas, layer_of)
    print(f"[01] fault_atlas.csv  {atlas.shape[0]} faults x {atlas.shape[1]} signals")
    return atlas


def _heatmap(atlas: pd.DataFrame, layer_of: dict) -> None:
    cols = sorted(atlas.columns, key=lambda k: (c.LAYERS.index(layer_of[k]), k))
    m = atlas[cols]
    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.34), max(7, len(m) * 0.34)))
    ax.imshow(m.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=7)
    # layer separators
    prev = None
    for i, k in enumerate(cols):
        L = layer_of[k]
        if L != prev:
            ax.axvline(i - 0.5, color="tab:red", lw=1.2)
            ax.text(i, -1.2, L[:5], color="tab:red", fontsize=8)
            prev = L
    ax.set_title("Fault atlas: signal manifestation by layer")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "atlas_heatmap.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
