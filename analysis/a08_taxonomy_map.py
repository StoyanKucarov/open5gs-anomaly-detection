"""08 - Master taxonomy map: the fault atlas table for the paper.

One row per fault tying the empirical observations to the Zhou 2018 / Silva
2022 / RQ5 axes:
  target NF, chaos, Zhou class, Silva code, origin layer (a priori) ->
  first-detected layer (observed), layer propagation order, blast radius,
  K8s-detected?, distinguishing signals (most specific that fired).

Consumes the CSVs produced by 01-06 so it is the single join point.

Outputs:
  data/analysis/fault_taxonomy.csv
"""
from __future__ import annotations

import pandas as pd

import analysis.common as c


def run() -> pd.DataFrame:
    c.ensure_dirs()
    atlas = pd.read_csv(c.OUT_DIR / "fault_atlas.csv", index_col=0)
    temporal = pd.read_csv(c.OUT_DIR / "temporal_layers.csv", index_col=0)
    firstsig = pd.read_csv(c.OUT_DIR / "first_signal_per_fault.csv", index_col=0)
    prop = pd.read_csv(c.OUT_DIR / "propagation_summary.csv", index_col=0)
    blind = pd.read_csv(c.OUT_DIR / "k8s_blindspot.csv", index_col=0)
    disc = pd.read_csv(c.OUT_DIR / "discriminability.csv", index_col=0)

    spec = disc["specificity"]
    spec = pd.to_numeric(spec, errors="coerce")

    rows = []
    for slug in c.fault_slugs():
        m = c.FAULTS[slug]
        fired = [k for k in atlas.columns if atlas.loc[slug, k] == 1]
        # most discriminative signals that actually fired for this fault
        ranked = sorted(fired, key=lambda k: spec.get(k, 0), reverse=True)
        rows.append({
            "fault": slug,
            "target_nf": m.target_nf,
            "chaos": m.chaos,
            "zhou_class": m.zhou,
            "silva_class": m.silva,
            "family": m.family,
            "origin_layer_apriori": m.origin,
            "first_detected_layer": firstsig.loc[slug, "first_layer"],
            "first_signal": firstsig.loc[slug, "first_signal"],
            "first_t_s": firstsig.loc[slug, "t_detect_s"],
            "layer_propagation": temporal.loc[slug, "layer_order"],
            "blast_radius_nfs": prop.loc[slug, "blast_radius"],
            "affected_nfs": prop.loc[slug, "affected_nfs"],
            "k8s_detected": bool(blind.loc[slug, "k8s_caught"]),
            "n_signals": len(fired),
            "distinguishing_signals": ";".join(ranked[:5]),
        })
    df = pd.DataFrame(rows).set_index("fault")
    df.to_csv(c.OUT_DIR / "fault_taxonomy.csv")
    print(f"[08] fault_taxonomy.csv  master atlas table ({len(df)} faults)")
    return df


if __name__ == "__main__":
    run()
