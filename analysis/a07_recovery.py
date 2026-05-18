"""07 - Recovery / fault-tolerance: does the system return to baseline post-fault?

Per fault, per scored continuous signal: compare the `post` phase against the
`pre` baseline band. recovered = signal back inside band by end of post.
time-to-recover measured from fault stop (Chaos `Recovered` / timeline end).
Discrete signals (events, counters, nrf): recovered = no anomaly in post.
Cross-checked against health_post.json (pods_not_running, restarts).

Outputs:
  data/analysis/recovery.csv
  data/analysis/plots/recovery_heatmap.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis.common as c


def _band(vals, sig):
    """Baseline band using the SAME thresholds the detector used, so a signal
    that never tripped detection cannot count as 'unrecovered' from noise."""
    s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(s) < 3:
        return None
    m, sd = float(s.mean()), float(s.std(ddof=0))
    half = max(c.Z * sd, c.FLAT_REL * abs(m), sig.floor, c.FLAT_EPS)
    return m, half


def _recovered_metric(ctx, sig):
    pre = c._read_prom(ctx.root, "pre", sig.source)
    post = c._read_prom(ctx.root, "post", sig.source)
    if pre is None or post is None:
        return None, None
    split = c._beyla_series_by_nf if sig.source.startswith("beyla_") else c._series_by_nf
    pre_nf, post_nf = split(pre), split(post)
    rec, ttr = True, 0.0
    for nf, ps in post_nf.items():
        band = _band(pre_nf.get(nf, pd.DataFrame({"value": []}))["value"], sig)
        if band is None:
            continue
        m, half = band
        ps = ps.sort_values("timestamp")
        tail = ps["value"].to_numpy()[-c.WIN:]
        out = len(tail) and abs(tail.mean() - m) > half
        if out and sig.ratio:
            am = abs(tail.mean()); ab = abs(m)
            out = (am >= sig.ratio * ab) or (ab > c.FLAT_EPS and am <= ab / sig.ratio)
        if out:
            rec = False
        else:
            ok = ps[(ps["value"] - m).abs() <= half]
            if len(ok):
                ttr = max(ttr, max(0.0, float(ok["timestamp"].iloc[0]) - ctx.fault_end))
    return rec, round(ttr, 1)


def run() -> pd.DataFrame:
    c.ensure_dirs()
    cont = [s for s in c.SCORED_SIGNALS if s.kind in ("zscore",)]
    disc = [s for s in c.SCORED_SIGNALS if s.kind in ("counter", "event", "nrf",
                                                       "podstep", "poddrop", "loki", "jaeger")]
    rows = []
    for slug in c.fault_slugs():
        ctx, res = c.get_detection(slug)
        not_rec, ttrs = [], []
        for s in cont:
            if not res[s.key].manifested:
                continue
            r, t = _recovered_metric(ctx, s)
            if r is False:
                not_rec.append(s.key)
            elif t is not None:
                ttrs.append(t)
        # discrete: assume recovered unless health_post says otherwise
        hp = ctx.root / "health_post.json"
        hpre = ctx.root / "health_pre.json"
        health_ok = True
        if hp.exists() and hpre.exists():
            try:
                a, b = json.loads(hpre.read_text()), json.loads(hp.read_text())
                health_ok = (b.get("pods_not_running", 0) <= a.get("pods_not_running", 0)
                             and b.get("gnb_connected", 1) >= 1)
            except Exception:
                pass
        rows.append({
            "fault": slug, "family": c.FAULTS[slug].family,
            "fully_recovered": (not not_rec) and health_ok,
            "unrecovered_signals": ";".join(not_rec),
            "n_unrecovered": len(not_rec),
            "max_time_to_recover_s": round(max(ttrs), 1) if ttrs else 0.0,
            "health_post_ok": health_ok,
        })
    df = pd.DataFrame(rows).set_index("fault")
    df.to_csv(c.OUT_DIR / "recovery.csv")
    _plot(df)
    bad = df.index[~df["fully_recovered"]].tolist()
    print(f"[07] recovery.csv  not fully recovered: {bad or 'none'}")
    return df


def _plot(df):
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 3.2))
    vals = df["n_unrecovered"].to_numpy(dtype=float)
    colors = ["tab:green" if df["fully_recovered"].iloc[i] else "tab:red"
              for i in range(len(df))]
    ax.bar(range(len(df)), np.maximum(vals, 0.15), color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=90, fontsize=7)
    ax.set_ylabel("# unrecovered signals")
    ax.set_title("Post-fault recovery  (green = fully recovered)")
    fig.tight_layout()
    fig.savefig(c.PLOT_DIR / "recovery_heatmap.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
