#!/usr/bin/env python3
"""
analysis/cross_nf_propagation.py

For each fault, builds a timeline of which NFs show signals and in what
order — revealing how faults propagate across the 5G service mesh.

Per-NF signals detected using:
  - Prometheus: CPU and memory spikes per pod
  - Jaeger: error rate and p95 latency increase per service
  - Loki: error-keyword line counts per app/pod

Outputs:
  <out>/propagation/<slug>.json   — per-fault propagation detail
  <out>/propagation_summary.csv   — blast radius + chain per fault
  stdout                          — summary table

Usage:
    python3 analysis/cross_nf_propagation.py \
        [--data reproduce/data/experiments/C-fault-detection] \
        [--out  reproduce/data/analysis]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import (
    EXPERIMENTS, NF_PODS,
    load_csv, load_json, floats,
    safe_mean, safe_max, percentile,
    prom_phase,
)

CPU_SPIKE_RATIO  = 2.0
MEM_SPIKE_RATIO  = 1.15
TRACE_ERR_DELTA  = 0.05
TRACE_LAT_RATIO  = 1.5
LOG_ERR_RATIO    = 2.0   # during error lines > N× pre error lines


ERROR_KEYWORDS = re.compile(
    r"(?i)(error|exception|refused|failed|fatal|oom|killed|panic|timeout|dropped)"
)


def prom_nf_vals(exp_dir: Path, metric: str, phase: str, pod_prefix: str) -> list:
    rows = prom_phase(exp_dir, metric, phase)
    return floats([r for r in rows if pod_prefix in r.get("pod", "")])


def jaeger_service_stats(exp_dir: Path, phase: str) -> dict:
    spans = load_csv(exp_dir / "jaeger" / phase / "spans_flat.csv")
    by_svc: dict = {}
    for s in spans:
        svc = s.get("service", "")
        if not svc:
            continue
        if svc not in by_svc:
            by_svc[svc] = {"errors": 0, "durations": []}
        by_svc[svc]["errors"] += int(s.get("error", 0) or 0)
        try:
            by_svc[svc]["durations"].append(int(s["duration_us"]))
        except (KeyError, ValueError):
            pass
    result = {}
    for svc, d in by_svc.items():
        n = len(d["durations"])
        result[svc] = {
            "error_rate": d["errors"] / n if n else 0,
            "p95_us": percentile(d["durations"], 95),
            "span_count": n,
        }
    return result


def loki_error_count_by_app(exp_dir: Path, phase: str) -> dict:
    rows = load_csv(exp_dir / "loki" / phase / "errors.csv")
    counts: dict = {}
    for r in rows:
        app = r.get("app", r.get("pod", "unknown"))
        # strip pod hash suffix to get NF name
        app = re.sub(r"-[a-z0-9]+-[a-z0-9]+$", "", app)
        counts[app] = counts.get(app, 0) + 1
    return counts


def analyze_nf_propagation(exp_dir: Path, target_nf: str) -> list:
    """Returns list of NF impact dicts, sorted by signal count descending."""
    tl = load_json(exp_dir / "timeline.json") or {}
    fault_start = tl.get("fault", {}).get("start", 0)

    pre_jaeger = jaeger_service_stats(exp_dir, "pre")
    dur_jaeger = jaeger_service_stats(exp_dir, "during")
    pre_loki   = loki_error_count_by_app(exp_dir, "pre")
    dur_loki   = loki_error_count_by_app(exp_dir, "during")

    nf_impacts = []
    for nf, pod_prefix in NF_PODS.items():
        signals_fired = []

        # Prometheus CPU spike
        pre_cpu = safe_mean(prom_nf_vals(exp_dir, "container_cpu_usage_rate.csv", "pre", pod_prefix))
        dur_cpu = safe_mean(prom_nf_vals(exp_dir, "container_cpu_usage_rate.csv", "during", pod_prefix))
        if dur_cpu > 0 and (pre_cpu < 1e-6 or dur_cpu / pre_cpu >= CPU_SPIKE_RATIO):
            signals_fired.append("cpu_spike")

        # Prometheus memory spike
        pre_mem = safe_max(prom_nf_vals(exp_dir, "container_memory_working_set_bytes.csv", "pre", pod_prefix))
        dur_mem = safe_max(prom_nf_vals(exp_dir, "container_memory_working_set_bytes.csv", "during", pod_prefix))
        if pre_mem > 0 and dur_mem / pre_mem >= MEM_SPIKE_RATIO:
            signals_fired.append("mem_spike")

        # Jaeger error rate
        svc = nf  # Beyla uses short NF name as service name
        if svc in dur_jaeger:
            p = pre_jaeger.get(svc, {"error_rate": 0, "p95_us": 0})
            d = dur_jaeger[svc]
            if d["error_rate"] - p["error_rate"] >= TRACE_ERR_DELTA:
                signals_fired.append("trace_errors")
            if p["p95_us"] > 0 and d["p95_us"] / p["p95_us"] >= TRACE_LAT_RATIO:
                signals_fired.append("trace_latency")

        # Loki error logs for this NF
        pre_err = pre_loki.get(nf, 0)
        dur_err = dur_loki.get(nf, 0)
        if dur_err > 0 and (pre_err == 0 or dur_err / max(pre_err, 1) >= LOG_ERR_RATIO):
            signals_fired.append("log_errors")

        if signals_fired:
            nf_impacts.append({
                "nf": nf,
                "is_target": nf == target_nf,
                "signals": signals_fired,
                "signal_count": len(signals_fired),
                "cpu_ratio": round(dur_cpu / max(pre_cpu, 1e-6), 2),
                "mem_ratio": round(dur_mem / max(pre_mem, 1.0), 2),
                "log_error_delta": dur_err - pre_err,
                "jaeger_error_rate_during": round(dur_jaeger.get(nf, {}).get("error_rate", 0), 4),
            })

    # Sort: target NF first, then by signal count
    nf_impacts.sort(key=lambda x: (-int(x["is_target"]), -x["signal_count"]))
    return nf_impacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(
        Path(__file__).parent.parent / "data/experiments/C-fault-detection"))
    parser.add_argument("--out", default=str(
        Path(__file__).parent.parent / "data/analysis"))
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    prop_dir = out_dir / "propagation"
    prop_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"{'Fault':<42} {'Target':>8}  {'Blast':>5}  Propagation chain")
    print("-" * 100)

    for slug, fault_type, target, fault_class in EXPERIMENTS:
        exp_dir = data_root / slug
        if not exp_dir.exists():
            continue

        try:
            impacts = analyze_nf_propagation(exp_dir, target)
        except Exception as e:
            print(f"  [WARN] {slug}: {e}", file=sys.stderr)
            continue

        blast_radius = len(impacts)
        chain = " → ".join(i["nf"] for i in impacts)

        print(f"  {slug:<40} {target:>8}  {blast_radius:>5}  {chain}")

        # Write per-fault JSON
        detail = {
            "slug": slug,
            "fault_class": fault_class,
            "target_nf": target,
            "blast_radius": blast_radius,
            "propagation_chain": [i["nf"] for i in impacts],
            "nf_impacts": impacts,
        }
        (prop_dir / f"{slug}.json").write_text(json.dumps(detail, indent=2))

        summary_rows.append({
            "slug": slug,
            "fault_class": fault_class,
            "target_nf": target,
            "blast_radius": blast_radius,
            "propagation_chain": " -> ".join(i["nf"] for i in impacts),
            "affected_nfs": ", ".join(i["nf"] for i in impacts),
        })

    # Write summary CSV
    if summary_rows:
        summary_path = out_dir / "propagation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[out] {summary_path}")
        print(f"[out] {prop_dir}/<slug>.json  ({len(summary_rows)} files)")

        # Print blast radius stats
        radii = [r["blast_radius"] for r in summary_rows]
        print(f"\nBlast radius: min={min(radii)}, max={max(radii)}, "
              f"mean={sum(radii)/len(radii):.1f}")


if __name__ == "__main__":
    main()
