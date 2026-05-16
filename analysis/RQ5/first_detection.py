#!/usr/bin/env python3
"""
analysis/first_detection.py

For each fault, find which observability signal fires FIRST after fault
injection starts, using timeline.json fault.start as t=0.

Detection methodology — sliding window anomaly detection:
  All signals use the same core approach:
    1. Bin raw data (Prometheus samples, Loki events, Jaeger spans) into 5s bins.
    2. Compute baseline distribution from pre-fault windows (all 30s windows
       whose last bin ends before fault_start).
    3. Slide a 30s window every 5s starting from fault_start - 25s (so the
       first windows straddle the pre/during boundary and can catch immediate
       signals). Report detection time = window_end - fault_start when the
       window stat first exceeds baseline_mean + Z_THRESH * baseline_std.
  This gives 5s timing resolution and uses actual pre variability as the
  threshold, not an arbitrary ratio.

  Special cases:
    - K8s events: discrete, timestamped — use first Warning event directly.
    - NRF drop: snapshot only — excluded (no timing available).
    - RTT: no per-sample timestamps — report fault_start if samples indicate spike.

Outputs:
  <out>/first_detection.csv   — first signal per fault with time-to-detect
  <out>/beyla_vs_infra.csv    — Beyla vs infra detection comparison
  stdout                      — summary table

Usage:
    python3 analysis/first_detection.py \
        [--data reproduce/data/experiments/C-fault-detection] \
        [--out  reproduce/data/analysis]
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import (
    EXPERIMENTS, SIGNAL_LAYERS, NF_PODS,
    load_csv, load_json, floats,
    safe_mean, safe_max, percentile,
    prom_phase, prom_vals_filtered, iso_to_unix,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BIN_S      = 5      # seconds per bin (matches Prometheus step)
WINDOW_S   = 30     # sliding window width in seconds
WINDOW_BINS = WINDOW_S // BIN_S   # 6 bins per window
PRE_S      = 600    # pre-window duration
Z_THRESH   = 3.0    # baseline_mean + Z * baseline_std to fire
MIN_STD    = 1e-6   # avoid division by zero

RTT_SPIKE_MS  = 50.0
RTT_LOSS_PCT  = 10.0

NOT_DETECTED = float("inf")


# ---------------------------------------------------------------------------
# Core sliding-window detector
# ---------------------------------------------------------------------------

def sliding_detect(bins: dict, fault_start: float,
                   stat_fn=None, z: float = Z_THRESH) -> float:
    """
    bins: {bin_start_ts: value}  — value is whatever stat_fn aggregates
          (mean for Prometheus, count for Loki/Jaeger)
    stat_fn: called on list of bin values in the window → scalar.
             Defaults to mean.
    Returns earliest window_end timestamp where window stat > baseline mean + z*std,
    or NOT_DETECTED.
    """
    if stat_fn is None:
        stat_fn = lambda vals: sum(vals) / len(vals) if vals else 0.0

    if not bins:
        return NOT_DETECTED

    all_ts = sorted(bins)
    t_min  = all_ts[0]
    t_max  = all_ts[-1]

    # --- Build baseline from pre windows ---
    # Slide over the pre region, collect per-window stats
    baseline_stats = []
    t = t_min
    while t + WINDOW_S <= fault_start:
        window_vals = [bins[b] for b in all_ts if t <= b < t + WINDOW_S and b in bins]
        if window_vals:
            baseline_stats.append(stat_fn(window_vals))
        t += BIN_S

    if not baseline_stats:
        return NOT_DETECTED

    b_mean = sum(baseline_stats) / len(baseline_stats)
    variance = sum((x - b_mean) ** 2 for x in baseline_stats) / len(baseline_stats)
    b_std  = max(math.sqrt(variance), MIN_STD)
    threshold = b_mean + z * b_std

    # --- Slide into fault region, starting WINDOW_S - BIN_S before fault_start ---
    # so first window straddles the boundary
    t = fault_start - (WINDOW_S - BIN_S)
    while t <= t_max:
        window_vals = [bins[b] for b in all_ts if t <= b < t + WINDOW_S and b in bins]
        if window_vals and stat_fn(window_vals) > threshold:
            # detection time = end of this window relative to fault_start
            detection_ts = t + WINDOW_S
            return max(0.0, detection_ts - fault_start)
        t += BIN_S

    return NOT_DETECTED


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def prom_to_bins(rows: list, t_min: float, stat="mean") -> dict:
    """Aggregate Prometheus rows into BIN_S bins by rounding timestamp down."""
    buckets: dict = defaultdict(list)
    for r in rows:
        try:
            ts  = float(r["timestamp"])
            val = float(r["value"])
            if math.isfinite(val):
                b = t_min + math.floor((ts - t_min) / BIN_S) * BIN_S
                buckets[b].append(val)
        except (KeyError, ValueError, TypeError):
            pass
    if stat == "mean":
        return {b: sum(v) / len(v) for b, v in buckets.items()}
    elif stat == "max":
        return {b: max(v) for b, v in buckets.items()}
    elif stat == "sum":
        return {b: sum(v) for b, v in buckets.items()}
    return {}


def loki_to_bins(rows: list, t_min: float) -> dict:
    """Count Loki log lines per BIN_S bin."""
    buckets: dict = defaultdict(int)
    for r in rows:
        try:
            ts = float(r["timestamp_ns"]) / 1e9
            b  = t_min + math.floor((ts - t_min) / BIN_S) * BIN_S
            buckets[b] += 1
        except (KeyError, ValueError, TypeError):
            pass
    return dict(buckets)


def jaeger_err_bins(spans: list, t_min: float) -> dict:
    """Error rate (errors/total) per BIN_S bin."""
    totals: dict  = defaultdict(int)
    errors: dict  = defaultdict(int)
    for s in spans:
        try:
            ts = float(s["start_us"]) / 1e6
            b  = t_min + math.floor((ts - t_min) / BIN_S) * BIN_S
            totals[b] += 1
            errors[b] += int(s.get("error", 0) or 0)
        except (KeyError, ValueError, TypeError):
            pass
    return {b: errors[b] / totals[b] for b in totals if totals[b] > 0}


def jaeger_lat_bins(spans: list, t_min: float, pct: float = 95) -> dict:
    """p95 latency (µs) per BIN_S bin."""
    buckets: dict = defaultdict(list)
    for s in spans:
        try:
            ts  = float(s["start_us"]) / 1e6
            dur = float(s["duration_us"])
            b   = t_min + math.floor((ts - t_min) / BIN_S) * BIN_S
            buckets[b].append(dur)
        except (KeyError, ValueError, TypeError):
            pass
    return {b: percentile(v, pct) for b, v in buckets.items() if v}


# ---------------------------------------------------------------------------
# Per-signal detection
# ---------------------------------------------------------------------------

def detect_all(exp_dir: Path, target_nf: str, fault_start: float) -> dict:
    results = {}
    pod_prefix = NF_PODS.get(target_nf, f"open5gs-{target_nf}")
    t_min = fault_start - PRE_S  # start of pre window

    def sd(bins):
        """Shorthand: run sliding_detect, return time-to-detect."""
        t = sliding_detect(bins, fault_start)
        return t if t != NOT_DETECTED else NOT_DETECTED

    # --- Infrastructure: Prometheus (mean value per bin) ---

    def prom_bins_nf(metric, stat="mean"):
        pre  = [r for r in prom_phase(exp_dir, metric, "pre")  if pod_prefix in r.get("pod", "")]
        dur  = [r for r in prom_phase(exp_dir, metric, "during") if pod_prefix in r.get("pod", "")]
        return prom_to_bins(pre + dur, t_min, stat=stat)

    results["cpu_spike"]    = sd(prom_bins_nf("container_cpu_usage_rate.csv"))
    results["mem_spike"]    = sd(prom_bins_nf("container_memory_working_set_bytes.csv", stat="max"))
    results["cpu_throttle"] = sd(prom_bins_nf("container_cpu_throttled_rate.csv"))

    # Network: per-NF, ignore NFs with near-zero baseline traffic
    def net_bins_nf(metric):
        pre = [r for r in prom_phase(exp_dir, metric, "pre")    if pod_prefix in r.get("pod", "")]
        dur = [r for r in prom_phase(exp_dir, metric, "during") if pod_prefix in r.get("pod", "")]
        bins = prom_to_bins(pre + dur, t_min, stat="mean")
        pre_vals = [v for ts, v in bins.items() if ts < fault_start]
        if not pre_vals or safe_mean(pre_vals) < 1.0:
            return {}
        return bins

    results["network_rx_anomaly"] = sd(net_bins_nf("network_rx_bytes_rate.csv"))
    results["network_tx_anomaly"] = sd(net_bins_nf("network_tx_bytes_rate.csv"))

    pre_node = prom_phase(exp_dir, "node_cpu_usage.csv", "pre")
    dur_node = prom_phase(exp_dir, "node_cpu_usage.csv", "during")
    results["node_cpu_spike"] = sd(prom_to_bins(pre_node + dur_node, t_min))

    # --- Orchestration ---

    # Pod restart: sum of restart counts per bin (step function → use max)
    pre_rst = prom_phase(exp_dir, "pod_restarts.csv", "pre")
    dur_rst = prom_phase(exp_dir, "pod_restarts.csv", "during")
    results["pod_restart"] = sd(prom_to_bins(pre_rst + dur_rst, t_min, stat="max"))

    # Pod ready drop: bin = fraction of pods that are ready (lower = worse)
    # Invert: detect when mean drops (use negated values so a drop = spike)
    pre_rdy = prom_phase(exp_dir, "pod_ready.csv", "pre")
    dur_rdy = prom_phase(exp_dir, "pod_ready.csv", "during")
    ready_bins_raw = prom_to_bins(pre_rdy + dur_rdy, t_min, stat="mean")
    # Negate so a drop in readiness looks like a spike to sliding_detect
    ready_bins = {b: -v for b, v in ready_bins_raw.items()}
    results["pod_ready_drop"] = sd(ready_bins)

    # K8s warning events: discrete, keep first-event approach
    events = load_json(exp_dir / "events" / "during" / "k8s_events.json") or []
    best_ts = NOT_DETECTED
    if isinstance(events, list):
        for e in events:
            if e.get("type") == "Warning" and e.get("reason") != "FailedGetResourceMetric":
                ets = iso_to_unix(e.get("time", ""))
                if ets >= fault_start:
                    best_ts = min(best_ts, ets)
    results["k8s_warning"] = best_ts - fault_start if best_ts != NOT_DETECTED else NOT_DETECTED

    # NRF drop: snapshot-only — no timing
    results["nrf_drop"] = NOT_DETECTED

    # --- Application: Loki (event count per bin) ---

    def loki_sd(filename):
        pre = load_csv(exp_dir / "loki" / "pre"    / filename)
        dur = load_csv(exp_dir / "loki" / "during" / filename)
        bins = loki_to_bins(pre + dur, t_min)
        return sd(bins)

    results["error_logs"]   = loki_sd("errors.csv")
    results["ue_failures"]  = loki_sd("ue_failures.csv")
    results["nrf_lifecycle"]= loki_sd("nrf_lifecycle.csv")

    dur_scp = load_csv(exp_dir / "loki" / "during" / "scp_routing.csv")
    pre_scp = load_csv(exp_dir / "loki" / "pre"    / "scp_routing.csv")
    results["scp_routing"]  = loki_sd("scp_routing.csv")

    # --- Application: Jaeger ---

    pre_spans = load_csv(exp_dir / "jaeger" / "pre"    / "spans_flat.csv")
    dur_spans = load_csv(exp_dir / "jaeger" / "during" / "spans_flat.csv")

    results["trace_errors"]  = sd(jaeger_err_bins(pre_spans + dur_spans, t_min))
    results["trace_latency"] = sd(jaeger_lat_bins(pre_spans + dur_spans, t_min))

    # Beyla metrics (Prometheus)
    pre_be = prom_phase(exp_dir, "beyla_http_server_error_rate.csv", "pre")
    dur_be = prom_phase(exp_dir, "beyla_http_server_error_rate.csv", "during")
    results["beyla_error_rate"] = sd(prom_to_bins(pre_be + dur_be, t_min))

    pre_bl = prom_phase(exp_dir, "beyla_http_server_duration.csv", "pre")
    dur_bl = prom_phase(exp_dir, "beyla_http_server_duration.csv", "during")
    results["beyla_latency"] = sd(prom_to_bins(pre_bl + dur_bl, t_min))

    # RTT: use ue_rtt.csv (timestamped) for sliding window; fall back to rtt_samples.txt
    rtt_csv = exp_dir / "rtt" / "during" / "ue_rtt.csv"
    rtt_txt = exp_dir / "rtt" / "during" / "rtt_samples.txt"
    if rtt_csv.exists():
        rtt_rows = load_csv(rtt_csv)
        # Build bins: RTT value for ok rows, RTT_SPIKE_MS*2 sentinel for loss rows
        rtt_bins: dict = defaultdict(list)
        for r in rtt_rows:
            try:
                ts = float(r["timestamp_ms"]) / 1000.0
                b  = t_min + math.floor((ts - t_min) / BIN_S) * BIN_S
                if r.get("status") == "loss":
                    rtt_bins[b].append(RTT_SPIKE_MS * 2)
                elif r.get("rtt_ms"):
                    rtt_bins[b].append(float(r["rtt_ms"]))
            except (KeyError, ValueError, TypeError):
                pass
        rtt_med_bins = {b: percentile(v, 50) for b, v in rtt_bins.items() if v}
        results["rtt_spike"] = sliding_detect(rtt_med_bins, fault_start,
                                              stat_fn=lambda v: sum(v)/len(v),
                                              z=Z_THRESH)
    elif rtt_txt.exists():
        lines = rtt_txt.read_text().splitlines()
        rtt_vals, has_loss = [], False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "packet loss" in line:
                try:
                    if float(line.split("%")[0]) >= RTT_LOSS_PCT:
                        has_loss = True
                except ValueError:
                    pass
                continue
            try:
                rtt_vals.append(float(line))
            except ValueError:
                pass
        median_rtt = percentile(rtt_vals, 50) if rtt_vals else 0.0
        results["rtt_spike"] = 0.0 if (has_loss or median_rtt > RTT_SPIKE_MS) else NOT_DETECTED
    else:
        results["rtt_spike"] = NOT_DETECTED

    # --- Native: Open5GS protocol metrics ---

    def native_prom_bins(metric, stat="mean"):
        pre = prom_phase(exp_dir, metric, "pre")
        dur = prom_phase(exp_dir, metric, "during")
        return prom_to_bins(pre + dur, t_min, stat=stat)

    # amf_sub_drop: detect when registered subscriber count drops
    sub_bins_raw = native_prom_bins("open5gs_amf_registered_subscribers.csv", stat="mean")
    sub_bins = {b: -v for b, v in sub_bins_raw.items()}  # negate: drop = spike
    results["amf_sub_drop"] = sliding_detect(sub_bins, fault_start)

    # amf_auth_fail: detect spike in auth failure rate
    fail_bins = native_prom_bins("open5gs_amf_auth_fail.csv")
    rej_bins  = native_prom_bins("open5gs_amf_auth_reject.csv")
    combined  = {b: fail_bins.get(b, 0) + rej_bins.get(b, 0)
                 for b in set(fail_bins) | set(rej_bins)}
    results["amf_auth_fail"] = sliding_detect(combined, fault_start)

    # pfcp_session_drop: detect PFCP session count drop
    pfcp_bins_raw = native_prom_bins("open5gs_pfcp_sessions_active.csv", stat="mean")
    if not pfcp_bins_raw:
        pfcp_bins_raw = native_prom_bins("open5gs_upf_session_nbr.csv", stat="mean")
    pfcp_bins = {b: -v for b, v in pfcp_bins_raw.items()}
    results["pfcp_session_drop"] = sliding_detect(pfcp_bins, fault_start)

    # gtp_data_anomaly: detect GTP packet rate change
    gtp_bins = native_prom_bins("open5gs_gtp_in_packets.csv")
    results["gtp_data_anomaly"] = sliding_detect(gtp_bins, fault_start)

    # smf_pdu_fail: detect PDU session success rate drop
    req_bins  = native_prom_bins("open5gs_smf_pdu_session_req.csv")
    succ_bins = native_prom_bins("open5gs_smf_pdu_session_succ.csv")
    all_ts = sorted(set(req_bins) | set(succ_bins))
    fail_rate_bins = {}
    for b in all_ts:
        req = req_bins.get(b, 0); succ = succ_bins.get(b, 0)
        if req > 0:
            fail_rate_bins[b] = -(succ / req)  # negate: lower success = spike
    results["smf_pdu_fail"] = sliding_detect(fail_rate_bins, fault_start)

    return results


# ---------------------------------------------------------------------------
# Summarise + main
# ---------------------------------------------------------------------------

def summarise(detection_times: dict) -> tuple:
    detected = {s: t for s, t in detection_times.items() if t != NOT_DETECTED}
    if not detected:
        return "none", "none", None
    first_sig = min(detected, key=detected.get)
    return first_sig, SIGNAL_LAYERS.get(first_sig, "unknown"), round(detected[first_sig], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(
        Path(__file__).parent.parent / "data/experiments/C-fault-detection"))
    parser.add_argument("--out", default=str(
        Path(__file__).parent.parent / "data/analysis"))
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, beyla_rows = [], []

    COL_SLUG  = 44
    COL_SIG   = 24
    COL_LAYER = 16
    COL_T     = 7
    hdr = f"  {'Fault':<{COL_SLUG}} {'First signal':<{COL_SIG}} {'Layer':<{COL_LAYER}} {'t(s)':>{COL_T}}"
    print(hdr)
    print("  " + "-" * (COL_SLUG + COL_SIG + COL_LAYER + COL_T + 3))

    for slug, fault_type, target, fault_class in EXPERIMENTS:
        exp_dir = data_root / slug
        if not exp_dir.exists():
            continue

        tl = load_json(exp_dir / "timeline.json") or {}
        fault_start = tl.get("fault", {}).get("start")
        if not fault_start:
            continue

        try:
            times = detect_all(exp_dir, target, float(fault_start))
        except Exception as e:
            print(f"  [WARN] {slug}: {e}", file=sys.stderr)
            continue

        first_sig, first_layer, ttd = summarise(times)
        all_det = {s: round(t, 1) for s, t in sorted(times.items(), key=lambda x: x[1])
                   if t != NOT_DETECTED}

        ttd_str = f"{ttd:.1f}" if ttd is not None else "—"
        print(f"  {slug:<{COL_SLUG}} {first_sig:<{COL_SIG}} {first_layer:<{COL_LAYER}} {ttd_str:>{COL_T}}")

        rows.append({
            "slug": slug,
            "fault_class": fault_class,
            "target_nf": target,
            "first_signal": first_sig,
            "first_layer": first_layer,
            "time_to_detect_s": ttd,
            "all_detection_times": str(all_det),
        })

        infra_times = {s: t for s, t in times.items()
                       if SIGNAL_LAYERS.get(s) == "infrastructure" and t != NOT_DETECTED}
        beyla_times = {s: t for s, t in times.items()
                       if s in ("beyla_error_rate", "beyla_latency") and t != NOT_DETECTED}
        if infra_times and beyla_times:
            first_infra_t = min(infra_times.values())
            first_infra_s = min(infra_times, key=infra_times.get)
            first_beyla_t = min(beyla_times.values())
            first_beyla_s = min(beyla_times, key=beyla_times.get)
            beyla_rows.append({
                "slug": slug,
                "fault_class": fault_class,
                "first_infra_signal": first_infra_s,
                "first_infra_t_s":    round(first_infra_t, 1),
                "first_beyla_signal": first_beyla_s,
                "first_beyla_t_s":    round(first_beyla_t, 1),
                "beyla_faster_by_s":  round(first_infra_t - first_beyla_t, 1),
            })

    det_path = out_dir / "first_detection.csv"
    with open(det_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "fault_class", "target_nf",
                                           "first_signal", "first_layer",
                                           "time_to_detect_s", "all_detection_times"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {det_path}")

    if beyla_rows:
        bv_path = out_dir / "beyla_vs_infra.csv"
        with open(bv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(beyla_rows[0].keys()))
            w.writeheader()
            w.writerows(beyla_rows)
        print(f"[out] {bv_path}")
        faster = sum(1 for r in beyla_rows if r["beyla_faster_by_s"] > 0)
        print(f"\nBeyla faster than infra in {faster}/{len(beyla_rows)} faults")


if __name__ == "__main__":
    main()
