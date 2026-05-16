#!/usr/bin/env python3
"""
analysis/fault_atlas.py

Produces a fault × signal matrix for all 22 experiments, with signals
grouped and sorted by observability layer (infrastructure → orchestration → application).

Outputs:
  <out>/fault_atlas.csv        — binary signal matrix
  <out>/fault_atlas_summary.txt — per-fault-class aggregates + dead signal report
  stdout                        — human-readable table

Usage:
    python3 analysis/fault_atlas.py \
        [--data reproduce/data/experiments/C-fault-detection] \
        [--out  reproduce/data/analysis]
"""

import argparse
import csv
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from lib import (
    EXPERIMENTS, ALL_SIGNALS, SIGNALS_INFRA, SIGNALS_ORCH, SIGNALS_APP, SIGNALS_NATIVE,
    SIGNAL_LAYERS, NF_PODS,
    load_csv, load_json, floats,
    safe_mean, safe_max, safe_min, percentile,
    prom_phase, prom_vals_filtered, iso_to_unix,
)

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
CPU_SPIKE_RATIO    = 2.0    # during_mean > N× pre_mean
MEM_SPIKE_RATIO    = 1.15   # during_max > N× pre_max
THROTTLE_THRESH    = 0.05   # any throttle ratio during > threshold
NET_HIGH_RATIO     = 2.0    # during_mean > N× pre_mean (surge)
NET_LOW_RATIO      = 0.3    # during_mean < N× pre_mean (drop)
NODE_CPU_RATIO     = 1.5
ERROR_LOG_DELTA    = 3
TRACE_ERR_DELTA    = 0.05
TRACE_LAT_RATIO    = 1.5
BEYLA_LAT_RATIO    = 1.5
RTT_SPIKE_MS       = 50.0
RTT_LOSS_PCT       = 10.0


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------

def detect_prometheus(exp_dir: Path, target_nf: str) -> dict:
    sigs = {s: False for s in SIGNALS_INFRA}
    pod_prefix = NF_PODS.get(target_nf, f"open5gs-{target_nf}")

    # CPU spike on target NF
    pre  = prom_vals_filtered(exp_dir, "container_cpu_usage_rate.csv", "pre",    pod_contains=pod_prefix)
    dur  = prom_vals_filtered(exp_dir, "container_cpu_usage_rate.csv", "during", pod_contains=pod_prefix)
    pre_mean = safe_mean(pre); dur_mean = safe_mean(dur)
    if dur_mean > 0 and (pre_mean < 1e-6 or dur_mean / pre_mean >= CPU_SPIKE_RATIO):
        sigs["cpu_spike"] = True

    # Memory spike on target NF
    pre  = prom_vals_filtered(exp_dir, "container_memory_working_set_bytes.csv", "pre",    pod_contains=pod_prefix)
    dur  = prom_vals_filtered(exp_dir, "container_memory_working_set_bytes.csv", "during", pod_contains=pod_prefix)
    pre_max = safe_max(pre); dur_max = safe_max(dur)
    if pre_max > 0 and dur_max / pre_max >= MEM_SPIKE_RATIO:
        sigs["mem_spike"] = True

    # CPU throttle on target NF
    dur = prom_vals_filtered(exp_dir, "container_cpu_throttled_rate.csv", "during", pod_contains=pod_prefix)
    if safe_max(dur) >= THROTTLE_THRESH:
        sigs["cpu_throttle"] = True

    # Network RX anomaly (namespace-wide — UPF/gNB faults show up here)
    pre  = floats(prom_phase(exp_dir, "network_rx_bytes_rate.csv", "pre"))
    dur  = floats(prom_phase(exp_dir, "network_rx_bytes_rate.csv", "during"))
    pre_m = safe_mean(pre); dur_m = safe_mean(dur)
    if pre_m > 0 and (dur_m / pre_m >= NET_HIGH_RATIO or dur_m / pre_m <= NET_LOW_RATIO):
        sigs["network_rx_anomaly"] = True

    # Network TX anomaly
    pre  = floats(prom_phase(exp_dir, "network_tx_bytes_rate.csv", "pre"))
    dur  = floats(prom_phase(exp_dir, "network_tx_bytes_rate.csv", "during"))
    pre_m = safe_mean(pre); dur_m = safe_mean(dur)
    if pre_m > 0 and (dur_m / pre_m >= NET_HIGH_RATIO or dur_m / pre_m <= NET_LOW_RATIO):
        sigs["network_tx_anomaly"] = True

    # Node CPU spike
    pre  = floats(prom_phase(exp_dir, "node_cpu_usage.csv", "pre"))
    dur  = floats(prom_phase(exp_dir, "node_cpu_usage.csv", "during"))
    pre_m = safe_mean(pre); dur_m = safe_mean(dur)
    if pre_m > 0 and dur_m / pre_m >= NODE_CPU_RATIO:
        sigs["node_cpu_spike"] = True

    return sigs


def detect_orchestration(exp_dir: Path) -> dict:
    sigs = {s: False for s in SIGNALS_ORCH}

    # Pod restart
    pre = floats(prom_phase(exp_dir, "pod_restarts.csv", "pre"))
    dur = floats(prom_phase(exp_dir, "pod_restarts.csv", "during"))
    if safe_max(dur) > safe_max(pre):
        sigs["pod_restart"] = True

    # Pod ready drop — any pod that was ready=1 in pre drops to 0 during
    pre_rows = prom_phase(exp_dir, "pod_ready.csv", "pre")
    dur_rows = prom_phase(exp_dir, "pod_ready.csv", "during")
    pre_ready = {r.get("pod") for r in pre_rows if float(r.get("value", 0) or 0) == 1}
    dur_not_ready = {r.get("pod") for r in dur_rows if float(r.get("value", 1) or 1) == 0}
    if pre_ready & dur_not_ready:
        sigs["pod_ready_drop"] = True

    # K8s warnings (exclude HPA FailedGetResourceMetric noise)
    events = load_json(exp_dir / "events" / "during" / "k8s_events.json") or []
    if isinstance(events, list):
        warnings = [e for e in events
                    if e.get("type") == "Warning"
                    and e.get("reason") != "FailedGetResourceMetric"]
        if warnings:
            sigs["k8s_warning"] = True
    elif isinstance(events, dict):
        # older format: {"total": N, "warnings": N}
        if events.get("warnings", 0) > 0:
            sigs["k8s_warning"] = True

    # NRF registration drop
    pre_nrf = load_json(exp_dir / "nrf" / "pre" / "nrf_registrations.json") or {}
    dur_nrf = load_json(exp_dir / "nrf" / "during" / "nrf_registrations.json") or {}
    if isinstance(pre_nrf, dict) and isinstance(dur_nrf, dict):
        for nf, cnt in pre_nrf.items():
            if isinstance(cnt, int) and cnt > 0:
                if isinstance(dur_nrf.get(nf), int) and dur_nrf[nf] < cnt:
                    sigs["nrf_drop"] = True
                    break

    return sigs


def detect_application(exp_dir: Path) -> dict:
    sigs = {s: False for s in SIGNALS_APP}

    def loki_count(name: str, phase: str) -> int:
        return len(load_csv(exp_dir / "loki" / phase / name))

    pre_err  = loki_count("errors.csv", "pre");    dur_err  = loki_count("errors.csv", "during")
    pre_ue   = loki_count("ue_failures.csv", "pre"); dur_ue  = loki_count("ue_failures.csv", "during")
    pre_nrf  = loki_count("nrf_lifecycle.csv", "pre"); dur_nrf = loki_count("nrf_lifecycle.csv", "during")
    pre_scp  = loki_count("scp_routing.csv", "pre");  dur_scp = loki_count("scp_routing.csv", "during")

    if dur_err  > pre_err  + ERROR_LOG_DELTA: sigs["error_logs"]   = True
    if dur_ue   > pre_ue:                     sigs["ue_failures"]  = True
    if dur_nrf  > pre_nrf + 1:                sigs["nrf_lifecycle"] = True
    if dur_scp  > 0:                          sigs["scp_routing"]  = True

    # Jaeger trace errors / latency
    def span_stats(phase: str) -> dict:
        spans = load_csv(exp_dir / "jaeger" / phase / "spans_flat.csv")
        by_svc = {}
        for s in spans:
            svc = s.get("service", "")
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
            }
        return result

    pre_s = span_stats("pre"); dur_s = span_stats("during")
    for svc, d in dur_s.items():
        p = pre_s.get(svc, {"error_rate": 0, "p95_us": 0})
        if d["error_rate"] - p["error_rate"] >= TRACE_ERR_DELTA:
            sigs["trace_errors"] = True
        if p["p95_us"] > 0 and d["p95_us"] / p["p95_us"] >= TRACE_LAT_RATIO:
            sigs["trace_latency"] = True

    # Beyla error rate (any 5xx during fault)
    dur_beyla_err = floats(prom_phase(exp_dir, "beyla_http_server_error_rate.csv", "during"))
    if safe_mean(dur_beyla_err) > 0:
        sigs["beyla_error_rate"] = True

    # Beyla latency spike
    pre_beyla = floats(prom_phase(exp_dir, "beyla_http_server_duration.csv", "pre"))
    dur_beyla = floats(prom_phase(exp_dir, "beyla_http_server_duration.csv", "during"))
    pre_m = safe_mean(pre_beyla); dur_m = safe_mean(dur_beyla)
    if pre_m > 0 and dur_m / pre_m >= BEYLA_LAT_RATIO:
        sigs["beyla_latency"] = True

    # RTT spike — prefer new ue_rtt.csv (timestamped CSV), fall back to rtt_samples.txt
    rtt_csv = exp_dir / "rtt" / "during" / "ue_rtt.csv"
    rtt_txt = exp_dir / "rtt" / "during" / "rtt_samples.txt"
    if rtt_csv.exists():
        rows = load_csv(rtt_csv)
        rtts = [float(r["rtt_ms"]) for r in rows if r.get("status") == "ok" and r.get("rtt_ms")]
        loss_count = sum(1 for r in rows if r.get("status") == "loss")
        loss_pct = 100.0 * loss_count / len(rows) if rows else 0.0
        if rtts and median(rtts) >= RTT_SPIKE_MS:
            sigs["rtt_spike"] = True
        if loss_pct >= RTT_LOSS_PCT:
            sigs["rtt_spike"] = True
    elif rtt_txt.exists():
        rtts = []; loss_pct = None
        for line in rtt_txt.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "packet loss" in line:
                try:
                    loss_pct = float(line.split("%")[0].rsplit(" ", 1)[-1])
                except (IndexError, ValueError):
                    pass
            else:
                try:
                    rtts.append(float(line))
                except ValueError:
                    pass
        if rtts and median(rtts) >= RTT_SPIKE_MS:
            sigs["rtt_spike"] = True
        if loss_pct is not None and loss_pct >= RTT_LOSS_PCT:
            sigs["rtt_spike"] = True

    return sigs


def detect_native(exp_dir: Path) -> dict:
    sigs = {s: False for s in SIGNALS_NATIVE}

    # AMF registered subscriber drop (>10% drop vs pre)
    pre = floats(prom_phase(exp_dir, "open5gs_amf_registered_subscribers.csv", "pre"))
    dur = floats(prom_phase(exp_dir, "open5gs_amf_registered_subscribers.csv", "during"))
    pre_min = safe_min(pre) if pre else None
    dur_min = safe_min(dur) if dur else None
    if pre_min and dur_min is not None and dur_min < pre_min * 0.9:
        sigs["amf_sub_drop"] = True

    # AMF auth failures (any non-zero rate during fault)
    dur_fail   = floats(prom_phase(exp_dir, "open5gs_amf_auth_fail.csv",   "during"))
    dur_reject = floats(prom_phase(exp_dir, "open5gs_amf_auth_reject.csv", "during"))
    if safe_mean(dur_fail) > 0 or safe_mean(dur_reject) > 0:
        sigs["amf_auth_fail"] = True

    # PFCP session drop (>10% drop in active sessions or UPF session count)
    pre_pfcp = floats(prom_phase(exp_dir, "open5gs_pfcp_sessions_active.csv", "pre"))
    dur_pfcp = floats(prom_phase(exp_dir, "open5gs_pfcp_sessions_active.csv", "during"))
    pre_upf  = floats(prom_phase(exp_dir, "open5gs_upf_session_nbr.csv",      "pre"))
    dur_upf  = floats(prom_phase(exp_dir, "open5gs_upf_session_nbr.csv",      "during"))
    pfcp_pre = safe_mean(pre_pfcp) or safe_mean(pre_upf)
    pfcp_dur = safe_min(dur_pfcp)  if dur_pfcp else safe_min(dur_upf) if dur_upf else None
    if pfcp_pre and pfcp_dur is not None and pfcp_dur < pfcp_pre * 0.9:
        sigs["pfcp_session_drop"] = True

    # GTP data plane anomaly (rate drops >50% or spikes >3×)
    pre_gtp = floats(prom_phase(exp_dir, "open5gs_gtp_in_packets.csv", "pre"))
    dur_gtp = floats(prom_phase(exp_dir, "open5gs_gtp_in_packets.csv", "during"))
    pre_m = safe_mean(pre_gtp); dur_m = safe_mean(dur_gtp)
    if pre_m > 0 and (dur_m / pre_m <= 0.5 or dur_m / pre_m >= 3.0):
        sigs["gtp_data_anomaly"] = True

    # SMF PDU session failure (success rate drops vs request rate)
    pre_req  = floats(prom_phase(exp_dir, "open5gs_smf_pdu_session_req.csv",  "pre"))
    dur_req  = floats(prom_phase(exp_dir, "open5gs_smf_pdu_session_req.csv",  "during"))
    pre_succ = floats(prom_phase(exp_dir, "open5gs_smf_pdu_session_succ.csv", "pre"))
    dur_succ = floats(prom_phase(exp_dir, "open5gs_smf_pdu_session_succ.csv", "during"))
    pre_rate = safe_mean(pre_succ) / safe_mean(pre_req) if safe_mean(pre_req) > 0 else None
    dur_rate = safe_mean(dur_succ) / safe_mean(dur_req) if safe_mean(dur_req) > 0 else None
    if pre_rate is not None and dur_rate is not None and dur_rate < pre_rate * 0.9:
        sigs["smf_pdu_fail"] = True

    return sigs


def analyze_fault(slug: str, target_nf: str, data_root: Path) -> dict:
    exp_dir = data_root / slug
    signals = {}
    for detector, args in [
        (detect_prometheus,   (exp_dir, target_nf)),
        (detect_orchestration,(exp_dir,)),
        (detect_application,  (exp_dir,)),
        (detect_native,       (exp_dir,)),
    ]:
        try:
            signals.update(detector(*args))
        except Exception as e:
            print(f"  [WARN] {slug}: {detector.__name__} failed: {e}", file=sys.stderr)
    return signals


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: list):
    # Use 2-digit column numbers so alignment is consistent regardless of signal count
    col_w  = 3   # " 1 ", " 2 ", ...
    slug_w = 42
    nf_w   = 10

    # Layer boundary indices
    layer_breaks = {}
    for i, s in enumerate(ALL_SIGNALS):
        layer = SIGNAL_LAYERS[s]
        if layer not in layer_breaks:
            layer_breaks[layer] = i

    # Header row: signal numbers
    header = f"  {'Fault':<{slug_w}} {'NF':<{nf_w}} "
    for i in range(len(ALL_SIGNALS)):
        header += f"{i+1:>{col_w}}"
    print(header)
    print("  " + "-" * (slug_w + nf_w + 1 + col_w * len(ALL_SIGNALS)))

    # Layer label row
    layer_row = "  " + " " * (slug_w + nf_w + 1)
    for i, s in enumerate(ALL_SIGNALS):
        layer = SIGNAL_LAYERS[s]
        labels = {"infrastructure": "I", "orchestration": "O",
                  "application": "A", "native": "N"}
        layer_row += f"{labels.get(layer, '?'):>{col_w}}"
    print(layer_row)
    print("  " + "-" * (slug_w + nf_w + 1 + col_w * len(ALL_SIGNALS)))

    # Data rows grouped by fault_class
    prev_class = None
    for slug, fault_type, target, fault_class, sigs in results:
        if fault_class != prev_class:
            print(f"\n  [{fault_class}]")
            prev_class = fault_class
        row = f"  {slug:<{slug_w}} {target:<{nf_w}} "
        for s in ALL_SIGNALS:
            row += f"{'✓':>{col_w}}" if sigs.get(s) else f"{'·':>{col_w}}"
        print(row)

    # Legend
    print()
    print(f"  {'#':<4} {'Signal':<28} Layer")
    print("  " + "-" * 50)
    prev_layer = None
    for i, s in enumerate(ALL_SIGNALS):
        layer = SIGNAL_LAYERS[s]
        if layer != prev_layer:
            print(f"  --- {layer.upper()} ---")
            prev_layer = layer
        print(f"  {i+1:<4} {s:<28} {layer}")


def write_outputs(results: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # fault_atlas.csv
    atlas_path = out_dir / "fault_atlas.csv"
    fieldnames = ["slug", "fault_type", "target_nf", "fault_class"] + ALL_SIGNALS
    with open(atlas_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for slug, fault_type, target, fault_class, sigs in results:
            row = {"slug": slug, "fault_type": fault_type,
                   "target_nf": target, "fault_class": fault_class}
            row.update({s: int(sigs.get(s, False)) for s in ALL_SIGNALS})
            w.writerow(row)
    print(f"\n[out] {atlas_path}")

    # summary text: per-class aggregates + dead signals
    summary_lines = []
    summary_lines.append("=== Per-fault-class signal profiles ===\n")
    by_class: dict = {}
    for slug, fault_type, target, fault_class, sigs in results:
        by_class.setdefault(fault_class, []).append(sigs)

    for cls, all_sigs in sorted(by_class.items()):
        n = len(all_sigs)
        summary_lines.append(f"{cls} ({n} faults):")
        for s in ALL_SIGNALS:
            fired = sum(1 for sg in all_sigs if sg.get(s))
            if fired > 0:
                summary_lines.append(f"  {s:<25} {fired}/{n}")
        summary_lines.append("")

    # dead signals (never fired)
    all_fired = set()
    for _, _, _, _, sigs in results:
        for s, v in sigs.items():
            if v:
                all_fired.add(s)
    dead = [s for s in ALL_SIGNALS if s not in all_fired]
    always = [s for s in ALL_SIGNALS if all(sg.get(s) for _, _, _, _, sg in results)]

    summary_lines.append(f"=== Signals that never fired ({len(dead)}): ===")
    for s in dead:
        summary_lines.append(f"  {s}")
    summary_lines.append(f"\n=== Signals that always fired ({len(always)}): ===")
    for s in always:
        summary_lines.append(f"  {s}")

    summary_path = out_dir / "fault_atlas_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[out] {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(
        Path(__file__).parent.parent / "data/experiments/C-fault-detection"))
    parser.add_argument("--out", default=str(
        Path(__file__).parent.parent / "data/analysis"))
    args = parser.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        sys.exit(f"Data directory not found: {data_root}")

    results = []
    for slug, fault_type, target, fault_class in EXPERIMENTS:
        exp_dir = data_root / slug
        if not exp_dir.exists():
            print(f"  [SKIP] {slug}", file=sys.stderr)
            continue
        sigs = analyze_fault(slug, target, data_root)
        results.append((slug, fault_type, target, fault_class, sigs))
        fired = [s for s in ALL_SIGNALS if sigs.get(s)]
        print(f"  [{slug}] {len(fired)} signals: {', '.join(fired) or 'none'}")

    print()
    print_table(results)
    write_outputs(results, Path(args.out))


if __name__ == "__main__":
    main()
