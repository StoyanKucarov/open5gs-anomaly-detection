#!/usr/bin/env python3
"""
Collect telemetry from Prometheus, Jaeger, and Loki for a given time window.

Usage:
  python collect.py \\
    --fault cpu-stress-amf \\
    --run 1 \\
    --phase baseline|fault|recovery \\
    --start 2026-05-01T14:00:00Z \\
    --end   2026-05-01T14:10:00Z

Expects port-forwards already running:
  Prometheus  localhost:9090
  Jaeger      localhost:16686
  Loki        localhost:3100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

PROMETHEUS_URL = "http://localhost:9090"
JAEGER_URL     = "http://localhost:16686"
LOKI_URL       = "http://localhost:3100"

# (output_name, PromQL expression)
PROMETHEUS_METRICS = [
    ("cpu_throttle_total",
     'container_cpu_cfs_throttled_seconds_total{namespace="open5gs",container!=""}'),
    ("cpu_throttle_rate",
     'rate(container_cpu_cfs_throttled_seconds_total{namespace="open5gs",container!=""}[1m])'),
    ("cpu_usage_total",
     'container_cpu_usage_seconds_total{namespace="open5gs",container!=""}'),
    ("cpu_usage_rate",
     'rate(container_cpu_usage_seconds_total{namespace="open5gs",container!=""}[1m])'),
    ("memory_usage_bytes",
     'container_memory_usage_bytes{namespace="open5gs",container!=""}'),
    # Pod-level memory: sums ALL containers in the pod including Chaos Mesh stress-ng sidecar.
    # Use this (not memory_usage_bytes) for memory-pressure-upf analysis.
    ("pod_memory_working_set_bytes",
     'sum by (pod, namespace) (container_memory_working_set_bytes{namespace="open5gs"})'),
    ("pod_restarts_total",
     'kube_pod_container_status_restarts_total{namespace="open5gs"}'),
    ("pod_status_ready",
     'kube_pod_status_ready{namespace="open5gs"}'),
    ("container_oom_events",
     'kube_pod_container_status_last_terminated_reason{namespace="open5gs"}'),
    # Network I/O through UE TUN interfaces (data plane signal)
    ("network_rx_bytes",
     'rate(container_network_receive_bytes_total{namespace="open5gs",pod=~"ueransim.*"}[30s])'),
    ("network_tx_bytes",
     'rate(container_network_transmit_bytes_total{namespace="open5gs",pod=~"ueransim.*"}[30s])'),
    # Pod running count per NF — drops instantly on crash/kill (better than ready metric)
    ("pod_running_count",
     'kube_pod_status_phase{namespace="open5gs",phase="Running"}'),
]

# Loki queries: (output_name, LogQL expression)
LOKI_QUERIES = [
    ("all",
     '{namespace="open5gs"}'),
    ("errors",
     '{namespace="open5gs"} |~ "(?i)(error|exception|refused|failed|fatal|oom|killed)"'),
    # 5G-specific fault signatures
    ("nrf_lifecycle",
     '{namespace="open5gs"} |~ "(?i)(heartbeat|de-registered|Retry registration|NF registered|NF de-registered)"'),
    ("ue_failures",
     '{namespace="open5gs"} |~ "(?i)(PAYLOAD_NOT_FORWARDED|Registration reject|UE_IDENTITY|FIVEG_SERVICES|Cannot receive SBI)"'),
    ("scp_routing",
     '{namespace="open5gs"} |~ "(?i)(Connection timer expired|Connection refused|Failed to connect|response_handler.*failed)"'),
]


def to_unix(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def prom_range(expr: str, start: float, end: float, step: str = "15s") -> dict:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": expr, "start": start, "end": end, "step": step},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def jaeger_services() -> list[str]:
    r = requests.get(f"{JAEGER_URL}/api/services", timeout=10)
    r.raise_for_status()
    return r.json().get("data", [])


def jaeger_traces(service: str, start_us: int, end_us: int, limit: int = 500) -> dict:
    r = requests.get(
        f"{JAEGER_URL}/api/traces",
        params={"service": service, "start": start_us, "end": end_us, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def loki_range(query: str, start_ns: int, end_ns: int, limit: int = 5000) -> dict:
    r = requests.get(
        f"{LOKI_URL}/loki/api/v1/query_range",
        params={"query": query, "start": start_ns, "end": end_ns,
                "limit": limit, "direction": "forward"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def collect_k8s_events(out: Path, start_str: str, end_str: str):
    """Collect Kubernetes events for open5gs namespace within the phase window."""
    import subprocess
    from datetime import datetime, timezone
    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    end_dt   = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    try:
        raw = subprocess.check_output(
            ["kubectl", "get", "events", "-n", "open5gs",
             "-o", "json", "--sort-by=.lastTimestamp"],
            timeout=15, text=True
        )
        all_events = json.loads(raw).get("items", [])
        # Filter to events whose lastTimestamp falls within the phase window
        phase_events = []
        for ev in all_events:
            ts_str = ev.get("lastTimestamp") or ev.get("eventTime", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if start_dt <= ts <= end_dt:
                    phase_events.append({
                        "time":    ts_str,
                        "reason":  ev.get("reason", ""),
                        "message": ev.get("message", ""),
                        "object":  ev.get("involvedObject", {}).get("name", ""),
                        "kind":    ev.get("involvedObject", {}).get("kind", ""),
                        "type":    ev.get("type", ""),  # Normal / Warning
                        "count":   ev.get("count", 1),
                    })
            except (ValueError, TypeError):
                continue
        (out / "k8s_events.json").write_text(json.dumps(phase_events, indent=2))
        warnings = [e for e in phase_events if e["type"] == "Warning"]
        print(f"  k8s   events                                  {len(phase_events)} total, {len(warnings)} warnings")
        # Print notable events
        notable = {"OOMKilling", "Killing", "BackOff", "Failed", "Evicted", "Unhealthy"}
        for ev in phase_events:
            if ev["reason"] in notable:
                print(f"    [{ev['reason']}] {ev['object']}: {ev['message'][:80]}")
        return phase_events
    except Exception as e:
        print(f"  k8s   events                                  ERROR: {e}", file=sys.stderr)
        return []


def collect_nrf_registrations(out: Path):
    """Query NRF API for current registered NF instance counts."""
    import subprocess
    NF_TYPES = ["AMF", "AUSF", "UDM", "UDR", "SMF", "PCF", "NSSF", "SCP"]
    result = {}
    try:
        nrf_ip = subprocess.check_output(
            ["kubectl", "get", "pod", "-n", "open5gs",
             "-l", "app.kubernetes.io/name=nrf",
             "-o", "jsonpath={.items[0].status.podIP}"],
            timeout=10, text=True
        ).strip()
        nrf_pod = subprocess.check_output(
            ["kubectl", "get", "pod", "-n", "open5gs",
             "-l", "app.kubernetes.io/name=nrf",
             "-o", "jsonpath={.items[0].metadata.name}"],
            timeout=10, text=True
        ).strip()
        for nf in NF_TYPES:
            try:
                raw = subprocess.check_output(
                    ["kubectl", "exec", "-n", "open5gs", nrf_pod, "-c", "open5gs-nrf",
                     "--", "curl", "-s", "--http2-prior-knowledge",
                     f"http://{nrf_ip}:7777/nnrf-nfm/v1/nf-instances?nf-type={nf}"],
                    timeout=10, text=True
                )
                d = json.loads(raw)
                count = len(d.get("_links", {}).get("item", []))
                result[nf] = count
            except Exception:
                result[nf] = -1
    except Exception as e:
        result["error"] = str(e)
    (out / "nrf_registrations.json").write_text(json.dumps(result, indent=2))
    total = sum(v for v in result.values() if isinstance(v, int) and v >= 0)
    print(f"  nrf   registrations                           {result}")
    return result


def collect(fault: str, run: int, phase: str, start_str: str, end_str: str, output: str):
    start_ts = to_unix(start_str)
    end_ts   = to_unix(end_str)
    out = Path(output) / fault / f"run_{run:02d}" / phase
    out.mkdir(parents=True, exist_ok=True)

    print(f"[collect] {fault}  run={run}  phase={phase}")
    print(f"          {start_str} → {end_str}")

    # ── Prometheus ────────────────────────────────────────────────────────────
    prom_dir = out / "prometheus"
    prom_dir.mkdir(exist_ok=True)
    for name, expr in PROMETHEUS_METRICS:
        try:
            data = prom_range(expr, start_ts, end_ts)
            (prom_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
            n = len(data.get("data", {}).get("result", []))
            print(f"  prom  {name:40s} {n} series")
        except Exception as e:
            print(f"  prom  {name:40s} ERROR: {e}", file=sys.stderr)

    # ── Jaeger ────────────────────────────────────────────────────────────────
    jaeger_dir = out / "jaeger"
    jaeger_dir.mkdir(exist_ok=True)
    start_us = int(start_ts * 1_000_000)
    end_us   = int(end_ts   * 1_000_000)
    try:
        services = jaeger_services()
        (jaeger_dir / "services.json").write_text(json.dumps(services, indent=2))
        for svc in services:
            safe = svc.replace("/", "_").replace(" ", "_")
            try:
                traces = jaeger_traces(svc, start_us, end_us)
                n = len(traces.get("data", []))
                (jaeger_dir / f"traces_{safe}.json").write_text(json.dumps(traces, indent=2))
                print(f"  jaeger {svc:39s} {n} traces")
            except Exception as e:
                print(f"  jaeger {svc:39s} ERROR: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  jaeger services ERROR: {e}", file=sys.stderr)

    # ── Loki ──────────────────────────────────────────────────────────────────
    loki_dir = out / "loki"
    loki_dir.mkdir(exist_ok=True)
    start_ns = int(start_ts * 1_000_000_000)
    end_ns   = int(end_ts   * 1_000_000_000)
    for name, query in LOKI_QUERIES:
        try:
            data = loki_range(query, start_ns, end_ns)
            streams = data.get("data", {}).get("result", [])
            n = sum(len(s.get("values", [])) for s in streams)
            (loki_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
            print(f"  loki  {name:40s} {n} log lines")
        except Exception as e:
            print(f"  loki  {name:40s} ERROR: {e}", file=sys.stderr)

    # ── NRF registrations ─────────────────────────────────────────────────────
    collect_nrf_registrations(out)

    # ── Kubernetes Events (orchestration layer) ────────────────────────────────
    collect_k8s_events(out, start_str, end_str)

    print(f"[collect] → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect fault-injection telemetry")
    ap.add_argument("--fault",  required=True, help="Fault name, e.g. cpu-stress-amf")
    ap.add_argument("--run",    required=True, type=int, help="Run number (1-based)")
    ap.add_argument("--phase",  required=True, choices=["baseline", "fault", "recovery"])
    ap.add_argument("--start",  required=True, help="Window start, ISO-8601 UTC")
    ap.add_argument("--end",    required=True, help="Window end,   ISO-8601 UTC")
    ap.add_argument("--output", default="experiments/data", help="Root output directory")
    args = ap.parse_args()
    collect(args.fault, args.run, args.phase, args.start, args.end, args.output)
