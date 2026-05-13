#!/usr/bin/env python3
"""
experiments/lib/collect_prometheus.py

Query Prometheus HTTP API for a set of standard metrics over a time window
and write each metric to a CSV file.

Usage:
    python3 collect_prometheus.py \
        --url http://127.0.0.1:9090 \
        --start <unix_ts> --end <unix_ts> \
        --step 5s \
        --out /path/to/output/dir \
        [--extra-metrics "label:query:filename.csv" ...]
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Standard metrics collected for every experiment window
# ---------------------------------------------------------------------------
STANDARD_METRICS = [
    # (output_filename, promql_query)
    ("container_cpu_usage_rate.csv",
     'rate(container_cpu_usage_seconds_total{namespace="open5gs",container!=""}[2m])'),
    ("container_memory_working_set_bytes.csv",
     'container_memory_working_set_bytes{namespace="open5gs",container!=""}'),
    # cAdvisor in newer kube-prometheus-stack dropped *_seconds_total in favour
    # of period counters. Throttle ratio = throttled-periods / total-periods,
    # bounded 0..1; spikes to ~1 when a container saturates its CPU limit.
    ("container_cpu_throttled_rate.csv",
     'rate(container_cpu_cfs_throttled_periods_total{namespace="open5gs",container!=""}[2m]) '
     '/ rate(container_cpu_cfs_periods_total{namespace="open5gs",container!=""}[2m])'),
    ("pod_restarts.csv",
     'kube_pod_container_status_restarts_total{namespace="open5gs"}'),
    ("monitoring_cpu_usage_rate.csv",
     'rate(container_cpu_usage_seconds_total{namespace="monitoring",container!=""}[2m])'),
    ("monitoring_memory_working_set.csv",
     'container_memory_working_set_bytes{namespace="monitoring",container!=""}'),
    ("node_cpu_usage.csv",
     'rate(node_cpu_seconds_total{mode!="idle"}[2m])'),
    ("node_memory_available.csv",
     'node_memory_MemAvailable_bytes'),
]


def query_range(url: str, query: str, start: int, end: int, step: str) -> list:
    params = urllib.parse.urlencode({
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    })
    req_url = f"{url}/api/v1/query_range?{params}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req_url, timeout=30) as resp:
                data = json.load(resp)
            if data.get("status") != "success":
                print(f"  [WARN] Prometheus query failed: {data.get('error', 'unknown')}", file=sys.stderr)
                return []
            return data["data"]["result"]
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] Query error after 3 attempts: {e}", file=sys.stderr)
                return []
            time.sleep(2)
    return []


def results_to_csv(results: list, out_path: Path):
    if not results:
        return
    rows = []
    for series in results:
        labels = series["metric"]
        for ts, val in series["values"]:
            row = {"timestamp": ts, "value": val}
            row.update(labels)
            rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [prom] {out_path.name}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9090")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", default="5s")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra-metrics", nargs="*", default=[],
                        help="label:query:filename triples")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(STANDARD_METRICS)
    for extra in args.extra_metrics:
        parts = extra.split(":", 2)
        if len(parts) == 3:
            _, query, fname = parts
            metrics.append((fname, query))

    for fname, query in metrics:
        results = query_range(args.url, query, args.start, args.end, args.step)
        results_to_csv(results, out_dir / fname)


if __name__ == "__main__":
    main()
