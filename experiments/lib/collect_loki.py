#!/usr/bin/env python3
"""
experiments/lib/collect_loki.py

Query Loki HTTP API for a fixed set of LogQL queries over a time window
and write each query result to a CSV file (one row per log line).

Usage:
    python3 collect_loki.py \
        --url http://127.0.0.1:3100 \
        --start <unix_ts> --end <unix_ts> \
        --out /path/to/output/dir
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# (output_filename, LogQL query)
LOKI_QUERIES = [
    ("all.csv",
     '{namespace="open5gs"}'),
    ("errors.csv",
     '{namespace="open5gs"} |~ "(?i)(error|exception|refused|failed|fatal|oom|killed)"'),
    ("nrf_lifecycle.csv",
     '{namespace="open5gs"} |~ "(?i)(heartbeat|de-registered|Retry registration|NF registered|NF de-registered)"'),
    ("ue_failures.csv",
     '{namespace="open5gs"} |~ "(?i)(PAYLOAD_NOT_FORWARDED|Registration reject|UE_IDENTITY|FIVEG_SERVICES|Cannot receive SBI)"'),
    ("scp_routing.csv",
     '{namespace="open5gs"} |~ "(?i)(Connection timer expired|Connection refused|Failed to connect|response_handler.*failed)"'),
]

LIMIT = 500000  # max lines per query (must match max_entries_limit_per_query
                # patched in cluster-start.sh — server cap is the binding one)


def query_range(url: str, query: str, start_ns: int, end_ns: int) -> dict:
    params = urllib.parse.urlencode({
        "query":     query,
        "start":     start_ns,
        "end":       end_ns,
        "limit":     LIMIT,
        "direction": "forward",
    })
    req_url = f"{url}/loki/api/v1/query_range?{params}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req_url, timeout=120) as resp:
                return json.load(resp)
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] Loki request failed: {e}", file=sys.stderr)
                return {}
            time.sleep(2)
    return {}


def streams_to_rows(data: dict) -> list:
    rows = []
    result = data.get("data", {}).get("result", []) or []
    for stream in result:
        labels = stream.get("stream", {}) or {}
        pod = labels.get("pod", "")
        container = labels.get("container", "")
        app = labels.get("app", "") or labels.get("app_kubernetes_io_name", "")
        for ts_ns, line in stream.get("values", []):
            rows.append({
                "timestamp_ns": ts_ns,
                "pod":          pod,
                "container":    container,
                "app":          app,
                "line":         line,
            })
    return rows


def write_csv(rows: list, out_path: Path):
    if not rows:
        # Still write an empty file with a header so downstream code doesn't
        # have to special-case missing files.
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp_ns", "pod", "container", "app", "line"]
            )
            writer.writeheader()
        print(f"  [loki] {out_path.name}: 0 lines")
        return
    fieldnames = ["timestamp_ns", "pod", "container", "app", "line"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [loki] {out_path.name}: {len(rows)} lines")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3100")
    parser.add_argument("--start", type=int, required=True, help="Window start, unix seconds")
    parser.add_argument("--end",   type=int, required=True, help="Window end,   unix seconds")
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ns = args.start * 1_000_000_000
    end_ns   = args.end   * 1_000_000_000

    for fname, query in LOKI_QUERIES:
        data = query_range(args.url, query, start_ns, end_ns)
        rows = streams_to_rows(data)
        write_csv(rows, out_dir / fname)
        if len(rows) >= LIMIT:
            print(f"  [loki] WARNING: {fname} hit {LIMIT}-line cap — may be truncated",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
