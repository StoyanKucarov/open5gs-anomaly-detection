#!/usr/bin/env python3
"""
models/metrics/data_loader.py

Loads Prometheus metric CSVs into MetricRecord objects for anomaly detection.

Each MetricRecord represents one 5-second sample for one experiment phase.
The `values` array holds all 45 metrics aligned to FEATURE_NAMES order.
Missing values (scrape gaps) are forward-filled within each phase.

Aggregation across pods/containers per timestamp:
  - Rate metrics          → SUM  (total system rate)
  - Duration/latency      → MEAN (average response time)
  - Node-level gauges     → MEAN (per-node average)
  - Everything else       → SUM  (total count/active sessions)

Labelling: during-phase = 1, pre/post = 0  (phase-level, consistent with logs).

Public API
----------
load_all(data_dir)  → {"train": [MetricRecord, ...], "test": [MetricRecord, ...]}
load_experiment(slug, data_dir) → list[MetricRecord]
"""

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from lib import EXPERIMENTS

_DEFAULT_DATA = ROOT / "data" / "experiments" / "C-fault-detection"

# metric catalogue: (csv_filename_stem, short_name, aggregation)
_METRICS: list[tuple[str, str, str]] = [
    ("beyla_http_server_request_rate",  "http_server_req_rate",   "sum"),
    ("beyla_http_client_request_rate",  "http_client_req_rate",   "sum"),
    ("beyla_http_server_duration",      "http_server_duration",   "mean"),
    ("beyla_http_client_duration",      "http_client_duration",   "mean"),
    ("beyla_http_server_error_rate",    "http_server_err_rate",   "sum"),
    ("beyla_http_client_error_rate",    "http_client_err_rate",   "sum"),
    ("container_cpu_usage_rate",        "cpu_usage",              "sum"),
    ("container_cpu_throttled_rate",    "cpu_throttled",          "sum"),
    ("node_cpu_usage",                  "cpu_node",               "mean"),
    ("beyla_cpu_usage_rate",            "cpu_beyla",              "sum"),
    ("monitoring_cpu_usage_rate",       "cpu_monitoring",         "sum"),
    ("container_memory_working_set_bytes", "mem_container",       "sum"),
    ("node_memory_available",           "mem_node_available",     "mean"),
    ("beyla_memory_working_set",        "mem_beyla",              "sum"),
    ("monitoring_memory_working_set",   "mem_monitoring",         "sum"),
    ("network_tx_bytes_rate",           "net_tx",                 "sum"),
    ("network_rx_bytes_rate",           "net_rx",                 "sum"),
    ("open5gs_amf_reg_init_req",        "amf_reg_req",            "sum"),
    ("open5gs_amf_reg_init_succ",       "amf_reg_succ",           "sum"),
    ("open5gs_amf_reg_init_fail",       "amf_reg_fail",           "sum"),
    ("open5gs_amf_auth_fail",           "amf_auth_fail",          "sum"),
    ("open5gs_amf_auth_reject",         "amf_auth_reject",        "sum"),
    ("open5gs_amf_sessions",            "amf_sessions",           "sum"),
    ("open5gs_amf_registered_subscribers", "amf_subscribers",     "sum"),
    ("open5gs_amf_ran_ue_count",        "amf_ran_ue_count",       "sum"),
    ("open5gs_amf_gnb_count",           "amf_gnb_count",          "sum"),
    ("open5gs_amf_paging_req",          "amf_paging_req",         "sum"),
    ("open5gs_pfcp_sessions_active",    "pfcp_sessions",          "sum"),
    ("open5gs_pfcp_peers_active",       "pfcp_peers",             "sum"),
    ("open5gs_smf_pdu_session_req",     "smf_pdu_req",            "sum"),
    ("open5gs_smf_pdu_session_succ",    "smf_pdu_succ",           "sum"),
    ("open5gs_smf_session_nbr",         "smf_sessions",           "sum"),
    ("open5gs_smf_ues_active",          "smf_ues",                "sum"),
    ("open5gs_smf_bearers_active",      "smf_bearers",            "sum"),
    ("open5gs_smf_qos_flow_nbr",        "smf_qos_flows",          "sum"),
    ("open5gs_smf_n4_session_estab",    "smf_n4_estab",           "sum"),
    ("open5gs_smf_n4_session_report",   "smf_n4_report",          "sum"),
    ("open5gs_smf_n4_session_report_succ", "smf_n4_report_succ",  "sum"),
    ("open5gs_upf_session_nbr",         "upf_sessions",           "sum"),
    ("open5gs_upf_qos_flows",           "upf_qos_flows",          "sum"),
    ("open5gs_upf_n4_session_estab",    "upf_n4_estab",           "sum"),
    ("open5gs_gtp_node_failed",         "gtp_failed",             "sum"),
]
# pod_running, pod_ready, pod_restarts excluded: cumulative counter behaviour
# and pod recreation semantics make these uninformative as anomaly signals.

FEATURE_NAMES: list[str] = [name for _, name, _ in _METRICS]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class MetricRecord:
    slug:        str
    fault_type:  str
    fault_class: str
    phase:       Literal["pre", "during", "post"]
    timestamp:   int        # unix seconds
    label:       int        # 0 = normal, 1 = anomalous (during-phase)
    values:      np.ndarray  # shape (N_FEATURES,), aligned to FEATURE_NAMES


def _load_csv_aggregated(path: Path, agg: str) -> dict[int, float]:
    """Load one metric CSV and aggregate across pods/containers per timestamp."""
    if not path.exists():
        return {}
    buckets: dict[int, list[float]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ts  = int(row["timestamp"])
                val = float(row["value"])
                buckets[ts].append(val)
            except (KeyError, ValueError):
                pass
    if agg == "sum":
        return {ts: sum(vals) for ts, vals in buckets.items()}
    else:  # mean
        return {ts: sum(vals) / len(vals) for ts, vals in buckets.items()}


def _load_phase(exp_dir: Path, phase: str,
                slug: str, fault_type: str, fault_class: str
                ) -> list[MetricRecord]:
    prom_dir   = exp_dir / "prometheus" / phase
    phase_label = 1 if phase == "during" else 0

    series: list[dict[int, float]] = []
    for fname, _name, agg in _METRICS:
        series.append(_load_csv_aggregated(prom_dir / f"{fname}.csv", agg))

    if not any(series):
        return []

    all_ts = sorted({ts for s in series for ts in s})
    if not all_ts:
        return []

    matrix = np.full((len(all_ts), N_FEATURES), np.nan, dtype=np.float32)
    for j, s in enumerate(series):
        for i, ts in enumerate(all_ts):
            if ts in s:
                matrix[i, j] = s[ts]
    for j in range(N_FEATURES):
        last = 0.0
        for i in range(len(all_ts)):
            if not np.isnan(matrix[i, j]):
                last = matrix[i, j]
            else:
                matrix[i, j] = last

    records = []
    for i, ts in enumerate(all_ts):
        records.append(MetricRecord(
            slug=slug, fault_type=fault_type, fault_class=fault_class,
            phase=phase, timestamp=ts, label=phase_label,
            values=matrix[i],
        ))
    return records


def load_experiment(slug: str, data_dir: Path | None = None) -> list[MetricRecord]:
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    meta = next(((ft, fc) for s, ft, _nf, fc in EXPERIMENTS if s == slug),
                ("unknown", "unknown"))
    fault_type, fault_class = meta
    exp_dir = data_dir / slug
    records: list[MetricRecord] = []
    for phase in ("pre", "during", "post"):
        records.extend(_load_phase(exp_dir, phase, slug, fault_type, fault_class))
    records.sort(key=lambda r: r.timestamp)
    return records


def base_slug(slug: str) -> str:
    """Strip run tag, e.g. '01-cpu-stress-amf__r3' → '01-cpu-stress-amf'."""
    return slug.split("__r")[0]


def load_all(data_dir: Path | None = None) -> dict[str, list[MetricRecord]]:
    """
    Returns {"train": pre-phase records, "test": during+post records}.
    Only includes slugs whose data directory exists.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
    train: list[MetricRecord] = []
    test:  list[MetricRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        exp_dir = data_dir / slug
        if not exp_dir.is_dir():
            continue
        train.extend(_load_phase(exp_dir, "pre",    slug, ft, fc))
        test.extend( _load_phase(exp_dir, "during", slug, ft, fc))
        test.extend( _load_phase(exp_dir, "post",   slug, ft, fc))

    train.sort(key=lambda r: r.timestamp)
    test.sort( key=lambda r: r.timestamp)
    return {"train": train, "test": test}


def load_multi(data_dirs: list[Path]) -> dict[str, list[MetricRecord]]:
    """
    Load from [run1, run2, run3, run4], preferring run2 over run1 per fault.

    For each fault:
      - If run2 has it  →  use run2 (primary, no tag) + run3 (__r3) + run4 (__r4)
      - If run2 missing →  use run1 (primary, no tag) + run3 (__r3) + run4 (__r4)
    """
    run1, run2, run3, run4 = (Path(d) for d in data_dirs[:4])
    train: list[MetricRecord] = []
    test:  list[MetricRecord] = []

    for slug, ft, _nf, fc in EXPERIMENTS:
        primary = run2 if (run2 / slug).is_dir() else run1
        sources = []
        if (primary / slug).is_dir():
            sources.append((primary / slug, ""))
        if run3.is_dir() and (run3 / slug).is_dir():
            sources.append((run3 / slug, "__r3"))
        if run4.is_dir() and (run4 / slug).is_dir():
            sources.append((run4 / slug, "__r4"))

        for exp_dir, tag in sources:
            tagged = slug + tag
            train.extend(_load_phase(exp_dir, "pre",    tagged, ft, fc))
            test.extend( _load_phase(exp_dir, "during", tagged, ft, fc))
            test.extend( _load_phase(exp_dir, "post",   tagged, ft, fc))

    train.sort(key=lambda r: r.timestamp)
    test.sort( key=lambda r: r.timestamp)

    n_base   = len({base_slug(r.slug) for r in train})
    n_tagged = len({r.slug for r in train})
    r2_used  = sum(1 for slug, *_ in EXPERIMENTS if (run2 / slug).is_dir())
    print(f"[data_loader] Multi-run: {n_base} faults — "
          f"{r2_used} from run2+run3+run4, {22-r2_used} from run1+run3+run4 "
          f"({n_tagged} run×fault combos), "
          f"{len(train):,} train / {len(test):,} test records")
    return {"train": train, "test": test}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    args = ap.parse_args()

    print(f"Loading from {args.data} ...")
    data  = load_all(args.data)
    train, test = data["train"], data["test"]
    n_anom = sum(r.label for r in test)
    print(f"Train: {len(train):,} records (all normal)")
    print(f"Test:  {len(test):,} records  "
          f"(anomalous={n_anom:,} [{100*n_anom/max(len(test),1):.1f}%])")
    print(f"Features ({N_FEATURES}): {FEATURE_NAMES[:5]} ...")
