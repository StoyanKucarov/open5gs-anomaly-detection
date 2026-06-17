#!/usr/bin/env python3
"""
analysis/lib.py — shared data loading utilities for all analysis scripts.
"""

import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Experiment registry — all 22 faults
# (slug, fault_type, target_nf, fault_class)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    ("01-cpu-stress-amf",                       "cpu_stress",      "amf",     "resource_exhaustion"),
    ("02-memory-pressure-upf",                  "mem_pressure",    "upf",     "resource_exhaustion"),
    ("03-pod-crash-amf",                        "pod_crash",       "amf",     "component_failure"),
    ("04-network-delay-gnb-amf",                "network_delay",   "amf",     "network_delay"),
    ("05-network-partition-amf-scp",            "net_partition",   "amf",     "network_partition"),
    ("06-packet-loss-upf",                      "packet_loss",     "upf",     "network_partition"),
    ("07-pod-crash-smf",                        "pod_crash",       "smf",     "component_failure"),
    ("08-cpu-stress-scp",                       "cpu_stress",      "scp",     "resource_exhaustion"),
    ("09-network-delay-nrf",                    "network_delay",   "nrf",     "network_delay"),
    ("10-pfcp-session-establishment-flood-upf", "pfcp_flood",      "upf",     "protocol_attack"),
    ("11-pfcp-session-deletion-upf",            "pfcp_deletion",   "upf",     "protocol_attack"),
    ("12-pfcp-session-modification-drop-upf",   "pfcp_drop",       "upf",     "protocol_attack"),
    ("13-pfcp-session-modification-dupl-upf",   "pfcp_dupl",       "upf",     "protocol_attack"),
    ("14-upf-infrastructure-packet-loss",       "packet_loss",     "upf",     "network_partition"),
    ("15-nrf-cascade",                          "pod_crash",       "nrf",     "component_failure"),
    ("16-cpu-stress-ausf",                      "cpu_stress",      "ausf",    "resource_exhaustion"),
    ("17-network-delay-scp",                    "network_delay",   "scp",     "network_delay"),
    ("18-cpu-stress-nrf",                       "cpu_stress",      "nrf",     "resource_exhaustion"),
    ("19-udm-pod-crash",                        "pod_crash",       "udm",     "component_failure"),
    ("20-mongodb-pod-kill",                     "pod_crash",       "mongodb", "component_failure"),
    ("21-n2-partition-amf-gnb",                 "net_partition",   "amf",     "network_partition"),
    ("22-memory-pressure-amf",                  "mem_pressure",    "amf",     "resource_exhaustion"),
]

# NF short name → pod name prefix
NF_PODS = {
    "amf":     "open5gs-amf",
    "smf":     "open5gs-smf",
    "upf":     "open5gs-upf",
    "nrf":     "open5gs-nrf",
    "scp":     "open5gs-scp",
    "ausf":    "open5gs-ausf",
    "udm":     "open5gs-udm",
    "udr":     "open5gs-udr",
    "pcf":     "open5gs-pcf",
    "bsf":     "open5gs-bsf",
    "nssf":    "open5gs-nssf",
    "sepp":    "open5gs-sepp",
    "mongodb": "open5gs-mongodb",
}

# Signal layer membership
SIGNAL_LAYERS = {
    # Infrastructure
    "cpu_spike":          "infrastructure",
    "mem_spike":          "infrastructure",
    "cpu_throttle":       "infrastructure",
    "network_rx_anomaly": "infrastructure",
    "network_tx_anomaly": "infrastructure",
    "node_cpu_spike":     "infrastructure",
    # Orchestration
    "pod_restart":        "orchestration",
    "pod_ready_drop":     "orchestration",
    "k8s_warning":        "orchestration",
    "nrf_drop":           "orchestration",
    # Application
    "error_logs":         "application",
    "ue_failures":        "application",
    "nrf_lifecycle":      "application",
    "scp_routing":        "application",
    "trace_errors":       "application",
    "trace_latency":      "application",
    "beyla_error_rate":   "application",
    "beyla_latency":      "application",
    "rtt_spike":          "application",
    # Open5GS native (5GC protocol layer)
    "amf_sub_drop":       "native",
    "amf_auth_fail":      "native",
    "pfcp_session_drop":  "native",
    "gtp_data_anomaly":   "native",
    "smf_pdu_fail":       "native",
}

SIGNALS_INFRA   = [s for s, l in SIGNAL_LAYERS.items() if l == "infrastructure"]
SIGNALS_ORCH    = [s for s, l in SIGNAL_LAYERS.items() if l == "orchestration"]
SIGNALS_APP     = [s for s, l in SIGNAL_LAYERS.items() if l == "application"]
SIGNALS_NATIVE  = [s for s, l in SIGNAL_LAYERS.items() if l == "native"]
ALL_SIGNALS     = SIGNALS_INFRA + SIGNALS_ORCH + SIGNALS_APP + SIGNALS_NATIVE


# ---------------------------------------------------------------------------
# CSV / JSON loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_timeline(exp_dir: Path) -> dict:
    tl = load_json(exp_dir / "timeline.json")
    if tl is None:
        return {}
    return tl


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def floats(rows: list, col: str = "value") -> list:
    out = []
    for r in rows:
        try:
            v = float(r[col])
            if math.isfinite(v):
                out.append(v)
        except (KeyError, ValueError, TypeError):
            pass
    return out


def safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def safe_max(vals: list) -> float:
    return max(vals) if vals else 0.0


def safe_min(vals: list) -> float:
    return min(vals) if vals else 0.0


def percentile(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


def entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v > 0:
            p = v / total
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def prom_phase(exp_dir: Path, metric: str, phase: str) -> list:
    return load_csv(exp_dir / "prometheus" / phase / metric)


def prom_vals_filtered(exp_dir: Path, metric: str, phase: str,
                        pod_contains: str = None, container: str = None) -> list:
    rows = prom_phase(exp_dir, metric, phase)
    if pod_contains:
        rows = [r for r in rows if pod_contains in r.get("pod", "")]
    if container:
        rows = [r for r in rows if r.get("container") == container]
    return floats(rows)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def iso_to_unix(s: str) -> float:
    """Parse RFC3339/ISO8601 string to unix seconds."""
    try:
        s = s.rstrip("Z")
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0
