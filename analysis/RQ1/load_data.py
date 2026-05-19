"""
analysis/load_data.py

Data loaders for every experiment output schema.
All functions return pandas DataFrames or plain dicts/lists.
Missing files return empty DataFrames / empty dicts with a warning.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASELINE_DIR, PROM_OVERHEAD_DIR, EBPF_OVERHEAD_DIR, BOTH_OVERHEAD_DIR,
    FAULT_DIR, SCALABILITY_DIR, ALL_FAULTS,
    PROM_INTERVALS, EBPF_SAMPLING_RATES, SCALABILITY_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return pd.DataFrame()


def _read_json(path: Path):
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return {}


def _parse_kubectl_top_cpu(value: str) -> float:
    """Convert kubectl top CPU string ('123m' or '1') to millicores float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s.endswith("m"):
        return float(s[:-1])
    try:
        return float(s) * 1000
    except ValueError:
        return np.nan


def _parse_kubectl_top_mem(value: str) -> float:
    """Convert kubectl top memory string ('128Mi', '1Gi', '512Ki') to MiB float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s.endswith("Ki"):
        return float(s[:-2]) / 1024
    if s.endswith("Mi"):
        return float(s[:-2])
    if s.endswith("Gi"):
        return float(s[:-2]) * 1024
    try:
        return float(s) / (1024 * 1024)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Baseline (A-01) — kubectl top CSVs
# ---------------------------------------------------------------------------

def load_baseline_top(phase: str = "steady") -> dict:
    """
    Load kubectl top snapshots from the no-telemetry baseline.

    Parameters
    ----------
    phase : 'steady' or 'bursty'

    Returns
    -------
    dict with keys:
        'pods'  : DataFrame(timestamp, pod, namespace, cpu_m, mem_mi)
        'nodes' : DataFrame(timestamp, node, cpu_m, mem_mi)
        'meta'  : dict
    """
    base = BASELINE_DIR / phase / "prometheus"
    pods = _read_csv(base / "pod_top.csv")
    nodes = _read_csv(base / "node_top.csv")
    meta = _read_json(base / "meta.json")

    if not pods.empty:
        pods["cpu_m"] = pods["cpu_cores_m"].apply(_parse_kubectl_top_cpu)
        pods["mem_mi"] = pods["memory_mi"].apply(_parse_kubectl_top_mem)

    if not nodes.empty:
        nodes["cpu_m"] = nodes["cpu_cores_m"].apply(_parse_kubectl_top_cpu)
        nodes["mem_mi"] = nodes["memory_mi"].apply(_parse_kubectl_top_mem)

    return {"pods": pods, "nodes": nodes, "meta": meta}


# ---------------------------------------------------------------------------
# Standard Prometheus CSV loader
# ---------------------------------------------------------------------------

def load_prometheus_csv(directory: Path, filename: str) -> pd.DataFrame:
    """
    Load a single standard Prometheus CSV (timestamp, value, ...labels).
    Converts timestamp to datetime and value to float.
    """
    df = _read_csv(directory / filename)
    if df.empty:
        return df
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_all_prometheus(directory: Path) -> dict:
    """
    Load all standard Prometheus CSVs from a directory.
    Returns dict mapping filename stem → DataFrame.
    """
    files = [
        "container_cpu_usage_rate.csv",
        "container_memory_working_set_bytes.csv",
        "container_cpu_throttled_rate.csv",
        "pod_restarts.csv",
        "monitoring_cpu_usage_rate.csv",
        "monitoring_memory_working_set.csv",
        "node_cpu_usage.csv",
        "node_memory_available.csv",
        "pod_ready.csv",
        "pod_running.csv",
        "network_rx_bytes_rate.csv",
        "network_tx_bytes_rate.csv",
        "beyla_http_server_duration.csv",
        "beyla_http_client_duration.csv",
        "beyla_http_server_request_rate.csv",
        "beyla_http_server_error_rate.csv",
        "beyla_http_client_request_rate.csv",
        "beyla_http_client_error_rate.csv",
        "beyla_cpu_usage_rate.csv",
        "beyla_memory_working_set.csv",
    ]
    return {Path(f).stem: load_prometheus_csv(directory, f) for f in files}


# ---------------------------------------------------------------------------
# Prometheus overhead (A-02)
# ---------------------------------------------------------------------------

def load_prometheus_overhead(interval: str) -> dict:
    """
    Load data for one Prometheus scrape interval condition.

    Parameters
    ----------
    interval : '1s', '5s', or '15s'

    Returns
    -------
    dict with keys:
        'prometheus' : dict of DataFrames (all standard metrics)
        'self'       : dict with keys 'head_chunks', 'active_appenders', 'wal_writes'
        'meta'       : dict
    """
    base = PROM_OVERHEAD_DIR / f"interval-{interval}"
    prom = load_all_prometheus(base)
    self_dir = base / "self_metrics"
    self_metrics = {
        "head_chunks":       load_prometheus_csv(self_dir, "prom_head_chunks.csv"),
        "active_appenders":  load_prometheus_csv(self_dir, "prom_active_appenders.csv"),
        "wal_writes":        load_prometheus_csv(self_dir, "prom_wal_writes.csv"),
    }
    meta = _read_json(base / "interval_meta.json")
    return {"prometheus": prom, "self": self_metrics, "meta": meta}


def load_all_prometheus_overhead() -> dict:
    """Returns dict mapping interval string → load_prometheus_overhead result."""
    return {iv: load_prometheus_overhead(iv) for iv in PROM_INTERVALS}


# ---------------------------------------------------------------------------
# eBPF/Beyla overhead (A-03)
# ---------------------------------------------------------------------------

def load_ebpf_overhead(sampling_rate: str) -> dict:
    """
    Load data for one Beyla sampling rate condition.

    Parameters
    ----------
    sampling_rate : '100pct', '50pct', or '10pct'

    Returns
    -------
    dict with keys:
        'prometheus'   : dict of DataFrames (all standard metrics)
        'beyla_cpu'    : DataFrame
        'beyla_mem'    : DataFrame
        'jaeger_spans' : DataFrame
        'jaeger_summary': dict
        'meta'         : dict
    """
    base = EBPF_OVERHEAD_DIR / f"sampling-{sampling_rate}"
    prom = load_all_prometheus(base)
    beyla_dir = base / "beyla_metrics"
    jaeger_dir = base / "jaeger"
    return {
        "prometheus":    prom,
        "beyla_cpu":     load_prometheus_csv(beyla_dir, "beyla_cpu.csv"),
        "beyla_mem":     load_prometheus_csv(beyla_dir, "beyla_mem.csv"),
        "jaeger_spans":  _read_csv(jaeger_dir / "spans_flat.csv"),
        "jaeger_summary": _read_json(jaeger_dir / "summary.json"),
        "meta":          _read_json(base / "rate_meta.json"),
    }


def load_all_ebpf_overhead() -> dict:
    """Returns dict mapping sampling rate string → load_ebpf_overhead result."""
    return {sr: load_ebpf_overhead(sr) for sr in EBPF_SAMPLING_RATES}


# ---------------------------------------------------------------------------
# Both stacks overhead (A-05)
# ---------------------------------------------------------------------------

def load_both_overhead() -> dict:
    """
    Load data for the combined Prometheus + Beyla overhead experiment.

    Returns
    -------
    dict with keys:
        'prometheus'    : dict of DataFrames (all standard metrics)
        'beyla_cpu'     : DataFrame
        'beyla_mem'     : DataFrame
        'jaeger_spans'  : DataFrame
        'jaeger_summary': dict
        'meta'          : dict
    """
    base = BOTH_OVERHEAD_DIR
    prom = load_all_prometheus(base)
    jaeger_dir = base / "jaeger"
    return {
        "prometheus":    prom,
        "beyla_cpu":     load_prometheus_csv(base, "beyla_cpu_usage_rate.csv"),
        "beyla_mem":     load_prometheus_csv(base, "beyla_memory_working_set.csv"),
        "jaeger_spans":  _read_csv(jaeger_dir / "spans_flat.csv"),
        "jaeger_summary": _read_json(jaeger_dir / "summary.json"),
        "meta":          _read_json(base / "both_meta.json"),
    }


# ---------------------------------------------------------------------------
# Scalability (A-04)
# ---------------------------------------------------------------------------

def load_scalability_scenario(ue_count: str, pattern: str) -> dict:
    """
    Load data for one scalability scenario.

    Parameters
    ----------
    ue_count : '10', '50', '100', '200'
    pattern  : 'steady' or 'bursty'

    Returns
    -------
    dict with keys:
        'prometheus'    : dict of DataFrames
        'beyla_cpu'     : DataFrame
        'beyla_mem'     : DataFrame
        'jaeger_spans'  : DataFrame
        'jaeger_summary': dict
        'meta'          : dict
    """
    slug = f"ues-{ue_count}-{pattern}"
    base = SCALABILITY_DIR / slug
    prom = load_all_prometheus(base)
    jaeger_dir = base / "jaeger"
    return {
        "prometheus":     prom,
        "beyla_cpu":      load_prometheus_csv(base, "beyla_cpu_usage_rate.csv"),
        "beyla_mem":      load_prometheus_csv(base, "beyla_memory_working_set.csv"),
        "jaeger_spans":   _read_csv(jaeger_dir / "spans_flat.csv"),
        "jaeger_summary": _read_json(jaeger_dir / "summary.json"),
        "meta":           _read_json(base / "scenario_meta.json"),
    }


def load_all_scalability() -> dict:
    """Returns dict mapping (ue_count, pattern) → load_scalability_scenario result."""
    return {
        (ue, pat): load_scalability_scenario(ue, pat)
        for ue, pat in SCALABILITY_SCENARIOS
    }


# ---------------------------------------------------------------------------
# Fault detection (C)
# ---------------------------------------------------------------------------

def load_fault_phase(fault_name: str, phase: str) -> dict:
    """
    Load all signals for one fault × phase combination.

    Parameters
    ----------
    fault_name : e.g. '01-cpu-stress-amf'
    phase      : 'pre', 'during', or 'post'

    Returns
    -------
    dict with keys:
        'prometheus'    : dict of DataFrames
        'jaeger_spans'  : DataFrame
        'jaeger_summary': dict
        'loki'          : dict with keys all/errors/nrf_lifecycle/ue_failures/scp_routing
        'events'        : list of dicts
        'nrf'           : dict
    """
    base = FAULT_DIR / fault_name
    prom_dir = base / "prometheus" / phase
    jaeger_dir = base / "jaeger" / phase
    loki_dir = base / "loki" / phase
    events_dir = base / "events" / phase
    nrf_dir = base / "nrf" / phase


    loki = {
        "all":           _read_csv(loki_dir / "all.csv"),
        "errors":        _read_csv(loki_dir / "errors.csv"),
        "nrf_lifecycle": _read_csv(loki_dir / "nrf_lifecycle.csv"),
        "ue_failures":   _read_csv(loki_dir / "ue_failures.csv"),
        "scp_routing":   _read_csv(loki_dir / "scp_routing.csv"),
    }

    events_raw = _read_json(events_dir / "k8s_events.json")
    events = events_raw if isinstance(events_raw, list) else []

    return {
        "prometheus":     load_all_prometheus(prom_dir),
        "jaeger_spans":   _read_csv(jaeger_dir / "spans_flat.csv"),
        "jaeger_summary": _read_json(jaeger_dir / "summary.json"),
        "loki":           loki,
        "events":         events,
        "nrf":            _read_json(nrf_dir / "nrf_registrations.json"),
    }


def load_fault(fault_name: str) -> dict:
    """
    Load all three phases for a fault.

    Returns
    -------
    dict with keys 'pre', 'during', 'post', 'timeline'
    """
    return {
        "pre":      load_fault_phase(fault_name, "pre"),
        "during":   load_fault_phase(fault_name, "during"),
        "post":     load_fault_phase(fault_name, "post"),
        "timeline": _read_json(FAULT_DIR / fault_name / "timeline.json"),
    }


def load_all_faults() -> dict:
    """Returns dict mapping fault_name → load_fault result."""
    return {f: load_fault(f) for f in ALL_FAULTS}


# ---------------------------------------------------------------------------
# Convenience aggregation helpers
# ---------------------------------------------------------------------------

def mean_cpu_millicores(prom_dict: dict, namespace_key: str = "container_cpu_usage_rate") -> float:
    """
    Return mean CPU usage in millicores across all containers in the given
    Prometheus CPU DataFrame (rate in cores → multiply by 1000).
    """
    df = prom_dict.get(namespace_key, pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    return float(df["value"].mean() * 1000)


def mean_memory_mib(prom_dict: dict, key: str = "container_memory_working_set_bytes") -> float:
    """Return mean memory usage in MiB."""
    df = prom_dict.get(key, pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    return float(df["value"].mean() / (1024 ** 2))


def beyla_mean_cpu_millicores(prom_dict: dict) -> float:
    # Prefer a dedicated beyla_cpu_usage_rate file; fall back to the beyla
    # container rows inside container_cpu_usage_rate (which is always present).
    dedicated = prom_dict.get("beyla_cpu_usage_rate", pd.DataFrame())
    if not dedicated.empty and "value" in dedicated.columns:
        return float(dedicated["value"].mean() * 1000)
    df = prom_dict.get("container_cpu_usage_rate", pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    beyla_rows = df[df.get("container", pd.Series(dtype=str)) == "beyla"] if "container" in df.columns else pd.DataFrame()
    if beyla_rows.empty:
        return np.nan
    return float(beyla_rows["value"].mean() * 1000)


def beyla_mean_memory_mib(prom_dict: dict) -> float:
    # Prefer a dedicated beyla_memory_working_set file; fall back to beyla
    # container rows inside container_memory_working_set_bytes.
    dedicated = prom_dict.get("beyla_memory_working_set", pd.DataFrame())
    if not dedicated.empty and "value" in dedicated.columns:
        return float(dedicated["value"].mean() / (1024 ** 2))
    df = prom_dict.get("container_memory_working_set_bytes", pd.DataFrame())
    if df.empty or "value" not in df.columns:
        return np.nan
    beyla_rows = df[df["container"] == "beyla"] if "container" in df.columns else pd.DataFrame()
    if beyla_rows.empty:
        return np.nan
    return float(beyla_rows["value"].mean() / (1024 ** 2))


def monitoring_mean_cpu_millicores(prom_dict: dict) -> float:
    return mean_cpu_millicores(prom_dict, "monitoring_cpu_usage_rate")


def monitoring_mean_memory_mib(prom_dict: dict) -> float:
    return mean_memory_mib(prom_dict, "monitoring_memory_working_set")


def per_pod_mean_cpu(prom_dict: dict) -> pd.Series:
    """Return mean CPU (millicores) per pod from container_cpu_usage_rate."""
    df = prom_dict.get("container_cpu_usage_rate", pd.DataFrame())
    if df.empty or "pod" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("pod")["value"].mean() * 1000


def per_pod_mean_memory(prom_dict: dict) -> pd.Series:
    """Return mean memory (MiB) per pod from container_memory_working_set_bytes."""
    df = prom_dict.get("container_memory_working_set_bytes", pd.DataFrame())
    if df.empty or "pod" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("pod")["value"].mean() / (1024 ** 2)
