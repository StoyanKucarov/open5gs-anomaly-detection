"""
analysis/config.py

Central configuration: paths, constants, plot styling.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "data" / "experiments"
FIGURES_DIR = REPO_ROOT / "analysis" / "RQ1" / "figures"
TABLES_DIR = REPO_ROOT / "analysis" / "RQ1" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment data directories
# ---------------------------------------------------------------------------
BASELINE_DIR = DATA_ROOT / "00-baseline"
PROM_OVERHEAD_DIR = DATA_ROOT / "01-overhead-prometheus"
EBPF_OVERHEAD_DIR = DATA_ROOT / "02-overhead-ebpf"
FAULT_DIR = DATA_ROOT / "03-fault-detection"
SCALABILITY_DIR = DATA_ROOT / "04-scalability"

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------
PROM_INTERVALS = ["1s", "5s", "15s"]          # scrape interval slugs
EBPF_SAMPLING_RATES = ["100pct", "50pct", "10pct"]  # Beyla sampling rate slugs

SCALABILITY_SCENARIOS = [
    ("10",  "steady"),
    ("50",  "steady"),
    ("100", "steady"),
    ("200", "steady"),
    ("50",  "bursty"),
    ("100", "bursty"),
]

# 200 UEs is the practical limit on a single-host kind cluster.
# The research plan targets 500 UEs, but that exceeds available memory
# (~16 GB) when running Open5GS + UERANSIM + full monitoring stack.
SCALABILITY_NOTE = (
    "Scalability experiments cover 10, 50, 100, 200 UEs (steady) and "
    "50, 100 UEs (bursty). The research plan target of 500 UEs was not "
    "feasible on the single-host kind cluster due to memory constraints."
)

ALL_FAULTS = [
    "01-cpu-stress-amf",
    "02-memory-pressure-upf",
    "03-pod-crash-amf",
    "04-network-delay-gnb-amf",
    "05-network-partition-amf-scp",
    "06-packet-loss-upf",
    "07-pod-crash-smf",
    "08-cpu-stress-scp",
    "09-network-delay-nrf",
    "10-pfcp-session-establishment-flood-upf",
    "11-pfcp-session-deletion-upf",
    "12-pfcp-session-modification-drop-upf",
    "13-pfcp-session-modification-dupl-upf",
    "14-upf-infrastructure-packet-loss",
    "15-nrf-cascade",
    "16-cpu-stress-ausf",
    "17-network-delay-scp",
    "18-cpu-stress-nrf",
    "19-udm-pod-crash",
    "20-mongodb-pod-kill",
    "21-n2-partition-amf-gnb",
    "22-memory-pressure-amf",
]

# Human-readable fault labels for tables/figures
FAULT_LABELS = {
    "01-cpu-stress-amf":                       "CPU stress – AMF",
    "02-memory-pressure-upf":                  "Mem pressure – UPF",
    "03-pod-crash-amf":                        "Pod crash – AMF",
    "04-network-delay-gnb-amf":                "Net delay – gNB↔AMF",
    "05-network-partition-amf-scp":            "Net partition – AMF↔SCP",
    "06-packet-loss-upf":                      "Packet loss – UPF",
    "07-pod-crash-smf":                        "Pod crash – SMF",
    "08-cpu-stress-scp":                       "CPU stress – SCP",
    "09-network-delay-nrf":                    "Net delay – NRF",
    "10-pfcp-session-establishment-flood-upf": "PFCP flood – UPF",
    "11-pfcp-session-deletion-upf":            "PFCP deletion – UPF",
    "12-pfcp-session-modification-drop-upf":   "PFCP mod drop – UPF",
    "13-pfcp-session-modification-dupl-upf":   "PFCP mod dup – UPF",
    "14-upf-infrastructure-packet-loss":       "Infra pkt loss – UPF",
    "15-nrf-cascade":                          "NRF cascade failure",
    "16-cpu-stress-ausf":                      "CPU stress – AUSF",
    "17-network-delay-scp":                    "Net delay – SCP",
    "18-cpu-stress-nrf":                       "CPU stress – NRF",
    "19-udm-pod-crash":                        "Pod crash – UDM",
    "20-mongodb-pod-kill":                     "MongoDB pod kill",
    "21-n2-partition-amf-gnb":                 "N2 partition – AMF↔gNB",
    "22-memory-pressure-amf":                  "Mem pressure – AMF",
}

# Fault categories for grouping in figures
FAULT_CATEGORIES = {
    "Resource stress": [
        "01-cpu-stress-amf", "08-cpu-stress-scp", "16-cpu-stress-ausf",
        "18-cpu-stress-nrf", "02-memory-pressure-upf", "22-memory-pressure-amf",
    ],
    "Pod crash": [
        "03-pod-crash-amf", "07-pod-crash-smf", "19-udm-pod-crash",
        "20-mongodb-pod-kill",
    ],
    "Network fault": [
        "04-network-delay-gnb-amf", "05-network-partition-amf-scp",
        "06-packet-loss-upf", "09-network-delay-nrf", "14-upf-infrastructure-packet-loss",
        "17-network-delay-scp", "21-n2-partition-amf-gnb",
    ],
    "PFCP / control-plane": [
        "10-pfcp-session-establishment-flood-upf", "11-pfcp-session-deletion-upf",
        "12-pfcp-session-modification-drop-upf", "13-pfcp-session-modification-dupl-upf",
        "15-nrf-cascade",
    ],
}

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
# Prometheus: flag a fault as detected if any metric deviates > SIGMA_THRESHOLD
# standard deviations from the pre-fault window mean.
SIGMA_THRESHOLD = 2.0

# eBPF/Beyla: flag as detected if p95 latency increases by more than this
# fraction relative to pre-fault baseline.
BEYLA_LATENCY_INCREASE_THRESHOLD = 0.50   # 50%

# eBPF/Beyla: flag as detected if error rate increases by more than this
# absolute percentage-point delta.
BEYLA_ERROR_RATE_DELTA_THRESHOLD = 0.10   # 10pp

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
PALETTE = {
    "prometheus": "#E6522C",   # Prometheus orange
    "ebpf":       "#00ADD8",   # Beyla/Go cyan
    "baseline":   "#888888",
    "steady":     "#2196F3",
    "bursty":     "#FF9800",
}

FIGURE_DPI = 150
FIGURE_SIZE_SINGLE = (7, 4)
FIGURE_SIZE_WIDE = (10, 4)
FIGURE_SIZE_TALL = (7, 8)
FIGURE_SIZE_SQUARE = (6, 6)

FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
