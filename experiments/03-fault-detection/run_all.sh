#!/usr/bin/env bash
# 03-fault-detection/run_all.sh
#
# Phase 3: Fault detection — runs all 8 faults in sequence.
#
# Each fault: 2 min pre → 5 min fault → 2 min post
# Total: ~8 faults × ~11 min = ~90 min + cooldowns
#
# Usage:
#   bash run_all.sh [--from N]   # skip faults 1..N-1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

FROM=1
if [[ "${1:-}" == "--from" && -n "${2:-}" ]]; then
    FROM="$2"
fi

UE_COUNT=50
OUT_BASE="$DATA_DIR/03-fault-detection"

echo "============================================================"
echo " Phase 3: Fault detection (8 faults)"
echo "============================================================"

check_cluster_ready

echo "[setup] Provisioning $UE_COUNT subscribers..."
bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"
scale_ues "$UE_COUNT"
wait_for_pods_stable open5gs 120

ensure_portforward_prometheus
ensure_portforward_jaeger

run_fault_experiment() {
    local num="$1" name="$2" manifest="$3"
    if [[ $num -lt $FROM ]]; then
        echo "[skip] Fault $num ($name)"
        return
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo " Fault $num: $name"
    echo "------------------------------------------------------------"
    bash "$LIB_DIR/run_fault.sh" \
        --name        "$name" \
        --manifest    "$CHAOS_DIR/$manifest" \
        --out         "$OUT_BASE/$name" \
        --pre-duration   120 \
        --fault-duration 300 \
        --post-duration  120 \
        --step           "5s"
    echo "[cooldown] 60s between faults..."
    sleep 60
}

run_fault_experiment 1 "01-cpu-stress-amf"            "01-cpu-stress-amf.yaml"
run_fault_experiment 2 "02-memory-pressure-upf"       "02-memory-pressure-upf.yaml"
run_fault_experiment 3 "03-pod-crash-amf"             "03-pod-crash-amf.yaml"
run_fault_experiment 4 "04-network-partition-amf-scp" "05-network-partition-amf-nrf.yaml"
run_fault_experiment 5 "05-dependency-failure-nrf"    "06-dependency-failure-nrf.yaml"
run_fault_experiment 6 "06-network-delay-nrf"         "08-network-delay-nrf.yaml"
run_fault_experiment 7 "07-packet-loss-upf"           "07-packet-loss-upf.yaml"
run_fault_experiment 8 "08-cpu-stress-scp"            "08-cpu-stress-scp.yaml"
run_fault_experiment 9 "03-pfcp-session-establishment-flood-upf"        "03-pfcp-session-establishment-flood-upf.yaml"
run_fault_experiment 10 "04-pfcp-session-deletion-upf"                  "04-pfcp-session-deletion-upf.yaml"
run_fault_experiment 11 "05-pfcp-session-modification-drop-upf"         "05-pfcp-session-modification-drop-upf.yaml"
run_fault_experiment 12 "07-pfcp-session-modification-dupl-upf"         "07-pfcp-session-modification-dupl-upf.yaml"
run_fault_experiment 13 "09-amf-internal-fault-pod-kill"                "09-amf-internal-fault-pod-kill.yaml"
run_fault_experiment 14 "10-smf-internal-fault-pod-kill"                "10-smf-internal-fault-pod-kill.yaml"
run_fault_experiment 15 "11-upf-infrastructure-packet-loss"             "11-upf-infrastructure-packet-loss.yaml"



echo ""
echo "============================================================"
echo " Phase 3 complete. Data in: $OUT_BASE"
echo "============================================================"
