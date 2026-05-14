#!/usr/bin/env bash
# C-fault-detection/run_all.sh
#
# Phase 3: Fault detection — runs all 22 faults in sequence.
#
# Each fault: PRE phase -> fault phase -> POST phase, with full
# Prometheus + Jaeger + Loki + K8s events + NRF API + RTT collection.
#
# Durations are env-overridable:
#   PRE_DURATION    (default 120s)
#   FAULT_DURATION  (default 300s)
#   POST_DURATION   (default 120s)
# Boyan's main pipeline uses 600/300/300:
#   PRE_DURATION=600 FAULT_DURATION=300 POST_DURATION=300 bash run_all.sh
#
# Usage:
#   bash run_all.sh [--from N]          # skip faults 1..N-1
#   bash run_all.sh --only 19,20        # run only the listed fault numbers

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/reset_workload.sh"

FROM=1
ONLY=""
while [[ $# -gt 0 ]]; do
    case "${1:-}" in
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) shift ;;
    esac
done

PRE_DURATION="${PRE_DURATION:-120}"
FAULT_DURATION="${FAULT_DURATION:-300}"
POST_DURATION="${POST_DURATION:-120}"

# Soft-reset the Open5GS+UERANSIM workload between every fault so each run
# starts from a clean baseline (no stale PFCP, ghost NF regs, broken UEs).
# Costs ~2-3 min per fault. Set RESET_BETWEEN_FAULTS=0 to disable.
RESET_BETWEEN_FAULTS="${RESET_BETWEEN_FAULTS:-1}"

UE_COUNT="${UE_COUNT:-50}"
OUT_BASE="$DATA_DIR/C-fault-detection"

echo "============================================================"
echo " Phase 3: Fault detection (22 faults)"
echo " durations: pre=${PRE_DURATION}s  fault=${FAULT_DURATION}s  post=${POST_DURATION}s"
echo "============================================================"

check_cluster_ready

echo "[setup] Provisioning $UE_COUNT subscribers..."
bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"
scale_ues "$UE_COUNT"
wait_for_pods_stable open5gs 120

ensure_portforward_prometheus
ensure_portforward_jaeger
ensure_portforward_loki

run_fault_experiment() {
    local num="$1" name="$2" manifest="$3"
    if [[ -n "$ONLY" ]] && ! echo ",$ONLY," | grep -q ",$num,"; then
        echo "[skip] Fault $num ($name)"
        return
    fi
    if [[ -z "$ONLY" && $num -lt $FROM ]]; then
        echo "[skip] Fault $num ($name)"
        return
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo " Fault $num: $name"
    echo "------------------------------------------------------------"
    if [[ "$RESET_BETWEEN_FAULTS" == "1" ]]; then
        reset_workload "$UE_COUNT"
    fi
    bash "$LIB_DIR/run_fault.sh" \
        --name        "$name" \
        --manifest    "$CHAOS_DIR/$manifest" \
        --out         "$OUT_BASE/$name" \
        --pre-duration   "$PRE_DURATION" \
        --fault-duration "$FAULT_DURATION" \
        --post-duration  "$POST_DURATION" \
        --step           "5s"
    echo "[cooldown] 60s between faults..."
    sleep 60
}

# Slug == chaos YAML basename, so lib/hooks/<slug>.sh resolves automatically.
run_fault_experiment 1  "01-cpu-stress-amf"                        "01-cpu-stress-amf.yaml"
run_fault_experiment 2  "02-memory-pressure-upf"                   "02-memory-pressure-upf.yaml"
run_fault_experiment 3  "03-pod-crash-amf"                         "03-pod-crash-amf.yaml"
run_fault_experiment 4  "04-network-delay-gnb-amf"                 "04-network-delay-gnb-amf.yaml"
run_fault_experiment 5  "05-network-partition-amf-scp"             "05-network-partition-amf-scp.yaml"
run_fault_experiment 6  "06-packet-loss-upf"                       "06-packet-loss-upf.yaml"
run_fault_experiment 7  "07-pod-crash-smf"                         "07-pod-crash-smf.yaml"
run_fault_experiment 8  "08-cpu-stress-scp"                        "08-cpu-stress-scp.yaml"
run_fault_experiment 9  "09-network-delay-nrf"                     "09-network-delay-nrf.yaml"
run_fault_experiment 10 "10-pfcp-session-establishment-flood-upf"  "10-pfcp-session-establishment-flood-upf.yaml"
run_fault_experiment 11 "11-pfcp-session-deletion-upf"             "11-pfcp-session-deletion-upf.yaml"
run_fault_experiment 12 "12-pfcp-session-modification-drop-upf"    "12-pfcp-session-modification-drop-upf.yaml"
run_fault_experiment 13 "13-pfcp-session-modification-dupl-upf"    "13-pfcp-session-modification-dupl-upf.yaml"
run_fault_experiment 14 "14-upf-infrastructure-packet-loss"        "14-upf-infrastructure-packet-loss.yaml"
run_fault_experiment 15 "15-nrf-cascade"                           "15-nrf-cascade.yaml"
run_fault_experiment 16 "16-cpu-stress-ausf"                       "16-cpu-stress-ausf.yaml"
run_fault_experiment 17 "17-network-delay-scp"                     "17-network-delay-scp.yaml"
run_fault_experiment 18 "18-cpu-stress-nrf"                        "18-cpu-stress-nrf.yaml"
run_fault_experiment 19 "19-udm-pod-crash"                         "19-udm-pod-crash.yaml"
run_fault_experiment 20 "20-mongodb-pod-kill"                      "20-mongodb-pod-kill.yaml"
run_fault_experiment 21 "21-n2-partition-amf-gnb"                  "21-n2-partition-amf-gnb.yaml"
run_fault_experiment 22 "22-memory-pressure-amf"                   "22-memory-pressure-amf.yaml"

echo ""
echo "============================================================"
echo " Phase 3 complete. Data in: $OUT_BASE"
echo "============================================================"
