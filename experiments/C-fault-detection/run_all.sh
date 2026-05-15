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

# Reset between faults is disabled: the post-phase (300s) + cooldown (60s) +
# pre-phase (600s) = 960s of recovery time is sufficient for all faults to
# self-heal via Kubernetes restart policies and PFCP/NRF heartbeats.
RESET_BETWEEN_FAULTS="${RESET_BETWEEN_FAULTS:-0}"

UE_COUNT="${UE_COUNT:-10}"
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

restart_ues() {
    echo "  [ue-restart] Restarting UDM to clear stale SDM subscriptions..."
    kubectl rollout restart deployment/open5gs-udm -n open5gs
    kubectl rollout status deployment/open5gs-udm -n open5gs --timeout=60s

    echo "  [ue-restart] Restarting UE pods for clean tunnel state..."
    kubectl rollout restart deployment/ueransim-gnb-ues deployment/ueransim-ues -n open5gs
    kubectl rollout status deployment/ueransim-gnb-ues -n open5gs --timeout=90s
    kubectl rollout status deployment/ueransim-ues -n open5gs --timeout=90s
    # Wait for uesimtun0 to appear on at least one pod
    local deadline=$(($(date +%s) + 90))
    until kubectl exec -n open5gs deployment/ueransim-gnb-ues -- \
            ip link show uesimtun0 >/dev/null 2>&1 || \
          kubectl exec -n open5gs deployment/ueransim-ues -- \
            ip link show uesimtun0 >/dev/null 2>&1; do
        [[ $(date +%s) -gt $deadline ]] && { echo "  [ue-restart] WARNING: uesimtun0 not ready after 90s"; break; }
        sleep 3
    done
    local gnb_tuns; gnb_tuns=$(kubectl exec -n open5gs deployment/ueransim-gnb-ues -- \
        ip link show 2>/dev/null | grep -c uesimtun || echo 0)
    local ues_tuns; ues_tuns=$(kubectl exec -n open5gs deployment/ueransim-ues -- \
        ip link show 2>/dev/null | grep -c uesimtun || echo 0)
    echo "  [ue-restart] Tunnels ready: gnb-ues=${gnb_tuns} ueransim-ues=${ues_tuns}"
}

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
    else
        restart_ues
    fi
    if ! bash "$LIB_DIR/health_check.sh" "pre-${name}" "$OUT_BASE/${name}/health_pre.json"; then
        echo "[ABORT] pre-fault health check failed for fault $num ($name)" >&2
        echo "[ABORT] Fix the cluster and re-run with: --from $num" >&2
        exit 1
    fi
    bash "$LIB_DIR/run_fault.sh" \
        --name        "$name" \
        --manifest    "$CHAOS_DIR/$manifest" \
        --out         "$OUT_BASE/$name" \
        --pre-duration   "$PRE_DURATION" \
        --fault-duration "$FAULT_DURATION" \
        --post-duration  "$POST_DURATION" \
        --step           "5s"
    bash "$LIB_DIR/health_check.sh" "post-${name}" "$OUT_BASE/${name}/health_post.json"
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
