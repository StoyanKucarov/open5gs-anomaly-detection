#!/usr/bin/env bash
# 03-system-visibility/run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"

STRATEGY="${1:-baseline}"
FROM=1
if [[ "${2:-}" == "--from" && -n "${3:-}" ]]; then
    FROM="$3"
fi

UE_COUNT=50
OUT_BASE="$DATA_DIR/03-visibility/$STRATEGY"

MANIFESTS=(
    "01-cpu-stress-amf.yaml"
    "02-memory-pressure-upf.yaml"
    "03-pod-crash-amf.yaml"
    "05-network-partition-amf-nrf.yaml"
    "06-dependency-failure-nrf.yaml"
    "08-network-delay-nrf.yaml"
    "07-packet-loss-upf.yaml"
    "08-cpu-stress-scp.yaml"
)

echo "============================================================"
echo " Phase 3: Visibility Analysis | Strategy: $STRATEGY"
echo "============================================================"

check_cluster_ready

apply_log_strategy "$STRATEGY"

echo "[setup] Provisioning $UE_COUNT subscribers..."
bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"
scale_ues "$UE_COUNT"
wait_for_pods_stable open5gs 120

for i in "${!MANIFESTS[@]}"; do
    NUM=$((i + 1))
    YAML="${MANIFESTS[$i]}"
    FAULT_NAME="${YAML%.*}"

    if [[ $NUM -lt $FROM ]]; then
        echo "[skip] Fault $NUM ($FAULT_NAME)"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " Fault $NUM: $FAULT_NAME"
    echo "------------------------------------------------------------"

    bash "$LIB_DIR/run_fault_loki.sh" \
        --name "$FAULT_NAME" \
        --strategy "$STRATEGY" \
        --manifest "$CHAOS_DIR/$YAML" \
        --out "$OUT_BASE/$FAULT_NAME" \
        --pre-duration 120 \
        --fault-duration 300 \
        --post-duration 120

    echo "[cooldown] 60s between faults..."
    sleep 60
done

apply_log_strategy "baseline"

echo "============================================================"
echo " Phase 3 complete. Data in: $OUT_BASE"
echo "============================================================"