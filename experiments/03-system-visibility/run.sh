#!/usr/bin/env bash
# 03-system-visibility/run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"

STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")
UE_COUNT=50

MANIFESTS=(
    "01-cpu-stress-amf.yaml"
    "03-pod-crash-amf.yaml"
    "05-network-partition-amf-nrf.yaml"
)

# ---------------------------------------------------------------------------
# Helper: Query Loki to verify if the fault was "seen"
# ---------------------------------------------------------------------------
perform_visibility_analysis() {
    local fault_name=$1
    local strategy=$2
    local out_file=$3
    
    local start_ts=$(jq -r '.timeline.fault.start' "$out_file/visibility_report.json")
    local end_ts=$(jq -r '.timeline.fault.end' "$out_file/visibility_report.json")

    echo "[analysis] Querying Loki for fault signatures ($fault_name)..."
    
    local logs=$(kubectl exec -n monitoring svc/loki -- \
        logcli query "{app='amf'}" --from-posix "$start_ts" --to-posix "$end_ts" --limit 100)
    
    local count=$(echo "$logs" | grep -iE "error|fail|panic|fatal|refused" | wc -l)
    
    python3 -c "
import json
with open('$out_file/visibility_report.json', 'r+') as f:
    data = json.load(f)
    data['detected'] = $count > 0
    data['log_sample_count'] = $count
    f.seek(0)
    json.dump(data, f, indent=2)
"
    
    if [ "$count" -gt 0 ]; then
        echo "  >> SUCCESS: Fault detected in reduced logs."
    else
        echo "  >> FAILURE: Fault obscured by reduction strategy!"
    fi
}

echo "============================================================"
echo " Phase 3: Multi-Strategy Visibility Analysis"
echo "============================================================"

check_cluster_ready

for STRATEGY in "${STRATEGIES[@]}"; do
    OUT_BASE="$DATA_DIR/03-visibility/$STRATEGY"

    reset_experiment_state "$STRATEGY" "$UE_COUNT"

    apply_log_strategy "$STRATEGY"
    
    echo "[setup] Scaling UEs to $UE_COUNT and stabilizing..."
    scale_ues "$UE_COUNT"
    wait_for_pods_stable open5gs 60

    for i in "${!MANIFESTS[@]}"; do
        YAML="${MANIFESTS[$i]}"
        FAULT_NAME="${YAML%.*}"
        FAULT_OUT="$OUT_BASE/$FAULT_NAME"

        echo ""
        echo ">>> Injecting Fault: $FAULT_NAME (Strategy: $STRATEGY)"
        
        bash "$LIB_DIR/run_fault_loki.sh" \
            --name "$FAULT_NAME" \
            --strategy "$STRATEGY" \
            --manifest "$CHAOS_DIR/$YAML" \
            --out "$FAULT_OUT" \
            --pre-duration 60 \
            --fault-duration 180 \
            --post-duration 60

        perform_visibility_analysis "$FAULT_NAME" "$STRATEGY" "$FAULT_OUT"

        echo "[cooldown] 30s between fault injections..."
        sleep 30
    done

    echo "[done] Finished all faults for $STRATEGY"
    echo "------------------------------------------------------------"
done

apply_log_strategy "baseline"
echo "Phase 3 complete."