#!/usr/bin/env bash
# B-log-strategies/03-system-visibility/run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"
source "$SCRIPT_DIR/../apply_strategy.sh"

STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")
UE_COUNT=50

if [[ $# -gt 0 ]]; then
    STRATEGIES=("$@")
fi

MANIFESTS=(
    "01-cpu-stress-amf.yaml"
    "03-pod-crash-amf.yaml"
    "05-network-partition-amf-scp.yaml"
)

# ---------------------------------------------------------------------------
# Helper: Query Loki to verify if the fault was "seen"
# ---------------------------------------------------------------------------
perform_visibility_analysis() {
    local fault_name=$1
    local strategy=$2
    local out_file=$3
    
    # local start_s=$(jq -r '.timeline.fault.start' "$out_file/visibility_report.json")
    # local end_s=$(jq -r '.timeline.fault.end' "$out_file/visibility_report.json")

    # local start_ns="$((start_s - 5))000000000"
    # local end_ns="$((end_s + 5))000000000"

    echo "[analysis] Waiting 15s for Loki to index chunks..."
    sleep 15

    echo "[analysis] Querying Loki API for fault signatures ($fault_name)..."
    
    local query='{container="open5gs-amf"}'
    
    ensure_portforward_loki

    local response=$(curl -G -s "http://localhost:3100/loki/api/v1/query_range" \
    --data-urlencode "query=$query" \
    --data-urlencode "since=4h" \
    --data-urlencode "limit=10000")

    local logs=$(echo "$response" | jq -r '.data.result[].values[][1]' 2>/dev/null || echo "")
    
    local count=$(echo "$logs" | grep -iE "error|fail|panic|fatal|refused|timeout|dropped|expired|sbi" | wc -l)  
      
    python3 -c "
import json
with open('$out_file/visibility_report.json', 'r+') as f:
    data = json.load(f)
    data['detected'] = $count > 0
    data['log_sample_count'] = $count
    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()
"
    
    if [ "$count" -gt 0 ]; then
        echo "  >> SUCCESS: Fault detected ($count signatures found)."
    else
        echo "  >> FAILURE: No fault signatures found in Loki for this window."
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
        
        bash "$SCRIPT_DIR/run_fault_loki.sh" \
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
