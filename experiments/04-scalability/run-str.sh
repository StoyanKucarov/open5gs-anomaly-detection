#!/usr/bin/env bash
# 04-scalability/run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"

WINDOW_DURATION=600

STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")

echo "============================================================"
echo " Phase 4: Scalability Analysis (Multi-Strategy)"
echo "============================================================"

check_cluster_ready

run_scenario() {
    local strat="$1" ue_count="$2" pattern="$3"
    local slug="ues-${ue_count}-${pattern}"
    local out_dir="$DATA_DIR/04-scalability/$strat/$slug"
    mkdir -p "$out_dir"

    rreset_experiment_state "$strat"

    echo ""
    echo "------------------------------------------------------------"
    echo " Scenario: $slug | Strategy: $strat"
    echo "------------------------------------------------------------"
    log_experiment_start "04-scalability-$slug" "$out_dir"
    
    bash "$LIB_DIR/provision_ues.sh" "$ue_count"
    scale_ues "$ue_count"
    wait_for_pods_stable open5gs 120

    local start_size
    start_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    
    echo "[run] Measuring for ${WINDOW_DURATION}s..."
    sleep_with_progress "$WINDOW_DURATION" "$slug"

    local end_size
    end_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    local storage_delta=$((end_size - start_size))

    echo "[collect] Capturing Promtail CPU..."
    echo "timestamp,pod,cpu_m,mem_mi" > "$out_dir/promtail_overhead.csv"
    local TS=$(date +%s)
    kubectl top pods -n monitoring --no-headers | grep "promtail" | while read -r name cpu mem; do
        echo "$TS,$name,$cpu,$mem" >> "$out_dir/promtail_overhead.csv"
    done

    echo "{\"ue_count\": $ue_count, \"strategy\": \"$strat\", \"storage_kb\": $storage_delta}" > "$out_dir/scalability_results.json"

    log_experiment_end "$out_dir"
}
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "############################################################"
    echo " Testing Scalability for Strategy: $STRATEGY"
    echo "############################################################"
    
    apply_log_strategy "$STRATEGY"
    
    kubectl rollout restart daemonset promtail -n monitoring
    kubectl rollout status daemonset promtail -n monitoring --timeout=2m

    run_scenario "$STRATEGY" 10  steady
    run_scenario "$STRATEGY" 50  steady
    run_scenario "$STRATEGY" 100 steady
    run_scenario "$STRATEGY" 200 steady
    
    echo "[cooldown] 60s between strategies..."
    sleep 60
done

apply_log_strategy "baseline"
echo "============================================================"
echo " Phase 4 complete. Data in: $DATA_DIR/04-scalability/"
echo "============================================================"