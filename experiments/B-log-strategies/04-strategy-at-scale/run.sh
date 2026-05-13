#!/usr/bin/env bash
# B-log-strategies/04-strategy-at-scale/run.sh
#
# Phase 4: Scalability — focuses on Loki storage and Promtail overhead.
#
# Scenarios:
#    10, 50, 100, 200 UEs — steady-state (10 min each)
#    50, 100 UEs          — bursty (10 min each)
#
# Output:
#    data/experiments/04-scalability/{strategy}/ues-{N}-{pattern}/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"
source "$SCRIPT_DIR/../../lib/apply_strategy.sh"

WINDOW_DURATION=600   # 10 minutes per scenario

STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")
if [[ $# -gt 0 ]]; then
    STRATEGIES=("$@")
fi

echo "============================================================"
echo " Phase 4: Scalability Analysis (Loki/Promtail Focus)"
echo "============================================================"

check_cluster_ready

run_scenario() {
    local strat="$1" ue_count="$2" pattern="$3"
    local slug="ues-${ue_count}-${pattern}"
    local out_dir="$DATA_DIR/04-scalability/$strat/$slug"
    mkdir -p "$out_dir"

    echo ""
    echo "--- Scenario: $slug | Strategy: $strat ---"
    log_experiment_start "04-scalability-$strat-$slug" "$out_dir"

    echo "[setup] Provisioning $ue_count subscribers..."
    bash "$LIB_DIR/provision_ues.sh" "$ue_count"
    scale_ues "$ue_count"
    wait_for_pods_stable open5gs 120

    echo "[wait] Stabilising for 30s..."
    sleep 30

    local start_size
    start_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    START_TS=$(now_ts)

    if [[ "$pattern" == "steady" ]]; then
        echo "[run] Steady-state for ${WINDOW_DURATION}s..."
        sleep_with_progress "$WINDOW_DURATION" "$slug"
    else
        echo "[run] Bursty for ${WINDOW_DURATION}s..."
        (
            end_ts=$(( $(date +%s) + WINDOW_DURATION ))
            while [[ $(date +%s) -lt $end_ts ]]; do
                scale_ues "$ue_count"
                sleep 30
                scale_ues 5
                sleep 30
            done
        ) &
        BURSTY_PID=$!
        sleep_with_progress "$WINDOW_DURATION" "$slug"
        wait "$BURSTY_PID" 2>/dev/null || true
    fi

    END_TS=$(now_ts)
    
    local end_size
    end_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    local storage_delta=$((end_size - start_size))

    echo "[collect] Capturing Promtail resource overhead..."
    echo "timestamp,pod,cpu,memory" > "$out_dir/promtail_overhead.csv"
    kubectl top pods -n monitoring | grep promtail | while read -r p c m; do
        echo "$(date +%s),$p,$c,$m" >> "$out_dir/promtail_overhead.csv"
    done

    python3 -c "
import json
meta = {
    'strategy': '$strat',
    'ue_count': $ue_count,
    'pattern': '$pattern',
    'storage_kb_delta': $storage_delta,
    'window_duration_s': $WINDOW_DURATION,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
}
with open('$out_dir/scenario_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    log_experiment_end "$out_dir"
    echo "[done] $slug complete. Storage Delta: ${storage_delta} KB"
}

# Main Loop
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "############################################################"
    echo " Strategy: $STRATEGY"
    echo "############################################################"

    reset_experiment_state "$STRATEGY" 10
    
    apply_log_strategy "$STRATEGY"

    for COUNT in 10 50 100 200; do
        run_scenario "$STRATEGY" "$COUNT" "steady"
    done
    
    for COUNT in 50 100; do
        run_scenario "$STRATEGY" "$COUNT" "bursty"
    done
    
    echo "[cooldown] 60s between strategy shifts..."
    sleep 60
done

apply_log_strategy "baseline"

echo "============================================================"
echo " Phase 4 complete. Data in: $DATA_DIR/04-scalability/"
echo "============================================================"
