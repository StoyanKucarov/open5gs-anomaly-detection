#!/usr/bin/env bash
# 04-scalability/run-str.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"

WINDOW_DURATION=600
STRATEGY="${1:-baseline}"
OUT_BASE="$DATA_DIR/04-scalability/$STRATEGY"

echo "============================================================"
echo " Phase 4: Scalability (Strategy: $STRATEGY)"
echo "============================================================"

check_cluster_ready
apply_log_strategy "$STRATEGY"

run_scenario() {
    local ue_count="$1" pattern="$2"
    local slug="ues-${ue_count}-${pattern}"
    local out_dir="$OUT_BASE/$slug"
    mkdir -p "$out_dir"

    echo ""
    echo "--- Scenario: $slug ---"
    log_experiment_start "04-scalability-$slug" "$out_dir"

    bash "$LIB_DIR/provision_ues.sh" "$ue_count"
    scale_ues "$ue_count"
    wait_for_pods_stable open5gs 120

    local start_size
    start_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    START_TS=$(now_ts)

    if [[ "$pattern" == "steady" ]]; then
        sleep_with_progress "$WINDOW_DURATION" "$slug"
    else
        sleep_with_progress "$WINDOW_DURATION" "$slug"
    fi

    END_TS=$(now_ts)
    
    local end_size
    end_size=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    local storage_delta=$((end_size - start_size))

    echo "[collect] Capturing Promtail CPU..."
    echo "timestamp,pod,cpu_m,mem_mi" > "$out_dir/promtail_overhead.csv"
    TS=$(date +%s)
    kubectl top pods -n monitoring --no-headers | grep "promtail" | while read -r name cpu mem; do
        echo "$TS,$name,$cpu,$mem" >> "$out_dir/promtail_overhead.csv"
    done

    echo "{\"ue_count\": $ue_count, \"strategy\": \"$STRATEGY\", \"storage_kb\": $storage_delta}" > "$out_dir/scalability_results.json"

    log_experiment_end "$out_dir"
}

run_scenario 10  steady
run_scenario 50  steady
run_scenario 100 steady
run_scenario 200 steady 