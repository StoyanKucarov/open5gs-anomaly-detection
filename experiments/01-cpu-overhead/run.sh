#!/usr/bin/env bash
# 01-cpu-overhead/run.sh
#
# Phase 1: Log Strategy Processing Overhead
# Measures the CPU/Memory cost of different Promtail strategies.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"

WINDOW_DURATION=600 
UE_COUNT=50
SAMPLE_INTERVAL=10    # Collection frequency
STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")

# Allow overriding strategies via CLI (e.g., ./run.sh denum)
if [[ $# -gt 0 ]]; then
    STRATEGIES=("$@")
fi

OUT_BASE="$DATA_DIR/01-cpu-overhead"

echo "============================================================"
echo " Phase 1: Log Strategy Processing Overhead"
echo " Strategies: ${STRATEGIES[*]}"
echo "============================================================"

check_cluster_ready

echo "[setup] Scaling down observability stack..."
kubectl scale statefulset -n monitoring \
    prometheus-kube-prom-kube-prometheus-prometheus --replicas=0 2>/dev/null || true
kubectl patch daemonset beyla -n open5gs \
    --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/nodeSelector","value":{"non-existing":"true"}}]' \
    2>/dev/null || true
echo "[setup] Observability scaled down"

for STRATEGY in "${STRATEGIES[@]}"; do
    OUT_DIR="$OUT_BASE/$STRATEGY"
    mkdir -p "$OUT_DIR"

    reset_experiment_state "$STRATEGY" "$UE_COUNT"

    echo ""
    echo "--- Strategy: $STRATEGY ---"
    log_experiment_start "log-overhead-$STRATEGY" "$OUT_DIR"

    apply_log_strategy "$STRATEGY"

    echo "[wait] Letting Promtail stabilize for 60s..."
    sleep 60

    START_TS=$(now_ts)
    END_TIME=$(( START_TS + WINDOW_DURATION ))
    
    echo "timestamp,pod,cpu_m,mem_mi" > "$OUT_DIR/promtail_overhead.csv"
    echo "[run] Collecting metrics for ${WINDOW_DURATION}s..."

    while [[ $(now_ts) -lt $END_TIME ]]; do
        TS=$(now_ts)
        kubectl top pods -n monitoring --no-headers 2>/dev/null | grep "promtail" | while read -r name cpu mem; do
            echo "$TS,$name,$cpu,$mem" >> "$OUT_DIR/promtail_overhead.csv"
        done || true
        
        echo -n "."
        sleep "$SAMPLE_INTERVAL"
    done

    echo ""
    END_TS=$(now_ts)

    python3 -c "
import json
meta = {
    'strategy': '$STRATEGY',
    'window_duration_s': $WINDOW_DURATION,
    'ue_count': $UE_COUNT,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
    'sample_interval': $SAMPLE_INTERVAL
}
with open('$OUT_DIR/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

    log_experiment_end "$OUT_DIR"
    echo "[done] Strategy $STRATEGY complete → $OUT_DIR"

    # 5. Cooldown between experiments
    if [[ "$STRATEGY" != "${STRATEGIES[-1]}" ]]; then
        echo "[cooldown] 60s between strategies..."
        sleep 60
    fi
done

echo "============================================================"
echo " Phase 1 complete. Data in: $OUT_BASE"
echo "============================================================"