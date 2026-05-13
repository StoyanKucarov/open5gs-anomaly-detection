#!/usr/bin/env bash
# 02-log-storage/run.sh
#
# Phase 2: Log Strategy Storage Efficiency
# Measures the actual disk growth in Loki for different reduction strategies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apply_strategy.sh"


WINDOW_DURATION=600
UE_COUNT=50
STRATEGIES=("baseline" "compression" "denum" "preprocessing" "dynamic-logging")

if [[ $# -gt 0 ]]; then
    STRATEGIES=("$@")
fi

OUT_BASE="$DATA_DIR/02-log-storage"

echo "============================================================"
echo " Phase 2: Log Strategy Storage Efficiency"
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
    log_experiment_start "log-storage-$STRATEGY" "$OUT_DIR"

    apply_log_strategy "$STRATEGY"

    echo "[wait] Flushing Loki and stabilizing (60s)..."
    kubectl exec -n monitoring svc/loki -- curl -X POST http://localhost:3100/flush>/dev/null 2>&1 || true
    sleep 60

    START_SIZE=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    START_TS=$(now_ts)

    echo "[run] Measuring storage growth for ${WINDOW_DURATION}s..."
    sleep_with_progress "$WINDOW_DURATION" "$STRATEGY storage"
    
    # Flush again so the logs generated during the window are written to disk
    echo -e "\n[collect] Finalizing storage capture..."
    kubectl exec -n monitoring svc/loki -- curl -X POST http://localhost:3100/flush >/dev/null 2>&1 || true
    
    END_SIZE=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
    END_TS=$(now_ts)
    DELTA=$((END_SIZE - START_SIZE))

    cat > "$OUT_DIR/storage_report.json" <<EOF
{
  "strategy": "$STRATEGY",
  "loki_growth_kb": $DELTA,
  "start_kb": $START_SIZE,
  "end_kb": $END_SIZE,
  "window_duration_s": $WINDOW_DURATION,
  "ue_count": $UE_COUNT,
  "bytes_per_second": $(echo "scale=2; ($DELTA * 1024) / $WINDOW_DURATION" | bc),
  "collected_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    log_experiment_end "$OUT_DIR"
    echo "[done] Strategy $STRATEGY complete. Delta: ${DELTA}KB → $OUT_DIR"

    # 7. Cooldown
    if [[ "$STRATEGY" != "${STRATEGIES[-1]}" ]]; then
        echo "[cooldown] 60s between strategies..."
        sleep 60
    fi
done

echo "============================================================"
echo " Phase 2 Storage complete. Data in: $OUT_BASE"
echo "============================================================"
