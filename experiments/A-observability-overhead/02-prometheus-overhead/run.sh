#!/usr/bin/env bash
# A-observability-overhead/02-prometheus-overhead/run.sh
#
# Phase 1: Prometheus overhead measurement.
#
# Measures the CPU and memory cost of the Prometheus observability stack
# at three scrape intervals: 1s, 5s, 15s.
# Beyla is scaled down so only Prometheus overhead is measured.
#
# Output:
#   data/experiments/01-overhead-prometheus/interval-{1s,5s,15s}/
#     container_cpu_usage_rate.csv
#     container_memory_working_set_bytes.csv
#     monitoring_cpu_usage_rate.csv
#     monitoring_memory_working_set.csv
#     meta.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

WINDOW_DURATION="${WINDOW_DURATION:-600}"   # 10 minutes per interval
UE_COUNT=50
STEP="5s"

INTERVALS=("1s" "5s" "15s")
if [[ "${1:-}" == "--interval" && -n "${2:-}" ]]; then
    INTERVALS=("$2")
fi

OUT_BASE="$DATA_DIR/01-overhead-prometheus"

echo "============================================================"
echo " Phase 1: Prometheus overhead measurement"
echo " Intervals: ${INTERVALS[*]}"
echo "============================================================"

check_cluster_ready

for INTERVAL in "${INTERVALS[@]}"; do
    SLUG="interval-${INTERVAL}"
    OUT_DIR="$OUT_BASE/$SLUG"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "--- Interval: $INTERVAL ---"

    # Full cluster reset before each interval so residual Prometheus TSDB state,
    # WAL backlog, or UE tunnel state from the previous interval cannot skew readings.
    echo "[reset] Full cluster reset before interval $INTERVAL..."
    bash "$SCRIPT_DIR/../../../cluster-start.sh"
    bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"

    # Disable Beyla to isolate Prometheus overhead
    echo "[setup] Disabling Beyla to isolate Prometheus overhead..."
    kubectl patch daemonset beyla -n open5gs \
        --type=json \
        -p='[{"op":"add","path":"/spec/template/spec/nodeSelector","value":{"non-existing":"true"}}]' \
        2>/dev/null || true

    scale_ues "$UE_COUNT"
    wait_for_pods_stable open5gs 120
    wait_for_ue_sessions "$UE_COUNT" 240

    ensure_portforward_prometheus

    if ! bash "$LIB_DIR/health_check.sh" "pre-prometheus-$SLUG" "$OUT_DIR/health_pre.json"; then
        echo "[ABORT] cluster not healthy before interval $INTERVAL" >&2
        exit 1
    fi

    log_experiment_start "01-overhead-prometheus-$SLUG" "$OUT_DIR"

    set_prometheus_scrape_interval "$INTERVAL"

    echo "[wait] Letting Prometheus stabilize for 60s..."
    sleep 60

    START_TS=$(now_ts)
    echo "[run] Collecting for ${WINDOW_DURATION}s..."
    sleep_with_progress "$WINDOW_DURATION" "measuring $INTERVAL"
    END_TS=$(now_ts)

    echo "[collect] Querying Prometheus API..."
    collect_prometheus "$START_TS" "$END_TS" "$STEP" "$OUT_DIR"

    # Also collect Prometheus self-metrics
    python3 "$LIB_DIR/collect_prometheus.py" \
        --url "$PROM_URL" \
        --start "$START_TS" \
        --end   "$END_TS" \
        --step  "$INTERVAL" \
        --out   "$OUT_DIR/self_metrics" \
        --extra-metrics \
            "prom_head_chunks:prometheus_tsdb_head_chunks:prom_head_chunks.csv" \
            "prom_active_appenders:prometheus_tsdb_head_active_appenders:prom_active_appenders.csv" \
            "prom_wal_writes:rate(prometheus_tsdb_wal_records_total[1m]):prom_wal_writes.csv"

    python3 -c "
import json
meta = {
    'scrape_interval': '$INTERVAL',
    'window_duration_s': $WINDOW_DURATION,
    'ue_count': $UE_COUNT,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
}
with open('$OUT_DIR/interval_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    log_experiment_end "$OUT_DIR"
    echo "[done] Interval $INTERVAL complete → $OUT_DIR"
done

# Restore
echo ""
echo "[restore] Restoring default scrape interval (5s)..."
set_prometheus_scrape_interval "5s"

echo "[restore] Re-enabling Beyla..."
kubectl patch daemonset beyla -n open5gs \
    --type=json \
    -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector/non-existing"}]' \
    2>/dev/null || true
kubectl rollout status daemonset/beyla -n open5gs --timeout=2m 2>/dev/null || true

echo ""
echo "============================================================"
echo " Phase 1 complete. Data in: $OUT_BASE"
echo "============================================================"
