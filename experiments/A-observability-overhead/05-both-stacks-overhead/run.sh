#!/usr/bin/env bash
# A-observability-overhead/05-both-stacks-overhead/run.sh
#
# Phase 5: Both stacks active — Prometheus (5s) + Beyla (100% sampling).
#
# This is the combined overhead measurement: both Prometheus and Beyla are
# running simultaneously at their default settings. This gives the true
# combined cost of the full observability stack.
#
# Output:
#   data/experiments/03-overhead-both/
#     container_cpu_usage_rate.csv
#     container_memory_working_set_bytes.csv
#     monitoring_cpu_usage_rate.csv
#     monitoring_memory_working_set.csv
#     beyla_cpu_usage_rate.csv
#     beyla_memory_working_set.csv
#     jaeger/spans_flat.csv
#     jaeger/summary.json
#     meta.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

WINDOW_DURATION="${WINDOW_DURATION:-600}"   # 10 minutes
UE_COUNT=50
STEP="5s"

OUT_DIR="$DATA_DIR/03-overhead-both"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Phase 5: Both stacks overhead (Prometheus 5s + Beyla 100%)"
echo "============================================================"

check_cluster_ready

# Full cluster reset to start from a clean known state
echo "[reset] Full cluster reset before both-stacks measurement..."
bash "$SCRIPT_DIR/../../../cluster-start.sh"
bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"

# Ensure Beyla is at 100% sampling (always_on)
echo "[setup] Ensuring Beyla is at always_on sampling..."
kubectl set env daemonset/beyla -n open5gs \
    OTEL_TRACES_SAMPLER=always_on \
    OTEL_TRACES_SAMPLER_ARG- 2>/dev/null || true
kubectl rollout status daemonset/beyla -n open5gs --timeout=2m 2>/dev/null || true

# Ensure Prometheus is at 5s scrape interval
echo "[setup] Ensuring Prometheus scrape interval is 5s..."
set_prometheus_scrape_interval "5s"

scale_ues "$UE_COUNT"
wait_for_pods_stable open5gs 120
wait_for_ue_sessions "$UE_COUNT" 240

ensure_portforward_prometheus
ensure_portforward_jaeger

if ! bash "$LIB_DIR/health_check.sh" "pre-both-stacks" "$OUT_DIR/health_pre.json"; then
    echo "[ABORT] cluster not healthy before both-stacks measurement" >&2
    exit 1
fi

log_experiment_start "03-overhead-both" "$OUT_DIR"

echo "[wait] Letting both stacks stabilize for 60s..."
sleep 60

START_TS=$(now_ts)
echo "[run] Collecting for ${WINDOW_DURATION}s..."
sleep_with_progress "$WINDOW_DURATION" "measuring both stacks"
END_TS=$(now_ts)

echo "[collect] Querying Prometheus..."
collect_prometheus "$START_TS" "$END_TS" "$STEP" "$OUT_DIR"

# Beyla-specific metrics
python3 "$LIB_DIR/collect_prometheus.py" \
    --url "$PROM_URL" \
    --start "$START_TS" \
    --end   "$END_TS" \
    --step  "$STEP" \
    --out   "$OUT_DIR" \
    --extra-metrics \
        "beyla_cpu:rate(container_cpu_usage_seconds_total{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}[2m]):beyla_cpu_usage_rate.csv" \
        "beyla_mem:container_memory_working_set_bytes{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}:beyla_memory_working_set.csv"

echo "[collect] Querying Jaeger..."
collect_jaeger "$START_TS" "$END_TS" "$OUT_DIR/jaeger"

python3 -c "
import json
meta = {
    'prometheus_interval': '5s',
    'beyla_sampling': '1.0',
    'window_duration_s': $WINDOW_DURATION,
    'ue_count': $UE_COUNT,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
}
with open('$OUT_DIR/both_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
log_experiment_end "$OUT_DIR"

echo ""
echo "============================================================"
echo " Phase 5 complete. Data in: $OUT_DIR"
echo "============================================================"
