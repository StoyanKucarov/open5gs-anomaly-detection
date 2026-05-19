#!/usr/bin/env bash
# A-observability-overhead/03-ebpf-overhead/run.sh
#
# Phase 2: Beyla/eBPF overhead measurement.
#
# Measures the CPU and memory cost of Beyla at three sampling rates:
# 100%, 50%, 10%. Prometheus is running (for metric collection) but
# Beyla is the subject under test.
#
# Output:
#   data/experiments/02-overhead-ebpf/sampling-{100,50,10}pct/
#     container_cpu_usage_rate.csv
#     container_memory_working_set_bytes.csv
#     beyla_metrics/beyla_cpu.csv
#     beyla_metrics/beyla_mem.csv
#     jaeger/spans_flat.csv
#     jaeger/summary.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

WINDOW_DURATION="${WINDOW_DURATION:-600}"   # 10 minutes per rate
UE_COUNT=50
STEP="5s"

# sampling rate → slug
declare -A RATE_SLUGS=(
    ["1.0"]="sampling-100pct"
    ["0.5"]="sampling-50pct"
    ["0.1"]="sampling-10pct"
)
RATES=("1.0" "0.5" "0.1")
if [[ "${1:-}" == "--rate" && -n "${2:-}" ]]; then
    RATES=("$2")
fi

OUT_BASE="$DATA_DIR/02-overhead-ebpf"

echo "============================================================"
echo " Phase 2: Beyla/eBPF overhead measurement"
echo "============================================================"

check_cluster_ready

for RATE in "${RATES[@]}"; do
    SLUG="${RATE_SLUGS[$RATE]}"
    OUT_DIR="$OUT_BASE/$SLUG"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "--- Sampling rate: $RATE ($SLUG) ---"

    # Full cluster reset before each sampling rate so Beyla's internal trace
    # buffer, GTP state, and UE tunnel state from the previous rate cannot
    # bleed into the next measurement window.
    echo "[reset] Full cluster reset before $SLUG..."
    bash "$SCRIPT_DIR/../../../cluster-start.sh"
    bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"

    scale_ues "$UE_COUNT"
    wait_for_pods_stable open5gs 120
    wait_for_ue_sessions "$UE_COUNT" 240

    ensure_portforward_prometheus
    ensure_portforward_jaeger

    if ! bash "$LIB_DIR/health_check.sh" "pre-ebpf-$SLUG" "$OUT_DIR/health_pre.json"; then
        echo "[ABORT] cluster not healthy before $SLUG" >&2
        exit 1
    fi

    log_experiment_start "02-overhead-ebpf-$SLUG" "$OUT_DIR"

    # Patch Beyla sampling rate via env var
    echo "[setup] Setting Beyla sampling rate to $RATE..."
    kubectl set env daemonset/beyla -n open5gs \
        OTEL_TRACES_SAMPLER=parentbased_traceidratio \
        OTEL_TRACES_SAMPLER_ARG="$RATE" 2>/dev/null || true
    kubectl rollout status daemonset/beyla -n open5gs --timeout=2m 2>/dev/null || true

    echo "[wait] Stabilising for 30s..."
    sleep 30

    START_TS=$(now_ts)
    echo "[run] Collecting for ${WINDOW_DURATION}s..."
    sleep_with_progress "$WINDOW_DURATION" "measuring $SLUG"
    END_TS=$(now_ts)

    echo "[collect] Querying Prometheus..."
    collect_prometheus "$START_TS" "$END_TS" "$STEP" "$OUT_DIR"

    # Beyla-specific metrics (CPU/mem of beyla pods)
    python3 "$LIB_DIR/collect_prometheus.py" \
        --url "$PROM_URL" \
        --start "$START_TS" \
        --end   "$END_TS" \
        --step  "$STEP" \
        --out   "$OUT_DIR/beyla_metrics" \
        --extra-metrics \
            "beyla_cpu:rate(container_cpu_usage_seconds_total{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}[2m]):beyla_cpu.csv" \
            "beyla_mem:container_memory_working_set_bytes{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}:beyla_mem.csv"

    echo "[collect] Querying Jaeger..."
    collect_jaeger "$START_TS" "$END_TS" "$OUT_DIR/jaeger"

    python3 -c "
import json
meta = {
    'sampling_rate': '$RATE',
    'slug': '$SLUG',
    'window_duration_s': $WINDOW_DURATION,
    'ue_count': $UE_COUNT,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
}
with open('$OUT_DIR/rate_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    log_experiment_end "$OUT_DIR"
    echo "[done] $SLUG complete → $OUT_DIR"
done

# Restore Beyla to always_on sampling
echo ""
echo "[restore] Restoring Beyla to always_on sampling..."
kubectl set env daemonset/beyla -n open5gs \
    OTEL_TRACES_SAMPLER=always_on \
    OTEL_TRACES_SAMPLER_ARG- 2>/dev/null || true
kubectl rollout status daemonset/beyla -n open5gs --timeout=2m 2>/dev/null || true

echo ""
echo "============================================================"
echo " Phase 2 complete. Data in: $OUT_BASE"
echo "============================================================"
