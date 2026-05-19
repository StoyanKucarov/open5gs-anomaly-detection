#!/usr/bin/env bash
# A-observability-overhead/04-overhead-at-scale/run.sh
#
# Phase 4: Scalability — measures NF and Beyla overhead across UE counts.
#
# Scenarios:
#   10, 50, 100, 200 UEs — steady-state (10 min each)
#   50, 100 UEs          — bursty (10 min each)
#
# Output:
#   data/experiments/04-scalability/ues-{N}-{pattern}/
#     container_cpu_usage_rate.csv
#     container_memory_working_set_bytes.csv
#     beyla_cpu_usage_rate.csv
#     beyla_memory_working_set.csv
#     jaeger/spans_flat.csv
#     jaeger/summary.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/common.sh"

WINDOW_DURATION="${WINDOW_DURATION:-600}"   # 10 minutes per scenario
STEP="5s"

OUT_BASE="$DATA_DIR/04-scalability"

echo "============================================================"
echo " Phase 4: Scalability"
echo "============================================================"

check_cluster_ready

run_scenario() {
    local ue_count="$1" pattern="$2"
    local slug="ues-${ue_count}-${pattern}"
    local out_dir="$OUT_BASE/$slug"
    mkdir -p "$out_dir"

    echo ""
    echo "--- Scenario: $slug ---"

    # Skip if already completed
    if [[ -f "$out_dir/meta.json" ]]; then
        echo "[skip] $slug already complete — skipping"
        return 0
    fi

    # Full cluster reset before each scenario so UE tunnel state, Beyla trace
    # buffers, and Prometheus TSDB from the previous scenario cannot skew readings.
    echo "[reset] Full cluster reset before $slug..."
    bash "$SCRIPT_DIR/../../../cluster-start.sh"
    bash "$LIB_DIR/provision_ues.sh" "$ue_count"

    scale_ues "$ue_count"
    wait_for_pods_stable open5gs 120
    if ! wait_for_ue_sessions "$ue_count" 240; then
        echo "[SKIP] $slug — cluster could not establish $ue_count UE sessions; skipping scenario" >&2
        return 0
    fi

    ensure_portforward_prometheus
    ensure_portforward_jaeger

    if ! bash "$LIB_DIR/health_check.sh" "pre-scalability-$slug" "$out_dir/health_pre.json"; then
        echo "[SKIP] $slug — cluster not healthy before scenario; skipping" >&2
        return 0
    fi

    log_experiment_start "04-scalability-$slug" "$out_dir"

    echo "[wait] Stabilising for 30s..."
    sleep 30

    if [[ "$pattern" == "steady" ]]; then
        START_TS=$(now_ts)
        echo "[run] Steady-state for ${WINDOW_DURATION}s..."
        sleep_with_progress "$WINDOW_DURATION" "$slug"
        END_TS=$(now_ts)
    else
        # Bursty: scale up/down cycles while collecting
        START_TS=$(now_ts)
        echo "[run] Bursty for ${WINDOW_DURATION}s..."
        (
            end_ts=$(( $(date +%s) + WINDOW_DURATION ))
            cycle=0
            while [[ $(date +%s) -lt $end_ts ]]; do
                cycle=$((cycle+1))
                helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
                    --version 0.1.2 --namespace open5gs --reuse-values \
                    --set ues.count="$ue_count" --wait --timeout=2m 2>/dev/null || true
                sleep 30
                helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
                    --version 0.1.2 --namespace open5gs --reuse-values \
                    --set ues.count=5 --wait --timeout=2m 2>/dev/null || true
                sleep 30
            done
        ) &
        BURSTY_PID=$!
        sleep_with_progress "$WINDOW_DURATION" "$slug"
        wait "$BURSTY_PID" 2>/dev/null || true
        END_TS=$(now_ts)
    fi

    echo "[collect] Querying Prometheus..."
    collect_prometheus "$START_TS" "$END_TS" "$STEP" "$out_dir"

    # Beyla-specific metrics
    python3 "$LIB_DIR/collect_prometheus.py" \
        --url "$PROM_URL" \
        --start "$START_TS" \
        --end   "$END_TS" \
        --step  "$STEP" \
        --out   "$out_dir" \
        --extra-metrics \
            "beyla_cpu:rate(container_cpu_usage_seconds_total{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}[2m]):beyla_cpu_usage_rate.csv" \
            "beyla_mem:container_memory_working_set_bytes{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}:beyla_memory_working_set.csv"

    echo "[collect] Querying Jaeger..."
    collect_jaeger "$START_TS" "$END_TS" "$out_dir/jaeger"

    python3 -c "
import json
meta = {
    'ue_count': $ue_count,
    'pattern': '$pattern',
    'window_duration_s': $WINDOW_DURATION,
    'start_ts': $START_TS,
    'end_ts': $END_TS,
}
with open('$out_dir/scenario_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
    log_experiment_end "$out_dir"
    echo "[done] $slug complete → $out_dir"
}

# Steady-state scenarios
run_scenario 10  steady
run_scenario 50  steady
run_scenario 100 steady
run_scenario 200 steady

# Bursty scenarios
run_scenario 50  bursty
run_scenario 100 bursty

echo ""
echo "============================================================"
echo " Phase 4 complete. Data in: $OUT_BASE"
echo "============================================================"
