#!/usr/bin/env bash
# A-observability-overhead/smoke_test.sh
#
# Smoke test: runs all four A experiments with 60s measurement windows
# (instead of 600s) to verify the full mechanics work end-to-end:
#   - cluster reset
#   - health checks
#   - data collection (kubectl top, Prometheus, Jaeger)
#   - port-forward re-establishment
#
# Writes to data/experiments/smoke-test/ (separate from real data).
# Exits non-zero if any experiment fails.
#
# Usage: bash smoke_test.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE_WINDOW=60   # 60s measurement windows

echo "============================================================"
echo " A-observability-overhead SMOKE TEST"
echo " Window duration: ${SMOKE_WINDOW}s per scenario"
echo " Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

PASS=0
FAIL=0
RESULTS=()

run_smoke() {
    local num="$1" name="$2" script="$3"
    shift 3
    local extra_args=("$@")
    echo ""
    echo "------------------------------------------------------------"
    echo " SMOKE A-$num: $name"
    echo "------------------------------------------------------------"
    local t0
    t0=$(date +%s)
    if WINDOW_DURATION="$SMOKE_WINDOW" \
       STEADY_DURATION="$SMOKE_WINDOW" \
       BURSTY_DURATION="$SMOKE_WINDOW" \
       bash "$script" "${extra_args[@]}"; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[PASS] A-$num: $name (${elapsed}s)"
        RESULTS+=("PASS  A-$num: $name  (${elapsed}s)")
        PASS=$((PASS+1))
    else
        local elapsed=$(( $(date +%s) - t0 ))
        echo "[FAIL] A-$num: $name (${elapsed}s)" >&2
        RESULTS+=("FAIL  A-$num: $name  (${elapsed}s)")
        FAIL=$((FAIL+1))
    fi
}

# A-01: only run steady window (skip bursty) to keep smoke fast
run_smoke 1 "No-telemetry baseline (steady only)" \
    "$SCRIPT_DIR/01-no-telemetry-baseline/run.sh" \
    --skip-bursty

# A-02: only test one interval (15s — fastest scrape config to stabilise)
run_smoke 2 "Prometheus overhead (15s interval only)" \
    "$SCRIPT_DIR/02-prometheus-overhead/run.sh" \
    --interval 15s

# A-03: only test 10% sampling rate (lowest Beyla load)
run_smoke 3 "eBPF/Beyla overhead (10pct sampling only)" \
    "$SCRIPT_DIR/03-ebpf-overhead/run.sh" \
    --rate 0.1

# A-04: only run the smallest scenario (10 UEs steady) via a wrapper
# We can't pass --only to run_scenario directly, so we run the full script
# but with SMOKE_SCENARIOS override. Instead, patch via a minimal wrapper.
echo ""
echo "------------------------------------------------------------"
echo " SMOKE A-4: Overhead at scale (10 UEs steady only)"
echo "------------------------------------------------------------"
t0=$(date +%s)
if WINDOW_DURATION="$SMOKE_WINDOW" bash -c '
    set -euo pipefail
    SCRIPT_DIR="'"$SCRIPT_DIR"'"
    source "$SCRIPT_DIR/../lib/common.sh"
    OUT_BASE="$DATA_DIR/04-scalability"

    run_scenario() {
        local ue_count="$1" pattern="$2"
        local slug="ues-${ue_count}-${pattern}"
        local out_dir="$OUT_BASE/$slug"
        mkdir -p "$out_dir"
        echo ""
        echo "--- Scenario: $slug ---"
        echo "[reset] Full cluster reset before $slug..."
        bash "$SCRIPT_DIR/../../cluster-start.sh"
        bash "$LIB_DIR/provision_ues.sh" "$ue_count"
        scale_ues "$ue_count"
        wait_for_pods_stable open5gs 120
        ensure_portforward_prometheus
        ensure_portforward_jaeger
        if ! bash "$LIB_DIR/health_check.sh" "pre-scalability-$slug" "$out_dir/health_pre.json"; then
            echo "[ABORT] cluster not healthy before scenario $slug" >&2
            exit 1
        fi
        log_experiment_start "04-scalability-$slug" "$out_dir"
        echo "[wait] Stabilising for 30s..."
        sleep 30
        START_TS=$(now_ts)
        echo "[run] Steady-state for '"$SMOKE_WINDOW"'s..."
        sleep_with_progress '"$SMOKE_WINDOW"' "$slug"
        END_TS=$(now_ts)
        collect_prometheus "$START_TS" "$END_TS" "5s" "$out_dir"
        python3 "$LIB_DIR/collect_prometheus.py" \
            --url "$PROM_URL" --start "$START_TS" --end "$END_TS" --step "5s" \
            --out "$out_dir" \
            --extra-metrics \
                "beyla_cpu:rate(container_cpu_usage_seconds_total{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}[2m]):beyla_cpu_usage_rate.csv" \
                "beyla_mem:container_memory_working_set_bytes{namespace=\"open5gs\",pod=~\"beyla.*\",container!=\"\"}:beyla_memory_working_set.csv"
        collect_jaeger "$START_TS" "$END_TS" "$out_dir/jaeger"
        python3 -c "
import json
meta = {\"ue_count\": $ue_count, \"pattern\": \"$pattern\", \"window_duration_s\": '"$SMOKE_WINDOW"', \"start_ts\": $START_TS, \"end_ts\": $END_TS}
with open(\"$out_dir/scenario_meta.json\", \"w\") as f: json.dump(meta, f, indent=2)
"
        log_experiment_end "$out_dir"
        echo "[done] $slug complete"
    }

    run_scenario 10 steady
'; then
    elapsed=$(( $(date +%s) - t0 ))
    echo "[PASS] A-4: Overhead at scale (${elapsed}s)"
    RESULTS+=("PASS  A-4: Overhead at scale (10 UEs steady)  (${elapsed}s)")
    PASS=$((PASS+1))
else
    elapsed=$(( $(date +%s) - t0 ))
    echo "[FAIL] A-4: Overhead at scale (${elapsed}s)" >&2
    RESULTS+=("FAIL  A-4: Overhead at scale (10 UEs steady)  (${elapsed}s)")
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================================================"
echo " SMOKE TEST RESULTS"
echo "============================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASS   Failed: $FAIL"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    echo "[SMOKE] FAILED — fix the above before running full experiments" >&2
    exit 1
fi

echo "[SMOKE] ALL PASSED — safe to run full experiments"
