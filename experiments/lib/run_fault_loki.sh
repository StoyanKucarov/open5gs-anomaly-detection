#!/usr/bin/env bash
# experiments/lib/run_fault.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Defaults
PRE_DURATION=120
FAULT_DURATION=300
POST_DURATION=120
STEP="5s"
NAME=""
MANIFEST=""
OUT_DIR=""
STRATEGY="unknown"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     NAME="$2";     shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --out)      OUT_DIR="$2";  shift 2 ;;
        --pre-duration)   PRE_DURATION="$2";   shift 2 ;;
        --fault-duration) FAULT_DURATION="$2"; shift 2 ;;
        --post-duration)  POST_DURATION="$2";  shift 2 ;;
        --step)     STEP="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR/logs"

echo "--- Executing Fault: $NAME (Strategy: $STRATEGY) ---"
log_experiment_start "$NAME" "$OUT_DIR"

# Internal Helper for Log/CPU Snapshots
capture_visibility_snapshot() {
    local phase="$1"
    
    kubectl top pods -n monitoring --no-headers | grep "promtail" >> "$OUT_DIR/promtail_cpu_$phase.csv" || true

    local LOG_FILE="$OUT_DIR/logs/amf_sample_$phase.log"
    
    kubectl logs -n open5gs -l "app.kubernetes.io/name=amf" --tail=200 --all-containers=true > "$LOG_FILE" 2>/dev/null || \
    kubectl logs -n open5gs -l "app=amf" --tail=200 --all-containers=true > "$LOG_FILE" 2>/dev/null || \
    echo "WARNING: Could not find AMF logs for phase $phase" > "$LOG_FILE"
}

START_SIZE=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
PRE_START=$(now_ts)
capture_visibility_snapshot "pre"
sleep_with_progress "$PRE_DURATION" "pre-fault baseline"
PRE_END=$(now_ts)

echo "[fault] Injecting: $MANIFEST"
kubectl apply -f "$MANIFEST"
FAULT_START=$(now_ts)
capture_visibility_snapshot "during"
sleep_with_progress "$FAULT_DURATION" "fault active"
FAULT_END=$(now_ts)

echo "[fault] Removing fault..."
kubectl delete -f "$MANIFEST" --ignore-not-found=true
REMOVE_TS=$(now_ts)
capture_visibility_snapshot "post"
sleep_with_progress "$POST_DURATION" "recovery"
POST_END=$(now_ts)

END_SIZE=$(kubectl exec -n monitoring svc/loki -- du -s /data/loki | awk '{print $1}')
STORAGE_DELTA=$((END_SIZE - START_SIZE))

python3 -c "
import json
res = {
    'name': '$NAME',
    'strategy': '$STRATEGY',
    'storage_delta_kb': $STORAGE_DELTA,
    'timeline': {
        'pre':   {'start': $PRE_START,   'end': $PRE_END},
        'fault': {'start': $FAULT_START, 'end': $FAULT_END},
        'post':  {'start': $REMOVE_TS,   'end': $POST_END}
    }
}
with open('$OUT_DIR/visibility_report.json', 'w') as f:
    json.dump(res, f, indent=2)
"

log_experiment_end "$OUT_DIR"
echo "[fault] $NAME complete (Growth: ${STORAGE_DELTA}KB) → $OUT_DIR"