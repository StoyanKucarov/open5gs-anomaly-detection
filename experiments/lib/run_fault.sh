#!/usr/bin/env bash
# experiments/lib/run_fault.sh
#
# Generic fault lifecycle runner.
# Applies a Chaos Mesh manifest, collects pre/during/post data, then removes it.
#
# Usage:
#   bash run_fault.sh \
#     --name    <fault-slug>          e.g. "01-cpu-stress-amf"
#     --manifest <path/to/chaos.yaml> \
#     --out     <output-dir>          \
#     --pre-duration  <seconds>       (default 120)
#     --fault-duration <seconds>      (default 300)
#     --post-duration <seconds>       (default 120)
#     --step    <prom-step>           (default "5s")

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)           NAME="$2";           shift 2 ;;
        --manifest)       MANIFEST="$2";       shift 2 ;;
        --out)            OUT_DIR="$2";        shift 2 ;;
        --pre-duration)   PRE_DURATION="$2";   shift 2 ;;
        --fault-duration) FAULT_DURATION="$2"; shift 2 ;;
        --post-duration)  POST_DURATION="$2";  shift 2 ;;
        --step)           STEP="$2";           shift 2 ;;
        *) echo "[run_fault] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$NAME" ]]     && { echo "[run_fault] --name required" >&2; exit 1; }
[[ -z "$MANIFEST" ]] && { echo "[run_fault] --manifest required" >&2; exit 1; }
[[ -z "$OUT_DIR" ]]  && { echo "[run_fault] --out required" >&2; exit 1; }

mkdir -p "$OUT_DIR"

ensure_portforward_prometheus
ensure_portforward_jaeger

echo ""
echo "--- Fault: $NAME ---"
log_experiment_start "$NAME" "$OUT_DIR"

# ---------------------------------------------------------------------------
# PRE window
# ---------------------------------------------------------------------------
echo "[fault] PRE window (${PRE_DURATION}s)..."
PRE_START=$(now_ts)
sleep_with_progress "$PRE_DURATION" "pre-fault baseline"
PRE_END=$(now_ts)

collect_prometheus "$PRE_START" "$PRE_END" "$STEP" "$OUT_DIR/prometheus/pre"
collect_jaeger     "$PRE_START" "$PRE_END"         "$OUT_DIR/jaeger/pre"

# ---------------------------------------------------------------------------
# Inject fault
# ---------------------------------------------------------------------------
echo "[fault] Injecting fault: $MANIFEST"
kubectl apply -f "$MANIFEST"
FAULT_START=$(now_ts)

sleep_with_progress "$FAULT_DURATION" "fault active"
FAULT_END=$(now_ts)

collect_prometheus "$FAULT_START" "$FAULT_END" "$STEP" "$OUT_DIR/prometheus/during"
collect_jaeger     "$FAULT_START" "$FAULT_END"         "$OUT_DIR/jaeger/during"

# ---------------------------------------------------------------------------
# Remove fault
# ---------------------------------------------------------------------------
echo "[fault] Removing fault..."
kubectl delete -f "$MANIFEST" --ignore-not-found=true
REMOVE_TS=$(now_ts)

# ---------------------------------------------------------------------------
# POST window
# ---------------------------------------------------------------------------
echo "[fault] POST window (${POST_DURATION}s)..."
sleep_with_progress "$POST_DURATION" "post-fault recovery"
POST_END=$(now_ts)

collect_prometheus "$REMOVE_TS" "$POST_END" "$STEP" "$OUT_DIR/prometheus/post"
collect_jaeger     "$REMOVE_TS" "$POST_END"         "$OUT_DIR/jaeger/post"

# ---------------------------------------------------------------------------
# Write timeline
# ---------------------------------------------------------------------------
python3 -c "
import json
timeline = {
    'name': '$NAME',
    'pre':   {'start': $PRE_START,   'end': $PRE_END},
    'fault': {'start': $FAULT_START, 'end': $FAULT_END},
    'post':  {'start': $REMOVE_TS,   'end': $POST_END},
}
with open('$OUT_DIR/timeline.json', 'w') as f:
    json.dump(timeline, f, indent=2)
"

log_experiment_end "$OUT_DIR"
echo "[fault] $NAME complete → $OUT_DIR"
