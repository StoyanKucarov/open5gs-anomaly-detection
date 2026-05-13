#!/usr/bin/env bash
# A-observability-overhead/run_all.sh
#
# Runs all four observability overhead experiments in sequence.
#
#   01 — No-telemetry baseline (kubectl top, zero observability)
#   02 — Prometheus overhead   (3 scrape intervals)
#   03 — eBPF/Beyla overhead   (3 sampling rates)
#   04 — Overhead at scale     (NF + Beyla across 4 UE counts)
#
# Estimated runtime: ~3.5 hours

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM=0
if [[ "${1:-}" == "--from" && -n "${2:-}" ]]; then FROM="$2"; fi

run_phase() {
    local num="$1" name="$2" script="$3"
    [[ $num -lt $FROM ]] && echo "[skip] Phase $num ($name)" && return
    echo ""
    echo "============================================================"
    echo " A-$num: $name"
    echo "============================================================"
    bash "$script"
    echo "[done] A-$num complete. Sleeping 2 min..."
    sleep 120
}

run_phase 1 "No-telemetry baseline"  "$SCRIPT_DIR/01-no-telemetry-baseline/run.sh"
run_phase 2 "Prometheus overhead"    "$SCRIPT_DIR/02-prometheus-overhead/run.sh"
run_phase 3 "eBPF/Beyla overhead"    "$SCRIPT_DIR/03-ebpf-overhead/run.sh"
run_phase 4 "Overhead at scale"      "$SCRIPT_DIR/04-overhead-at-scale/run.sh"

echo ""
echo "============================================================"
echo " Group A complete."
echo "============================================================"
