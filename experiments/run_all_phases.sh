#!/usr/bin/env bash
# experiments/run_all_phases.sh
#
# Top-level orchestrator: runs all four experimental phases in sequence,
# then runs the analysis script automatically.
#
# Usage:
#   bash run_all_phases.sh [--from-phase <0|1|2|3|4>]
#
# Phases:
#   0 — Baseline (no telemetry)
#   1 — Prometheus overhead (3 scrape intervals)
#   2 — Beyla/eBPF overhead (3 sampling rates)
#   3 — Fault detection (8 faults)
#   4 — Scalability (6 scenarios)
#
# Estimated total runtime: ~6-7 hours
# Run in a tmux/screen session.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FROM_PHASE=0
if [[ "${1:-}" == "--from-phase" && -n "${2:-}" ]]; then
    FROM_PHASE="$2"
fi

run_phase() {
    local num="$1" name="$2" script="$3"
    if [[ $num -lt $FROM_PHASE ]]; then
        echo "[skip] Phase $num ($name)"
        return
    fi
    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo " PHASE $num: $name"
    echo "████████████████████████████████████████████████████████████"
    bash "$script"
    echo ""
    echo "[phase $num done] Sleeping 2 minutes before next phase..."
    sleep 120
}

run_phase 0 "Baseline"              "$SCRIPT_DIR/00-baseline/run.sh"
# run_phase 1 "Prometheus Overhead"   "$SCRIPT_DIR/01-overhead-prometheus/run.sh"
# run_phase 2 "Beyla/eBPF Overhead"   "$SCRIPT_DIR/02-overhead-ebpf/run.sh"
# run_phase 3 "Fault Detection"       "$SCRIPT_DIR/03-fault-detection/run_all.sh"
# run_phase 4 "Scalability"           "$SCRIPT_DIR/04-scalability/run.sh"
run_phase 1 "CPU overhead"          "$SCRIPT_DIR/01-cpu-overhead/run.sh"
run_phase 2 "Storage requirements"  "$SCRIPT_DIR/02-storage-requirements/run.sh"
run_phase 3 "System visibility"     "$SCRIPT_DIR/03-system-visibility/run.sh"
run_phase 4 "Scalability (strategies)" "$SCRIPT_DIR/04-scalability/run-str.sh"

echo ""
echo "████████████████████████████████████████████████████████████"
echo " ALL PHASES COMPLETE"
echo " Data in: $REPO_ROOT/data/experiments/"
echo "████████████████████████████████████████████████████████████"

# echo ""
# echo "████████████████████████████████████████████████████████████"
# echo " RUNNING ANALYSIS"
# echo "████████████████████████████████████████████████████████████"
# python3 "$REPO_ROOT/data/analysis/analyse.py" \
#     --data-dir "$REPO_ROOT/data/experiments" \
#     --out "$REPO_ROOT/data/analysis"

# echo ""
# echo "████████████████████████████████████████████████████████████"
# echo " ALL_DONE"
# echo " Figures : $REPO_ROOT/data/analysis/figures"
# echo " Tables  : $REPO_ROOT/data/analysis/tables"
# echo " Summary : $REPO_ROOT/data/analysis/summary"
# echo "████████████████████████████████████████████████████████████"