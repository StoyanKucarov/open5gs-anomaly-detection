#!/usr/bin/env bash
# B-log-strategies/run_all.sh
#
# Runs all four log strategy experiments in sequence.
#
#   01 — CPU overhead        (Promtail CPU per strategy)
#   02 — Storage requirements (Loki storage per strategy)
#   03 — System visibility   (fault detection per strategy, 3 faults × 5 strategies)
#   04 — Strategy at scale   (Loki storage + Promtail CPU across UE counts)
#
# Estimated runtime: ~9.5 hours

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM=0
if [[ "${1:-}" == "--from" && -n "${2:-}" ]]; then FROM="$2"; fi

run_phase() {
    local num="$1" name="$2" script="$3"
    [[ $num -lt $FROM ]] && echo "[skip] Phase $num ($name)" && return
    echo ""
    echo "============================================================"
    echo " B-$num: $name"
    echo "============================================================"
    bash "$script"
    echo "[done] B-$num complete. Sleeping 2 min..."
    sleep 120
}

run_phase 1 "CPU overhead"          "$SCRIPT_DIR/01-cpu-overhead/run.sh"
run_phase 2 "Storage requirements"  "$SCRIPT_DIR/02-storage-requirements/run.sh"
run_phase 3 "System visibility"     "$SCRIPT_DIR/03-system-visibility/run.sh"
run_phase 4 "Strategy at scale"     "$SCRIPT_DIR/04-strategy-at-scale/run.sh"

echo ""
echo "============================================================"
echo " Group B complete."
echo "============================================================"
