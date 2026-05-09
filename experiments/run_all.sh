#!/usr/bin/env bash
# Run all 8 fault injection experiments sequentially.
#
# Usage: ./run_all.sh <run_number>
#   e.g. ./run_all.sh 1
set -euo pipefail

RUN="${1:?Usage: $0 <run_number>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS="$REPO_ROOT/experiments"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_experiment() {
  local name="$1" yaml="$2"
  log "════ START: $name (run $RUN) ════"
  BASELINE_DURATION=600 FAULT_DURATION=300 RECOVERY_DURATION=300 \
  bash "$EXPERIMENTS/run_experiment.sh" "$name" "$yaml" "$RUN"
  log "════ DONE: $name ════"
}

run_experiment "cpu-stress-amf"         "k8s/chaos/01-cpu-stress-amf.yaml"
run_experiment "memory-pressure-upf"    "k8s/chaos/02-memory-pressure-upf.yaml"
run_experiment "pod-crash-amf"          "k8s/chaos/03-pod-crash-amf.yaml"
run_experiment "pod-crash-smf"          "k8s/chaos/07-pod-crash-smf.yaml"
run_experiment "network-delay"          "k8s/chaos/04-network-delay-gnb-amf.yaml"
run_experiment "network-partition"      "k8s/chaos/05-network-partition-amf-nrf.yaml"
run_experiment "dependency-failure-nrf" "k8s/chaos/06-dependency-failure-nrf.yaml"
run_experiment "network-delay-nrf"      "k8s/chaos/08-network-delay-nrf.yaml"

log "ALL EXPERIMENTS COMPLETE"
log "Data in: $REPO_ROOT/experiments/data/"
