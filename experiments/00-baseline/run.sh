#!/usr/bin/env bash
# 00-baseline/run.sh
#
# Phase 0: Baseline measurement — no observability stack active.
#
# Establishes the reference CPU/memory footprint of the 5G core under
# load with ZERO telemetry collection. All overhead numbers in later phases
# are deltas from this baseline.
#
# What this does:
#   1. Scales down Prometheus and Beyla (disables telemetry)
#   2. Provisions 50 subscribers and scales UERANSIM to 50 UEs
#   3. Runs a 10-minute STEADY-STATE window — collects kubectl top snapshots
#   4. Runs a 10-minute BURSTY window (UE attach/detach cycles)
#   5. Restores Prometheus and Beyla
#
# Data collected via kubectl top (not Prometheus — that's the point):
#   data/experiments/00-baseline/steady/prometheus/  → pod_top.csv, node_top.csv
#   data/experiments/00-baseline/bursty/prometheus/  → pod_top.csv, node_top.csv
#
# Usage: bash run.sh [--skip-steady] [--skip-bursty]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

STEADY_DURATION=600   # 10 minutes
BURSTY_DURATION=600   # 10 minutes
SAMPLE_INTERVAL=10    # kubectl top every 10s
UE_COUNT=50

SKIP_STEADY=false
SKIP_BURSTY=false
for arg in "$@"; do
    [[ "$arg" == "--skip-steady" ]] && SKIP_STEADY=true
    [[ "$arg" == "--skip-bursty" ]] && SKIP_BURSTY=true
done

OUT_BASE="$DATA_DIR/00-baseline"

echo "============================================================"
echo " Phase 0: Baseline (no telemetry)"
echo "============================================================"

check_cluster_ready

# ---------------------------------------------------------------------------
# Provision subscribers
# ---------------------------------------------------------------------------
echo "[setup] Provisioning $UE_COUNT subscribers..."
bash "$LIB_DIR/provision_ues.sh" "$UE_COUNT"

# ---------------------------------------------------------------------------
# Scale down observability
# ---------------------------------------------------------------------------
echo "[setup] Scaling down observability stack..."
kubectl scale statefulset -n monitoring \
    prometheus-kube-prom-kube-prometheus-prometheus --replicas=0 2>/dev/null || true
kubectl patch daemonset beyla -n open5gs \
    --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/nodeSelector","value":{"non-existing":"true"}}]' \
    2>/dev/null || true
echo "[setup] Observability scaled down"

# ---------------------------------------------------------------------------
# Scale UEs
# ---------------------------------------------------------------------------
scale_ues "$UE_COUNT"
wait_for_pods_stable open5gs 120

# ---------------------------------------------------------------------------
# Helper: collect kubectl top snapshots for a duration
# ---------------------------------------------------------------------------
collect_top_snapshots() {
    local out_dir="$1" duration="$2" label="$3"
    mkdir -p "$out_dir"
    local end_ts=$(( $(now_ts) + duration ))
    local snap=0

    echo "[collect] Sampling kubectl top every ${SAMPLE_INTERVAL}s for ${duration}s → $out_dir"

    local pods_csv="$out_dir/pod_top.csv"
    echo "timestamp,pod,namespace,cpu_cores_m,memory_mi" > "$pods_csv"
    local nodes_csv="$out_dir/node_top.csv"
    echo "timestamp,node,cpu_cores_m,memory_mi" > "$nodes_csv"

    while [[ $(now_ts) -lt $end_ts ]]; do
        local ts
        ts=$(now_ts)

        kubectl top pods -n open5gs --no-headers 2>/dev/null | while read -r pod cpu mem; do
            echo "$ts,$pod,open5gs,$cpu,$mem"
        done >> "$pods_csv" || true

        kubectl top nodes --no-headers 2>/dev/null | while read -r node cpu _cpupct mem _mempct; do
            echo "$ts,$node,$cpu,$mem"
        done >> "$nodes_csv" || true

        snap=$((snap+1))
        sleep "$SAMPLE_INTERVAL"
    done

    echo "[collect] $snap snapshots written"

    cat > "$out_dir/meta.json" <<EOF
{
  "phase": "00-baseline",
  "label": "$label",
  "ue_count": $UE_COUNT,
  "duration_s": $duration,
  "sample_interval_s": $SAMPLE_INTERVAL,
  "snapshots": $snap,
  "collected_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# ---------------------------------------------------------------------------
# Steady-state window
# ---------------------------------------------------------------------------
if ! $SKIP_STEADY; then
    echo ""
    echo "--- Steady-state window (${STEADY_DURATION}s) ---"
    log_experiment_start "00-baseline-steady" "$OUT_BASE/steady"
    collect_top_snapshots "$OUT_BASE/steady/prometheus" "$STEADY_DURATION" "steady"
    log_experiment_end "$OUT_BASE/steady"
    echo "[done] Steady-state baseline complete"
fi

# ---------------------------------------------------------------------------
# Bursty window
# ---------------------------------------------------------------------------
if ! $SKIP_BURSTY; then
    echo ""
    echo "--- Bursty window (${BURSTY_DURATION}s) ---"
    echo "[bursty] Simulating bursty traffic via UE scale up/down cycles..."
    log_experiment_start "00-baseline-bursty" "$OUT_BASE/bursty"

    (
        bursty_end=$(( $(date +%s) + BURSTY_DURATION ))
        cycle=0
        while [[ $(date +%s) -lt $bursty_end ]]; do
            cycle=$((cycle+1))
            echo "  [bursty] Cycle $cycle: scaling to $UE_COUNT UEs"
            helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
                --version 0.1.2 --namespace open5gs --reuse-values \
                --set ues.count="$UE_COUNT" --wait --timeout=2m 2>/dev/null || true
            sleep 30
            echo "  [bursty] Cycle $cycle: scaling to 5 UEs"
            helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
                --version 0.1.2 --namespace open5gs --reuse-values \
                --set ues.count=5 --wait --timeout=2m 2>/dev/null || true
            sleep 30
        done
    ) &
    BURSTY_PID=$!

    collect_top_snapshots "$OUT_BASE/bursty/prometheus" "$BURSTY_DURATION" "bursty"
    wait "$BURSTY_PID" 2>/dev/null || true

    log_experiment_end "$OUT_BASE/bursty"
    echo "[done] Bursty baseline complete"
fi

# ---------------------------------------------------------------------------
# Restore observability stack
# ---------------------------------------------------------------------------
echo ""
echo "[restore] Restoring observability stack..."
kubectl scale statefulset -n monitoring \
    prometheus-kube-prom-kube-prometheus-prometheus --replicas=1 2>/dev/null || true
echo "[restore] Waiting for Prometheus to be ready..."
kubectl rollout status statefulset/prometheus-kube-prom-kube-prometheus-prometheus \
    -n monitoring --timeout=5m 2>/dev/null || true
kubectl patch daemonset beyla -n open5gs \
    --type=json \
    -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector/non-existing"}]' 2>/dev/null || true
kubectl rollout status daemonset/beyla -n open5gs --timeout=2m 2>/dev/null || true
echo "[restore] Done"

echo ""
echo "============================================================"
echo " Phase 0 complete. Data in: $OUT_BASE"
echo "============================================================"
