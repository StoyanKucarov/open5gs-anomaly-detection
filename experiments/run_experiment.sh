#!/usr/bin/env bash
# Run one fault-injection experiment: baseline → inject → fault phase → recover.
# Generates continuous data-plane and control-plane traffic throughout.
#
# Usage:
#   ./run_experiment.sh <fault_name> <chaos_yaml> <run_number>
#
# Tunable durations (seconds):
#   BASELINE_DURATION   default 300  (5 min)
#   FAULT_DURATION      default 300  (5 min)
#   RECOVERY_DURATION   default 180  (3 min)

set -euo pipefail

FAULT_NAME="${1:?Usage: $0 <fault_name> <chaos_yaml> <run_number>}"
CHAOS_YAML="${2:?Usage: $0 <fault_name> <chaos_yaml> <run_number>}"
RUN_NUMBER="${3:?Usage: $0 <fault_name> <chaos_yaml> <run_number>}"

BASELINE_DURATION="${BASELINE_DURATION:-300}"
FAULT_DURATION="${FAULT_DURATION:-300}"
RECOVERY_DURATION="${RECOVERY_DURATION:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/experiments/data"
RUN_DIR="$DATA_DIR/$FAULT_NAME/run_$(printf '%02d' "$RUN_NUMBER")"

UE_POD=""

# ── Port-forward helpers ──────────────────────────────────────────────────────

PF_PROM_PID=""
PF_JAEGER_PID=""
PF_LOKI_PID=""

start_port_forwards() {
  echo "[pf] starting port-forwards..."
  kubectl port-forward -n monitoring svc/kube-prom-kube-prometheus-prometheus 9090:9090 \
    >"$RUN_DIR/pf-prom.log" 2>&1 &
  PF_PROM_PID=$!

  kubectl port-forward -n monitoring svc/jaeger 16686:16686 \
    >"$RUN_DIR/pf-jaeger.log" 2>&1 &
  PF_JAEGER_PID=$!

  kubectl port-forward -n monitoring svc/loki 3100:3100 \
    >"$RUN_DIR/pf-loki.log" 2>&1 &
  PF_LOKI_PID=$!

  sleep 4
  echo "[pf] pids: prom=$PF_PROM_PID jaeger=$PF_JAEGER_PID loki=$PF_LOKI_PID"
}

stop_port_forwards() {
  echo "[pf] stopping port-forwards..."
  kill "$PF_PROM_PID" "$PF_JAEGER_PID" "$PF_LOKI_PID" 2>/dev/null || true
}

# ── Traffic generation ────────────────────────────────────────────────────────

TRAFFIC_PID=""
REREGISTER_PID=""

start_traffic() {
  UE_POD=$(kubectl get pods -n open5gs -l app.kubernetes.io/component=ues \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || \
    kubectl get pods -n open5gs | grep -m1 ues | awk '{print $1}')
  echo "[traffic] UE pod: $UE_POD"

  # 1. Continuous pings through all TUN interfaces (data plane via UPF)
  kubectl exec -n open5gs "$UE_POD" -- bash -c '
    for i in $(seq 0 9); do
      ip link show uesimtun$i >/dev/null 2>&1 && \
        ping -i 0.5 -W 1 -I uesimtun$i 8.8.8.8 >/dev/null 2>&1 &
    done
    wait
  ' >"$RUN_DIR/traffic-pids.txt" 2>&1 &
  TRAFFIC_PID=$!
  echo "[traffic] data-plane pings started"

  # 2. Periodic re-registration loop (control plane: NGAP + SBI auth chain)
  #    Cycles 4 UEs through deregister→register every 15s
  (
    UEs=("imsi-999700000000003" "imsi-999700000000004" \
         "imsi-999700000000005" "imsi-999700000000006")
    while true; do
      sleep 15
      for ue in "${UEs[@]}"; do
        kubectl exec -n open5gs "$UE_POD" -- \
          nr-cli "$ue" --exec "deregister normal" 2>/dev/null || true
      done
      sleep 5
      for ue in "${UEs[@]}"; do
        kubectl exec -n open5gs "$UE_POD" -- \
          nr-cli "$ue" --exec "register" 2>/dev/null || true
      done
    done
  ) &
  REREGISTER_PID=$!
  echo "[traffic] control-plane re-registration loop started (pid=$REREGISTER_PID)"
}

stop_traffic() {
  echo "[traffic] stopping..."
  kill "$REREGISTER_PID" 2>/dev/null || true
  # Kill pings inside the UE pod
  kubectl exec -n open5gs "$UE_POD" -- \
    bash -c 'pkill ping 2>/dev/null; true' 2>/dev/null || true
}

# ── Recovery helpers ──────────────────────────────────────────────────────────

warn_post_recovery() {
  case "$FAULT_NAME" in
    memory-pressure-upf)
      [ -n "$MEM_ALLOC_PID" ] && kill "$MEM_ALLOC_PID" 2>/dev/null || true
      echo "  NOTE: UPF OOM-killed. Waiting for restart then restarting SMF to clear stale PFCP state..."
      until kubectl get pod -n open5gs -l app.kubernetes.io/name=upf \
          --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; do
        sleep 3
      done
      sleep 5
      kubectl rollout restart deployment/open5gs-smf -n open5gs
      kubectl rollout status  deployment/open5gs-smf -n open5gs --timeout=60s
      ;;
    pod-crash-amf)
      echo "  NOTE: AMF was killed. Restarting gNB and UEs..."
      kubectl rollout restart deployment/ueransim-gnb -n open5gs
      kubectl rollout status  deployment/ueransim-gnb -n open5gs --timeout=60s
      kubectl rollout restart deployment/ueransim-gnb-ues -n open5gs
      kubectl rollout status  deployment/ueransim-gnb-ues -n open5gs --timeout=60s
      sleep 15
      ;;
    pod-crash-smf)
      echo "  NOTE: SMF was killed. Restarting SMF and clearing PFCP state..."
      kubectl rollout restart deployment/open5gs-smf -n open5gs
      kubectl rollout status  deployment/open5gs-smf -n open5gs --timeout=60s
      sleep 15
      ;;
    dependency-failure-nrf)
      echo "  NOTE: NRF was killed. Waiting for NF re-registration..."
      sleep 30
      ;;
  esac
}

# ── Helpers ───────────────────────────────────────────────────────────────────

now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

collect_phase() {
  local phase="$1" start="$2" end="$3"
  python3 "$SCRIPT_DIR/collect.py" \
    --fault  "$FAULT_NAME" \
    --run    "$RUN_NUMBER" \
    --phase  "$phase" \
    --start  "$start" \
    --end    "$end" \
    --output "$DATA_DIR"
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────

cleanup() {
  stop_traffic
  stop_port_forwards
}
trap cleanup EXIT

# ── Main ──────────────────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR"
start_port_forwards
start_traffic

echo ""
echo "════════════════════════════════════════════════════"
echo "  fault : $FAULT_NAME"
echo "  yaml  : $CHAOS_YAML"
echo "  run   : $RUN_NUMBER"
echo "  output: $RUN_DIR"
echo "════════════════════════════════════════════════════"

# ── Phase 1: Baseline ─────────────────────────────────────────────────────────
echo ""
echo "── [1/4] BASELINE (${BASELINE_DURATION}s) ───────────────────────────────"
BASELINE_START=$(now_iso)
echo "  start: $BASELINE_START"
sleep "$BASELINE_DURATION"
BASELINE_END=$(now_iso)
echo "  end:   $BASELINE_END"
collect_phase baseline "$BASELINE_START" "$BASELINE_END"

# ── Phase 2: Inject ───────────────────────────────────────────────────────────
echo ""
echo "── [2/4] INJECTING ──────────────────────────────────────────────────────"
INJECTION_TIME=$(now_iso)
echo "  injection_time: $INJECTION_TIME"
kubectl apply -f "$REPO_ROOT/$CHAOS_YAML"
sleep 6
echo "  chaos status:"
kubectl get stresschaos,podchaos,networkchaos -n open5gs --no-headers 2>/dev/null || true

# For memory-pressure-upf: also allocate memory INSIDE the UPF container to
# trigger the 128Mi container memory limit and force an OOM kill.
# StressChaos alone runs in chaos-daemon's cgroup and cannot hit the container limit.
MEM_ALLOC_PID=""
if [[ "$FAULT_NAME" == "memory-pressure-upf" ]]; then
  UPF_POD=$(kubectl get pods -n open5gs -l app.kubernetes.io/name=upf \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  echo "  [mem] allocating 150MB inside UPF container ($UPF_POD) to trigger OOM..."
  kubectl exec -n open5gs "$UPF_POD" -c open5gs-upf -- \
    perl -e 'my $x = "a" x (150*1024*1024); print "allocated 150MB\n"; sleep 400' \
    2>/dev/null &
  MEM_ALLOC_PID=$!
  echo "  [mem] allocator pid=$MEM_ALLOC_PID (will be OOM-killed with the container)"
fi

# ── Phase 3: Fault phase ──────────────────────────────────────────────────────
echo ""
echo "── [3/4] FAULT PHASE (${FAULT_DURATION}s) ──────────────────────────────"
FAULT_START=$(now_iso)

# For network-delay experiments: collect RTT samples from AMF→SCP during fault.
# Chaos Mesh applies delay at TC layer (below Beyla), so RTT is the only metric
# that captures it. Saves to rtt_samples.json in the fault phase directory.
RTT_PID=""
if [[ "$FAULT_NAME" == network-delay* || "$FAULT_NAME" == network-partition* ]]; then
  (
    AMF_POD=$(kubectl get pods -n open5gs -l app.kubernetes.io/name=amf \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    SCP_IP=$(kubectl get pods -n open5gs -l app.kubernetes.io/name=scp \
      -o jsonpath='{.items[0].status.podIP}' 2>/dev/null)
    OUT="$DATA_DIR/$FAULT_NAME/run_$(printf '%02d' "$RUN_NUMBER")/fault/rtt_samples.txt"
    mkdir -p "$(dirname "$OUT")"
    echo "# RTT samples (ms): AMF→SCP ping during fault phase" > "$OUT"
    if [[ "$FAULT_NAME" == network-partition* ]]; then
      # For partition: short burst to confirm packet loss (no replies expected)
      PING_OUT=$(kubectl exec -n open5gs "$AMF_POD" -c open5gs-amf -- \
        ping -i 1 -W 1 -c 10 "$SCP_IP" 2>/dev/null || true)
      echo "$PING_OUT" | grep -oP '\d+% packet loss' >> "$OUT" || true
    else
      # For delay: full-duration ping to measure RTT samples
      PING_OUT=$(kubectl exec -n open5gs "$AMF_POD" -c open5gs-amf -- \
        ping -i 1 -W 3 -c "$FAULT_DURATION" "$SCP_IP" 2>/dev/null || true)
      echo "$PING_OUT" | grep -oP 'time=\K[\d.]+' >> "$OUT" || true
      echo "$PING_OUT" | grep -oP '\d+% packet loss' >> "$OUT" || true
    fi
  ) &
  RTT_PID=$!
  echo "  [rtt] measuring AMF→SCP ping RTT (pid=$RTT_PID)"
fi

sleep "$FAULT_DURATION"
FAULT_END=$(now_iso)

# Kill RTT collector if still running
[ -n "$RTT_PID" ] && kill "$RTT_PID" 2>/dev/null || true

collect_phase fault "$FAULT_START" "$FAULT_END"

# ── Phase 4: Recovery ─────────────────────────────────────────────────────────
echo ""
echo "── [4/4] RECOVERY (${RECOVERY_DURATION}s) ──────────────────────────────"
RECOVERY_START=$(now_iso)
kubectl delete -f "$REPO_ROOT/$CHAOS_YAML" --ignore-not-found &
DELETE_PID=$!
sleep 15
if kill -0 "$DELETE_PID" 2>/dev/null; then
  echo "  [warn] kubectl delete still running after 15s — patching finalizers..."
  for kind in networkchaos stresschaos podchaos; do
    kubectl get "$kind" -n open5gs --no-headers 2>/dev/null \
      | awk '{print $1}' | while read -r name; do
        kubectl patch "$kind/$name" -n open5gs --type='json' \
          -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null || true
      done
  done
  wait "$DELETE_PID" 2>/dev/null || true
fi
echo "  chaos deleted at: $RECOVERY_START"
warn_post_recovery
sleep "$RECOVERY_DURATION"
RECOVERY_END=$(now_iso)
collect_phase recovery "$RECOVERY_START" "$RECOVERY_END"

# ── Save timestamps ───────────────────────────────────────────────────────────
printf '{
  "fault":          "%s",
  "run":            %d,
  "baseline_start": "%s",
  "baseline_end":   "%s",
  "injection_time": "%s",
  "fault_start":    "%s",
  "fault_end":      "%s",
  "recovery_start": "%s",
  "recovery_end":   "%s"
}\n' \
  "$FAULT_NAME" "$RUN_NUMBER" \
  "$BASELINE_START" "$BASELINE_END" \
  "$INJECTION_TIME" \
  "$FAULT_START" "$FAULT_END" \
  "$RECOVERY_START" "$RECOVERY_END" \
  > "$RUN_DIR/timestamps.json"

echo ""
echo "════════════════════════════════════════════════════"
echo "  COMPLETE — $FAULT_NAME run $RUN_NUMBER"
echo "  data: $RUN_DIR"
echo "════════════════════════════════════════════════════"
