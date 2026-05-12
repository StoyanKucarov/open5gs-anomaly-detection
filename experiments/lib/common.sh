#!/usr/bin/env bash
# experiments/lib/common.sh
#
# Shared helpers for all experiment scripts.
# Source this file at the top of each experiment script:
#   source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/experiments"
CHAOS_DIR="$REPO_ROOT/kind/chaos"

# Port-forward PIDs (tracked for cleanup)
_PF_PIDS=()

# Prometheus, Jaeger, and Loki URLs (set by ensure_portforward_*)
PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
JAEGER_URL="${JAEGER_URL:-http://127.0.0.1:16686}"
LOKI_URL="${LOKI_URL:-http://127.0.0.1:3100}"

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
_cleanup() {
    for pid in "${_PF_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap _cleanup EXIT

# ---------------------------------------------------------------------------
# Experiment Reset
# ---------------------------------------------------------------------------
reset_experiment_state() {
    local strategy="${1:-unknown}"
    local ue_count="${2:-50}" 
    echo "[reset] Executing Sequential Cold Start: $strategy"

    kkubectl exec -n monitoring svc/loki -- rm -rf /data/loki/chunks /data/loki/index /data/loki/boltdb-shipper-active /data/loki/compactor 2>/dev/null || true
    kubectl delete configmap -n monitoring loki-promtail-positions 2>/dev/null || true

    kubectl rollout restart -n monitoring statefulset/loki
    
    echo "  [reset] Waiting for Loki Rollout (180s)..."
    kubectl rollout status -n monitoring statefulset/loki --timeout=180s >/dev/null

    echo -n "  [reset] Waiting for Loki Pod readiness..."
    if kubectl wait --for=condition=ready pod/loki-0 -n monitoring --timeout=90s >/dev/null 2>&1; then
        echo " ready."
    else
        # Fallback: if name wait fails, try the label again with a broader scope
        echo -n " (using fallback selector)..."
        kubectl wait --for=condition=ready pod -n monitoring -l "app.kubernetes.io/instance=loki" --timeout=60s >/dev/null 2>&1 || \
        kubectl wait --for=condition=ready pod -n monitoring -l "app=loki" --timeout=60s >/dev/null 2>&1
        echo " ready."
    fi

    kubectl rollout restart daemonset -n monitoring loki-promtail

    echo "  [reset] Tier 1: Forced Restart of MongoDB and NRF..."
    local mongo_label="app.kubernetes.io/name=mongodb"
    kubectl delete pod -n open5gs -l "$mongo_label" --force --grace-period=0 2>/dev/null || true
    kubectl rollout restart deployment -n open5gs open5gs-nrf
    
    echo -n "  [reset] Waiting for MongoDB readiness..."
    kubectl wait --for=condition=ready pod -n open5gs -l "$mongo_label" --timeout=120s >/dev/null 2>&1
    echo " ready."

    echo "  [reset] Provisioning $ue_count subscribers..."
    kubectl scale deployment open5gs-populate -n open5gs --replicas=0 2>/dev/null || true
    kubectl delete pod -n open5gs -l app=open5gs-populate --force --grace-period=0 2>/dev/null || true
    
    bash "$LIB_DIR/provision_ues.sh" "$ue_count"

    echo "  [reset] Tier 2: Restarting remaining Network Functions..."
    kubectl get deployments -n open5gs -o name | grep -vE 'mongodb|nrf|populate' | xargs -r kubectl rollout restart -n open5gs

    echo -n "  [reset] Waiting for final stability..."
    wait_for_pods_stable open5gs 300
    
    sleep 20
    echo " done."
}
# ---------------------------------------------------------------------------
# Port-forward helpers
# ---------------------------------------------------------------------------

# start_portforward <namespace> <resource> <local_port> <remote_port>
start_portforward() {
    local ns="$1" resource="$2" local_port="$3" remote_port="$4"
    # Kill any stale process holding the port
    local stale
    stale=$(lsof -ti tcp:"$local_port" 2>/dev/null || true)
    [[ -n "$stale" ]] && kill "$stale" 2>/dev/null && sleep 1 || true
    kubectl port-forward -n "$ns" "$resource" "${local_port}:${remote_port}" \
        --address=127.0.0.1 >/dev/null 2>&1 &
    local pid=$!
    _PF_PIDS+=("$pid")
    # Wait until the port is actually open (max 60s)
    local i=0
    while ! (echo > /dev/tcp/127.0.0.1/"$local_port") 2>/dev/null; do
        sleep 1
        i=$((i+1))
        if [[ $i -ge 60 ]]; then
            echo "[ERROR] Port-forward to $resource:$remote_port never became ready" >&2
            return 1
        fi
    done
    echo "[pf] $resource → localhost:$local_port (pid $pid)"
}

# ensure_portforward_prometheus — idempotent, sets PROM_URL
ensure_portforward_prometheus() {
    PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
    if ! (echo > /dev/tcp/127.0.0.1/9090) 2>/dev/null; then
        start_portforward monitoring \
            svc/kube-prom-kube-prometheus-prometheus 9090 9090
    else
        echo "[pf] Prometheus already reachable at localhost:9090"
    fi
}

# ensure_portforward_jaeger — idempotent, sets JAEGER_URL
ensure_portforward_jaeger() {
    JAEGER_URL="${JAEGER_URL:-http://127.0.0.1:16686}"
    if ! (echo > /dev/tcp/127.0.0.1/16686) 2>/dev/null; then
        start_portforward monitoring svc/jaeger 16686 16686
    else
        echo "[pf] Jaeger already reachable at localhost:16686"
    fi
}

# ensure_portforward_loki — idempotent, sets LOKI_URL
ensure_portforward_loki() {
    LOKI_URL="${LOKI_URL:-http://127.0.0.1:3100}"
    if ! (echo > /dev/tcp/127.0.0.1/3100) 2>/dev/null; then
        start_portforward monitoring svc/loki 3100 3100
    else
        echo "[pf] Loki already reachable at localhost:3100"
    fi
}

# stop_portforward <local_port>
stop_portforward() {
    local port="$1"
    local pid
    pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

now_ts() { date +%s; }

# sleep_with_progress <seconds> <label>
sleep_with_progress() {
    local secs="$1" label="${2:-waiting}"
    echo -n "  [$label] ${secs}s "
    local i=0
    while [[ $i -lt $secs ]]; do
        sleep 10
        i=$((i+10))
        echo -n "."
    done
    echo " done"
}

# ---------------------------------------------------------------------------
# Cluster readiness
# ---------------------------------------------------------------------------

check_cluster_ready() {
    echo "[check] Verifying cluster context..."
    kubectl cluster-info --context kind-open5gs >/dev/null 2>&1 || {
        echo "[ERROR] Cluster kind-open5gs not reachable" >&2
        exit 1
    }
    kubectl config use-context kind-open5gs >/dev/null 2>&1
    echo "[check] Cluster ready"
}

# ---------------------------------------------------------------------------
# Pod stability
# ---------------------------------------------------------------------------

# wait_for_pods_stable <namespace> <timeout_seconds>
wait_for_pods_stable() {
    local ns="$1" timeout="${2:-120}"
    echo -n "  [wait] Waiting for all pods in $ns to be Running "
    local i=0
    while true; do
        local not_ready
        not_ready=$(kubectl get pods -n "$ns" --no-headers 2>/dev/null \
            | { grep -v -E "Running|Completed|Succeeded" || true; } | wc -l)
        if [[ "$not_ready" -eq 0 ]]; then
            echo " stable"
            return 0
        fi
        sleep 5
        i=$((i+5))
        echo -n "."
        if [[ $i -ge $timeout ]]; then
            echo " [WARN] pods not stable after ${timeout}s"
            return 0
        fi
    done
}

# ---------------------------------------------------------------------------
# UERANSIM UE scaling
# ---------------------------------------------------------------------------

# scale_ues <count>
scale_ues() {
    local count="$1"
    echo "[ues] Scaling UERANSIM UEs to $count..."
    helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
        --version 0.1.2 \
        --namespace open5gs \
        --reuse-values \
        --set ues.count="$count" \
        --wait --timeout=3m 2>/dev/null || \
    helm upgrade ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
        --version 0.1.2 \
        --namespace open5gs \
        --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
        --set ues.count="$count" \
        --wait --timeout=3m
    echo "[ues] Scaled to $count UEs"
}

# ---------------------------------------------------------------------------
# Prometheus scrape interval reconfiguration
# ---------------------------------------------------------------------------

# set_prometheus_scrape_interval <interval>  e.g. "1s", "5s", "15s"
set_prometheus_scrape_interval() {
    local interval="$1"
    echo "[prom] Scrape interval set to $interval"
    helm upgrade kube-prom prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --reuse-values \
        --set prometheus.prometheusSpec.scrapeInterval="$interval" \
        --wait --timeout=3m 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Data collection wrappers
# ---------------------------------------------------------------------------

# collect_prometheus <start_ts> <end_ts> <step> <out_dir>
collect_prometheus() {
    local start="$1" end="$2" step="$3" out_dir="$4"
    mkdir -p "$out_dir"
    python3 "$LIB_DIR/collect_prometheus.py" \
        --url "$PROM_URL" \
        --start "$start" \
        --end   "$end" \
        --step  "$step" \
        --out   "$out_dir"
}

# collect_jaeger <start_ts> <end_ts> <out_dir>
collect_jaeger() {
    local start="$1" end="$2" out_dir="$3"
    mkdir -p "$out_dir"
    python3 "$LIB_DIR/collect_jaeger.py" \
        --url   "$JAEGER_URL" \
        --start "$start" \
        --end   "$end" \
        --out   "$out_dir"
}

# collect_loki <start_ts> <end_ts> <out_dir>
collect_loki() {
    local start="$1" end="$2" out_dir="$3"
    mkdir -p "$out_dir"
    python3 "$LIB_DIR/collect_loki.py" \
        --url   "$LOKI_URL" \
        --start "$start" \
        --end   "$end" \
        --out   "$out_dir"
}

# collect_events <start_ts> <end_ts> <out_dir>
collect_events() {
    local start="$1" end="$2" out_dir="$3"
    mkdir -p "$out_dir"
    python3 "$LIB_DIR/collect_events.py" \
        --namespace open5gs \
        --start "$start" \
        --end   "$end" \
        --out   "$out_dir"
}

# collect_nrf <out_dir> — snapshots current NRF instance counts (no time window)
collect_nrf() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    python3 "$LIB_DIR/collect_nrf.py" \
        --namespace open5gs \
        --out   "$out_dir"
}

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

log_experiment_start() {
    local name="$1" out_dir="$2"
    mkdir -p "$out_dir"
    echo "{\"experiment\": \"$name\", \"started_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
        > "$out_dir/meta.json"
    echo "[meta] $name started"
}

log_experiment_end() {
    local out_dir="$1"
    local meta="$out_dir/meta.json"
    if [[ -f "$meta" ]]; then
        python3 -c "
import json, datetime
with open('$meta') as f: d = json.load(f)
d['ended_at'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
with open('$meta', 'w') as f: json.dump(d, f, indent=2)
" 2>/dev/null || true
    fi
}
