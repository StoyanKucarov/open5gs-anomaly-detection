#!/usr/bin/env bash
# Bring up the open5gs k3d cluster and fix DNS inside k3s nodes.
# Run this after every reboot or Docker restart.
#
# Usage: ./cluster-start.sh [--skip-dns-fix]
set -euo pipefail

CLUSTER=open5gs
NODES=(k3d-open5gs-server-0 k3d-open5gs-agent-0 k3d-open5gs-agent-1)
DNS=8.8.8.8

# --- 1. Fix Docker iptables chains if missing (corrupted after Docker restart) ---
echo "[1/4] Checking Docker iptables chains..."
if ! sudo iptables -t filter -L DOCKER-ISOLATION-STAGE-2 &>/dev/null; then
  echo "  -> Chains missing, pre-creating..."
  sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-1 2>/dev/null || true
  sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-2 2>/dev/null || true
  echo "  -> Done"
else
  echo "  -> OK"
fi

# --- 2. Start cluster ---
echo "[2/4] Starting cluster '$CLUSTER'..."
k3d cluster start "$CLUSTER"

# --- 3. Fix DNS in k3s nodes (ephemeral — wiped on each stop/start) ---
if [[ "${1:-}" != "--skip-dns-fix" ]]; then
  echo "[3/4] Fixing DNS in k3s nodes (nameserver $DNS)..."
  for node in "${NODES[@]}"; do
    docker exec "$node" sh -c "echo 'nameserver $DNS' > /etc/resolv.conf"
    echo "  -> $node OK"
  done
fi

# --- 4. Check Chaos Mesh health (informational — it persists in the cluster) ---
echo "[4/4] Checking Chaos Mesh status..."
CHAOS_NOTREADY=$(kubectl get pods -n chaos-mesh --no-headers 2>/dev/null \
  | grep -v "Running\|Completed" | wc -l)
if [[ "$CHAOS_NOTREADY" -eq 0 ]]; then
  echo "  -> OK (all chaos-mesh pods Running)"
else
  echo "  -> WARNING: some chaos-mesh pods are not Ready:"
  kubectl get pods -n chaos-mesh --no-headers | grep -v "Running\|Completed" || true
  echo "  -> If controller-manager is crashing, check inotify limits:"
  echo "       sudo sysctl fs.inotify.max_user_instances=512"
  echo "       sudo sysctl fs.inotify.max_user_watches=524288"
  echo "     Then: kubectl rollout restart deployment -n chaos-mesh"
fi

echo ""
echo "Cluster ready. Verifying nodes..."
kubectl get nodes
