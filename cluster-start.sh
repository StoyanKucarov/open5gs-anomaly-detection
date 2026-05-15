#!/usr/bin/env bash
# Recreate the open5gs kind cluster and redeploy the full stack.
# Run this after every reboot or Docker restart (Option A: always recreate).
#
# Usage: ./cluster-start.sh [--skip-deploy]
#   --skip-deploy   Recreate the cluster only; skip Helm installs (useful if
#                   you want to deploy manually or iterate on values).
set -euo pipefail

CLUSTER=open5gs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND_CONFIG="$SCRIPT_DIR/kind/kind-config.yaml"

SKIP_DEPLOY=false
[[ "${1:-}" == "--skip-deploy" ]] && SKIP_DEPLOY=true

# --- 1. Fix Docker iptables chains if missing (Docker 29 nftables bug) --------
echo "[1/5] Checking Docker iptables chains..."
if ! sudo iptables -t filter -L DOCKER-ISOLATION-STAGE-2 &>/dev/null; then
  echo "  -> Chains missing, pre-creating..."
  sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-1 2>/dev/null || true
  sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-2 2>/dev/null || true
  echo "  -> Done"
else
  echo "  -> OK"
fi

# --- 2. Tear down any existing cluster and recreate ---------------------------
echo "[2/5] Recreating kind cluster '$CLUSTER'..."
kind delete cluster --name "$CLUSTER" 2>/dev/null || true
kind create cluster --config "$KIND_CONFIG"
echo "  -> Cluster created"
kubectl get nodes

# --- 3. Raise inotify limits (required for Promtail + Chaos Mesh controller) --
echo "[3/5] Checking inotify limits..."
INSTANCES=$(sysctl -n fs.inotify.max_user_instances)
WATCHES=$(sysctl -n fs.inotify.max_user_watches)
if [[ "$INSTANCES" -lt 512 || "$WATCHES" -lt 524288 ]]; then
  echo "  -> Raising limits (current: instances=$INSTANCES watches=$WATCHES)..."
  sudo sysctl fs.inotify.max_user_instances=512
  sudo sysctl fs.inotify.max_user_watches=524288
else
  echo "  -> OK (instances=$INSTANCES watches=$WATCHES)"
fi

# --- 4. Deploy full stack (unless --skip-deploy) ------------------------------
if $SKIP_DEPLOY; then
  echo "[4/5] Skipping deploy (--skip-deploy)"
else
  echo "[4/5] Deploying full stack..."

  # ── Observability ──────────────────────────────────────────────────────────
  # Must come before Open5GS — Open5GS references ServiceMonitor CRDs which
  # are installed by kube-prometheus-stack.
  echo "  [4a] Observability stack..."
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
  helm repo add grafana               https://grafana.github.io/helm-charts             2>/dev/null || true
  helm repo add jaegertracing         https://jaegertracing.github.io/helm-charts       2>/dev/null || true
  helm repo update

  kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

  helm install kube-prom prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --set grafana.adminPassword=admin \
    --set prometheus.prometheusSpec.scrapeInterval=5s \
    --timeout=10m

  # ── Open5GS ────────────────────────────────────────────────────────────────
  echo "  [4b] Open5GS..."
  kubectl create namespace open5gs --dry-run=client -o yaml | kubectl apply -f -
  helm install open5gs oci://registry-1.docker.io/gradiantcharts/open5gs \
    --version 2.3.4 \
    --namespace open5gs \
    -f "$SCRIPT_DIR/kind/open5gs-values.yaml" \
    --wait --timeout=10m
  kubectl delete deployment -n open5gs open5gs-webui --ignore-not-found

  # ── UERANSIM ───────────────────────────────────────────────────────────────
  echo "  [4c] UERANSIM gNB + UEs..."
  helm install ueransim-gnb oci://registry-1.docker.io/gradiant/ueransim-gnb \
    --version 0.2.6 --namespace open5gs \
    --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
    --set ues.count=10 \
    --wait --timeout=5m
  helm install ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
    --version 0.1.2 --namespace open5gs \
    --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
    --wait --timeout=5m

  echo "  [4d] Provisioning subscribers..."
  bash "$SCRIPT_DIR/experiments/lib/provision_ues.sh" 10
  kubectl rollout restart deployment/ueransim-gnb-ues -n open5gs
  kubectl rollout status  deployment/ueransim-gnb-ues -n open5gs --timeout=60s

  helm install loki grafana/loki-stack \
    --namespace monitoring \
    --set promtail.enabled=true \
    --set loki.persistence.enabled=false \
    --set grafana.enabled=false \
    --set loki.isDefault=false

  helm install jaeger jaegertracing/jaeger \
    --namespace monitoring \
    --set allInOne.enabled=true \
    --set storage.type=memory \
    --set agent.enabled=false --set collector.enabled=false --set query.enabled=false \
    --timeout=5m

  kubectl apply -f "$SCRIPT_DIR/kind/monitoring/beyla-daemonset.yaml"

  # ── Chaos Mesh ─────────────────────────────────────────────────────────────
  echo "  [4d] Chaos Mesh..."
  helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
  helm repo update
  helm install chaos-mesh chaos-mesh/chaos-mesh \
    --namespace chaos-mesh --create-namespace \
    --version 2.7.2 \
    --set chaosDaemon.runtime=containerd \
    --set chaosDaemon.socketPath=/run/containerd/containerd.sock

  echo "  -> Waiting for Chaos Mesh to be ready..."
  kubectl rollout status deployment/chaos-controller-manager -n chaos-mesh --timeout=7m
  
  # ── Metrics Server (Required for kubectl top) ──────────────────────────────
  echo "  [4e] Metrics Server..."
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  kubectl patch -n kube-system deployment metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

fi

# --- 5. Sanity checks ---------------------------------------------------------
echo "[5/5] Sanity checks..."
echo "  Nodes:"
kubectl get nodes
echo "  open5gs pods:"
kubectl get pods -n open5gs
echo "  monitoring pods:"
kubectl get pods -n monitoring
echo "  chaos-mesh pods:"
kubectl get pods -n chaos-mesh

echo ""
echo "Cluster ready."
echo "  Port-forward Grafana:    kubectl port-forward -n monitoring deployment/kube-prom-grafana 3000:3000"
echo "  Port-forward Prometheus: kubectl port-forward -n monitoring svc/kube-prom-kube-prometheus-prometheus 9090:9090"
echo "  Port-forward Jaeger:     kubectl port-forward -n monitoring svc/jaeger 16686:16686"
