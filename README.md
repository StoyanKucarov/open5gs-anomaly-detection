# Cloud-Native 5G Fault Injection — Reproduction Pack

Everything needed to stand up the platform and run the fault-injection experiments
that produce the raw telemetry for the fault atlas study.

This pack only covers **platform setup + data collection**. Analysis is out of scope here.

> **Extended setup notes:** See [`EXTENSIONS.md`](./EXTENSIONS.md) for the full
> list of changes added on top of this base — extra signal collectors (Loki,
> K8s events, NRF API, RTT), synthetic traffic generators, per-fault hooks,
> bug fixes encountered during bring-up, and the Phase-3 fault dispatch.
>
> ⚠️ **Current state (2026-05-16): read [`EXTENSIONS.md` §10](./EXTENSIONS.md#10-pipeline-hardening--2026-05-16) first.**
> It supersedes several numbers below — the pipeline now runs **22 faults**,
> default durations are **600/300/300** (no env vars needed), `run_all.sh`
> does a **full cluster recreate per fault** (soft-reset is shelved), bring-up
> is gated on strict 10/10 readiness, Loki collection is **paginated** (the
> 5000-line cap is gone) with Beyla logs excluded, and **every teammate must
> create their own gitignored `kind/.dockerhub-auth`** (username + read-only
> PAT, two lines) or the cluster will hit Docker Hub pull rate limits.

---

## 1. Stack

| Layer           | Tool                                     | Version       | Why this one                                                                                                                                                                                                                                          |
| --------------- | ---------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5G core         | Open5GS (Gradiant Helm chart)            | 2.3.4         | Most complete OSS 5G core; all SBI NFs as separate pods → clean per-NF fault targeting                                                                                                                                                                |
| RAN sim         | UERANSIM gNB + UEs (Gradiant Helm chart) | 0.2.6 / 0.1.2 | Standard pairing for Open5GS; works headless in k8s                                                                                                                                                                                                   |
| Orchestration   | kind (Kubernetes in Docker)              | latest        | Single-host k8s, no cloud dependency, fast teardown                                                                                                                                                                                                   |
| Metrics         | kube-prometheus-stack                    | latest        | Bundles Prometheus + Grafana + node_exporter + cAdvisor + kube-state-metrics                                                                                                                                                                          |
| Logs            | Loki + Promtail                          | latest        | Promtail DaemonSet ships every pod's stdout to Loki; no app changes                                                                                                                                                                                   |
| Traces          | Jaeger (all-in-one, in-memory)           | v4.7.0        | No code changes; in-mem is fine for short experiment windows                                                                                                                                                                                          |
| Span source     | Beyla (eBPF auto-instrumentation)        | ≥ 3.9.5       | Hooks at kernel socket layer, sees Open5GS HTTP/2 prior-knowledge SBI calls. **Istio sidecars don't work** (Envoy can't proxy raw HTTP/2 without TLS/ALPN). 3.9.5+ is required — earlier versions hit the kernel verifier's 1M-instruction BPF limit. |
| Fault injection | Chaos Mesh                               | 2.7.2         | Native k8s CRDs (StressChaos / PodChaos / NetworkChaos), no test-harness code                                                                                                                                                                         |
| Collection      | Python 3 + `requests`                    | —             | `experiments/collect.py` queries Prometheus / Loki / Jaeger HTTP APIs                                                                                                                                                                                 |

### Key design decisions

- **kind, not minikube/k3d.** Multi-node out of the box via `kind-config.yaml`; containerd socket at `/run/containerd/containerd.sock` integrates cleanly with Chaos Mesh.
- **Indirect SBI (Model D) via SCP.** All NF-to-NF SBI traffic in Open5GS goes through the SCP. AMF never talks directly to NRF, so the network-partition experiment targets **AMF↔SCP**, not AMF↔NRF.
- **Synthetic traffic during every run.** Without traffic, a fault produces no observable signal. `run_experiment.sh` runs continuous data-plane pings (UE TUN → 8.8.8.8 via UPF) and a control-plane re-registration loop (4 UEs cycling deregister/register every 15s, exercising NGAP + AMF + AUSF + UDM + NRF + SCP).
- **4-phase model.** Each experiment: baseline 600s → inject → fault 300s → recovery 300s. Each phase is collected separately so deltas vs baseline are computable.
- **Pod-level memory metric for memory-pressure.** StressChaos memory runs in chaos-daemon's cgroup, _not_ the target container's. We additionally allocate memory **inside** the UPF container (perl trick in `run_experiment.sh`) to actually hit the 128Mi limit and trigger an OOM kill.
- **RTT collection for network-delay/partition.** Chaos Mesh applies delay at the TC kernel layer; Beyla eBPF sits above TC and is blind to it. The script runs `ping AMF→SCP` during the fault phase as the only way to confirm the delay.
- **Resource limits enforced on every NF.** Without limits, CPU stressors don't throttle and memory stressors don't OOM. Limits are set in `k8s/open5gs-values.yaml` (AMF/UPF 128 Mi & 500m; NRF 64 Mi & 200m; etc).
- **inotify limits raised.** Chaos Mesh controller and Promtail both consume many inotify watches; the kernel default crashes them.

---

## 2. Host prerequisites (Linux + Docker)

- Docker installed and running (Docker 29 with nftables works; see iptables note in §6).
- Raise inotify limits — required for Promtail + Chaos Mesh controller, otherwise both crash with `too many open files`:

```bash
sudo sysctl fs.inotify.max_user_instances=512
sudo sysctl fs.inotify.max_user_watches=524288

# Persist across reboots
echo "fs.inotify.max_user_instances=512"   | sudo tee -a /etc/sysctl.conf
echo "fs.inotify.max_user_watches=524288"  | sudo tee -a /etc/sysctl.conf
```

- Python 3 with `requests`:

```bash
pip install --user requests
```

---

## 3. Install CLIs

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl

# kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
sudo install -o root -g root -m 0755 kind /usr/local/bin/kind && rm kind

# helm
curl -L https://get.helm.sh/helm-v3.20.2-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/helm && rm -rf linux-amd64
```

---

## 4. Create the cluster

```bash
kind create cluster --config k8s/kind-config.yaml

kubectl get nodes   # expect: open5gs-control-plane, open5gs-worker, open5gs-worker2 all Ready
```

Flag rationale (set in `k8s/kind-config.yaml`):

- `evictionHard: nodefs.available: "5%"` — kind nodes share host disk; default ~10–15% threshold trips DiskPressure on a moderately full disk and blocks pod scheduling.

---

## 5. Deploy everything (run from the repo root containing this `reproduce/` folder)

```bash
cd reproduce

# ── Open5GS ───────────────────────────────────────────────────────────────────
kubectl create namespace open5gs
helm install open5gs oci://registry-1.docker.io/gradiantcharts/open5gs \
  --version 2.3.4 \
  --namespace open5gs \
  -f k8s/open5gs-values.yaml \
  --wait --timeout=10m

# Optional: webui pod isn't needed for experiments
kubectl delete deployment -n open5gs open5gs-webui --ignore-not-found

# ── UERANSIM gNB + UEs ───────────────────────────────────────────────────────
helm install ueransim-gnb oci://registry-1.docker.io/gradiant/ueransim-gnb \
  --version 0.2.6 --namespace open5gs \
  --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
  --wait --timeout=5m

helm install ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
  --version 0.1.2 --namespace open5gs \
  --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
  --wait --timeout=5m

# Sanity check — should print "PDU Session establishment is successful"
kubectl logs -n open5gs deployment/ueransim-ues | grep -i "PDU Session"
kubectl exec  -n open5gs deployment/ueransim-ues -- ping -I uesimtun0 -c 3 8.8.8.8

# ── Observability stack ──────────────────────────────────────────────────────
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana               https://grafana.github.io/helm-charts
helm repo add jaegertracing         https://jaegertracing.github.io/helm-charts
helm repo update

kubectl create namespace monitoring

helm install kube-prom prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.scrapeInterval=5s \
  --timeout=10m

helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set promtail.enabled=true \
  --set loki.persistence.enabled=false \
  --set grafana.enabled=false   # avoid datasource conflict with kube-prom's Grafana

helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --set allInOne.enabled=true \
  --set storage.type=memory \
  --set agent.enabled=false --set collector.enabled=false --set query.enabled=false \
  --timeout=5m

# Beyla DaemonSet (eBPF tracer → Jaeger)
kubectl apply -f k8s/monitoring/beyla-daemonset.yaml

# ── Chaos Mesh ───────────────────────────────────────────────────────────────
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update

helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace chaos-mesh --create-namespace \
  --version 2.7.2 \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock
# kind uses containerd directly at /run/containerd/containerd.sock

# ── Verify ───────────────────────────────────────────────────────────────────
kubectl get pods -n open5gs
kubectl get pods -n monitoring
kubectl get pods -n chaos-mesh
```

Add Loki and Jaeger as Grafana datasources (Prometheus is already default):

```bash
kubectl port-forward -n monitoring deployment/kube-prom-grafana 3000:3000
# → http://localhost:3000  (admin / admin)
#   Connections → Data sources → Add:
#     Loki    http://loki.monitoring.svc.cluster.local:3100
#     Jaeger  http://jaeger.monitoring.svc.cluster.local:16686
```

---

## 6. Daily lifecycle — `./cluster-start.sh`

After every reboot or Docker restart, run:

```bash
./cluster-start.sh              # recreate cluster + redeploy everything (~10–15 min)
./cluster-start.sh --skip-deploy  # recreate cluster only, deploy manually afterwards
```

It does five things:

1. **Pre-creates `DOCKER-ISOLATION-STAGE-{1,2}` iptables chains.** Docker 29 (nftables backend) sometimes drops these on dirty restart, then all Docker network creation fails.
2. **Deletes and recreates the kind cluster** (`kind delete cluster` + `kind create cluster --config k8s/kind-config.yaml`). This is always a clean slate — no stale state from the previous run.
3. **Checks and raises inotify limits** if below the required thresholds (512 instances / 524288 watches). Promtail and Chaos Mesh controller both crash without these.
4. **Redeploys the full stack** (Open5GS, UERANSIM, kube-prometheus-stack, Loki, Jaeger, Beyla, Chaos Mesh) unless `--skip-deploy` is passed.
5. **Sanity-checks** all three namespaces.

> **Note:** kind has no `cluster start/stop` command. The cluster lives as Docker containers; when Docker stops (reboot, `docker restart`), those containers stop. The only safe recovery is to recreate — hence Option A.

**Don't restart Docker while experiments are running** — it will corrupt the cluster state. Stop experiments first.

---

## 7. Run experiments

Run all 8 sequentially (each takes ~20 min: 600 + 300 + 300 + overhead):

```bash
bash experiments/run_all.sh 1 > experiments/run1.log 2>&1 &
tail -f experiments/run1.log
```

Or one at a time:

```bash
BASELINE_DURATION=600 FAULT_DURATION=300 RECOVERY_DURATION=300 \
  bash experiments/run_experiment.sh <fault_name> k8s/chaos/<file>.yaml <run_number>
```

### What each experiment does

| #   | Fault name               | YAML                                | Class            | Target          | Notes                                                          |
| --- | ------------------------ | ----------------------------------- | ---------------- | --------------- | -------------------------------------------------------------- |
| 1   | `cpu-stress-amf`         | `01-cpu-stress-amf.yaml`            | Resource         | AMF             | StressChaos saturates AMF's 500m CPU limit                     |
| 2   | `memory-pressure-upf`    | `02-memory-pressure-upf.yaml`       | Resource / Crash | UPF             | StressChaos + in-container 150 MB allocation → OOM kill        |
| 3   | `pod-crash-amf`          | `03-pod-crash-amf.yaml`             | Crash            | AMF             | Tears down gNB SCTP — script auto-restarts gNB+UEs in recovery |
| 4   | `pod-crash-smf`          | `07-pod-crash-smf.yaml`             | Crash            | SMF             | Stale PFCP — script auto-restarts SMF in recovery              |
| 5   | `network-delay`          | `04-network-delay-gnb-amf.yaml`     | Network          | AMF→SCP, 500 ms | Invisible to Beyla; RTT confirmed via ping                     |
| 6   | `network-partition`      | `05-network-partition-amf-scp.yaml` | Network          | AMF↔SCP         | Target is SCP, not NRF (Model D indirect SBI)                  |
| 7   | `dependency-failure-nrf` | `06-dependency-failure-nrf.yaml`    | Dependency       | NRF (kill)      | Recovery waits 30 s for NF re-registration                     |
| 8   | `network-delay-nrf`      | `08-network-delay-nrf.yaml`         | Slow dependency  | NRF, 500 ms     | Gradual SBI latency vs hard kill                               |

### What happens during a run

`run_experiment.sh` per experiment:

1. Starts `kubectl port-forward` for Prometheus :9090, Jaeger :16686, Loki :3100.
2. Starts traffic generators inside the UE pod:
   - data-plane: 10 parallel `ping`s through `uesimtun0..9` → 8.8.8.8;
   - control-plane: re-registration loop (4 UEs deregister/register every 20 s).
3. **Baseline** (600 s) → `collect.py --phase baseline`.
4. **Inject**: `kubectl apply` the chaos YAML. For memory-pressure, also exec a 150 MB perl alloc inside the UPF container.
5. **Fault** (300 s) → optional ping RTT collection (network-delay/partition only) → `collect.py --phase fault`.
6. **Recovery**: `kubectl delete` the chaos resource (with finalizer-patching fallback if it hangs); fault-specific NF restart (SMF after UPF OOM, gNB+UEs after AMF kill, etc.); wait 300 s → `collect.py --phase recovery`.
7. Writes `timestamps.json` with all phase boundaries.

### What gets collected (per phase)

`experiments/collect.py` queries each backend's HTTP API and writes JSON to
`experiments/data/<fault>/run_NN/<phase>/`:

- **Prometheus** (one file per metric): CPU usage/throttle rates, container & pod memory, pod restarts, ready/running counts, OOM events, UE TUN RX/TX bytes.
- **Loki** (one file per query): all logs in `open5gs` namespace (Beyla excluded); targeted error queries; NRF heartbeat lifecycle; UE-visible failures (Registration reject, etc.); SCP routing errors. **Cursor-paginated — no line cap** (see EXTENSIONS §10.5).
- **Jaeger**: services list + up to 20000 traces per service per phase (raised from 2000; see EXTENSIONS §10.7).
- **NRF API snapshot**: registered NF instance counts per NF type (drops to 0 during NRF kill).
- **K8s Events**: `kubectl get events -n open5gs` filtered to the phase window. The collector filters out two known noise sources: `open5gs-populate` (permanent CrashLoopBackOff since cluster init) and `FailedGetScale` from a misconfigured HPA targeting a non-existent StatefulSet.
- **`fault/rtt_samples.txt`** for network-delay/partition only.

Output layout:

```
experiments/data/<fault-name>/run_NN/
├── timestamps.json
├── baseline/    prometheus/  loki/  jaeger/  k8s_events.json  nrf_registrations.json
├── fault/       (same +)    rtt_samples.txt   (for network experiments)
└── recovery/   prometheus/  loki/  jaeger/  k8s_events.json  nrf_registrations.json
```

---

## 8. Known limitations / gotchas

- **`pod_status_ready` misses brief crashes.** Prometheus scrapes every 5–15 s; pod restarts often complete in 10–50 s. Use the memory drop or `kube_pod_container_status_restarts_total` instead.
- **StressChaos memory ≠ container memory.** stress-ng runs in chaos-daemon's cgroup. `container_memory_*` for the target won't budge unless you also force allocation inside the target (we do this for UPF in `run_experiment.sh`).
- **Beyla can't see TC-layer network delay.** Jaeger spans look normal even with 500 ms confirmed. Ping RTT samples are the only signal.
- **Jaeger in-memory storage.** No persistence across pod restart. Collection must happen while port-forwards are active — collected at end-of-phase before tearing down.
- ~~**Loki 5000-line cap per query.**~~ **Resolved (2026-05-16)** — `collect_loki.py` now cursor-paginates; no cap, no undercount. See EXTENSIONS §10.5.
- **Signal caveats** (use the right metric): `upf_session_nbr`/`upf_qos_flows` over-count — use `pfcp_sessions_active`/`smf_ues_active`; GTP N3 packet counters and AMF `rm_regtime` are not exported by this build; control-plane workload is light so AMF/registration faults are under-stimulated. Full list: EXTENSIONS §10.10.
- **K8s Events background noise.** `open5gs-populate` (permanent CrashLoop) and HPA `FailedGetScale` are filtered in `collect.py`; if you add NFs, watch for new noise.
- **Network-partition target = SCP.** Open5GS uses indirect SBI (Model D). AMF↔NRF directly partition has zero effect — everything goes via SCP.

---

## 9. Recovery procedures (run if something is wedged)

```bash
# UPF after OOM (clears stale PFCP)
kubectl rollout restart deployment/open5gs-smf -n open5gs

# AMF kill (rebuilds gNB SCTP and UE PDU sessions)
kubectl rollout restart deployment/ueransim-gnb deployment/ueransim-gnb-ues -n open5gs

# General sanity
kubectl exec -n open5gs deployment/ueransim-gnb-ues -- ping -I uesimtun0 -c 3 8.8.8.8
kubectl get pods -n open5gs
```

If `kubectl delete` of a chaos resource hangs, patch the finalizers (the script does this automatically after 15 s):

```bash
kubectl patch <kind>/<name> -n open5gs --type=json \
  -p='[{"op":"remove","path":"/metadata/finalizers"}]'
```

---

## 10. File map

```
reproduce/
├── README.md                    ← this file
├── cluster-start.sh             ← daily lifecycle (recreate cluster + redeploy)
├── k8s/
│   ├── kind-config.yaml         ← kind cluster config: 3 nodes, eviction thresholds
│   ├── open5gs-values.yaml      ← Helm values: NF resource limits, MCC/MNC, slice id
│   ├── monitoring/
│   │   └── beyla-daemonset.yaml ← eBPF tracer DaemonSet (privileged, hostPID)
│   └── chaos/
│       ├── 01-cpu-stress-amf.yaml
│       ├── 02-memory-pressure-upf.yaml
│       ├── 03-pod-crash-amf.yaml
│       ├── 04-network-delay-gnb-amf.yaml
│       ├── 05-network-partition-amf-scp.yaml
│       ├── 06-dependency-failure-nrf.yaml
│       ├── 07-pod-crash-smf.yaml
│       └── 08-network-delay-nrf.yaml
└── experiments/
    ├── run_all.sh               ← all 8 experiments sequentially
    ├── run_experiment.sh        ← one experiment, 4-phase, with traffic + RTT
    └── collect.py               ← Prometheus + Loki + Jaeger + K8s events fetcher
```
