# Cloud-Native 5G Fault Detection — Reproduction Pack

Covers the **C-fault-detection experiment (RQ3)**: Can existing anomaly detection algorithms accurately detect faults in cloud-native 5G core networks using logs, metrics, and traces?

> **Extended notes:** [`EXTENSIONS.md`](./EXTENSIONS.md) — extra collectors, traffic generators, per-fault hooks, pipeline hardening (2026-05-16).

---

## Quick Start

```bash
# Full pipeline: cluster + 22 faults + model evaluation + figures (~12 h)
./cluster-start.sh                                         # bring up cluster    (~15 min)
./experiments/C-fault-detection/run_all.sh                 # collect 22 faults   (~8 h)
python models/logs/evaluate.py     --multi-run             # log model eval       (~30 min)
python models/metrics/evaluate.py  --multi-run             # metric model eval    (~45 min)
python models/traces/evaluate.py   --multi-run             # trace model eval     (~30 min)
python visualizations/logs/run_all.py                      # log figures
python visualizations/metrics/run_all.py                   # metric figures
python visualizations/robustness/run_sweep.py              # robustness figures
```

Figures → `models/*/out/`, `visualizations/*/out/`.

> **Docker Hub auth:** create `kind/.dockerhub-auth` (two lines: username + read-only PAT) to avoid the 100/6h unauthenticated pull-rate limit. The cluster start script injects it automatically.

---

## 1. Stack

| Layer           | Tool                                     | Version       |
| --------------- | ---------------------------------------- | ------------- |
| 5G core         | Open5GS (Gradiant Helm chart)            | 2.3.4         |
| RAN sim         | UERANSIM gNB + UEs (Gradiant Helm chart) | 0.2.6 / 0.1.2 |
| Orchestration   | kind (Kubernetes in Docker)              | latest        |
| Metrics         | kube-prometheus-stack                    | latest        |
| Logs            | Loki + Promtail                          | latest        |
| Traces          | Jaeger (all-in-one, in-memory)           | v4.7.0        |
| Span source     | Beyla (eBPF auto-instrumentation)        | ≥ 3.9.5       |
| Fault injection | Chaos Mesh                               | 2.7.2         |
| Collection      | Python 3 + `requests`                   | —             |

---

## 2. Host prerequisites (Linux + Docker)

- Docker 29+ installed and running.
- Raise inotify limits (required for Promtail + Chaos Mesh, otherwise both crash):

```bash
sudo sysctl fs.inotify.max_user_instances=512
sudo sysctl fs.inotify.max_user_watches=524288

echo "fs.inotify.max_user_instances=512"  | sudo tee -a /etc/sysctl.conf
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
```

- Python 3.10+:

```bash
pip install torch numpy scikit-learn requests
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

## 4. Daily lifecycle — `./cluster-start.sh`

After every reboot or Docker restart:

```bash
./cluster-start.sh              # recreate cluster + redeploy everything (~15 min)
./cluster-start.sh --skip-deploy  # recreate cluster only
```

The script deletes and recreates the kind cluster, checks/raises inotify limits, redeploys the full stack (Open5GS, UERANSIM, kube-prometheus-stack, Loki, Jaeger, Beyla, Chaos Mesh), and sanity-checks all namespaces. kind has no start/stop — when Docker stops, recreate with this script.

**Don't restart Docker while experiments are running.**

---

## 5. Run the fault detection experiment

```bash
./experiments/C-fault-detection/run_all.sh              # all 22 faults sequentially (~8 h)
./experiments/C-fault-detection/run_all.sh --from 10    # resume from fault 10
./experiments/C-fault-detection/run_all.sh --only 3,7   # run specific faults
```

Each fault runs three phases (600 s pre → 300 s fault → 300 s post) with full telemetry collection: Prometheus (41 KPIs), Loki (paginated, no cap), Jaeger (up to 20 000 traces/service/phase), NRF API snapshots, and K8s events.

Output: `data/experiments/C-fault-detection/<fault>/run_NN/{pre,fault,post}/`

### Fault catalogue

| # | Fault | Class | Target | Notes |
|---|-------|-------|--------|-------|
| 01 | cpu-stress-amf | Resource | AMF | StressChaos saturates 500m CPU limit |
| 02 | memory-pressure-upf | Resource/Crash | UPF | StressChaos + in-pod 150 MB alloc → OOM kill |
| 03 | pod-crash-amf | Crash | AMF | Tears down gNB SCTP; auto-restarts gNB+UEs in post |
| 04 | network-delay-gnb-amf | Network | AMF↔gNB, 500 ms | Invisible to Beyla; RTT confirmed via ping |
| 05 | network-partition-amf-scp | Network | AMF↔SCP | Targets SCP — all SBI goes through SCP (Model D) |
| 06 | packet-loss-upf | Network | UPF | Data-plane packet loss |
| 07 | pod-crash-smf | Crash | SMF | Stale PFCP; auto-restarts SMF in post |
| 08 | cpu-stress-scp | Resource | SCP | SBI routing bottleneck |
| 09 | network-delay-nrf | Network | NRF, 500 ms | Slow NF discovery |
| 10 | pfcp-flood | Protocol attack | UPF | Session establishment flood |
| 11 | pfcp-deletion | Protocol attack | UPF | Spurious session deletions |
| 12 | pfcp-drop | Protocol attack | UPF | Session modification drops |
| 13 | pfcp-duplication | Protocol attack | UPF | Session modification duplications |
| 14 | upf-infra-packet-loss | Network | UPF node | Infrastructure-level loss |
| 15 | nrf-cascade | Dependency | NRF kill | NF re-registration cascade; 30 s recovery wait |
| 16 | cpu-stress-ausf | Resource | AUSF | Auth bottleneck |
| 17 | network-delay-scp | Network | SCP, 500 ms | SBI routing latency |
| 18 | cpu-stress-nrf | Resource | NRF | Discovery bottleneck |
| 19 | udm-pod-crash | Crash | UDM | Subscription data unavailable |
| 20 | mongodb-pod-kill | Crash | MongoDB | Backing store for UDR/UDM |
| 21 | n2-partition-amf-gnb | Network | AMF↔gNB | N2 interface severed |
| 22 | memory-pressure-amf | Resource | AMF | OOM pressure on control plane |

---

## 6. Train and evaluate anomaly detection models

All three evaluate scripts share the same interface. Results are written to `models/*/out/`.

```bash
# Standard evaluation (all available runs merged)
python models/logs/evaluate.py    --multi-run
python models/metrics/evaluate.py --multi-run
python models/traces/evaluate.py  --multi-run

# Robustness: feature dropout and Gaussian noise (metrics + traces)
python models/metrics/evaluate.py --dropout 5g_control
python models/metrics/evaluate.py --noise-std 0.1
python models/traces/evaluate.py  --dropout latency
python models/traces/evaluate.py  --noise-std 0.1
```

### Models

**Logs** — input: Drain-parsed template sequences (30 s windows per NF)

| Model | Method | Reference |
|-------|--------|-----------|
| DeepLog | LSTM next-key prediction | Du et al., CCS 2017 |
| LogBERT | MLM Transformer | Guo et al., BigData 2021 |
| LogRobust | BiLSTM + attention autoencoder | Zhang et al., WWW 2019 (adapted unsupervised) |
| Logs2Graphs | DiGCN + Deep SVDD | Li et al., ICSE 2024 |
| FeatureModel | Heartbeat-aware Isolation Forest | Monika Steidl et al., IEEE 2024s |

**Metrics** — input: 41 Prometheus KPIs (5 s scrape, 60 s windows)

| Model | Method | Reference |
|-------|--------|-----------|
| MetricPCA | PCA reconstruction error | Xu et al., 2009 |
| USAD | Dual AE adversarial training | Audibert et al., KDD 2020 |
| TranAD | Transformer + 2-decoder | Tuli et al., VLDB 2022 |
| OmniAnomaly | GRU + VAE | Su et al., KDD 2019 |
| AnomalyTransformer | Series/prior association | Xu et al., ICLR 2022 |

**Traces** — input: per-service span features (span count, error rate, latency p50/p95/p99) over 60 s windows

| Model | Method | Reference |
|-------|--------|-----------|
| TraceRPCA | Robust PCA (inexact ALM) | Candès et al., 2011 |
| TraceAnomaly | Real NVP normalizing flow | Liu et al., ISSRE 2020 |
| GAL-MAD | GAT + BiLSTM | Attanayake et al., arXiv 2504.00058 |
| TraceDAE | Dual AE with GAT | Li et al., TNSM 2025 |
| TraceSieve | VGAE + GAN noise filter | Zhang et al., ISSRE 2023 |

Evaluation metrics: AUROC, Average Precision (PR-AUC), Recall@optimal-F1. All threshold-free.

---

## 7. Generate figures

```bash
python visualizations/logs/run_all.py                        # log feature heatmap, clustering, templates, timeline
python visualizations/metrics/run_all.py                     # metric delta heatmap, timelines, fault-class profiles
python visualizations/robustness/run_sweep.py                # noise curves, dropout heatmaps
python visualizations/cross_modality/01_cross_modality_coverage.py
python visualizations/feature_importance/01_feature_importance.py
```

All figures land in `visualizations/*/out/`.

---

## 8. Recovery

```bash
# UPF after OOM (clears stale PFCP)
kubectl rollout restart deployment/open5gs-smf -n open5gs

# AMF kill (rebuilds gNB SCTP and UE PDU sessions)
kubectl rollout restart deployment/ueransim-gnb deployment/ueransim-gnb-ues -n open5gs

# General sanity
kubectl exec -n open5gs deployment/ueransim-gnb-ues -- ping -I uesimtun0 -c 3 8.8.8.8
kubectl get pods -n open5gs
```

If `kubectl delete` of a chaos resource hangs, patch the finalizers (the run script does this automatically after 15 s):

```bash
kubectl patch <kind>/<name> -n open5gs --type=json \
  -p='[{"op":"remove","path":"/metadata/finalizers"}]'
```

---

## 9. File map

```
.
├── cluster-start.sh                  ← daily lifecycle (recreate cluster + redeploy)
├── kind/
│   ├── kind-config.yaml              ← 3-node kind cluster, eviction thresholds
│   ├── open5gs-values.yaml           ← NF resource limits, MCC/MNC, slice config
│   ├── monitoring/
│   │   └── beyla-daemonset.yaml      ← eBPF tracer DaemonSet (privileged, hostPID)
│   └── chaos/                        ← 22 Chaos Mesh YAMLs (one per fault)
├── experiments/
│   ├── lib/                          ← shared shell + Python helpers (Loki/Prom/Jaeger collection, health checks)
│   └── C-fault-detection/
│       └── run_all.sh                ← 22-fault orchestrator (--from N, --only N,M)
├── models/
│   ├── logs/
│   │   ├── data_loader.py            ← Loki JSON → LogRecord, Drain parsing, vocab
│   │   ├── log_parser.py             ← self-contained Drain implementation
│   │   ├── evaluate.py               ← train + eval all log models, write out/
│   │   ├── plot_results.py           ← AUROC/F1/recall heatmaps from eval_results.json
│   │   ├── deeplog_model.py
│   │   ├── logbert_model.py
│   │   ├── logrobust_model.py
│   │   ├── logs2graphs_model.py
│   │   ├── feature_model.py
│   │   └── out/                      ← eval_results*.json, heatmap PNGs
│   ├── metrics/
│   │   ├── data_loader.py            ← Prometheus JSON → MetricRecord (41 KPIs)
│   │   ├── evaluate.py               ← train + eval all metric models, write out/
│   │   ├── plot_results.py
│   │   ├── pca_model.py
│   │   ├── usad_model.py
│   │   ├── tranad_model.py
│   │   ├── omnianomaly_model.py
│   │   ├── anomaly_transformer_model.py
│   │   └── out/                      ← eval_results*.json, dropout/noise variants
│   └── traces/
│       ├── data_loader.py            ← Jaeger JSON → TraceRecord (per-service features)
│       ├── evaluate.py               ← train + eval all trace models, write out/
│       ├── plot_results.py
│       ├── rpca_model.py
│       ├── trace_anomaly_model.py
│       ├── galmad_model.py
│       ├── tracedae_model.py
│       ├── tracesieve_model.py
│       └── out/                      ← eval_results*.json, dropout/noise variants
├── visualizations/
│   ├── logs/                         ← feature heatmap, clustering, template dist, timeline
│   ├── metrics/                      ← delta heatmap, timelines, fault-class profiles
│   ├── robustness/                   ← noise curves, feature dropout heatmaps
│   ├── cross_modality/               ← coverage overlap, optimal model combinations
│   └── feature_importance/           ← Prometheus KPI importance by fault class
└── analysis/
    ├── lib.py                        ← EXPERIMENTS registry (22 faults, fault classes)
    └── run_all.py
```
