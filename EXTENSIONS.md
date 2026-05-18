# Extensions to the Reproduce Pack

Victor's `reproduce/` started as a stripped-down, kind-based version of Boyan's
original k3d pipeline. This document records everything that was added or
fixed on top of that base so it can produce the same dataset Boyan's main
pipeline (`../experiments/`) produces, plus a couple of new faults, and run
unattended for hours producing a shared dataset all teammates can analyse.

The original `README.md` still describes the underlying stack and deployment
steps. **Read it first.** This file is the diff.

---

## 1. What changed in one paragraph

Phase 3 (`03-fault-detection`) previously collected Prometheus + Jaeger only,
ran 8 faults, no synthetic traffic, no recovery hooks. It now collects
Prometheus + Jaeger + **Loki + K8s events + NRF API + RTT samples**, runs
**10 faults** (Boyan's 8 ∪ Victor's `packet-loss-upf` + `cpu-stress-scp`),
generates continuous **data-plane + control-plane traffic** during every
run, applies **fault-specific bash hooks** (memory-pressure in-container
alloc, gNB/UE/SMF rollouts after crashes, RTT ping during network faults),
and optionally **soft-resets** the Open5GS + UERANSIM workload between every
fault so each run starts from a clean baseline.

---

## 2. New files

```
reproduce/experiments/lib/
├── collect_loki.py         NEW — 5 LogQL queries, 5000-line cap, writes CSV
├── collect_events.py       NEW — kubectl get events, phase-window filter, drops
│                                noise (open5gs-populate, FailedGetScale)
├── collect_nrf.py          NEW — snapshots NRF NF-instance counts via kubectl
│                                exec + curl --http2-prior-knowledge
├── traffic.sh              NEW — start_traffic / stop_traffic (10× uesimtun
│                                pings + 4-UE deregister/register loop @15s)
├── reset_workload.sh       NEW — helm uninstall/reinstall open5gs + ueransim,
│                                wipe MongoDB PVC, re-provision subscribers,
│                                roll SMF + UEs to clear PFCP startup race
└── hooks/                  NEW — per-fault bash glue, sourced by run_fault.sh
    ├── 02-memory-pressure-upf.sh    during_fault: 150 MB perl alloc in UPF
    │                                post_delete:  rollout restart SMF
    ├── 03-pod-crash-amf.sh          post_delete:  rollout restart gNB + UEs
    ├── 04-network-delay-gnb-amf.sh  during_fault: AMF→SCP ping, RTT to file
    ├── 05-network-partition-amf-scp.sh during_fault: AMF→SCP ping, capture
    │                                                 packet-loss %
    ├── 06-dependency-failure-nrf.sh post_delete:  sleep 30 for NF re-reg
    ├── 07-pod-crash-smf.sh          post_delete:  rollout restart SMF
    └── 08-network-delay-nrf.sh      during_fault: AMF→NRF ping, RTT to file
```

Existing files were also modified — see §5 for the diff.

---

## 3. Hook contract

Each `lib/hooks/<slug>.sh` defines any of three bash functions; `run_fault.sh`
sources the file and calls them at the right point:

| Function       | When run_fault.sh calls it                                  |
| -------------- | ----------------------------------------------------------- |
| `pre_inject`   | After port-forwards + traffic are up, before `kubectl apply` |
| `during_fault` | Right after `kubectl apply`, before the `FAULT_DURATION` sleep. Hooks that need to run for the full fault window (RTT ping, memory alloc) launch a background process and return immediately. |
| `post_delete`  | After `kubectl delete`, before the `POST_DURATION` sleep    |

Unhooked faults still work — `run_fault.sh` installs no-op defaults before
sourcing. The hook filename **must** match the `--name` argument exactly,
which now equals the chaos YAML basename for every fault.

---

## 4. New behaviour in `run_fault.sh` and `03-fault-detection/run_all.sh`

### `run_fault.sh` — what was added

- Sources `traffic.sh` and `lib/hooks/$NAME.sh` (no-op defaults if missing)
- Adds `ensure_portforward_loki`
- Starts `start_traffic` after port-forwards; combined cleanup trap that
  calls `stop_traffic` then the existing `_cleanup` (PF killer)
- Calls `pre_inject` → `kubectl apply` → `during_fault` → fault sleep
- After `kubectl delete`, runs a 15-second deadline check; if delete hangs,
  patches `metadata.finalizers` out of every `networkchaos/stresschaos/
  podchaos` resource and resumes (same fallback as `../experiments/run_
  experiment.sh`)
- Then calls `post_delete` → post sleep
- Per phase, `collect_phase` calls **five** collectors instead of two:
  `collect_prometheus`, `collect_jaeger`, `collect_loki`, `collect_events`,
  `collect_nrf`
- Durations are env-overridable: `PRE_DURATION`, `FAULT_DURATION`,
  `POST_DURATION`, `STEP` (defaults 120 / 300 / 120 / 5s for back-compat)

### `03-fault-detection/run_all.sh` — what was added

- Same env-overridable durations propagate from caller to `run_fault.sh`
- New env `RESET_BETWEEN_FAULTS` (default `1`). When set, `reset_workload
  "$UE_COUNT"` runs before every fault. Set to `0` for the previous
  behaviour (no isolation, faster).
- Fault dispatch grew from 8 → **10**. Slugs are now equal to the chaos
  YAML basename, so `lib/hooks/<slug>.sh` resolves automatically.

The runtime numbers:

| Mode                                          | Per fault | Per Phase 3 run |
|-----------------------------------------------|-----------|-----------------|
| Boyan default (PRE=600 FAULT=300 POST=300, RESET=1) | ~24 min   | ~4 hours        |
| Victor default (PRE=120 FAULT=300 POST=120, RESET=0) | ~9 min    | ~90 min         |

---

## 5. Modifications to existing files

### `lib/common.sh`

- **Bug fix**: `check_cluster_ready` used context `k3d-open5gs`; the kind
  cluster is named `open5gs`, which kubectl exposes as context `kind-open5gs`.
  Fixed lines 117 + 121.
- Added `LOKI_URL` env + `ensure_portforward_loki` (svc/loki :3100)
- Added shell wrappers `collect_loki`, `collect_events`, `collect_nrf`
  alongside existing `collect_prometheus`/`collect_jaeger`

### `lib/run_fault.sh`

Substantially rewritten — see §4 above.

### `03-fault-detection/run_all.sh`

Substantially rewritten — see §4 above.

### `lib/collect_jaeger.py`

- **Bug fix**: hardcoded the service list as `open5gs-amf`, `open5gs-smf`,
  etc. — but Beyla autodetect strips the `open5gs-` prefix, so Jaeger
  receives traces under service names `amf`, `smf`, `nrf`, … Switched
  to dynamic discovery via `/api/services` with a small blocklist for
  jaeger-self traces.

### `lib/provision_ues.sh`

- **Bug fix**: the mongosh upsert wrote subscribers with slice `{sst: 1,
  default_indicator: true}` only, no `sd`. UE configs request slice
  `{sst:1, sd:0x111111}`, so AMF rejected every UE with "Bad constructed
  NAS message". Cluster-start.sh's `populate` Job *did* write IMSI 1+2 with
  `sd:111111` correctly via `add_ue_with_slice ... 1 111111`, but Victor's
  `provision_ues.sh` ran later and overwrote them. Added `sd: "111111"`
  to the slice config.

### `cluster-start.sh`

- **Bug fix**: referenced `$SCRIPT_DIR/k8s/...` for the open5gs values and
  Beyla daemonset, but the directory was renamed `kind/` when Victor moved
  off k3d. Helm install failed at line 59 without this. Updated both paths.

### `kind/open5gs-values.yaml`

- Tried `bitnamilegacy/mongodb:4.4.1` to dodge Bitnami's August 2025
  Docker Hub auth changes — but the 4.4.1 image layout doesn't match what
  Bitnami's chart wrapper scripts expect (`sed: mongodb.conf` failure).
  **Reverted** to the chart default (`bitnami/mongodb:latest`), which
  works after a slow pull. Long-term fix: pin to a `bitnamilegacy` tag
  that matches the chart's expectations (probably 6.x or 7.x); not done
  yet.

---

## 6. Output layout (per fault)

```
reproduce/data/experiments/03-fault-detection/<slug>/
├── meta.json                       experiment name + start/end ISO timestamps
├── timeline.json                   unix-seconds boundaries for pre/fault/post
├── prometheus/{pre,during,post}/   one CSV per metric (unchanged from Victor)
│       container_cpu_usage_rate.csv
│       container_memory_working_set_bytes.csv
│       container_cpu_throttled_rate.csv
│       pod_restarts.csv
│       monitoring_cpu_usage_rate.csv
│       monitoring_memory_working_set.csv
│       node_cpu_usage.csv
│       node_memory_available.csv
├── jaeger/{pre,during,post}/       NEW: services discovered dynamically
│       spans_flat.csv              flattened spans across all NF services
│       summary.json                per-service counts + p50/p95/p99 µs
├── loki/{pre,during,post}/         NEW: 5 LogQL queries
│       all.csv                     {namespace="open5gs"}
│       errors.csv                  errors|exceptions|refused|fatal|oom|killed
│       nrf_lifecycle.csv           heartbeat|de-registered|Retry registration
│       ue_failures.csv             PAYLOAD_NOT_FORWARDED|Registration reject|…
│       scp_routing.csv             Connection timer|Connection refused|…
├── events/{pre,during,post}/       NEW
│       k8s_events.json             kubectl get events, phase-window filter
└── nrf/{pre,during,post}/          NEW
        nrf_registrations.json      live NF-instance counts via NRF /nf-instances
```

For `04-network-delay-gnb-amf`, `05-network-partition-amf-scp`,
`08-network-delay-nrf`, the hook also writes:

```
└── rtt/during/rtt_samples.txt      ping RTTs (ms) and packet-loss % lines
```

---

## 7. Operational gotchas hit during setup

These are documented so the next person bringing this stack up doesn't
hit the same dead ends:

1. **kind alpha bug.** `https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64`
   returned an alpha build (`v0.32.0-alpha`) that fails on cgroup v2 + Docker
   29 with the error
   `could not find a log line that matches "Reached target …Multi-User System…|
   detected cgroup v1"`. Use a stable tag: `dl/v0.27.0/kind-linux-amd64`.

2. **inotify limits.** `cluster-start.sh` raises them at step [3/5], which is
   *after* `kind create cluster` at step [2/5]. Nodes will fail to boot
   ("Failed to create control group inotify object: Too many open files")
   on the first run if the host limits are low. Raise them manually first:

   ```bash
   sudo sysctl fs.inotify.max_user_instances=512
   sudo sysctl fs.inotify.max_user_watches=524288
   ```

3. **Old k3d cluster cleanup.** If a previous k3d-based session is still
   running, its containers consume inotify watches that prevent kind nodes
   from booting. `docker rm -f $(docker ps -aq --filter "name=k3d-*")`
   before running `cluster-start.sh`.

4. **Bitnami MongoDB pull is slow.** The chart references
   `bitnami/mongodb:latest`; the pull can take 5–10 min from a cold cache
   and may exceed the `--wait --timeout=10m` helm install timeout — leaving
   the release in status `failed` even though pods are healthy. The
   workaround is to let it finish and either `helm uninstall + reinstall`
   or just continue manually with the remaining stack pieces.

5. **Disk pressure on full host disks.** The `kind-config.yaml` sets
   `evictionHard.nodefs.available: 5%`. On a host that's >90% full, the
   first batch of image pulls (kube-prom + chaos-mesh + Bitnami images)
   can trip eviction, killing chaos-mesh and Grafana mid-install. They
   recover after disk pressure clears, but you may need to
   `kubectl delete pod` the Errored/Evicted ones to force re-creation.

6. **SMF/UPF PFCP startup race.** After `helm install open5gs`, SMF
   sometimes establishes PFCP with UPF, loses the heartbeat as UPF
   finishes booting, and re-associates. Any UE that registered in the
   gap has a stale PDU session and its data plane is dead. Fix: after the
   install settles, `kubectl rollout restart deployment/open5gs-smf` then
   `... deployment/ueransim-ues deployment/ueransim-gnb-ues`. This is
   what `reset_workload.sh` does at the end, and what `cluster-start.sh`
   does *not* yet do (so the very first deploy can produce dead data
   planes — restart SMF + UEs manually after `cluster-start.sh` if
   `ping -I uesimtun0 8.8.8.8` returns 100% loss).

7. **Cluster context name.** Kind exposes `kind create cluster --name foo`
   as kubectl context `kind-foo`. Victor's `check_cluster_ready` looked
   for `k3d-open5gs` — copy-pasted from the k3d era. Fixed; if you see
   "context not found", verify with `kubectl config current-context`.

---

## 8. How to invoke (Boyan's settings)

Fresh start, full 10-fault run that produces the shared dataset:

```bash
cd reproduce
./cluster-start.sh                                  # ~15 min, one-time per session
# (manually restart SMF + UEs after; see gotcha #6, or just trust reset_workload to do it)
PRE_DURATION=600 FAULT_DURATION=300 POST_DURATION=300 \
  nohup bash experiments/03-fault-detection/run_all.sh \
  > /tmp/phase3.log 2>&1 < /dev/null & disown
```

Wall-clock: ~4 hours. Output: `data/experiments/03-fault-detection/<slug>/...`.

To skip a fault that already ran successfully:

```bash
PRE_DURATION=600 FAULT_DURATION=300 POST_DURATION=300 \
  bash experiments/03-fault-detection/run_all.sh --from 5
```

To run with isolation OFF (faster, ~3 h, drift accumulates):

```bash
RESET_BETWEEN_FAULTS=0 PRE_DURATION=600 FAULT_DURATION=300 POST_DURATION=300 \
  bash experiments/03-fault-detection/run_all.sh
```

To smoke a single fault:

```bash
PRE_DURATION=60 FAULT_DURATION=60 POST_DURATION=60 \
  bash experiments/lib/run_fault.sh \
    --name 01-cpu-stress-amf \
    --manifest "$(pwd)/kind/chaos/01-cpu-stress-amf.yaml" \
    --out /tmp/smoke/01 --step 5s
```

---

## 9. Differences from the original `experiments/` pipeline

Functionally equivalent signal coverage, but schema differs (intentionally
kept Victor's CSV layout so teammates' analysis built on the reproduce/
data still works). Key deltas:

| Aspect              | `experiments/` (Boyan, k3d)              | `reproduce/` (shared, kind)            |
| ------------------- | ---------------------------------------- | -------------------------------------- |
| Orchestration       | k3d                                      | kind                                   |
| Phase dir names     | `baseline / fault / recovery`            | `pre / during / post`                  |
| Phase metadata file | `timestamps.json` (ISO strings)          | `timeline.json` (unix seconds)         |
| Prometheus output   | one JSON per metric (raw response)       | one CSV per metric (flat rows)         |
| Loki output         | one JSON per query (raw response)        | one CSV per query (flat rows)          |
| Jaeger output       | one JSON per service (raw traces)        | `spans_flat.csv` + `summary.json`      |
| K8s events output   | `k8s_events.json`                        | `events/<phase>/k8s_events.json`       |
| NRF output          | `nrf_registrations.json`                 | `nrf/<phase>/nrf_registrations.json`   |
| Fault count         | 8                                        | 10 (`packet-loss-upf`, `cpu-stress-scp` new) |
| Per-fault isolation | none                                     | soft-reset via `reset_workload`        |

Analysis code built against the original `experiments/` layout will NOT
run unchanged against `reproduce/data/`. Either adapt the existing
`experiments/analyze.py` to the CSV/`pre`/`during`/`post` schema, or
write an adjacent `analyze_reproduce.py`.

---

## 10. Pipeline hardening — 2026-05-16

A second hardening pass after the first full-batch attempts produced
contaminated baselines, silent collection failures, and Docker Hub
rate-limit stalls. **This section supersedes the relevant numbers in
§1–§9 above** (fault count, durations, reset model, Loki/Jaeger caps).
All changes are committed; scripts were not reverted.

### 10.1 Fault count & dispatch

- Phase 3 now dispatches **22 faults** (not 10). Output path is
  `data/experiments/C-fault-detection/<NN-slug>/`.

### 10.2 Phase durations are now 600 / 300 / 300 by default

- `run_all.sh` and `run_fault.sh` default `PRE/FAULT/POST = 600/300/300`
  (previously 120/300/120). No env vars needed; the old
  `PRE_DURATION=600 …` invocation is now redundant.

### 10.3 Reset model: full recreate per fault (soft-reset shelved)

- `run_all.sh` calls **`cluster-start.sh` (full `kind delete`+`create`)
  before every fault**, not `reset_workload`. `RESET_BETWEEN_FAULTS`
  defaults to `0`; `reset_workload.sh` is now **legacy/unused**.
- Rationale: soft-reset (rolling restart) reintroduced the documented
  SCP→AUSF stale-pod-IP routing failures and gNB no-reconnect race;
  full recreate is the proven-clean path. Cost: ~15–20 min recreate
  per fault → **~45–55 min/fault, ~16–20 h for the 22-fault batch**.

### 10.4 Bring-up ordering gates (cluster-start.sh)

Cold bring-up has a race (gNB NG-Setup → UE register → SMF↔UPF PFCP →
PDU/GTP). UERANSIM's gNB does **not** auto-reconnect after an SCTP drop,
so a lost race silently yields a degraded baseline (the old
`health_check.sh` accepted ≥5/10 tunnels and never checked PFCP). Three
state-probe gates were added to `cluster-start.sh`, each polling the NF
Prometheus endpoint (bound to pod eth0 IP, curled in-pod):

| Gate | Check | Position | On miss |
|---|---|---|---|
| A | `upf pfcp_peers_active ≥ 1` | after Open5GS install, before RAN | FATAL after 240s |
| B | `amf gnb ≥ 1` (NG Setup) | after gNB install, before UEs | restart gNB ×3, then FATAL |
| C | `smf pfcp_sessions_active ≥ UE_COUNT` (**strict 10/10**) | after provision+rollout | restart UEs ×3, then FATAL |

`UE_COUNT=10` is centralized. FATAL aborts the batch (resume with
`--from N`) rather than ever measuring a degraded baseline. Note Gate C
queries **SMF** (`pfcp_sessions_active` is an SMF metric; the UPF does
not expose it).

### 10.5 Loki collection rewritten (`collect_loki.py`)

- **Cursor pagination**: `PER_PAGE=5000`, forward cursor, cross-page
  dedupe by `(ts,pod,line)`, `MAX_PAGES` guard. Replaces the single
  capped query that timed out / 400'd on large windows. Works
  regardless of volume and **without** depending on the server-cap
  patch. The old "5000-line cap / undercount" limitation is **gone**.
- **Beyla excluded at the selector**: all 5 queries use
  `{namespace="open5gs", pod!~"beyla-.+"}`. At `DEBUG`, Beyla emitted
  ~85% of namespace log volume (eBPF name-resolution chatter, zero
  atlas signal) and was the cause of the unfiltered-query timeouts.
  ueransim is **kept** (UE/gNB failure logs are signal).
- `all.csv` restored. Server-cap patch in `cluster-start.sh` reduced
  500000 → 50000 (no longer load-bearing — pagination handles it).

### 10.6 Beyla quieted to INFO (`kind/monitoring/beyla-daemonset.yaml`)

- `OTEL_EBPF_LOG_LEVEL` / `BEYLA_LOG_LEVEL`: `DEBUG → INFO`. Removes
  ~85% of log volume at source. **Beyla metrics & traces are
  unaffected** (log level does not gate metric/trace emission).

### 10.7 Jaeger trace cap raised (`collect_jaeger.py`)

- Per-service `limit` `2000 → 20000`; request timeout `30 → 120s`.
  The 2000 cap was truncating SCP (the busiest NF, every inter-NF SBI
  call) in exactly the SCP-relevant faults. Verify SCP `span_count`
  is well under 20000 in `summary.json` (not truncated).

### 10.8 Prometheus metrics (`collect_prometheus.py`)

- **Removed** the dead GTP-packet queries
  (`fivegs_ep_n3_gtp_*datapktn3upf`): they exist in Open5GS 2.7.5
  source but read **flat zero** in this Gradiant UPF datapath
  (verified live). Use UPF container `network_rx_bytes_rate` /
  `network_tx_bytes_rate` as the data-plane traffic signal instead.
- **Added 10 metrics**: failure counters
  `smf_pdu_session_fail`, `smf_n4_session_estab_fail`,
  `upf_n4_session_estab_fail`, `amf_reg_init_fail`, `amf_reg_mob_fail`,
  `amf_reg_period_fail`, `amf_reg_emerg_fail`; clean session gauges
  `smf_ues_active`, `smf_bearers_active`, `smf_qos_flow_nbr`. Failure
  counters are zero-suppressed: empty/absent CSV in a clean phase,
  populated only when the fault actually induces failures (correct,
  expected behaviour).

### 10.9 Docker Hub authentication (`cluster-start.sh`, `.gitignore`)

- Recreate-per-fault re-pulls ~10–20 docker.io images each time;
  unauthenticated Docker Hub = **100 pulls / 6 h / IP** → the batch
  stalls on `ImagePullBackOff`/`429`.
- `cluster-start.sh` now injects credentials from a **gitignored**
  `kind/.dockerhub-auth` (two lines: username, then PAT) into a
  runtime kind config (`containerdConfigPatches`) before
  `kind create`. Authenticated = **200 / 6 h**, billed to a separate
  account bucket (independent of the exhausted per-IP bucket).
- **Every teammate must create their own** `kind/.dockerhub-auth`
  (it is gitignored and never committed). Without it the script warns
  and falls back to unauthenticated. Read-only PAT scope is sufficient.

### 10.10 Known signal caveats (for analysis & the paper)

These are measurement properties of this stack, not bugs to fix —
state them explicitly in the paper rather than treat as missing data:

- **`upf_session_nbr` / `upf_qos_flows` over-count.** The UPF gauge
  increments on session add but not on delete, so it climbs within a
  run (real ≈10 → 25+) while the system is healthy. **Use
  `pfcp_sessions_active` / `smf_ues_active` / `smf_sessionnbr`** — all
  three independently track the true count and respond to faults.
- **GTP N3 packet/volume counters are not exported** by this build —
  data-plane traffic is observable only via UE RTT + UPF network
  rx/tx bytes.
- **`fivegs_amffunction_rm_regtime` (registration latency) is not
  exported.** AMF CPU/stress faults degrade registration *latency*,
  but only failure *counters* are collectable, and registrations
  eventually succeed — so control-plane faults look "resource-metric-
  only."
- **The control-plane workload is light.** `traffic.sh` cycles only
  **3 of 10 UEs** through deregister/register every ~70 s (~24
  transactions per 300 s fault). Control-plane faults
  (AMF/AUSF/UDM/registration) are under-stimulated; the atlas may
  under-report their blast radius. This is a methodology limitation to
  disclose, or to address by raising re-registration intensity in
  `traffic.sh` (trades off against PFCP/GTP stability — see the
  in-file comment).
- **Non-crash faults are quiet at the orchestration layer** by nature
  (no pod events unless a liveness probe trips). Chaos Mesh
  `Applied`/`Recovered` events *are* captured in the `open5gs`
  namespace and provide ground-truth fault timing; the `chaos-mesh`
  namespace is not scraped (low priority — `timeline.json` covers
  timing).
