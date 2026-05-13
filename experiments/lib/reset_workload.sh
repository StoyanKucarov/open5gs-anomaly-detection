#!/usr/bin/env bash
# experiments/lib/reset_workload.sh
#
# Soft-reset the open5gs workload: tear down Open5GS + UERANSIM (uninstalls
# their Helm releases — also drops MongoDB and all subscribers), then
# reinstall fresh, reprovision subscribers, scale UEs. Leaves kind itself,
# monitoring (Prom/Loki/Jaeger/Beyla), and Chaos Mesh untouched.
#
# Use to wipe per-fault drift (stale PFCP sessions, ghost NF registrations,
# UE TUN interfaces in weird states) without paying the ~15 min cost of a
# full cluster recreate. Costs ~2-3 min per call.
#
# Caller is expected to source common.sh first so $LIB_DIR is set.
#
# Usage:
#   reset_workload <ue_count>      # e.g. reset_workload 50

reset_workload() {
    local ue_count="${1:-50}"
    local repo_root
    repo_root="$(cd "$LIB_DIR/../.." && pwd)"
    local values="$repo_root/kind/open5gs-values.yaml"

    echo "[reset] tearing down workload..."
    helm uninstall ueransim-ues -n open5gs --ignore-not-found 2>/dev/null || true
    helm uninstall ueransim-gnb -n open5gs --ignore-not-found 2>/dev/null || true
    helm uninstall open5gs      -n open5gs --ignore-not-found 2>/dev/null || true

    # Force-delete the PVC so MongoDB starts from an empty volume.
    kubectl delete pvc -n open5gs -l app.kubernetes.io/name=mongodb \
        --ignore-not-found --wait=true 2>/dev/null || true

    # Wait for pods to fully drain (helm uninstall returns before pods are gone).
    local i=0
    while kubectl get pods -n open5gs --no-headers 2>/dev/null \
            | grep -qE 'open5gs-|ueransim-'; do
        sleep 3
        i=$((i + 3))
        [[ $i -ge 120 ]] && { echo "[reset] WARN: pods still terminating after 120s, continuing"; break; }
    done

    echo "[reset] reinstalling Open5GS..."
    helm install open5gs oci://registry-1.docker.io/gradiantcharts/open5gs \
        --version 2.3.4 --namespace open5gs \
        -f "$values" \
        --wait --timeout=10m
    kubectl delete deployment -n open5gs open5gs-webui --ignore-not-found

    echo "[reset] reinstalling UERANSIM..."
    helm install ueransim-gnb oci://registry-1.docker.io/gradiant/ueransim-gnb \
        --version 0.2.6 --namespace open5gs \
        --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
        --wait --timeout=5m
    helm install ueransim-ues oci://registry-1.docker.io/gradiant/ueransim-ues \
        --version 0.1.2 --namespace open5gs \
        --values https://gradiant.github.io/5g-charts/docs/open5gs-ueransim-gnb/gnb-ues-values.yaml \
        --set ues.count="$ue_count" \
        --wait --timeout=5m

    echo "[reset] reprovisioning $ue_count subscribers..."
    bash "$LIB_DIR/provision_ues.sh" "$ue_count"

    wait_for_pods_stable open5gs 120

    # SMF/UPF startup race: SMF often loses initial PFCP heartbeat and
    # re-associates, leaving any PDU sessions from UEs that registered
    # in between in a stale state (data plane dead). Restart SMF then
    # UEs so PDU sessions are established against the stable PFCP.
    echo "[reset] settling PFCP + UE PDU sessions..."
    kubectl rollout restart deployment/open5gs-smf -n open5gs
    kubectl rollout status  deployment/open5gs-smf -n open5gs --timeout=60s
    kubectl rollout restart deployment/ueransim-ues -n open5gs
    kubectl rollout restart deployment/ueransim-gnb-ues -n open5gs
    kubectl rollout status  deployment/ueransim-ues -n open5gs --timeout=60s
    sleep 10
    echo "[reset] workload ready"
}
