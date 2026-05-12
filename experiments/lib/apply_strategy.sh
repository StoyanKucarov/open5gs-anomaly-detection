#!/usr/bin/env bash
# Usage: source lib/apply_strategy.sh && apply_log_strategy "denum"

apply_log_strategy() {
    local strategy_name="$1"
    local strategy_file="$LIB_DIR/reduction_strategies/${strategy_name}.yaml"

    if [[ ! -f "$strategy_file" ]]; then
        echo "[ERROR] Strategy file $strategy_file not found!"
        return 1
    fi

    apply_dynamic_logging() {
        local level=$1 
        echo "[dynamic] Patching Open5GS ConfigMap to level: $level"
    }

    echo "[setup] Applying strategy: $strategy_name"
    helm upgrade loki grafana/loki-stack \
        --namespace monitoring \
        --reuse-values \
        -f "$strategy_file" \
        --timeout=10m
    
    echo "[wait] Waiting for Promtail rollout..."
    kubectl rollout status daemonset loki-promtail -n monitoring --timeout=5m
}