#!/usr/bin/env bash
# Hook: 06-dependency-failure-nrf
#
# After NRF is killed and the chaos resource is deleted, NFs need ~30s
# to re-register with the new NRF instance. Without this wait, the
# post-fault collection window catches a half-degraded mesh.

post_delete() {
    echo "  [hook 06] waiting 30s for NF re-registration with NRF..."
    sleep 30
}
