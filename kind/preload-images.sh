#!/usr/bin/env bash
# kind/preload-images.sh — make per-fault cluster recreates network-independent.
#
# kind nodes pull images into the node's own containerd, which is wiped on
# every `kind create cluster`. On a slow network each workload image takes
# 5-7 min to re-pull, repeatedly breaking the 22-fault recreate-per-fault run.
#
# Strategy: keep the images in the HOST docker daemon (persists across kind
# recreations) and `kind load` them into each fresh cluster locally (fast, no
# network). The image list is self-bootstrapping: snapshotted from the first
# fully-healthy cluster and reused thereafter.
#
# Usage:
#   preload-images.sh load     <cluster>   # host docker -> fresh kind node
#   preload-images.sh snapshot <cluster>   # cluster in-use images -> list + host docker
set -uo pipefail

CMD="${1:-}"
CLUSTER="${2:-open5gs}"
LIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preload-images.txt"

case "$CMD" in
  load)
    [[ -f "$LIST" ]] || { echo "  [preload] no list yet — skipping (first run)"; exit 0; }
    n=0; loaded=0
    while IFS= read -r img; do
      [[ -z "$img" || "$img" == \#* ]] && continue
      n=$((n+1))
      # Only images already in host docker can be loaded; skip the rest
      # (they'll pull over the network this once, then get snapshotted).
      if docker image inspect "$img" >/dev/null 2>&1; then
        if kind load docker-image "$img" --name "$CLUSTER" >/dev/null 2>&1; then
          loaded=$((loaded+1))
        fi
      fi
    done < "$LIST"
    echo "  [preload] loaded $loaded/$n cached images into kind '$CLUSTER'"
    ;;
  snapshot)
    # Authoritative list = every image actually in use, across all namespaces.
    imgs=$(kubectl get pods -A -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.image}{"\n"}{end}{range .spec.initContainers[*]}{.image}{"\n"}{end}{end}' 2>/dev/null \
           | grep -v '^$' | sort -u)
    [[ -z "$imgs" ]] && { echo "  [preload] snapshot: no images found — skipped"; exit 0; }
    printf '%s\n' "$imgs" > "$LIST"
    echo "  [preload] snapshot: $(wc -l < "$LIST") images -> $LIST"
    # Pull each into host docker so the next recreate can `kind load` it
    # offline. kindest/* and the k8s control-plane images are provided by the
    # kind node image itself — no need to host-cache them.
    while IFS= read -r img; do
      case "$img" in kindest/*|registry.k8s.io/*|docker.io/kindest/*) continue ;; esac
      docker image inspect "$img" >/dev/null 2>&1 && continue
      echo "  [preload] caching $img ..."
      docker pull "$img" >/dev/null 2>&1 || echo "  [preload] WARN: pull failed $img"
    done < "$LIST"
    echo "  [preload] host docker cache populated"
    ;;
  *)
    echo "usage: preload-images.sh {load|snapshot} <cluster>" >&2
    exit 2
    ;;
esac
