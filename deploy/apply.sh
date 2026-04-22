#!/usr/bin/env bash
# One-shot installer for the host-side infrastructure changes required to wire
# wikipedia-retriever into the existing monitoring and gateway stack.
#
# Run from the project root:   sudo bash deploy/apply.sh
#
# Idempotent: reapplying is safe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 1/5 creating /var/log/wiki (UID 10001 so the container can write to it)"
install -d -o 10001 -g 10001 -m 0755 /var/log/wiki

echo "==> 2/5 installing /opt/monitoring/prometheus/prometheus.yml"
install -o root -g root -m 0644 "$HERE/prometheus.yml" /opt/monitoring/prometheus/prometheus.yml

echo "==> 3/5 patching /opt/monitoring/alloy/config.alloy"
ALLOY_CFG=/opt/monitoring/alloy/config.alloy
if grep -q 'WIKIPEDIA RETRIEVER SECTION' "$ALLOY_CFG"; then
  echo "    already patched, skipping"
else
  cp "$ALLOY_CFG" "${ALLOY_CFG}.bak.$(date +%Y%m%d%H%M%S)"
  # Insert our block BEFORE the loki.write block so all forward_to targets are defined.
  awk '
    /^loki\.write "local_loki"/ && !inserted {
      while ((getline line < "'"$HERE/alloy-wiki.snippet.alloy"'") > 0) print line
      print ""
      inserted=1
    }
    { print }
  ' "$ALLOY_CFG" > "${ALLOY_CFG}.new"
  mv "${ALLOY_CFG}.new" "$ALLOY_CFG"
  chmod 0644 "$ALLOY_CFG"
fi

echo "==> 4/5 installing /opt/nginx/conf.d/wiki.conf"
install -o root -g root -m 0644 "$HERE/wiki.conf" /opt/nginx/conf.d/wiki.conf

echo "==> 5/5 reloading prometheus, alloy, nginx-gateway"
docker kill -s HUP prometheus
docker restart alloy
docker exec nginx-gateway nginx -t
docker exec nginx-gateway nginx -s reload

echo
echo "Done. Verification:"
echo "  curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job==\"wikipedia-retriever\")'"
echo "  docker logs --tail 30 alloy"
echo "  curl -sI http://wiki.sgcore.dev   # via cloudflared / nginx"
