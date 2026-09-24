#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  80|8877) port="$1" ;;
  *) exit 64 ;;
esac
pid=$(/usr/bin/docker -H unix:///run/mas-shopping-docker/docker.sock inspect \
  --format '{{if .State.Running}}{{.State.Pid}}{{else}}0{{end}}' mas-shopping)
test "$pid" -gt 0
exec /usr/bin/nsenter --target "$pid" --net --no-fork \
  /lib/systemd/systemd-socket-proxyd "127.0.0.1:$port"
