#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
LABEL="${1:-__latest__}"
WAIT_S="${2:-45}"
REASON="${STOP_REASON:-operator_stop}"

cd "$PROJECT" || exit 1

if [ "$LABEL" = "__latest__" ] || [ -z "$LABEL" ]; then
  latest_file="$(ls -t sync_ipc/latest_*_run_dir.txt 2>/dev/null | head -1 || true)"
else
  latest_file="sync_ipc/latest_${LABEL}_run_dir.txt"
fi

if [ -z "${latest_file:-}" ] || [ ! -f "$latest_file" ]; then
  echo "[remote_ops][ERROR] latest run file not found for label='$LABEL'" >&2
  exit 2
fi

RUN="$(tr -d '\r\n' < "$latest_file")"
echo "[remote_ops] run=$RUN"

mkdir -p sync_ipc
stop_payload="$(printf '{"schema_version":1,"source":"remote_ops_stop_fullsys_test","label":"%s","reason":"%s","created_epoch":%s}\n' "$LABEL" "$REASON" "$(date +%s)")"
printf '%s' "$stop_payload" > sync_ipc/remote_stop_request.json
if [ -n "$RUN" ] && [ -d "$RUN" ]; then
  printf '%s' "$stop_payload" > "$RUN/stop_requested.json"
fi

if [ -f remote_ops/check_fullsys_run.py ]; then
  "$PY" remote_ops/check_fullsys_run.py set-control --project . --hit-enabled false --source remote_ops_stop_safe_off >/dev/null 2>&1 || true
fi

echo "[remote_ops] launch_fusion pids before:"
ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1, $0}' || true

pids="$(ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1}' | tr '\n' ' ')"
if [ -z "$pids" ]; then
  echo "[remote_ops] no launch_fusion_system.py process found"
else
  for pid in $pids; do
    kill -INT "$pid" 2>/dev/null || true
  done
fi

deadline=$((SECONDS + WAIT_S))
while [ "$SECONDS" -lt "$deadline" ]; do
  if [ -f "$RUN/launcher_rc.txt" ] && [ -f "$RUN.tar.gz" ]; then
    break
  fi
  sleep 1
done

echo "[remote_ops] launch_fusion pids after:"
ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1, $0}' || true

echo "[remote_ops] launcher_rc:"
cat "$RUN/launcher_rc.txt" 2>/dev/null || true

echo "[remote_ops] archive:"
ls -lh "$RUN.tar.gz" 2>/dev/null || true

if [ -f "$RUN/runner.out" ]; then
  echo "[remote_ops] runner tail:"
  tail -n 80 "$RUN/runner.out"
fi
