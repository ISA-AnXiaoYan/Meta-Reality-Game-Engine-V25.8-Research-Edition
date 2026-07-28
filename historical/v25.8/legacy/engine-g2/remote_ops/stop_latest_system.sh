#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
LABEL="${LABEL:-desktop_latest}"
WAIT_S="${1:-90}"
REASON="${STOP_REASON:-operator_stop_latest}"

cd "$PROJECT" || exit 1

PID_FILE="sync_ipc/${LABEL}_launcher.pid"
WRAPPER_PID_FILE="sync_ipc/${LABEL}_wrapper.pid"
LATEST_FILE="sync_ipc/latest_${LABEL}_run_dir.txt"

RUN=""
if [ -f "$LATEST_FILE" ]; then
  RUN="$(tr -d '\r\n' < "$LATEST_FILE" 2>/dev/null || true)"
fi

if [ -z "$RUN" ] || { [ -n "$RUN" ] && [ -f "$RUN/launcher_rc.txt" ]; }; then
  latest_any="$(ls -t sync_ipc/latest_*_run_dir.txt 2>/dev/null | head -1 || true)"
  if [ -n "$latest_any" ]; then
    fallback_run="$(tr -d '\r\n' < "$latest_any" 2>/dev/null || true)"
    if [ -n "$fallback_run" ] && [ ! -f "$fallback_run/launcher_rc.txt" ]; then
      echo "[one_click_stop] fallback latest run from $latest_any"
      LATEST_FILE="$latest_any"
      RUN="$fallback_run"
    fi
  fi
fi

mkdir -p sync_ipc
stop_payload="$(printf '{"schema_version":1,"source":"remote_ops_stop_latest_system","label":"%s","reason":"%s","created_epoch":%s}\n' "$LABEL" "$REASON" "$(date +%s)")"
printf '%s' "$stop_payload" > sync_ipc/remote_stop_request.json
if [ -n "$RUN" ] && [ -d "$RUN" ]; then
  printf '%s' "$stop_payload" > "$RUN/stop_requested.json"
fi

if [ -f remote_ops/check_fullsys_run.py ]; then
  "$PY" remote_ops/check_fullsys_run.py set-control --project . --hit-enabled false --source one_click_stop_safe_off >/dev/null 2>&1 || true
fi

launcher_pid=""
if [ -f "$PID_FILE" ]; then
  launcher_pid="$(tr -d '\r\n' < "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$launcher_pid" ] && ps -p "$launcher_pid" >/dev/null 2>&1; then
  echo "[one_click_stop] stopping launcher_pid=$launcher_pid"
  kill -INT "$launcher_pid" 2>/dev/null || true
else
  echo "[one_click_stop] launcher pid file missing or stale; searching launch_fusion_system.py"
  pids="$(ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1}' | tr '\n' ' ')"
  if [ -n "$pids" ]; then
    for pid in $pids; do
      echo "[one_click_stop] stopping found pid=$pid"
      kill -INT "$pid" 2>/dev/null || true
    done
  else
    echo "[one_click_stop] no running launch_fusion_system.py found"
  fi
fi

deadline=$((SECONDS + WAIT_S))
while [ "$SECONDS" -lt "$deadline" ]; do
  if [ -n "$RUN" ] && [ -f "$RUN/launcher_rc.txt" ] && [ -f "$RUN.tar.gz" ]; then
    break
  fi
  if [ -n "$launcher_pid" ] && ! ps -p "$launcher_pid" >/dev/null 2>&1 && [ -z "$RUN" ]; then
    break
  fi
  sleep 1
done

if [ -n "$launcher_pid" ] && ps -p "$launcher_pid" >/dev/null 2>&1; then
  echo "[one_click_stop][WARN] launcher still running after ${WAIT_S}s; sending TERM"
  kill -TERM "$launcher_pid" 2>/dev/null || true
fi

if [ -n "$RUN" ]; then
  echo "[one_click_stop] run=$RUN"
  echo "[one_click_stop] launcher_rc:"
  cat "$RUN/launcher_rc.txt" 2>/dev/null || true
  echo
  echo "[one_click_stop] archive:"
  ls -lh "$RUN.tar.gz" 2>/dev/null || true
  if [ -f "$RUN/test_report.stdout.json" ]; then
    echo "[one_click_stop] compact summary:"
    cat "$RUN/test_report.stdout.json"
  elif [ -d "$RUN" ]; then
    "$PY" remote_ops/check_fullsys_run.py summary --project . --run-dir "$RUN" --compact 2>/dev/null || true
  fi
  if [ -f "$RUN/runner.out" ]; then
    echo "[one_click_stop] runner tail:"
    tail -n 60 "$RUN/runner.out"
  fi
else
  echo "[one_click_stop] latest run not found"
fi
