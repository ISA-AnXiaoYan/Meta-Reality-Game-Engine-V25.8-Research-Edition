#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
LABEL="${LABEL:-desktop_latest}"
WAIT_S="${1:-${WAIT_S:-120}}"
MAX_STATUS_AGE_MS="${MAX_STATUS_AGE_MS:-5000}"
EXPECT_EVENT_READY="${EXPECT_EVENT_READY:-1}"
EXPECT_EVENT_COUNT="${EXPECT_EVENT_COUNT:-8}"

cd "$PROJECT" || exit 1

PID_FILE="sync_ipc/${LABEL}_launcher.pid"
LATEST_FILE="sync_ipc/latest_${LABEL}_run_dir.txt"

if [ ! -f "$LATEST_FILE" ]; then
  latest_any="$(ls -t sync_ipc/latest_*_run_dir.txt 2>/dev/null | head -1 || true)"
  if [ -n "$latest_any" ]; then
    LATEST_FILE="$latest_any"
  fi
fi

RUN=""
if [ -f "$LATEST_FILE" ]; then
  RUN="$(tr -d '\r\n' < "$LATEST_FILE" 2>/dev/null || true)"
fi

echo "[wait_ready] label=$LABEL run=${RUN:-unknown} wait=${WAIT_S}s max_status_age_ms=$MAX_STATUS_AGE_MS"

status_probe() {
  "$PY" - "$MAX_STATUS_AGE_MS" <<'PY'
import json
import sys
import time
from pathlib import Path

max_age_ms = float(sys.argv[1])
p = Path("sync_ipc/system_status_latest.json")
if not p.exists():
    print("[wait_ready] state=MISSING age_ms=-1 broken=status_file_missing")
    raise SystemExit(2)
try:
    obj = json.loads(p.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"[wait_ready] state=BAD_JSON age_ms=-1 broken={type(exc).__name__}")
    raise SystemExit(2)
summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
state = str(summary.get("overall_state") or obj.get("overall_state") or "UNKNOWN").upper()
broken = summary.get("broken_links") if isinstance(summary.get("broken_links"), list) else []
degraded = summary.get("degraded_links") if isinstance(summary.get("degraded_links"), list) else []
ts_us = float(obj.get("ts_wall_us") or obj.get("timestamp_us") or 0.0)
age_ms = ((time.time() * 1_000_000.0 - ts_us) / 1000.0) if ts_us > 0 else -1.0
print(
    "[wait_ready] "
    f"state={state} age_ms={age_ms:.1f} "
    f"broken={','.join(map(str, broken[:8])) or '-'} "
    f"degraded={','.join(map(str, degraded[:8])) or '-'}"
)
if state == "OK" and 0.0 <= age_ms <= max_age_ms:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

ready_event_count() {
  find sync_ipc -maxdepth 1 -name 'event_ready_*.flag' -printf '.' 2>/dev/null | wc -c | tr -d ' '
}

launcher_alive() {
  local pid=""
  if [ -f "$PID_FILE" ]; then
    pid="$(tr -d '\r\n' < "$PID_FILE" 2>/dev/null || true)"
  fi
  if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
    return 0
  fi
  ps -eo args | grep -F 'launch_fusion_system.py' | grep -v grep >/dev/null 2>&1
}

deadline=$((SECONDS + WAIT_S))
last_print=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! launcher_alive; then
    echo "[wait_ready][WARN] launcher process not observed yet or already exited"
  fi
  status_probe
  status_rc=$?
  ready_count="$(ready_event_count)"
  if [ "$EXPECT_EVENT_READY" = "1" ]; then
    echo "[wait_ready] event_ready=${ready_count}/${EXPECT_EVENT_COUNT}"
  fi
  if [ "$status_rc" -eq 0 ]; then
    if [ "$EXPECT_EVENT_READY" != "1" ] || [ "$ready_count" -ge "$EXPECT_EVENT_COUNT" ]; then
      echo "[wait_ready] ready"
      exit 0
    fi
  fi
  now="$SECONDS"
  if [ $((now - last_print)) -ge 10 ]; then
    last_print="$now"
  fi
  sleep 2
done

echo "[wait_ready][ERROR] timeout after ${WAIT_S}s"
echo "===== launcher pids ====="
ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $0}' || true
echo "===== event ready flags ====="
ls -l sync_ipc/event_ready_*.flag 2>/dev/null || true
echo "===== system status head ====="
sed -n '1,80p' sync_ipc/system_status_latest.json 2>/dev/null || true
echo "===== event log key errors ====="
grep -n -E 'bad HAL bridge magic|Traceback|FAIL|ERROR|Failed|Exception' sync_ipc/launcher_logs/event.log 2>/dev/null | tail -80 || true
echo "===== native HAL broker logs ====="
tail -80 sync_ipc/launcher_logs/event_native_hal_broker.err 2>/dev/null || true
tail -80 sync_ipc/launcher_logs/event_native_hal_broker.out 2>/dev/null || true
if [ -n "$RUN" ] && [ -f "$RUN/runner.out" ]; then
  echo "===== runner tail ====="
  tail -120 "$RUN/runner.out" || true
fi
exit 1
