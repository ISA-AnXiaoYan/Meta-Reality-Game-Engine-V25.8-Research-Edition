#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
DURATION_S="${1:-45}"
RUN_ID="v6b_8route_smoke_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/sync_ipc/$RUN_ID"

SERIALS=(4110040581 4110046946 4110046952 4110046954 4110046945 4110046947 4110035688 4110046939)
ALIASES=(A B C D E F G H)

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT"
echo "$OUT"

PIDS=()
for i in "${!SERIALS[@]}"; do
  serial="${SERIALS[$i]}"
  alias="${ALIASES[$i]}"
  log="$OUT/event_${alias}.log"
  python3 metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py \
    --tsf1-camera-id "$serial" \
    --hit-camera-alias "$alias" \
    --headless \
    --processing-accumulation-time 4000 \
    --event-native-hal-bridge-enable \
    --event-native-hal-bridge-wait-mode poll \
    --event-native-hal-bridge-poll-sleep-us 100 \
    --event-native-hal-bridge-stderr-log "$OUT/event_hal_bridge_{camera}.log" \
    --enable-ext-trigger-in \
    --enable-erc \
    --erc-cd-event-rate 10000000 \
    --enable-hw-trail \
    --hw-trail-type stc_keep_trail \
    --hw-trail-threshold-us 10000 \
    --disable-spatter-processing \
    --print-event-stats \
    --event-stats-interval-ms 1000 \
    --event-worker-status-enable \
    --event-worker-status-file "$OUT/event_worker_{camera}_status.json" \
    > "$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "$alias $serial $pid" >> "$OUT/python_pids.txt"
  sleep 1.0
done

PIDS_CSV="$(IFS=,; echo "${PIDS[*]}")"

if command -v pidstat >/dev/null 2>&1; then
  pidstat -p "$PIDS_CSV" -u -r 1 "$DURATION_S" > "$OUT/pidstat_python.txt" 2>&1 &
  PIDSTAT_PID=$!
else
  echo "pidstat not installed" > "$OUT/pidstat_python.txt"
  PIDSTAT_PID=""
fi

{
  end=$((SECONDS + DURATION_S))
  while [ "$SECONDS" -lt "$end" ]; do
    echo "----- $(date +%H:%M:%S) -----"
    ps -p "$PIDS_CSV" -o pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu 2>/dev/null || true
    pgrep -af hal_event_pipe_bridge || true
    sleep 2
  done
} > "$OUT/top_timeseries.txt" 2>&1 &
TOP_PID=$!

sleep "$DURATION_S"

for pid in "${PIDS[@]}"; do
  kill -INT "$pid" 2>/dev/null || true
done
sleep 2
for pid in "${PIDS[@]}"; do
  kill "$pid" 2>/dev/null || true
done

for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null
  echo "$pid $?" >> "$OUT/python_rc.txt"
done

if [ -n "${PIDSTAT_PID:-}" ]; then wait "$PIDSTAT_PID" 2>/dev/null || true; fi
wait "$TOP_PID" 2>/dev/null || true

pgrep -af 'hal_event_pipe_bridge|metavision_spatter_tracking_linefilter_tsf1_autosender_backfill' > "$OUT/processes_after.txt" 2>&1 || true
tar -czf "$OUT.tar.gz" -C "$PROJECT_ROOT/sync_ipc" "$RUN_ID"
echo "[done] $OUT"
echo "[archive] $OUT.tar.gz"
