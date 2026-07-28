#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
DURATION_S="${1:-45}"
RUN_ID="v6c_event_worker_fifo_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/sync_ipc/$RUN_ID"
FIFO_DIR="$OUT/fifos"

SERIALS_CSV="${SERIALS_CSV:-4110040581,4110046946,4110046952,4110046954,4110046945,4110046947,4110035688,4110046939}"
ALIASES_CSV="${ALIASES_CSV:-A,B,C,D,E,F,G,H}"
OPEN_RETRIES="${OPEN_RETRIES:-1}"
OPEN_RETRY_DELAY_MS="${OPEN_RETRY_DELAY_MS:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ysxq/miniconda3/envs/ids_recv/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

IFS=',' read -r -a SERIALS <<< "$SERIALS_CSV"
IFS=',' read -r -a ALIASES <<< "$ALIASES_CSV"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT" "$FIFO_DIR"
echo "$OUT"

export MV_HAL_PLUGIN_PATH="${MV_HAL_PLUGIN_PATH:-/usr/lib/ids/ueye_evs/hal/plugins}"

native_event_bridge/build_native_event_bridge.sh "$OUT/build" > "$OUT/build.out" 2>&1
BROKER="$OUT/build/hal_event_fifo_broker"

WORKER_PIDS=()
for i in "${!SERIALS[@]}"; do
  serial="${SERIALS[$i]}"
  alias="${ALIASES[$i]:-cam${i}}"
  fifo="$FIFO_DIR/event_fifo_${alias}.evb"
  rm -f "$fifo"
  mkfifo "$fifo"
  "$PYTHON_BIN" metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py \
    --tsf1-camera-id "$serial" \
    --hit-camera-alias "$alias" \
    --headless \
    --processing-accumulation-time 4000 \
    --event-native-hal-fifo-path "$fifo" \
    --enable-ext-trigger-in \
    --enable-erc \
    --erc-cd-event-rate 10000000 \
    --enable-hw-trail \
    --hw-trail-type stc_keep_trail \
    --hw-trail-threshold-us 10000 \
    --disable-spatter-processing \
    --print-event-stats \
    --event-stats-interval-ms 1000 \
    --event-loop-profiler-enable \
    --event-loop-profiler-interval-ms 1000 \
    --event-loop-profiler-jsonl-template "$OUT/event_loop_profile_{camera}.jsonl" \
    --event-worker-status-enable \
    --event-worker-status-interval-ms 200 \
    --event-worker-status-file "$OUT/event_worker_{camera}_status.json" \
    > "$OUT/event_${alias}.log" 2>&1 &
  pid=$!
  WORKER_PIDS+=("$pid")
  echo "$alias $serial $pid $fifo" >> "$OUT/event_worker_pids.txt"
done

"$BROKER" \
  --serials "$SERIALS_CSV" \
  --aliases "$ALIASES_CSV" \
  --output-mode fifo \
  --out-dir "$FIFO_DIR" \
  --fifo-open-mode rdwr \
  --duration-sec "$DURATION_S" \
  --slice-us 4000 \
  --wait-mode poll \
  --poll-sleep-us 100 \
  --open-retries "$OPEN_RETRIES" \
  --open-retry-delay-ms "$OPEN_RETRY_DELAY_MS" \
  --stats-jsonl "$OUT/broker_stats.jsonl" \
  --enable-trigger-in \
  --enable-erc \
  --erc-rate 10000000 \
  --enable-trail \
  --trail-type stc_keep_trail \
  --trail-threshold-us 10000 \
  > "$OUT/broker.out" 2> "$OUT/broker.err" &
BROKER_PID=$!
echo "$BROKER_PID" > "$OUT/broker.pid"

PIDS_CSV="$(IFS=,; echo "${WORKER_PIDS[*]}")"
ALL_PIDS_CSV="$BROKER_PID"
if [ -n "$PIDS_CSV" ]; then
  ALL_PIDS_CSV="$BROKER_PID,$PIDS_CSV"
fi

if command -v pidstat >/dev/null 2>&1; then
  pidstat -p "$ALL_PIDS_CSV" -u -r 1 "$DURATION_S" > "$OUT/pidstat_all.txt" 2>&1 &
  PIDSTAT_PID=$!
else
  echo "pidstat not installed" > "$OUT/pidstat_all.txt"
  PIDSTAT_PID=""
fi

{
  end=$((SECONDS + DURATION_S))
  while [ "$SECONDS" -lt "$end" ]; do
    echo "----- $(date +%H:%M:%S) -----"
    ps -p "$ALL_PIDS_CSV" -o pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu 2>/dev/null || true
    sleep 2
  done
} > "$OUT/top_processes.txt" 2>&1 &
TOP_PID=$!

wait "$BROKER_PID"
BROKER_RC=$?
echo "$BROKER_RC" > "$OUT/broker_rc.txt"

deadline=$((SECONDS + 15))
for pid in "${WORKER_PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
    sleep 0.2
  done
done
for pid in "${WORKER_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
  fi
done
sleep 2
for pid in "${WORKER_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
done

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid" 2>/dev/null
  echo "$pid $?" >> "$OUT/event_worker_rc.txt"
done

if [ -n "${PIDSTAT_PID:-}" ]; then wait "$PIDSTAT_PID" 2>/dev/null || true; fi
wait "$TOP_PID" 2>/dev/null || true

pgrep -af 'hal_event_fifo_broker|hal_event_pipe_bridge|smoke_hal_fifo_consumer|metavision_spatter_tracking_linefilter_tsf1_autosender_backfill' > "$OUT/processes_after.txt" 2>&1 || true
tar -czf "$OUT.tar.gz" -C "$PROJECT_ROOT/sync_ipc" "$RUN_ID"
echo "[done] $OUT"
echo "[archive] $OUT.tar.gz"
exit "$BROKER_RC"
