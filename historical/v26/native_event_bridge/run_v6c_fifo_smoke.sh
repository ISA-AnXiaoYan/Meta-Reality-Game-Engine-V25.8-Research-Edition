#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
DURATION_S="${1:-45}"
SLICES="${2:-$((DURATION_S * 300))}"
RUN_ID="v6c_fifo_smoke_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/sync_ipc/$RUN_ID"
FIFO_DIR="$OUT/fifos"

SERIALS_CSV="${SERIALS_CSV:-4110040581,4110046946,4110046952,4110046954,4110046945,4110046947,4110035688,4110046939}"
ALIASES_CSV="${ALIASES_CSV:-A,B,C,D,E,F,G,H}"
OPEN_RETRIES="${OPEN_RETRIES:-15}"
OPEN_RETRY_DELAY_MS="${OPEN_RETRY_DELAY_MS:-1500}"
IFS=',' read -r -a ALIASES <<< "$ALIASES_CSV"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT" "$FIFO_DIR"
echo "$OUT"

export MV_HAL_PLUGIN_PATH="${MV_HAL_PLUGIN_PATH:-/usr/lib/ids/ueye_evs/hal/plugins}"

native_event_bridge/build_native_event_bridge.sh "$OUT/build" > "$OUT/build.out" 2>&1
BROKER="$OUT/build/hal_event_fifo_broker"

CONSUMER_PIDS=()
for alias in "${ALIASES[@]}"; do
  fifo="$FIFO_DIR/event_fifo_${alias}.evb"
  rm -f "$fifo"
  mkfifo "$fifo"
  python3 native_event_bridge/smoke_hal_fifo_consumer.py \
    --fifo "$fifo" \
    --alias "$alias" \
    --slices "$SLICES" \
    > "$OUT/consumer_${alias}.json" 2> "$OUT/consumer_${alias}.err" &
  pid=$!
  CONSUMER_PIDS+=("$pid")
  echo "$alias $pid $fifo" >> "$OUT/consumer_pids.txt"
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

if command -v pidstat >/dev/null 2>&1; then
  pidstat -p "$BROKER_PID" -u -r 1 "$DURATION_S" > "$OUT/pidstat_broker.txt" 2>&1 &
  PIDSTAT_PID=$!
else
  echo "pidstat not installed" > "$OUT/pidstat_broker.txt"
  PIDSTAT_PID=""
fi

{
  end=$((SECONDS + DURATION_S))
  while [ "$SECONDS" -lt "$end" ]; do
    echo "----- $(date +%H:%M:%S) -----"
    ps -p "$BROKER_PID" -L -o pid,tid,psr,pcpu,pmem,comm,args --sort=-pcpu 2>/dev/null || true
    ps -p "$(IFS=,; echo "${CONSUMER_PIDS[*]}")" -o pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu 2>/dev/null || true
    sleep 2
  done
} > "$OUT/top_threads.txt" 2>&1 &
TOP_PID=$!

wait "$BROKER_PID"
echo "$?" > "$OUT/broker_rc.txt"

for pid in "${CONSUMER_PIDS[@]}"; do
  wait "$pid" 2>/dev/null
  echo "$pid $?" >> "$OUT/consumer_rc.txt"
done

if [ -n "${PIDSTAT_PID:-}" ]; then wait "$PIDSTAT_PID" 2>/dev/null || true; fi
wait "$TOP_PID" 2>/dev/null || true

pgrep -af 'hal_event_fifo_broker|hal_event_pipe_bridge|smoke_hal_fifo_consumer|metavision_spatter_tracking_linefilter_tsf1_autosender_backfill' > "$OUT/processes_after.txt" 2>&1 || true
tar -czf "$OUT.tar.gz" -C "$PROJECT_ROOT/sync_ipc" "$RUN_ID"
echo "[done] $OUT"
echo "[archive] $OUT.tar.gz"
