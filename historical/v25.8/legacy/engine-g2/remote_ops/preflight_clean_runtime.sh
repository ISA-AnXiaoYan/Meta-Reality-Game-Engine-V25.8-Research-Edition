#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
LABEL="${1:-remote_ops}"
SUMMARY_JSON="${2:-}"
LOG_PATH="${3:-}"
WAIT_S="${4:-12}"
OBS_SRT_PORT_START="${OBS_SRT_PORT_START:-6101}"
OBS_SRT_PORT_END="${OBS_SRT_PORT_END:-6108}"

cd "$PROJECT" || exit 1

if [ -z "$LOG_PATH" ]; then
  LOG_PATH="sync_ipc/preflight_clean_${LABEL}.log"
fi
mkdir -p "$(dirname "$LOG_PATH")"

now_us() {
  python3 -c 'import time; print(int(time.time()*1000000))' 2>/dev/null || date +%s000000
}

count_lines() {
  sed '/^[[:space:]]*$/d' | wc -l | tr -d ' '
}

pgrep_lines() {
  local pattern="$1"
  pgrep -af "$pattern" 2>/dev/null | grep -v "remote_ops/preflight_clean_runtime.sh" || true
}

kill_pattern() {
  local pattern="$1"
  local signal="${2:-TERM}"
  local lines pids
  lines="$(pgrep_lines "$pattern")"
  if [ -z "$lines" ]; then
    return 0
  fi
  printf '%s\n' "$lines" >> "$LOG_PATH"
  pids="$(printf '%s\n' "$lines" | awk '{print $1}' | tr '\n' ' ')"
  if [ -n "$pids" ]; then
    kill "-$signal" $pids 2>/dev/null || true
  fi
}

list_obs_srt_ffmpeg() {
  local port
  for port in $(seq "$OBS_SRT_PORT_START" "$OBS_SRT_PORT_END" 2>/dev/null); do
    pgrep_lines "ffmpeg.*srt://0\\.0\\.0\\.0:${port}.*mode=listener"
  done | awk '!seen[$1]++'
}

kill_obs_srt_ffmpeg() {
  local signal="${1:-TERM}"
  local lines pids
  lines="$(list_obs_srt_ffmpeg)"
  if [ -z "$lines" ]; then
    return 0
  fi
  {
    echo "[preflight] OBS-SRT ffmpeg listeners signal=$signal ports=${OBS_SRT_PORT_START}-${OBS_SRT_PORT_END}:"
    printf '%s\n' "$lines"
  } >> "$LOG_PATH"
  pids="$(printf '%s\n' "$lines" | awk '{print $1}' | tr '\n' ' ')"
  if [ -n "$pids" ]; then
    kill "-$signal" $pids 2>/dev/null || true
  fi
}

started_us="$(now_us)"
{
  echo "[preflight] start $(date -Is) label=$LABEL wait_s=$WAIT_S project=$PROJECT"
  echo "[preflight] existing launch_fusion_system.py:"
  pgrep_lines 'launch_fusion_system.py'
  echo "[preflight] existing Global BEV:"
  pgrep_lines 'global_bev_fusion_renderer.py'
  echo "[preflight] existing OBS-SRT ffmpeg listeners:"
  list_obs_srt_ffmpeg
} > "$LOG_PATH" 2>&1

before_launchers="$(pgrep_lines 'launch_fusion_system.py' | count_lines)"
before_bev="$(pgrep_lines 'global_bev_fusion_renderer.py' | count_lines)"
before_obs="$(list_obs_srt_ffmpeg | count_lines)"

# Stop managed desktop/latest wrappers first so their finalizers can archive
# cleanly.  Direct process cleanup below is the safety net for stale children.
for pid_file in sync_ipc/*_launcher.pid sync_ipc/*_wrapper.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(tr -d '\r\n' < "$pid_file" 2>/dev/null || true)"
  [ -n "$pid" ] || continue
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "[preflight] INT managed pid=$pid file=$pid_file" >> "$LOG_PATH"
    kill -INT "$pid" 2>/dev/null || true
  fi
done

kill_pattern 'launch_fusion_system.py' INT
sleep 2
kill_pattern 'multi_camera_launcher.py' TERM
kill_pattern 'global_bev_fusion_renderer.py' TERM
kill_pattern 'pyopengl_cuda_preview_renderer.py' TERM
kill_pattern 'global_yolo_infer_server.py' TERM
kill_pattern 'global_person_id_service.py' TERM
kill_pattern 'hit_judge_server.py' TERM
kill_pattern 'game_event_bridge_service.py' TERM
kill_pattern 'damage_attribution_service.py' TERM
kill_pattern 'metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py' TERM
kill_pattern 'hal_event_fifo_broker' TERM
kill_obs_srt_ffmpeg TERM

deadline=$(( $(date +%s) + ${WAIT_S:-12} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  remaining="$(
    {
      pgrep_lines 'launch_fusion_system.py'
      pgrep_lines 'global_bev_fusion_renderer.py'
      pgrep_lines 'global_yolo_infer_server.py'
      pgrep_lines 'metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py'
      pgrep_lines 'hal_event_fifo_broker'
      list_obs_srt_ffmpeg
    } | count_lines
  )"
  [ "$remaining" = "0" ] && break
  sleep 1
done

kill_pattern 'launch_fusion_system.py' KILL
kill_pattern 'global_bev_fusion_renderer.py' KILL
kill_pattern 'global_yolo_infer_server.py' KILL
kill_pattern 'metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py' KILL
kill_pattern 'hal_event_fifo_broker' KILL
kill_obs_srt_ffmpeg KILL

mkdir -p sync_ipc/native_hal_fifos
find sync_ipc/native_hal_fifos -maxdepth 1 \( -type p -o -type f \) \( -name 'event_fifo_*.evb' -o -name '*.tmp' \) -delete 2>/dev/null || true
find sync_ipc -maxdepth 1 \( -name 'event_ready_*.flag' -o -name 'event_trigger_*.csv' -o -name 'event_trigger_map_*.jsonl' -o -name 'event_sync_health_*.json' -o -name 'event_sync_map_ring_*.json' -o -name 'event_overlay_sparse_*_history.json' \) -delete 2>/dev/null || true
for shm_pattern in \
  /dev/shm/event_overlay_latest_*.mmap \
  /dev/shm/event_overlay_masked_latest_*.mmap \
  /dev/shm/event_overlay_masked_hist_*.mmap \
  /dev/shm/event_overlay_sparse_hist_*.mmap \
  /dev/shm/event_human_mask_raster_*.mmap; do
  for shm_file in $shm_pattern; do
    [ -e "$shm_file" ] || continue
    rm -f "$shm_file" 2>/dev/null || true
  done
done
stale_dir="sync_ipc/preflight_stale_status/$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "$stale_dir"
for status_file in \
  global_bev_status.json global_bev_status_debug.json system_status_latest.json \
  source_mode_effective.json v25_resolved_manifest_latest.json \
  event_overlay_overview_status.json global_yolo_status.json global_person_status.json \
  hit_judge_status.json trigger_status.json test_run_status_latest.json; do
  if [ -f "sync_ipc/$status_file" ]; then
    mv "sync_ipc/$status_file" "$stale_dir/$status_file" 2>/dev/null || rm -f "sync_ipc/$status_file" 2>/dev/null || true
  fi
done
for runtime_status_file in source_mode_effective.json v25_resolved_manifest.json; do
  if [ -f "sync_ipc/_runtime_config/$runtime_status_file" ]; then
    mkdir -p "$stale_dir/_runtime_config"
    mv "sync_ipc/_runtime_config/$runtime_status_file" "$stale_dir/_runtime_config/$runtime_status_file" 2>/dev/null || rm -f "sync_ipc/_runtime_config/$runtime_status_file" 2>/dev/null || true
  fi
done
for status_pattern in \
  event_worker_*_status.json \
  event_overlay_*_latest.json \
  event_overlay_masked_*_latest.json \
  event_overlay_masked_*_history.json \
  event_obs_*_latest.json \
  event_viz_*_latest.json; do
  for status_path in sync_ipc/$status_pattern; do
    [ -f "$status_path" ] || continue
    status_name="$(basename "$status_path")"
    mv "$status_path" "$stale_dir/$status_name" 2>/dev/null || rm -f "$status_path" 2>/dev/null || true
  done
done

after_launchers="$(pgrep_lines 'launch_fusion_system.py' | count_lines)"
after_bev="$(pgrep_lines 'global_bev_fusion_renderer.py' | count_lines)"
after_hal="$(pgrep_lines 'hal_event_fifo_broker' | count_lines)"
after_obs="$(list_obs_srt_ffmpeg | count_lines)"
finished_us="$(now_us)"

{
  echo "[preflight] after launch_fusion_system.py:"
  pgrep_lines 'launch_fusion_system.py'
  echo "[preflight] after Global BEV:"
  pgrep_lines 'global_bev_fusion_renderer.py'
  echo "[preflight] after HAL broker:"
  pgrep_lines 'hal_event_fifo_broker'
  echo "[preflight] after OBS-SRT ffmpeg listeners:"
  list_obs_srt_ffmpeg
  echo "[preflight] end $(date -Is)"
} >> "$LOG_PATH" 2>&1

if [ -n "$SUMMARY_JSON" ]; then
  mkdir -p "$(dirname "$SUMMARY_JSON")"
  cat > "$SUMMARY_JSON" <<EOF
{"schema_version":1,"kind":"remote_preflight_clean","label":"$LABEL","started_wall_us":$started_us,"finished_wall_us":$finished_us,"before":{"launchers":$before_launchers,"global_bev":$before_bev,"obs_srt_ffmpeg":$before_obs},"after":{"launchers":$after_launchers,"global_bev":$after_bev,"hal_event_fifo_broker":$after_hal,"obs_srt_ffmpeg":$after_obs},"obs_srt_port_start":$OBS_SRT_PORT_START,"obs_srt_port_end":$OBS_SRT_PORT_END,"stale_status_dir":"$stale_dir","log":"$LOG_PATH"}
EOF
fi

if [ "$after_launchers" != "0" ] || [ "$after_bev" != "0" ] || [ "$after_obs" != "0" ]; then
  echo "[preflight][WARN] stale processes remain; see $LOG_PATH" >&2
  exit 76
fi
exit 0
