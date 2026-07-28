#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
PROFILE="${PROFILE:-launch_profiles/627_event_overlay_bev.json}"
LABEL="${LABEL:-desktop_latest}"
LAUNCH_EXTRA_ARGS="${LAUNCH_EXTRA_ARGS:-}"
LAUNCH_EXTRA_ARGS_JSON="${LAUNCH_EXTRA_ARGS_JSON:-}"
V25_DATASET_DIR="${V25_DATASET_DIR:-}"
MVS_SOURCE_MODE_REQUESTED="${MVS_SOURCE_MODE_REQUESTED:-}"
SNAPSHOT_DELAY_S="${SNAPSHOT_DELAY_S:-60}"
CLEAN_NATIVE_HAL_FIFO="${CLEAN_NATIVE_HAL_FIFO:-1}"
LIVE_STATUS_INTERVAL_S="${LIVE_STATUS_INTERVAL_S:-5}"
PREFLIGHT_CLEAN="${PREFLIGHT_CLEAN:-1}"
PREFLIGHT_CLEAN_WAIT_S="${PREFLIGHT_CLEAN_WAIT_S:-12}"

cd "$PROJECT" || exit 1
mkdir -p sync_ipc/desktop_runs sync_ipc/launcher_logs sync_ipc/locks

launch_extra_array=()
if [ -n "$LAUNCH_EXTRA_ARGS_JSON" ]; then
  launch_args_tmp="$(mktemp)"
  if ! "$PY" - "$LAUNCH_EXTRA_ARGS_JSON" > "$launch_args_tmp" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise SystemExit("LAUNCH_EXTRA_ARGS_JSON must be a JSON string array")
for item in value:
    if "\n" in item or "\r" in item:
        raise SystemExit("launch arguments must not contain newlines")
    print(item)
PY
  then
    rm -f "$launch_args_tmp"
    echo "[one_click_start][ERROR] invalid LAUNCH_EXTRA_ARGS_JSON"
    exit 64
  fi
  mapfile -t launch_extra_array < "$launch_args_tmp"
  rm -f "$launch_args_tmp"
elif [ -n "$LAUNCH_EXTRA_ARGS" ]; then
  # Legacy contract retained for existing remote_test callers.
  # shellcheck disable=SC2206
  launch_extra_array=($LAUNCH_EXTRA_ARGS)
fi

LOCK="sync_ipc/locks/${LABEL}_start.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[one_click_start][ERROR] another start operation is active for label=$LABEL"
  exit 75
fi

PID_FILE="sync_ipc/${LABEL}_launcher.pid"
WRAPPER_PID_FILE="sync_ipc/${LABEL}_wrapper.pid"
LATEST_FILE="sync_ipc/latest_${LABEL}_run_dir.txt"

if [ -f "$PID_FILE" ]; then
  old_pid="$(tr -d '\r\n' < "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "[one_click_start] already running launcher_pid=$old_pid"
    cat "$LATEST_FILE" 2>/dev/null || true
    exit 0
  fi
fi

running_launchers="$(ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1}' | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
if [ -n "$running_launchers" ]; then
  echo "[one_click_start][ERROR] launch_fusion_system.py already running: $running_launchers"
  echo "[one_click_start][HINT] stop it first with: LABEL=<label> bash remote_ops/stop_latest_system.sh 90"
  exit 75
fi

if [ "$CLEAN_NATIVE_HAL_FIFO" = "1" ]; then
  echo "[one_click_start] cleaning native HAL FIFO runtime state"
  pkill -x hal_event_fifo_broker 2>/dev/null || true
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
fi

RUN_ID="${LABEL}_$(date +%Y%m%d_%H%M%S)_display1_keepstdin"
OUT="sync_ipc/desktop_runs/${RUN_ID}"
mkdir -p "$OUT/status" "$OUT/status_log_latest" "$OUT/control_checks" "$OUT/logs"
echo "$OUT" > "$LATEST_FILE"

if [ "$PREFLIGHT_CLEAN" = "1" ]; then
  echo "[one_click_start] preflight clean start label=$LABEL"
  if ! bash remote_ops/preflight_clean_runtime.sh "$LABEL" "$OUT/preflight_clean_summary.json" "$OUT/preflight_clean.log" "$PREFLIGHT_CLEAN_WAIT_S"; then
    preflight_rc=$?
    echo "[one_click_start][WARN] preflight clean returned rc=$preflight_rc; continuing so runtime status captures remaining issues"
  fi
fi

CHECKER="remote_ops/check_fullsys_run.py"
STOP_REQUEST="$OUT/stop_requested.json"
GLOBAL_STOP_REQUEST="sync_ipc/remote_stop_request.json"
RUNNING_FLAG="$OUT/launcher_running.flag"

(
  live_status() {
    local tag="${1:-live}"
    "$PY" "$CHECKER" live-status --project . --run-dir "$OUT" --label "$LABEL" --tag "$tag" --write-latest --compact >> "$OUT/live_status_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
  }

  finalize() {
    local rc="$1"
    rm -f "$RUNNING_FLAG"
    echo "$rc" > "$OUT/launcher_rc.txt"
    date +%s > "$OUT/end_epoch.txt"
    live_status "final"
    "$PY" "$CHECKER" snapshot --project . --run-dir "$OUT" --tag final >> "$OUT/snapshot_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
    obs_dir="$OUT/logs/obs_srt"
    mkdir -p "$obs_dir"
    : > "$obs_dir/index.txt"
    obs_start_epoch="$(cat "$OUT/start_epoch.txt" 2>/dev/null || echo 0)"
    for obs_log in obs_srt_*.log; do
      [ -f "$obs_log" ] || continue
      obs_mtime="$(stat -c '%Y' "$obs_log" 2>/dev/null || echo 0)"
      if [ "${obs_start_epoch:-0}" -gt 0 ] 2>/dev/null && [ "${obs_mtime:-0}" -lt "${obs_start_epoch:-0}" ] 2>/dev/null; then
        continue
      fi
      obs_base="$(basename "$obs_log")"
      stat -c '%n %s %y' "$obs_log" >> "$obs_dir/index.txt" 2>/dev/null || true
      tail -n 500 "$obs_log" > "$obs_dir/${obs_base}.tail.log" 2>/dev/null || true
    done
    find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort > "$OUT/event_human_mask_stats_after.txt" 2>/dev/null || true

    latest_status="$(ls -td sync_ipc/status_logs/* 2>/dev/null | head -1 || true)"
    echo "$latest_status" > "$OUT/latest_status_dir.txt"

    cp -a sync_ipc/launcher_logs "$OUT/" 2>/dev/null || true
    cp -a sync_ipc/global_bev_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_overlay_overview_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/system_status_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/source_mode_effective.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/v25_resolved_manifest_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/v25_replay_control.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/v25_replay_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/v25_panel_launch_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/v25_replay_launch_command.sh "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/_runtime_config/source_mode_effective.json "$OUT/status/source_mode_effective_runtime.json" 2>/dev/null || true
    cp -a sync_ipc/_runtime_config/v25_resolved_manifest.json "$OUT/status/v25_resolved_manifest_runtime.json" 2>/dev/null || true
    cp -a sync_ipc/global_yolo_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/global_yolo_worker_0_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/global_yolo_worker_1_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/global_person_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/person_id_dataset_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/person_id_dataset_control.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/person_id_dataset_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_mask_evidence_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_mask_control.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_worker_*_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_obs_*_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_viz_*_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_overlay_*_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_overlay_masked_*_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_overlay_masked_*_history.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_overlay_sparse_*_history.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_sync_health_*.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/event_sync_map_ring_*.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/test_run_status_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/test_run_events.jsonl "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/experiment_control.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/experiment_status_latest.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/experiment_events.jsonl "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/hit_judge_status.json "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/pseudo_hit_*.jsonl "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/pseudo_hit_all.jsonl "$OUT/status/" 2>/dev/null || true
    cp -a sync_ipc/trigger_status.json "$OUT/status/" 2>/dev/null || true

    if [ -n "$latest_status" ] && [ -d "$latest_status" ]; then
      cp -a "$latest_status/status_latest.json" "$OUT/status_log_latest/" 2>/dev/null || true
      cp -a "$latest_status/status_summary.csv" "$OUT/status_log_latest/" 2>/dev/null || true
      cp -a "$latest_status/status_tabs.jsonl" "$OUT/status_log_latest/" 2>/dev/null || true
      cp -a "$latest_status/status_snapshot.jsonl" "$OUT/status_log_latest/" 2>/dev/null || true
    fi

    "$PY" "$CHECKER" summary --project . --run-dir "$OUT" --compact > "$OUT/test_report.stdout.json" 2>> "$OUT/remote_ops_errors.log" || true
    tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    rm -f "$PID_FILE" "$WRAPPER_PID_FILE"
    echo "[one_click_start] archive=$OUT.tar.gz"
  }

  {
    echo "[one_click_start] start $(date -Is)"
    echo "[one_click_start] label=$LABEL run_id=$RUN_ID profile=$PROFILE"
    printf '[one_click_start] launch_extra_argv='
    printf '%q ' "${launch_extra_array[@]}"
    printf '\n'
    echo "[one_click_start] project=$PROJECT"
    echo "[one_click_start] deployed marker:"
    cat GIT_BRANCH_DEPLOYED_REMOTE_OPS.txt 2>/dev/null || true

    date +%s > "$OUT/start_epoch.txt"
    rm -f "$STOP_REQUEST" "$GLOBAL_STOP_REQUEST"
    find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort > "$OUT/event_human_mask_stats_before.txt" 2>/dev/null || true
    "$PY" "$CHECKER" set-control --project . --overview-visible true --show-event-layer true --bev-dilate-px 16 --hit-enabled true --final-hit-authority strong_gate --strong-gate-enabled true --strong-reasoning-enabled true --strong-reasoning-shadow-only true --human-noise-filter-enable true --human-noise-filter-mode soft --human-noise-track-start-bbox-filter-enable true --human-noise-track-start-bbox-expand-px 24 --final-hit-cross-camera-dedup-ms 700 --projected-mask-final-mode pending_only --projected-mask-require-terminal-or-turn true --same-track-episode-dedup-enable true --same-track-episode-dedup-ms 1500 --strong-reasoning-track-project-enable true --strong-reasoning-track-project-extend-px 100 --strong-reasoning-track-project-max-last-dist-px 40 --source one_click_start >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
    "$PY" "$CHECKER" snapshot --project . --run-dir "$OUT" --tag before >> "$OUT/snapshot_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
    live_status "before"

    export DISPLAY="${DISPLAY:-:1}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
    export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
    export __GL_SYNC_TO_VBLANK="${__GL_SYNC_TO_VBLANK:-0}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export MRG_RUN_ID="$RUN_ID"

    launch_cmd=("$PY" launch_fusion_system.py --launch-profile "$PROFILE" "${launch_extra_array[@]}")
    printf '%q ' "${launch_cmd[@]}" > "$OUT/launch_command.sh"
    printf '\n' >> "$OUT/launch_command.sh"
    "$PY" - "$OUT/source_mode_request.json" "$PROFILE" "$LAUNCH_EXTRA_ARGS_JSON" "$V25_DATASET_DIR" "$MVS_SOURCE_MODE_REQUESTED" <<'PY'
import json
import sys
import time
from pathlib import Path

extra_argv = json.loads(sys.argv[3]) if sys.argv[3] else []
payload = {
    "schema_version": 1,
    "kind": "v25_source_mode_request",
    "ts_wall_us": int(time.time() * 1_000_000),
    "profile": sys.argv[2],
    "launch_extra_argv": extra_argv,
    "dataset_dir": sys.argv[4],
    "mvs_source_mode_requested": sys.argv[5],
}
Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

    (
      sleep "$SNAPSHOT_DELAY_S"
      "$PY" "$CHECKER" snapshot --project . --run-dir "$OUT" --tag "startup_${SNAPSHOT_DELAY_S}s" >> "$OUT/snapshot_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
    ) &

    touch "$RUNNING_FLAG"
    if [ "$LIVE_STATUS_INTERVAL_S" -gt 0 ] 2>/dev/null; then
      (
        while [ -f "$RUNNING_FLAG" ]; do
          live_status "live"
          sleep "$LIVE_STATUS_INTERVAL_S"
        done
      ) &
      live_status_pid="$!"
    else
      live_status_pid=""
    fi
    tail -f /dev/null | "${launch_cmd[@]}" &
    launcher_pid="$!"
    echo "$launcher_pid" > "$PID_FILE"
    echo "[one_click_start] launcher_pid=$launcher_pid"
    (
      last_seen=""
      while [ -f "$RUNNING_FLAG" ] && ps -p "$launcher_pid" >/dev/null 2>&1; do
        if [ -f "$STOP_REQUEST" ] || [ -f "$GLOBAL_STOP_REQUEST" ]; then
          if [ -f "$STOP_REQUEST" ]; then
            reason="$(tr -d '\r\n' < "$STOP_REQUEST" 2>/dev/null | head -c 500 || true)"
          else
            reason="$(tr -d '\r\n' < "$GLOBAL_STOP_REQUEST" 2>/dev/null | head -c 500 || true)"
          fi
          if [ "$reason" != "$last_seen" ]; then
            echo "[one_click_start] stop requested reason=$reason at $(date -Is)"
            "$PY" "$CHECKER" set-control --project . --hit-enabled false --source one_click_stop_request_safe_off >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
            kill -INT "$launcher_pid" 2>/dev/null || true
            last_seen="$reason"
          fi
          break
        fi
        sleep 1
      done
    ) &
    stop_watch_pid="$!"
    wait "$launcher_pid"
    rc="$?"
    rm -f "$RUNNING_FLAG"
    wait "$stop_watch_pid" 2>/dev/null || true
    if [ -n "${live_status_pid:-}" ]; then
      wait "$live_status_pid" 2>/dev/null || true
    fi
    echo "[one_click_start] launcher rc=$rc"
    finalize "$rc"
    echo "[one_click_start] end $(date -Is)"
  } > "$OUT/runner.out" 2>&1
) </dev/null >/dev/null 2>&1 &

wrapper_pid="$!"
echo "$wrapper_pid" > "$WRAPPER_PID_FILE"
echo "[one_click_start] run=$OUT"
echo "[one_click_start] wrapper_pid=$wrapper_pid"
echo "[one_click_start] latest_file=$LATEST_FILE"
