#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
PROFILE="${PROFILE:-launch_profiles/627_event_overlay_bev.json}"
LAUNCH_EXTRA_ARGS="${LAUNCH_EXTRA_ARGS:-}"
V25_DATASET_DIR="${V25_DATASET_DIR:-}"
MVS_SOURCE_MODE_REQUESTED="${MVS_SOURCE_MODE_REQUESTED:-live_camera}"
LABEL="${1:-smoke_remote_ops}"
DURATION_S="${2:-370}"
CONTROL_DELAY_S="${CONTROL_DELAY_S:-90}"
CONTROL_ENABLE="${CONTROL_ENABLE:-1}"
FULL_RECORDING_ENABLE="${FULL_RECORDING_ENABLE:-0}"
FULL_RECORDING_SESSION_ID="${FULL_RECORDING_SESSION_ID:-}"
RAW_RECORD_ENABLE="${RAW_RECORD_ENABLE:-0}"
RAW_RECORD_TARGETS="${RAW_RECORD_TARGETS:-event_raw,mvs_raw}"
RAW_RECORD_SESSION_ID="${RAW_RECORD_SESSION_ID:-}"
RAW_RECORD_START_DELAY_S="${RAW_RECORD_START_DELAY_S:-$CONTROL_DELAY_S}"
RAW_RECORD_STOP_MARGIN_S="${RAW_RECORD_STOP_MARGIN_S:-15}"
PERSON_ID_DATASET_ENABLE="${PERSON_ID_DATASET_ENABLE:-0}"
PERSON_ID_DATASET_KIND="${PERSON_ID_DATASET_KIND:-person_id}"
PERSON_ID_DATASET_TYPE_RAW="${PERSON_ID_DATASET_TYPE:-}"
PERSON_ID_DATASET_TYPE="${PERSON_ID_DATASET_TYPE:-negative}"
PERSON_ID_DATASET_LABEL="${PERSON_ID_DATASET_LABEL:-${LABEL}_no_person}"
PERSON_ID_DATASET_SESSION_ID="${PERSON_ID_DATASET_SESSION_ID:-}"
PERSON_ID_DATASET_START_DELAY_S="${PERSON_ID_DATASET_START_DELAY_S:-$CONTROL_DELAY_S}"
PERSON_ID_DATASET_STOP_MARGIN_S="${PERSON_ID_DATASET_STOP_MARGIN_S:-15}"
PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS="${PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS:-}"
PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY="${PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY:-}"
PERSON_ID_DATASET_INCLUDE_HIT_DEBUG="${PERSON_ID_DATASET_INCLUDE_HIT_DEBUG:-}"
LIVE_STATUS_INTERVAL_S="${LIVE_STATUS_INTERVAL_S:-5}"
PREFLIGHT_CLEAN="${PREFLIGHT_CLEAN:-1}"
PREFLIGHT_CLEAN_WAIT_S="${PREFLIGHT_CLEAN_WAIT_S:-12}"
FINAL_HIT_AUTHORITY="${FINAL_HIT_AUTHORITY:-strong_gate}"
STRONG_REASONING_SHADOW_ONLY="${STRONG_REASONING_SHADOW_ONLY:-true}"
HUMAN_NOISE_FILTER_MODE="${HUMAN_NOISE_FILTER_MODE:-soft}"
HUMAN_NOISE_TRACK_START_BBOX_EXPAND_PX="${HUMAN_NOISE_TRACK_START_BBOX_EXPAND_PX:-24}"
HUMAN_NOISE_TRACK_START_BBOX_POLICY="${HUMAN_NOISE_TRACK_START_BBOX_POLICY:-soft}"
PROJECTED_MASK_FINAL_MODE="${PROJECTED_MASK_FINAL_MODE:-pending_only}"
PROJECTED_MASK_REQUIRE_TERMINAL_OR_TURN="${PROJECTED_MASK_REQUIRE_TERMINAL_OR_TURN:-true}"
SAME_TRACK_EPISODE_DEDUP_MS="${SAME_TRACK_EPISODE_DEDUP_MS:-1500}"
STRONG_REASONING_TRACK_PROJECT_EXTEND_PX="${STRONG_REASONING_TRACK_PROJECT_EXTEND_PX:-100}"
STRONG_REASONING_TRACK_PROJECT_MAX_LAST_DIST_PX="${STRONG_REASONING_TRACK_PROJECT_MAX_LAST_DIST_PX:-40}"

cd "$PROJECT" || exit 1
mkdir -p sync_ipc/full_system_tests sync_ipc/locks

LOCK="sync_ipc/locks/remote_ops_fullsys.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[remote_ops][ERROR] another full-system test appears to be running; lock=$LOCK" >&2
  exit 75
fi

RUN_ID="${LABEL}_$(date +%Y%m%d_%H%M%S)_display1_keepstdin"
OUT="sync_ipc/full_system_tests/${RUN_ID}"
mkdir -p "$OUT/status" "$OUT/status_log_latest" "$OUT/control_checks" "$OUT/perf_samples" "$OUT/logs"
echo "$OUT" > "sync_ipc/latest_${LABEL}_run_dir.txt"
date +%s > "$OUT/start_epoch.txt"
V26_B_LOG_OFFSETS="$OUT/v26_b_log_offsets.tsv"

snapshot_v26_b_log_offsets() {
  find sync_ipc -maxdepth 1 -type f -name 'v26_b_*.jsonl' -printf '%f\t%s\n' | sort > "$V26_B_LOG_OFFSETS" 2>/dev/null || true
}

archive_v26_b_log_deltas() {
  local out_dir="$OUT/status/v26_b"
  local name size offset start_byte
  mkdir -p "$out_dir"
  while IFS=$'\t' read -r name size; do
    [ -n "$name" ] || continue
    offset=0
    if [ -f "$V26_B_LOG_OFFSETS" ]; then
      offset="$(awk -F '\t' -v n="$name" '$1 == n {print $2; exit}' "$V26_B_LOG_OFFSETS" 2>/dev/null || true)"
      offset="${offset:-0}"
    fi
    if [ "$size" -gt "$offset" ] 2>/dev/null; then
      start_byte=$((offset + 1))
      tail -c +"$start_byte" "sync_ipc/$name" > "$out_dir/$name" 2>/dev/null || true
    fi
  done < <(find sync_ipc -maxdepth 1 -type f -name 'v26_b_*.jsonl' -printf '%f\t%s\n' | sort)
}

snapshot_v26_b_log_offsets

if [ "$FULL_RECORDING_ENABLE" = "1" ]; then
  FULL_RECORDING_SESSION_ID="${FULL_RECORDING_SESSION_ID:-${RUN_ID}_full}"
  RAW_RECORD_ENABLE="1"
  RAW_RECORD_SESSION_ID="${RAW_RECORD_SESSION_ID:-$FULL_RECORDING_SESSION_ID}"
  RAW_RECORD_START_DELAY_S="${RAW_RECORD_START_DELAY_S:-$CONTROL_DELAY_S}"
  RAW_RECORD_STOP_MARGIN_S="${RAW_RECORD_STOP_MARGIN_S:-$PERSON_ID_DATASET_STOP_MARGIN_S}"
  PERSON_ID_DATASET_ENABLE="1"
  PERSON_ID_DATASET_KIND="evidence_replay"
  if [ -z "$PERSON_ID_DATASET_TYPE_RAW" ]; then
    PERSON_ID_DATASET_TYPE="mixed"
  fi
  PERSON_ID_DATASET_LABEL="${PERSON_ID_DATASET_LABEL:-${LABEL}_full_evidence}"
  PERSON_ID_DATASET_SESSION_ID="${PERSON_ID_DATASET_SESSION_ID:-$FULL_RECORDING_SESSION_ID}"
  PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS="true"
  PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY="true"
  PERSON_ID_DATASET_INCLUDE_HIT_DEBUG="true"
fi

CHECKER="remote_ops/check_fullsys_run.py"
RUNNING_FLAG="$OUT/launcher_running.flag"
STOP_REQUEST="$OUT/stop_requested.json"
GLOBAL_STOP_REQUEST="sync_ipc/remote_stop_request.json"
LAUNCHER_WRAPPER_PID_FILE="$OUT/launcher_wrapper.pid"
LAUNCHER_PID_FILE="$OUT/launcher_pids.txt"

snapshot() {
  local tag="$1"
  "$PY" "$CHECKER" snapshot --project . --run-dir "$OUT" --tag "$tag" >> "$OUT/snapshot_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

live_status() {
  local tag="${1:-live}"
  "$PY" "$CHECKER" live-status --project . --run-dir "$OUT" --label "$LABEL" --tag "$tag" --write-latest --compact >> "$OUT/live_status_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

set_control() {
  local visible="$1"
  local dilate="$2"
  "$PY" "$CHECKER" set-control --project . --overview-visible "$visible" --show-event-layer true --bev-dilate-px "$dilate" --source remote_ops >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

set_person_dataset_control() {
  local cmd="$1"
  normalize_bool_arg() {
    case "${1:-}" in
      1|true|TRUE|True|yes|YES|Yes|on|ON|On) printf 'true' ;;
      0|false|FALSE|False|no|NO|No|off|OFF|Off) printf 'false' ;;
      *) printf '%s' "${1:-}" ;;
    esac
  }
  "$PY" "$CHECKER" set-control --project . --person-dataset-cmd "$cmd" --person-dataset-kind "$PERSON_ID_DATASET_KIND" --person-dataset-type "$PERSON_ID_DATASET_TYPE" --person-dataset-label "$PERSON_ID_DATASET_LABEL" --person-dataset-session-id "$PERSON_ID_DATASET_SESSION_ID" --person-dataset-include-mask-polygons "$(normalize_bool_arg "$PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS")" --person-dataset-include-bullet-trajectory "$(normalize_bool_arg "$PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY")" --person-dataset-include-hit-debug "$(normalize_bool_arg "$PERSON_ID_DATASET_INCLUDE_HIT_DEBUG")" --source remote_ops_person_dataset >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

set_raw_record_control() {
  local cmd="$1"
  "$PY" "$CHECKER" set-control --project . --raw-record-cmd "$cmd" --raw-record-targets "$RAW_RECORD_TARGETS" --raw-record-session-id "$RAW_RECORD_SESSION_ID" --source remote_ops_raw_record >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

stop_background_pid() {
  local pid="${1:-}"
  local name="${2:-background}"
  [ -n "$pid" ] || return 0
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "[runner] stopping $name pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 0.2
  fi
  if ps -p "$pid" >/dev/null 2>&1; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

stop_background_tasks() {
  stop_background_pid "${STOP_WATCH_PID:-}" "stop-watch"
  stop_background_pid "${CONTROL_PID:-}" "control-sequence"
  stop_background_pid "${RAW_RECORD_PID:-}" "raw-record-sequence"
  stop_background_pid "${PERSON_DATASET_PID:-}" "person-dataset-sequence"
  stop_background_pid "${LIVE_STATUS_PID:-}" "live-status"
}

init_controls() {
  "$PY" "$CHECKER" set-control --project . --overview-visible true --show-event-layer true --bev-dilate-px 16 --hit-enabled true --final-hit-authority "$FINAL_HIT_AUTHORITY" --strong-gate-enabled true --strong-reasoning-enabled true --strong-reasoning-shadow-only "$STRONG_REASONING_SHADOW_ONLY" --human-noise-filter-enable true --human-noise-filter-mode "$HUMAN_NOISE_FILTER_MODE" --human-noise-track-start-bbox-filter-enable true --human-noise-track-start-bbox-expand-px "$HUMAN_NOISE_TRACK_START_BBOX_EXPAND_PX" --human-noise-track-start-bbox-policy "$HUMAN_NOISE_TRACK_START_BBOX_POLICY" --final-hit-cross-camera-dedup-ms 700 --projected-mask-final-mode "$PROJECTED_MASK_FINAL_MODE" --projected-mask-require-terminal-or-turn "$PROJECTED_MASK_REQUIRE_TERMINAL_OR_TURN" --same-track-episode-dedup-enable true --same-track-episode-dedup-ms "$SAME_TRACK_EPISODE_DEDUP_MS" --strong-reasoning-track-project-enable true --strong-reasoning-track-project-extend-px "$STRONG_REASONING_TRACK_PROJECT_EXTEND_PX" --strong-reasoning-track-project-max-last-dist-px "$STRONG_REASONING_TRACK_PROJECT_MAX_LAST_DIST_PX" --source remote_ops_init >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

sample_perf() {
  local tag="$1"
  {
    echo "===== ps $tag $(date -Is) ====="
    ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -80
    echo "===== du sync_ipc $tag ====="
    du -sh sync_ipc 2>/dev/null || true
    echo "===== event_human_mask_stats $tag ====="
    find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort || true
  } > "$OUT/perf_samples/${tag}.txt" 2>&1
}

collect_obs_srt_logs() {
  local obs_dir="$OUT/logs/obs_srt"
  local f base start_epoch mtime
  mkdir -p "$obs_dir"
  : > "$obs_dir/index.txt"
  start_epoch="$(cat "$OUT/start_epoch.txt" 2>/dev/null || echo 0)"
  for f in obs_srt_*.log; do
    [ -f "$f" ] || continue
    mtime="$(stat -c '%Y' "$f" 2>/dev/null || echo 0)"
    if [ "${start_epoch:-0}" -gt 0 ] 2>/dev/null && [ "${mtime:-0}" -lt "${start_epoch:-0}" ] 2>/dev/null; then
      continue
    fi
    base="$(basename "$f")"
    stat -c '%n %s %y' "$f" >> "$obs_dir/index.txt" 2>/dev/null || true
    tail -n 500 "$f" > "$obs_dir/${base}.tail.log" 2>/dev/null || true
  done
}

copy_replay_sync_anchors_to_dataset() {
  local session_id="${RAW_RECORD_SESSION_ID:-${FULL_RECORDING_SESSION_ID:-}}"
  local target anchor_dir copied_any f base
  [ -n "$session_id" ] || return 0

  for target in "raw_records/$session_id" "recordings/$session_id"; do
    [ -d "$target" ] || continue
    anchor_dir="$target/replay_sync"
    mkdir -p "$anchor_dir" 2>/dev/null || continue
    copied_any=0
    for f in sync_ipc/event_trigger_map_*.jsonl sync_ipc/event_sync_map_ring_*.json sync_ipc/global_trigger_timeline*.json* sync_ipc/sync_start.flag sync_ipc/sync_start_latest.json; do
      [ -f "$f" ] || continue
      base="$(basename "$f")"
      cp -a "$f" "$anchor_dir/$base" 2>/dev/null && copied_any=1
    done
    if [ "$copied_any" = "1" ]; then
      "$PY" - "$anchor_dir/manifest.json" "$session_id" <<'PY' 2>/dev/null || true
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "v25_replay_sync_anchor_manifest",
    "session_id": sys.argv[2],
    "created_wall_us": int(time.time() * 1_000_000),
    "contract": "event_trigger_map_jsonl",
    "files": sorted(p.name for p in path.parent.iterdir() if p.is_file() and p.name != path.name),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
      echo "[runner] replay_sync_anchors copied to $anchor_dir"
    fi
  done
}

finalize() {
  local rc="$1"
  rm -f "$RUNNING_FLAG"
  echo "$rc" > "$OUT/launcher_rc.txt"
  date +%s > "$OUT/end_epoch.txt"
  live_status "final"
  snapshot "final"
  sample_perf "final"
  collect_obs_srt_logs
  find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort > "$OUT/event_human_mask_stats_after.txt" 2>/dev/null || true

  latest_status="$(ls -td sync_ipc/status_logs/* 2>/dev/null | head -1 || true)"
  echo "$latest_status" > "$OUT/latest_status_dir.txt"

  cp -a sync_ipc/launcher_logs "$OUT/" 2>/dev/null || true
  cp -a sync_ipc/launcher_logs/person_id_dataset.log "$OUT/logs/" 2>/dev/null || true
  cp -a sync_ipc/global_bev_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_overlay_overview_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/system_status_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/source_mode_effective.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/v25_resolved_manifest_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/v25_replay_control.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/v25_replay_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/v25_replay_launch_command.sh "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/_runtime_config/source_mode_effective.json "$OUT/status/source_mode_effective_runtime.json" 2>/dev/null || true
  cp -a sync_ipc/_runtime_config/v25_resolved_manifest.json "$OUT/status/v25_resolved_manifest_runtime.json" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_worker_0_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_worker_1_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_infer_perf.csv "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_exposure_audit.csv "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_worker_*_perf.csv "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_yolo_worker_*_exposure_audit.csv "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_person_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/person_id_dataset_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/person_id_dataset_control.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/person_id_dataset_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/record_control.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/record_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/record_status_*.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_mask_evidence_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_mask_control.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_worker_*_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_obs_*_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_viz_*_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_overlay_*_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_overlay_masked_*_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_overlay_masked_*_history.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_overlay_sparse_*_history.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_trigger_map_*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_sync_health_*.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_sync_map_ring_*.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_native_hal_broker_stats*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/event_file_fifo_broker_stats*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/mvs_file_frame_broker_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/mvs_file_frame_broker_stats*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/test_run_status_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/test_run_events.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/experiment_control.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/experiment_status_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/experiment_events.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_bullet_fusion_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_bullet_fusion_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/global_bullet_tracks.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/damage_attribution_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/damage_attribution_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/damage_attribution_events.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/game_event_bridge_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/game_event_bridge_latest.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/game_event_bridge_sent.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/hit_judge_status.json "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/hit_candidate_*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/hit_candidate_all.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/hit_candidate_shadow_*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/pseudo_hit_*.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/pseudo_hit_all.jsonl "$OUT/status/" 2>/dev/null || true
  cp -a sync_ipc/trigger_status.json "$OUT/status/" 2>/dev/null || true
  archive_v26_b_log_deltas

  person_dataset_dir="$("$PY" -c 'import json; from pathlib import Path; p=Path("sync_ipc/person_id_dataset_status.json"); d="";
try:
    o=json.loads(p.read_text(encoding="utf-8"))
    d=str(o.get("session_dir") or "")
except Exception:
    d=""
print(d)' 2>/dev/null || true)"
  if [ -n "$person_dataset_dir" ] && [ -d "$person_dataset_dir" ]; then
    mkdir -p "$OUT/person_id_dataset"
    cp -a "$person_dataset_dir" "$OUT/person_id_dataset/" 2>/dev/null || true
  fi

  copy_replay_sync_anchors_to_dataset

  if [ -n "$latest_status" ] && [ -d "$latest_status" ]; then
    cp -a "$latest_status/status_latest.json" "$OUT/status_log_latest/" 2>/dev/null || true
    cp -a "$latest_status/status_summary.csv" "$OUT/status_log_latest/" 2>/dev/null || true
    cp -a "$latest_status/status_tabs.jsonl" "$OUT/status_log_latest/" 2>/dev/null || true
    cp -a "$latest_status/status_snapshot.jsonl" "$OUT/status_log_latest/" 2>/dev/null || true
  fi

  "$PY" "$CHECKER" summary --project . --run-dir "$OUT" > "$OUT/test_report.stdout.json" 2>> "$OUT/remote_ops_errors.log" || true
  tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "[runner] archive=$OUT.tar.gz"
}

request_hit_safe_off() {
  "$PY" "$CHECKER" set-control --project . --hit-enabled false --source remote_ops_stop_request_safe_off >> "$OUT/control_events.jsonl" 2>> "$OUT/remote_ops_errors.log" || true
}

send_launcher_int() {
  local reason="$1"
  echo "[runner] stop requested reason=$reason at $(date -Is)"
  request_hit_safe_off
  if [ -f "$LAUNCHER_WRAPPER_PID_FILE" ]; then
    local wrapper_pid
    wrapper_pid="$(tr -d '\r\n' < "$LAUNCHER_WRAPPER_PID_FILE" 2>/dev/null || true)"
    if [ -n "$wrapper_pid" ] && ps -p "$wrapper_pid" >/dev/null 2>&1; then
      echo "[runner] sending INT to timeout wrapper pid=$wrapper_pid"
      kill -INT "$wrapper_pid" 2>/dev/null || true
    fi
  fi
  ps -eo pid,args | awk '/launch_fusion_system.py/ && !/awk/ {print $1}' > "$LAUNCHER_PID_FILE" 2>/dev/null || true
  while read -r pid; do
    [ -n "$pid" ] || continue
    echo "[runner] sending INT to launcher pid=$pid"
    kill -INT "$pid" 2>/dev/null || true
  done < "$LAUNCHER_PID_FILE"
}

{
  echo "[runner] start $(date -Is)"
  echo "[runner] label=$LABEL run_id=$RUN_ID duration=${DURATION_S}s profile=$PROFILE"
  echo "[runner] project=$PROJECT"
  echo "[runner] deployed marker:"
  cat GIT_BRANCH_DEPLOYED_REMOTE_OPS.txt 2>/dev/null || cat GIT_BRANCH_DEPLOYED_20260702_V17A_ATEST.txt 2>/dev/null || true

  if [ "$PREFLIGHT_CLEAN" = "1" ]; then
    echo "[runner] preflight clean start label=$LABEL"
    if ! bash remote_ops/preflight_clean_runtime.sh "$LABEL" "$OUT/preflight_clean_summary.json" "$OUT/preflight_clean.log" "$PREFLIGHT_CLEAN_WAIT_S"; then
      preflight_rc=$?
      echo "[runner][WARN] preflight clean returned rc=$preflight_rc; continuing so the run report captures the failure"
    fi
  fi

  find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort > "$OUT/event_human_mask_stats_before.txt" 2>/dev/null || true
  init_controls
  sample_perf "before"
  snapshot "before"
  live_status "before"

  export DISPLAY=:1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GL_SYNC_TO_VBLANK=0
  export CUDA_VISIBLE_DEVICES=0
  export PY_BIN="$PY"
  export PROFILE_PATH="$PROFILE"
  export MRG_RUN_ID="$RUN_ID"

  launch_cmd=("$PY" launch_fusion_system.py --launch-profile "$PROFILE")
  if [ -n "$LAUNCH_EXTRA_ARGS" ]; then
    # Intentionally simple whitespace splitting.  V25 replay paths should avoid spaces;
    # richer dataset/profile generation will replace this in V25-A/D/E.
    # shellcheck disable=SC2206
    launch_extra_array=($LAUNCH_EXTRA_ARGS)
    launch_cmd+=("${launch_extra_array[@]}")
  fi
  printf '%q ' "${launch_cmd[@]}" > "$OUT/launch_command.sh"
  printf '\n' >> "$OUT/launch_command.sh"
  "$PY" - "$OUT/source_mode_request.json" "$PROFILE" "$LAUNCH_EXTRA_ARGS" "$V25_DATASET_DIR" "$MVS_SOURCE_MODE_REQUESTED" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "v25_source_mode_request",
    "ts_wall_us": int(time.time() * 1_000_000),
    "profile": sys.argv[2],
    "launch_extra_args": sys.argv[3],
    "dataset_dir": sys.argv[4],
    "mvs_source_mode_requested": sys.argv[5],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

  touch "$RUNNING_FLAG"
  rm -f "$STOP_REQUEST" "$GLOBAL_STOP_REQUEST" "$LAUNCHER_WRAPPER_PID_FILE" "$LAUNCHER_PID_FILE"
  if [ "$LIVE_STATUS_INTERVAL_S" -gt 0 ] 2>/dev/null; then
    (
      exec 9>&-
      while [ -f "$RUNNING_FLAG" ]; do
        live_status "live"
        sleep "$LIVE_STATUS_INTERVAL_S"
      done
    ) &
    LIVE_STATUS_PID=$!
  else
    LIVE_STATUS_PID=""
  fi
  if [ "$CONTROL_ENABLE" = "1" ]; then
    (
      exec 9>&-
      sleep "$CONTROL_DELAY_S"
      [ -f "$RUNNING_FLAG" ] || exit 0
      snapshot "startup_${CONTROL_DELAY_S}s"
      sample_perf "startup_${CONTROL_DELAY_S}s"
      set_control false 32
      sleep 5
      [ -f "$RUNNING_FLAG" ] || exit 0
      snapshot "toggle_hidden_dilate32"
      sample_perf "toggle_hidden_dilate32"
      set_control true 16
      sleep 5
      [ -f "$RUNNING_FLAG" ] || exit 0
      snapshot "toggle_visible_dilate16"
      sample_perf "toggle_visible_dilate16"
    ) &
    CONTROL_PID=$!
  else
    CONTROL_PID=""
  fi
  if [ "$RAW_RECORD_ENABLE" = "1" ]; then
    (
      exec 9>&-
      sleep "$RAW_RECORD_START_DELAY_S"
      [ -f "$RUNNING_FLAG" ] || exit 0
      set_raw_record_control start
      snapshot "raw_record_start"
      if [ "$DURATION_S" -gt 0 ] 2>/dev/null; then
        raw_stop_sleep=$((DURATION_S - RAW_RECORD_START_DELAY_S - RAW_RECORD_STOP_MARGIN_S))
        if [ "$raw_stop_sleep" -gt 0 ]; then
          sleep "$raw_stop_sleep"
        fi
      else
        while [ -f "$RUNNING_FLAG" ]; do
          sleep 2
        done
      fi
      [ -f "$RUNNING_FLAG" ] || exit 0
      set_raw_record_control stop
      sleep 2
      snapshot "raw_record_stop"
    ) &
    RAW_RECORD_PID=$!
  else
    RAW_RECORD_PID=""
  fi
  if [ "$PERSON_ID_DATASET_ENABLE" = "1" ]; then
    (
      exec 9>&-
      sleep "$PERSON_ID_DATASET_START_DELAY_S"
      [ -f "$RUNNING_FLAG" ] || exit 0
      set_person_dataset_control start
      snapshot "person_dataset_start"
      if [ "$DURATION_S" -gt 0 ] 2>/dev/null; then
        stop_sleep=$((DURATION_S - PERSON_ID_DATASET_START_DELAY_S - PERSON_ID_DATASET_STOP_MARGIN_S))
        if [ "$stop_sleep" -gt 0 ]; then
          sleep "$stop_sleep"
        fi
      else
        while [ -f "$RUNNING_FLAG" ]; do
          sleep 2
        done
      fi
      [ -f "$RUNNING_FLAG" ] || exit 0
      set_person_dataset_control stop
      sleep 2
      snapshot "person_dataset_stop"
    ) &
    PERSON_DATASET_PID=$!
  else
    PERSON_DATASET_PID=""
  fi

  if [ "$DURATION_S" -gt 0 ] 2>/dev/null; then
    timeout -s INT "${DURATION_S}s" bash -c 'tail -f /dev/null | "$@"' _ "${launch_cmd[@]}" &
  else
    echo "[runner] longrun mode enabled; waiting for explicit stop request"
    bash -c 'tail -f /dev/null | "$@"' _ "${launch_cmd[@]}" &
  fi
  LAUNCH_WRAPPER_PID=$!
  echo "$LAUNCH_WRAPPER_PID" > "$LAUNCHER_WRAPPER_PID_FILE"
  (
    exec 9>&-
    last_seen=""
    while [ -f "$RUNNING_FLAG" ] && ps -p "$LAUNCH_WRAPPER_PID" >/dev/null 2>&1; do
      if [ -f "$STOP_REQUEST" ] || [ -f "$GLOBAL_STOP_REQUEST" ]; then
        if [ -f "$STOP_REQUEST" ]; then
          reason="$(tr -d '\r\n' < "$STOP_REQUEST" 2>/dev/null | head -c 500 || true)"
        else
          reason="$(tr -d '\r\n' < "$GLOBAL_STOP_REQUEST" 2>/dev/null | head -c 500 || true)"
        fi
        if [ "$reason" != "$last_seen" ]; then
          send_launcher_int "$reason"
          last_seen="$reason"
        fi
        break
      fi
      sleep 1
    done
  ) &
  STOP_WATCH_PID=$!
  wait "$LAUNCH_WRAPPER_PID"
  rc=$?
  echo "[runner] launcher rc=$rc"
  rm -f "$RUNNING_FLAG"
  if [ "$RAW_RECORD_ENABLE" = "1" ]; then
    set_raw_record_control stop
    sleep 1
    snapshot "raw_record_stop_final"
  fi
  if [ "$PERSON_ID_DATASET_ENABLE" = "1" ]; then
    set_person_dataset_control stop
    sleep 2
    snapshot "person_dataset_stop_final"
  fi
  stop_background_tasks
  finalize "$rc"
  echo "[runner] end $(date -Is)"
} > "$OUT/runner.out" 2>&1

echo "$OUT"
