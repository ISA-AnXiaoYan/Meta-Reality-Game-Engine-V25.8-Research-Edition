#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-1}"
DURATION_S="${2:-120}"
CAMS="${3:-A,B,C,D,E,F,G,H}"
PROFILE="${4:-${PROFILE:-launch_profiles/627_event_overlay_bev.json}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ysxq/miniconda3/envs/ids_recv/bin/python3}"
REQUIRE_EVENT_DEVICES="${REQUIRE_EVENT_DEVICES:-0}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-round${ROUND}_${STAMP}}"
RUN_DIR="eval_event_mask_buffer/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[ROUND] id=${RUN_ID} duration=${DURATION_S}s cams=${CAMS} profile=${PROFILE}"
"${PYTHON_BIN}" tools/set_event_mask_buffer_round.py --round "${ROUND}" --profile "${PROFILE}" > "${RUN_DIR}/round_config.json"

export DISPLAY="${DISPLAY:-:1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GL_SYNC_TO_VBLANK="${__GL_SYNC_TO_VBLANK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[ROUND] stopping old processes"
pkill -INT  -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true
sleep 2
pkill -TERM -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true
sleep 1
pkill -KILL -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true

echo "[ROUND] cleaning sync_ipc"
mkdir -p sync_ipc
rm -f sync_ipc/event_ready_*.flag sync_ipc/event_trigger_*.csv sync_ipc/human_result_*.jsonl
rm -f sync_ipc/mvs_latest_*.json sync_ipc/mvs_frame_audit_*.csv sync_ipc/frame_bundle_audit_*.csv
rm -f sync_ipc/hit_judge_debug_*.jsonl sync_ipc/hit_candidate_*.jsonl sync_ipc/overlay_bullet_event_*.jsonl sync_ipc/bullet_effect_*.csv
rm -f sync_ipc/event_human_mask_stats_*.jsonl sync_ipc/event_worker_*_status.json sync_ipc/display_audit_*.csv
rm -rf sync_ipc/_runtime_config
rm -f sync_ipc/record_control.json sync_ipc/record_status.json
rm -f /dev/shm/mvs_latest_* /dev/shm/event_overlay_latest_* /dev/shm/human_latest_* /dev/shm/fusion_* 2>/dev/null || true

echo "[ROUND] event device probe"
DEVICE_PROBE="${RUN_DIR}/event_device_probe_before.txt"
if command -v metavision_platform_info >/dev/null 2>&1; then
  timeout 12s metavision_platform_info > "${DEVICE_PROBE}" 2>&1 || true
  if grep -q "No systems USB connected" "${DEVICE_PROBE}"; then
    echo "[ROUND][WARN] metavision_platform_info reports no USB systems; continuing because serial open is the runtime truth source" | tee "${RUN_DIR}/hardware_probe_warning.txt"
    if [ "${REQUIRE_EVENT_DEVICES}" = "1" ]; then
      echo "[ROUND][BLOCKED] no Metavision USB systems detected before launch" | tee "${RUN_DIR}/hardware_blocked.txt"
      printf '{"hardware_blocked":true,"reason":"no_metavision_usb_systems","probe":"%s"}\n' "${DEVICE_PROBE}" \
        > "${RUN_DIR}/evaluation_stdout.json"
      exit 3
    fi
  fi
else
  echo "metavision_platform_info not found" > "${DEVICE_PROBE}"
fi

echo "[ROUND] launching"
set +e
export PYTHON_BIN PROFILE CAMS
timeout -s INT "${DURATION_S}s" bash -lc \
  'tail -f /dev/null | "$PYTHON_BIN" launch_fusion_system.py --launch-profile "$PROFILE" --cams "$CAMS"' \
  > "${RUN_DIR}/launcher_stdout.log" 2>&1
RC=$?
set -e
echo "[ROUND] launcher_rc=${RC}" | tee "${RUN_DIR}/launcher_rc.txt"

echo "[ROUND] preflight"
"${PYTHON_BIN}" tools/preflight_event_mask_buffer_run.py \
  --project-root . \
  --profile "${PROFILE}" \
  --round-config "${RUN_DIR}/round_config.json" \
  --cams "${CAMS}" \
  --max-age-sec 600 \
  --probe-event-devices \
  --out-json "${RUN_DIR}/preflight_after.json" \
  --out-md "${RUN_DIR}/preflight_after.md" \
  > "${RUN_DIR}/preflight_stdout.json" 2> "${RUN_DIR}/preflight_stderr.log" || true

echo "[ROUND] archiving telemetry"
cp -f sync_ipc/event_human_mask_stats_*.jsonl "${RUN_DIR}/" 2>/dev/null || true
cp -f sync_ipc/event_worker_*_status.json "${RUN_DIR}/" 2>/dev/null || true
cp -f sync_ipc/global_yolo*_status.json sync_ipc/global_yolo*_latest.json sync_ipc/global_yolo*_perf.csv "${RUN_DIR}/" 2>/dev/null || true
cp -f sync_ipc/status_logs/*.zip "${RUN_DIR}/" 2>/dev/null || true
if [ -d sync_ipc/_runtime_config ]; then
  mkdir -p "${RUN_DIR}/_runtime_config"
  cp -f sync_ipc/_runtime_config/* "${RUN_DIR}/_runtime_config/" 2>/dev/null || true
fi
if [ -d sync_ipc/launcher_logs ]; then
  mkdir -p "${RUN_DIR}/launcher_logs"
  cp -f sync_ipc/launcher_logs/* "${RUN_DIR}/launcher_logs/" 2>/dev/null || true
fi

echo "[ROUND] evaluating"
"${PYTHON_BIN}" tools/evaluate_mask_buffer_run.py "${RUN_DIR}" --run-name "${RUN_ID}" --out-dir "${RUN_DIR}/reports" \
  > "${RUN_DIR}/evaluation_stdout.json" 2> "${RUN_DIR}/evaluation_stderr.log" || true

echo "[ROUND] done ${RUN_DIR}"
