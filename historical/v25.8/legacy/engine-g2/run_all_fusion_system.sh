#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "========== STOP OLD PROCESSES =========="
pkill -INT  -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true
sleep 2
pkill -TERM -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true
sleep 1
pkill -KILL -f "launch_fusion_system.py|multi_camera_launcher.py|metavision_spatter_tracking|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|hit_judge_server.py|fusion_renderer_gl.py" 2>/dev/null || true

echo
echo "========== CLEAN SYNC FILES =========="
mkdir -p sync_ipc
rm -f sync_ipc/event_ready_*.flag sync_ipc/event_trigger_*.csv sync_ipc/human_result_*.jsonl
rm -f sync_ipc/mvs_latest_*.json sync_ipc/mvs_frame_audit_*.csv sync_ipc/frame_bundle_audit_*.csv
rm -f sync_ipc/hit_judge_debug_*.jsonl sync_ipc/hit_candidate_*.jsonl sync_ipc/overlay_bullet_event_*.jsonl sync_ipc/bullet_effect_*.csv
rm -f sync_ipc/event_human_mask_stats_*.jsonl sync_ipc/display_audit_*.csv
rm -rf sync_ipc/_runtime_config
rm -f sync_ipc/record_control.json sync_ipc/record_status.json
rm -f /dev/shm/mvs_latest_* /dev/shm/event_overlay_latest_* /dev/shm/human_latest_* /dev/shm/fusion_* 2>/dev/null || true

echo
echo "========== START 5-CAMERA FUSION SYSTEM =========="
python3 launch_fusion_system.py \
  --config ids_5cam_fusion_config.json \
  --cams A,B,C,D,E \
  --clean-shm \
  --clean-jsonl \
  --ready-timeout 120 \
  --preview-fps 60 \
  "$@"
