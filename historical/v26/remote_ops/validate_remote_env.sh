#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"

cd "$PROJECT"

echo "===== project ====="
pwd

echo "===== deployed marker ====="
cat GIT_BRANCH_DEPLOYED_REMOTE_OPS.txt 2>/dev/null || cat GIT_BRANCH_DEPLOYED_20260702_V17A_ATEST.txt 2>/dev/null || true

echo "===== py_compile ====="
"$PY" -m py_compile \
  global_bev_fusion_renderer.py \
  pyopengl_cuda_preview_renderer.py \
  global_person_id_server.py \
  person_id_dataset_recorder.py \
  remote_ops/check_fullsys_run.py \
  remote_ops/check_v25_dataset_manifest.py \
  remote_ops/start_system_from_panel.py \
  remote_ops/list_v25_datasets.py \
  remote_ops/mvs_file_frame_broker.py \
  event_overlay_overview_renderer.py \
  event_mask_evidence_service.py \
  multi_camera_launcher.py \
  launch_fusion_system.py \
  metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py \
  cpp_bullet_core/metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py

echo "===== normalize shell line endings ====="
find remote_ops native_event_bridge -maxdepth 1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} +

echo "===== native event bridge build ====="
bash native_event_bridge/build_native_event_bridge.sh native_event_bridge/build

echo "===== shell syntax ====="
bash -n remote_ops/run_fullsys_test.sh
bash -n remote_ops/build_native_brokers.sh
bash -n remote_ops/start_latest_system.sh
bash -n remote_ops/stop_fullsys_test.sh
bash -n remote_ops/stop_latest_system.sh
bash -n remote_ops/wait_latest_system_ready.sh
bash -n remote_ops/open_startup_control_panel.sh

echo "===== json/cv2 ====="
"$PY" - <<'PY'
import json
from pathlib import Path

for path in [
    "launch_profiles/627_event_overlay_bev.json",
    "ids_8cam_fusion_config.json",
    "ids_8cam_fusion_config_627_runtime.json",
]:
    json.loads(Path(path).read_text(encoding="utf-8"))
    print("json_ok", path)

try:
    import cv2
    print("cv2", cv2.__version__)
except Exception as exc:
    print("cv2_unavailable", repr(exc))
PY

echo "===== relevant processes ====="
pgrep -af 'launch_fusion_system|global_bev_fusion_renderer|event_overlay_overview_renderer|pyopengl_cuda_preview_renderer|metavision_spatter|global_yolo|hit_judge|mvs_cam_worker|hal_event_fifo_broker' || true

echo "===== event_human_mask_stats baseline ====="
find sync_ipc -maxdepth 1 -name 'event_human_mask_stats_*.jsonl' -printf '%f %s %T@\n' | sort || true
