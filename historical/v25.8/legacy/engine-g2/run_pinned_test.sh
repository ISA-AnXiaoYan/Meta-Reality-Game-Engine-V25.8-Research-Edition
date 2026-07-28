#!/usr/bin/env bash
set -e

cd ~/PycharmProjects/Ids_Test_3.9/test607

mkdir -p logs
RUN_TAG=$(date +%Y%m%d_%H%M%S)
echo "[pin-test] RUN_TAG=$RUN_TAG"

export FUSION_CHILD_LOG_MODE=none

# 限制底层库乱开线程，避免 OpenCV / MKL / OpenBLAS / PyTorch 抢占 GL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=4
export OPENCV_FOR_THREADS_NUM=1

rm -f sync_ipc/display_audit_*.csv
rm -f sync_ipc/composite_fps_*.csv
rm -f sync_ipc/gl_paint_fps.csv
rm -f sync_ipc/mvs_display_trace_*.csv
rm -f sync_ipc/gl_stage_perf.csv

./run_all_fusion_system.sh &
LAUNCH_PID=$!

echo "[pin-test] launcher pid=$LAUNCH_PID"
echo "[pin-test] waiting 10s for child processes..."
sleep 10

echo "[pin-test] pin GL to CPU 4-7"
for p in $(pgrep -f "fusion_renderer_gl.py" || true); do
  sudo taskset -cp 4-7 "$p" || true
  sudo renice -n -10 -p "$p" || true
done

echo "[pin-test] pin MVS/YOLO to CPU 8-23"
for p in $(pgrep -f "hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py" || true); do
  sudo taskset -cp 8-23 "$p" || true
  sudo renice -n -5 -p "$p" || true
done

echo "[pin-test] pin EVENT to CPU 24-39"
for p in $(pgrep -f "metavision_spatter_tracking_linefilter" || true); do
  sudo taskset -cp 24-39 "$p" || true
  sudo renice -n -3 -p "$p" || true
done

echo "[pin-test] pin HIT judge to CPU 40-43"
for p in $(pgrep -f "hit_judge_server.py" || true); do
  sudo taskset -cp 40-43 "$p" || true
  sudo renice -n -3 -p "$p" || true
done

echo "[pin-test] pin launcher to CPU 44-47"
sudo taskset -cp 44-47 "$LAUNCH_PID" || true

echo "[pin-test] current processes:"
pgrep -af "fusion_renderer_gl.py|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|metavision_spatter_tracking_linefilter|hit_judge_server.py|launch_fusion_system.py" | tee "logs/pinned_processes_${RUN_TAG}.txt"

echo "[pin-test] current affinity:"
for p in $(pgrep -f "fusion_renderer_gl.py|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|metavision_spatter_tracking_linefilter|hit_judge_server.py|launch_fusion_system.py" || true); do
  taskset -cp "$p" || true
done | tee "logs/pinned_affinity_${RUN_TAG}.txt"

echo "[pin-test] now observe the GL window for 1-2 minutes."
echo "[pin-test] press Ctrl+C here to stop."

wait "$LAUNCH_PID"
