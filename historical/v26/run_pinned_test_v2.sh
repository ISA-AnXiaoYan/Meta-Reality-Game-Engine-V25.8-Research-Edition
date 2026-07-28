#!/usr/bin/env bash
set -e

cd ~/PycharmProjects/Ids_Test_3.9/test607

mkdir -p logs
RUN_TAG=$(date +%Y%m%d_%H%M%S)
echo "[pin-test] RUN_TAG=$RUN_TAG"

# 先用 file 模式，不刷终端，但保留子进程日志，方便判断有没有启动卡住
export FUSION_CHILD_LOG_MODE=file

# 限制底层库乱开线程
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
RUNALL_PID=$!

echo "[pin-test] run_all pid=$RUNALL_PID"
echo "[pin-test] waiting until EVENT/MVS/HIT/GL are all started..."

deadline=$((SECONDS + 180))
while true; do
  EVENT_PID=$(pgrep -f "metavision_spatter_tracking_linefilter" | head -1 || true)
  MVS_PID=$(pgrep -f "hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py" | head -1 || true)
  HIT_PID=$(pgrep -f "hit_judge_server.py" | head -1 || true)
  GL_PID=$(pgrep -f "fusion_renderer_gl.py" | head -1 || true)
  LAUNCH_PID=$(pgrep -f "launch_fusion_system.py" | head -1 || true)

  echo "[pin-test] pids launch=${LAUNCH_PID:-none} event=${EVENT_PID:-none} mvs=${MVS_PID:-none} hit=${HIT_PID:-none} gl=${GL_PID:-none}"

  if [[ -n "$EVENT_PID" && -n "$MVS_PID" && -n "$HIT_PID" && -n "$GL_PID" ]]; then
    break
  fi

  if (( SECONDS > deadline )); then
    echo "[pin-test][ERROR] timed out waiting for all processes."
    echo "[pin-test] latest process list:"
    pgrep -af "fusion_renderer_gl.py|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|metavision_spatter_tracking_linefilter|hit_judge_server.py|launch_fusion_system.py" || true
    echo "[pin-test] latest logs:"
    find logs -maxdepth 2 -type f | sort | tail -20
    exit 1
  fi

  sleep 3
done

echo "[pin-test] all child processes found. applying affinity..."

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
for p in $(pgrep -f "launch_fusion_system.py" || true); do
  sudo taskset -cp 44-47 "$p" || true
done

echo "[pin-test] current processes:"
pgrep -af "fusion_renderer_gl.py|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|metavision_spatter_tracking_linefilter|hit_judge_server.py|launch_fusion_system.py" | tee "logs/pinned_processes_${RUN_TAG}.txt"

echo "[pin-test] current affinity:"
for p in $(pgrep -f "fusion_renderer_gl.py|hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py|metavision_spatter_tracking_linefilter|hit_judge_server.py|launch_fusion_system.py" || true); do
  taskset -cp "$p" || true
done | tee "logs/pinned_affinity_${RUN_TAG}.txt"

echo "[pin-test] GL window should be visible now."
echo "[pin-test] observe for 1-2 minutes, then press Ctrl+C here to stop."

wait "$RUNALL_PID"
