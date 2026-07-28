# PyOpenGL CUDA Preview 40 FPS 视频 / 250 FPS 子弹轨迹测试方案

## 目标

本次修改新增独立进程 `pyopengl_cuda_preview_renderer.py`，用于替代旧 `fusion_renderer_gl.py` preview：

- 视频流只按 40 FPS 更新，不跟随 250 FPS 重复上传。
- 子弹轨迹按 250 FPS 重绘。
- MVS worker 输出 raw Bayer8 preview shm。
- preview renderer 从 raw Bayer8 shm 读取视频，在 CUDA 中完成 Bayer8 -> RGBA + resize，并通过 CUDA-OpenGL PBO interop 更新 OpenGL texture。
- hit judge 将 `overlay_bullet_point` 同步写入 `preview_trajectory` 二进制 shm，renderer 用 VBO 绘制轨迹。

## 修改文件

新增：

```text
preview_gpu_shm.py
pyopengl_cuda_preview_renderer.py
PYOPENGL_CUDA_PREVIEW_TEST_PLAN.md
```

修改：

```text
launch_fusion_system.py
mvs_cam_worker_external_trigger.py
hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py
hit_judge_server.py
```

## 环境依赖

必须具备：

```bash
python3 - <<'PY'
import OpenGL, PySide6
print('PyOpenGL / PySide6 OK')
PY
```

CUDA-GL interop 推荐依赖 PyCUDA：

```bash
python3 - <<'PY'
import pycuda.driver as cuda
import pycuda.gl as cudagl
print('PyCUDA GL OK')
PY
```

如果 PyCUDA 不可用，`pyopengl-cuda` backend 默认会退出。只有显式加 `--preview-allow-cpu-upload-fallback` 才允许 CPU fallback；但 fallback 不满足“全链路 GPU”目标，只用于定位问题。

## NVIDIA OpenGL 检查

renderer 启动时会打印：

```text
GL_VENDOR
GL_RENDERER
GL_VERSION
```

目标：`GL_VENDOR` 或 `GL_RENDERER` 中必须包含 `NVIDIA`。如果显示器走核显，CUDA-OpenGL interop 可能失败。

可尝试：

```bash
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__NV_PRIME_RENDER_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 launch_fusion_system.py ...
```

## 推荐启动命令

```bash
python3 launch_fusion_system.py \
  --config ids_8cam_fusion_config.json \
  --cams A,B,C,D,E,F,G,H \
  --mvs-split-workers \
  --mvs-trigger-fps 40.0 \
  --preview-backend pyopengl-cuda \
  --preview-video-fps 40 \
  --preview-render-fps 250 \
  --preview-trajectory-fps 250 \
  --preview-vsync 0 \
  --preview-cuda-device 0 \
  --preview-layout grid2x4 \
  --preview-tile-width 640 \
  --preview-tile-height 360 \
  --clean-shm \
  --clean-jsonl \
  --overlay-ws-enable \
  --overlay-ws-host 0.0.0.0 \
  --overlay-ws-port 8765
```

## 进程检查

```bash
pgrep -af "mvs_cam_worker|mvs_trigger_coordinator|pyopengl_cuda_preview|hit_judge|hik_rgb_yolo"
```

目标进程：

```text
mvs_cam_worker_external_trigger.py --cam A ... --raw-preview-shm-enable
...
mvs_cam_worker_external_trigger.py --cam H ... --raw-preview-shm-enable
mvs_trigger_coordinator.py
pyopengl_cuda_preview_renderer.py --video-fps 40 --render-fps 250
hit_judge_server.py
hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py
```

## 共享内存检查

```bash
ls -lh /dev/shm/mvs_raw_latest_*.mmap
ls -lh /dev/shm/preview_trajectory.mmap
```

目标：

- 每个 `mvs_raw_latest_{A..H}.mmap` 存在，大小约为 header + 3 * width * height。
- `preview_trajectory.mmap` 存在。

## renderer 日志目标

查看日志：

```bash
tail -f sync_ipc/launcher_logs/gl_cuda.log
```

目标日志：

```text
[pygl_cuda_preview] GL_VENDOR=...NVIDIA...
[pygl_cuda_preview] CUDA-GL backend ready: device=...
[pygl_cuda_preview][A] raw shm opened...
[pygl_cuda_preview] trajectory shm opened...
[pygl_cuda_preview] started video_fps=40.0 render_fps=250.0 trajectory_fps=250.0
```

不能出现：

```text
using CPU upload fallback
CUDA-GL init failed
OpenGL context not on NVIDIA GPU
```

## MVS worker 日志目标

```bash
tail -f sync_ipc/launcher_logs/mvs_a.log
```

目标日志：

```text
[mvs_worker][A] raw preview shm enabled: mvs_raw_latest_A shape=... pattern=BG
[mvs_worker][A] frames=... fps≈40 trigger_fps≈40
```

## hit judge 轨迹 shm 目标

```bash
tail -f sync_ipc/launcher_logs/hit.log
```

目标日志：

```text
[hit_judge] preview trajectory shm enabled: preview_trajectory max_segments=65536
```

当有子弹点时，`preview_trajectory.mmap` 的 seq 应持续变化。可用简短脚本检查：

```bash
python3 - <<'PY'
import time
from preview_gpu_shm import PreviewTrajectoryReader
r = PreviewTrajectoryReader('preview_trajectory')
last = -1
for _ in range(50):
    seq = int(r.header[6])
    cnt = int(r.header[5])
    if seq != last:
        print('seq', seq, 'count', cnt)
        last = seq
    time.sleep(0.1)
PY
```

## 性能验证

### 1. 确认旧 GL renderer 未启动

```bash
pgrep -af fusion_renderer_gl.py
```

目标：无输出。

### 2. 确认新 renderer CPU 不再出现旧热点

```bash
GLPID=$(pgrep -f pyopengl_cuda_preview_renderer.py | head -1)
sudo py-spy top --pid $GLPID
```

目标：不再出现旧 renderer 的：

```text
poll_frames
_update_texture
wrapperCall
_draw_background
```

新 renderer 的 Python CPU 不应长期单核 100%。

### 3. 确认视频不是 250 FPS 上传

renderer 日志每 5 秒打印：

```text
render_fps=...
last_frame_ids={...}
```

`last_frame_ids` 每秒增量应约为 40，而不是 250。

### 4. 确认轨迹 250 FPS 重绘

因为显示器刷新率可能不是 250Hz，物理显示不一定真的 present 250 FPS。测试目标是：

- renderer render loop 按 250 FPS 调用。
- 轨迹 VBO 可在 250 Hz 被刷新。
- 视频 texture 仍只在 40 FPS 更新。

建议使用：

```bash
pidstat -p $GLPID 1
nvidia-smi dmon -s u
```

## 通过标准

满足以下条件可认为目标达成：

1. `pyopengl_cuda_preview_renderer.py` 独立运行，`fusion_renderer_gl.py` 不运行。
2. 8 路 `mvs_raw_latest_{camera}.mmap` 正常生成。
3. `preview_trajectory.mmap` 正常生成，并在子弹点到来时 seq 增长。
4. renderer 日志显示 `CUDA-GL backend ready`，没有 CPU fallback。
5. 视频 frame_id 增长约 40 FPS。
6. render loop 目标为 250 FPS，子弹轨迹在旧视频帧上高频重绘。
7. 旧 GL profiling 热点 `poll_frames/_update_texture/wrapperCall` 消失。
8. MVS worker 不因 preview renderer 阻塞，worker 仍稳定约 40 FPS。

## 常见问题

### CUDA-GL init failed

多半是 OpenGL context 没跑在 NVIDIA GPU。检查 `GL_VENDOR/GL_RENDERER`。

### raw shm not ready

MVS worker 尚未启动或未开启 `--raw-preview-shm-enable`。检查 launcher 是否使用：

```bash
--preview-backend pyopengl-cuda
--mvs-split-workers
```

### preview_trajectory.mmap 不存在

hit judge 未启动，或者没有启用 trajectory shm。默认会启用；也可显式设置：

```bash
PREVIEW_TRAJECTORY_SHM_ENABLE=1
PREVIEW_TRAJECTORY_SHM=preview_trajectory
```

### 颜色异常

调整 Bayer pattern：

```bash
BAYER_PATTERN=BG / RG / GB / GR
```

或在 config 的 sync 中设置 `bayer_pattern`。

## Overlay / 文字层新增验收项

本版 `pyopengl-cuda` renderer 已接入以下 overlay 源：

- `preview_trajectory.mmap`：高频子弹轨迹线段。
- `overlay_ws`：`hit_judge_server.py` 广播的 `overlay_bullet_point`、`overlay_bullet_marker`、`overlay_hit_candidate`、`overlay_frame_state`。
- `hit_candidate_{camera}.jsonl` / `hit_candidate_all.jsonl`：作为 websocket 丢包或未连接时的 hit marker fallback。
- 状态文字层：显示 renderer FPS、trajectory vertex count、WS 连接状态、各路 camera frame id、近期 hit/marker 文本。

### 1. 检查 overlay_ws 是否连接

```bash
tail -f sync_ipc/launcher_logs/gl_cuda.log
```

目标看到：

```text
[pygl_cuda_preview] overlay_ws connected: ws://127.0.0.1:8765
```

如果没有这行，仍可通过 hit jsonl fallback 显示 hit marker，但实时 marker/文字会不完整。

### 2. 检查 trajectory shm 可见段

```bash
python3 - <<'PY'
import time
from preview_gpu_shm import PreviewTrajectoryReader
r = PreviewTrajectoryReader('preview_trajectory')
for _ in range(30):
    segs = r.read_segments(max_age_ms=1000, require_new=False)
    print('visible_segments', 0 if segs is None else len(segs))
    time.sleep(0.1)
PY
```

子弹事件发生后，`visible_segments` 应短时间大于 0，并在 preview 中显示轨迹。

### 3. 检查 hit_candidate fallback

```bash
ls -lh sync_ipc/hit_candidate_*.jsonl sync_ipc/hit_candidate_all.jsonl 2>/dev/null
```

有 hit candidate 写入时，preview 中应显示 `HIT` marker 和文字提示。

### 4. 视觉验收

窗口中应同时显示：

- 8 路可见光视频。
- 左上角状态文本：`CUDA preview video=40Hz ... ws=ON/OFF ...`。
- 每个 tile 左上角 camera id 和 frame id。
- 子弹轨迹线。
- turn/end/hit 等 marker 圆圈和文字标签。

