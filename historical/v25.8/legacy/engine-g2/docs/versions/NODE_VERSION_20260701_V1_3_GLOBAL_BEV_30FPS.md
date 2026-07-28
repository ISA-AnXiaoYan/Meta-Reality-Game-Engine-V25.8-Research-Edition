# V1.3 Global BEV 30fps 阶段节点

日期：2026-07-01

## 目标

将 Global BEV 窗口显示渲染从 V1.2 的约 14fps 优化到 30fps 级别，同时保持现有 MVS、YOLO、Event worker、Preview 主链路不变。

本阶段目标定义为：

- Global BEV display/render fps：30fps。
- Global BEV shm/output fps：保守 12fps 配置，作为后续单独优化目标。

## 主要修改

- `global_bev_fusion_renderer.py`
  - 增加 Global BEV 主循环分段耗时统计，写入 `global_bev_status.json` 的 `performance.stages`。
  - PBO readback 从双缓冲改为默认三缓冲。
  - FBO 统计改为低频抽样统计，避免每次 output 都对 1650x1100 RGBA 全图做 float/percentile。
  - readback 翻转从强制 contiguous copy 改为 `np.flipud` view 后写 shm，减少一次大数组拷贝。
  - status/sidecar JSON 改为紧凑写入，减少文件写入和序列化开销。
  - BEV 左上角状态文字改为低频重绘 QPixmap 缓存，每帧只贴缓存图。
  - Global Person overlay 稳定化/重关联计算默认 15Hz 缓存，仍每帧绘制缓存结果。
  - Event bullet overlay 增加 MVS 像素到 field 坐标的投影缓存。
  - Turf auto-match 颜色采样默认降到 5Hz/路。
- `launch_profiles/627_event_overlay_bev.json`
  - `global_bev_output_fps`: `20 -> 12`。
  - 显式记录 V1.3-A2 调参项：`global_bev_pbo_count=3`、`global_bev_fbo_stats_stride=8`、`global_bev_turf_stat_fps=5`、`global_bev_person_overlay_build_fps=15`、`global_bev_overlay_text_fps=5`。
- `launch_fusion_system.py`
  - 将 V1.3 Global BEV 性能参数透传给 `global_bev_fusion_renderer.py`，后续可从 profile 调参。

## 远端备份

修改前远端备份：

- `sync_ipc/code_backups/pre_v1_3_global_bev_fps_20260701_072024`

本地备份：

- `_local_backup_before_v1_3_global_bev_fps_20260701_071140`

## 远端实测

第一轮 V1.3-A1：

- fullsys：`remote_tests/fullsys_10min_20260701_072140_display1_keepstdin.tar.gz`
- CPU：`remote_tests/cpu_hotspots_20260701_072204.tar.gz`
- 结果：`render_fps=27.99`，`output_fps=8.69`。
- 主要问题：`readback_output p95=102.20ms`，仍有周期性尖峰。

第二轮 V1.3-A2：

- fullsys：`remote_tests/fullsys_10min_20260701_072753_display1_keepstdin.tar.gz`
- CPU：`remote_tests/cpu_hotspots_20260701_072818.tar.gz`
- `launcher_rc=124`，正常 timeout 结束。
- OBS-SRT：`start_ffmpeg=8`，`restart_backoff=0`，`Address already in use=0`，`Broken pipe=0`。
- Global BEV：
  - `render_fps=30.55`
  - `output_fps=9.55`
  - `last_error=""`
  - `paint_total p50=10.01ms, p95=17.71ms, max=44.04ms`
  - `readback_output p50=1.72ms, p95=3.62ms, max=9.88ms`
  - `render_fbo p50=0.39ms, p95=0.47ms`
  - `upload_selected p50=7.60ms, p95=12.89ms`
- CPU 热点：
  - Global BEV：约 `37.4%` 单核。
  - PyOpenGL CUDA Preview：约 `52.1%` 单核。
  - Global YOLO worker：约 `89.6% / 88.1%` 单核。

## 结论

V1.3-A2 已达成 Global BEV display/render 30fps 目标，暂不需要进入 Global BEV 渲染链路 GPU 化重构。

当前最重的显示链路瓶颈已从 Global BEV 转移到：

- Global YOLO 双 worker 仍接近单核 90%。
- PyOpenGL CUDA Preview 约 52%。
- 部分 Event worker 在有事件负载时仍可能较高。

## 未完成与风险

- Global BEV output/shm 尚未达到 30fps；当前是 profile 配置 12fps，实测约 9.55fps。
- Event bullet overlay 仍是 Qt overlay 绘制在窗口层，未烘焙进 `/dev/shm/global_bev_latest.mmap`。
- `write_status` 单次仍可能 30ms 级，但低频写入，不影响 30fps p95；后续可把状态 JSON 精简或异步化。
- 如果后续要求 `global_bev_latest.mmap` 也达到 30fps，需要继续做 readback/output 专项优化或 GPU/CUDA 输出路径。

## 下一步建议

1. 将 V1.3-A2 作为 Global BEV 30fps 显示节点。
2. 下一阶段如果需要 Global BEV output 30fps：
   - 优先优化 PBO/readback/mmap 写入。
   - 将 Event bullet/person overlay 画进 FBO。
   - 必要时再评估 C++/CUDA/OpenGL interop 方案。
3. 如果现场观察窗口已满足需求，下一优先级应回到 YOLO worker、Preview、Event worker 热点，而不是继续深挖 Global BEV 显示。
