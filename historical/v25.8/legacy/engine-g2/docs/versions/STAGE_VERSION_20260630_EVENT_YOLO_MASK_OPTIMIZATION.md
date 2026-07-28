# 阶段版本说明：20260630 Event/YOLO Human Mask 优化

## 版本定位

阶段版本：`20260630-human-mask-preview-polygon-gfix`

本阶段目标是先把人体 mask 的可视化与诊断链路补齐，并验证 YOLO polygon 输出对性能和延迟的影响；同时修复事件相机 G 链路中由事件渲染时间戳回退导致 worker 退出的问题。

本阶段不包含事件侧 mask 过滤策略修复。当前事件 mask filter 仍保持 `current-only + wait_ms=0` 语义，用于保留原始问题现场和后续对照。

追加范围：本阶段也纳入 Global BEV Fusion 窗口的人体唯一 ID 显示稳定性修复。PyOpenGL CUDA Preview 主预览窗口的人体 ID 显示相对稳定，但 Global BEV Fusion 依赖 Global Person ID 聚合与 BEV overlay 展示，存在 ID 跳变和短时丢失重建问题。

## 主要优化内容

### 1. PyOpenGL CUDA Preview 主预览补人体 mask 图层

改动文件：

- `pyopengl_cuda_preview_renderer.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

新增能力：

- Preview 支持 `human_mask_render` 状态输出。
- `--human-mask-render-mode auto|polygon|bbox|off`。
- `auto` 模式优先绘制 YOLO `mask_polygons`。
- 当 YOLO 未输出 polygon 时，自动使用 bbox 半透明填充作为显示兜底。
- 状态窗口和 `status_summary.csv` 增加：
  - `human_mask_drawn`
  - `human_mask_source`
  - `human_mask_count`
  - `human_mask_polygon_fill_count`
  - `human_mask_bbox_fill_count`
  - `human_mask_latest_age_ms`

验证结果：

- bbox fallback 测试中，Preview 可以显示人体 mask 面。
- polygon 开启后，Preview 运行中显示 `source=polygon`，`bbox_fill_count=0`。

### 2. Global YOLO 真实输出 mask polygons

改动文件：

- `global_yolo_infer_server.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

新增能力：

- `global_yolo_include_mask_polygons=true`。
- 从 Ultralytics `result.masks.xy/xyn` 提取 polygon。
- 新增 polygon 点数上限：
  - `global_yolo_mask_polygon_max_points=96`
  - worker 参数：`--mask-polygon-max-points`
- polygon 会写入 `human_result_{camera}.jsonl` 的 `persons[].mask_polygons`。

实测结果：

- 5 分钟 polygon run 中，所有有人的 YOLO 记录均写入 polygon：
  - `polygon_records/person_records = 15590/15590`
- Preview 状态序列：
  - `status_rows=197`
  - `polygon drawn rows=135`
  - `drawn_ratio=0.685`
  - `max_polygon_fill=7`
  - `max_bbox_fill=0`
  - `human_mask_latest_age_ms p50≈21.3ms`

### 3. 修复 Event G 链路 timestamp rollback 崩溃

改动文件：

- `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py`

问题表现：

- 之前 G worker 出现：
  - `ValueError: Call the method reset() before generating frames in the past`
  - `Last frame was generated at 7560000 and current frame generation is at 7480000`
- 直接导致 G 事件 worker 启动失败或中途退出。

修复方式：

- `_maybe_render()` 中记录 `last_generate_ts_us`。
- 当渲染时间戳回退时：
  - reset frame generator；
  - 将生成时间 clamp 到最近一次生成时间；
  - 若 Metavision 内部 last timestamp 更大，则解析异常文本后用 internal last 再 retry；
  - retry 仍失败时跳过该帧事件背景渲染，不再杀死 worker。

实测结果：

- polygon + G fix 5 分钟 run：
  - `Traceback=0`
  - `ValueError=0`
  - `[FAIL]=0`
  - G/H 均进入稳定运行，G 不再断链。

### 4. 建立人体 mask 延迟分析脚本

新增文件：

- `tools/analyze_human_mask_latency.py`

能力：

- 自动读取 run 目录中的：
  - `human_result_*.jsonl`
  - `event_human_mask_stats_*.jsonl`
  - `event_worker_*_status.json`
  - `global_yolo_worker_*_perf.csv`
  - `trigger_status.json`
  - `global_trigger_timeline.jsonl`
- 自动根据 `start_epoch.txt/end_epoch.txt` 做时间窗过滤，避免追加型历史日志污染本次 run。
- 输出 JSON/Markdown 汇总。
- 统计：
  - human result 延迟
  - polygon 记录数/person 数
  - event mask lag frames/ms
  - current mask missing ratio
  - Global YOLO FPS、batch、partial batch、postprocess/total latency

### 5. Global BEV Fusion 人体 ID 显示稳定性修复

改动文件：

- `global_person_id_server.py`
- `global_bev_fusion_renderer.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

问题表现：

- PyOpenGL CUDA Preview 主预览窗口中，每路人体 ID 显示相对稳定。
- Global BEV Fusion 窗口中，人体 ID 显示和轨迹跟随存在跳变、短时丢失后重建、显示 ID 不连续的问题。

原因分析：

- Preview 主窗口主要显示单相机当前人体结果，显示 ID 压力较低。
- Global BEV Fusion 依赖 `global_person_latest.json` 的全局 track。
- Global Person ID 原逻辑会给同相机 `stable_id/local_id` 加分，但当前 Global YOLO 输出的 `stable_id` 本质是单帧检测序号，不是跨帧 ReID。这会诱发错误匹配和 ID 跳变。
- BEV overlay 原先直接绘制 Global Person ID 输出，缺少显示层 hold、平滑和近距离续接机制。

修复内容：

- launcher/profile 暴露并配置 Global Person ID 参数：
  - `global_person_id_local_id_bonus=0.0`
  - `global_person_id_local_id_bonus_max=0.0`
  - `global_person_id_position_alpha=0.45`
  - `global_person_id_velocity_alpha=0.20`
- 修复 duplicate suppress 排序中错误使用不存在状态 `confirmed` 的问题，改为 `tracking`。
- Global BEV 增加显示层 ID 稳定器：
  - 独立 display slot ID，例如 `P01/P02`；
  - 短时丢失 hold；
  - 新 gid 近距离重现时续接旧 display ID；
  - 坐标显示平滑；
  - 诊断字段写入 `global_person_overlay`。
- 新增 BEV overlay 参数：
  - `global_bev_global_person_overlay_display_stabilize_enable`
  - `global_bev_global_person_overlay_display_alpha`
  - `global_bev_global_person_overlay_reid_gate_m`
  - `global_bev_global_person_overlay_reid_ttl_ms`
  - `global_bev_global_person_overlay_hold_ms`

当前默认参数：

- `display_alpha=0.45`
- `reid_gate_m=1.2`
- `reid_ttl_ms=2200`
- `hold_ms=900`

预期效果：

- BEV 窗口人体显示 ID 不再跟随短时 Global Person gid 抖动立即跳号。
- 轨迹点短时丢失时保留 display slot，减少视觉闪烁。
- 不改变 Global Person ID 原始输出，只在 BEV 显示层增加稳定展示 ID。

## 远端实测结果

### Run A：显示兜底验证

归档：

- `sync_ipc/full_system_tests/fullsys_5min_mask_latency_20260630_213602_display1_keepstdin.tar.gz`

关键结果：

- Preview bbox fallback 生效：
  - `human_mask_source=bbox_fallback`
  - `polygon_fill_count=0`
  - `bbox_fill_count>0`
- Global YOLO：
  - 合计约 `316.239 FPS`
- Event mask 延迟：
  - `lag_p50_frames=5`
  - `lag_p50_ms=125`
  - `current_mask_missing_ratio_avg=0.891`
  - `applied_ratio_avg=0.0`

### Run B：polygon + G fix 验证

归档：

- `sync_ipc/full_system_tests/fullsys_5min_mask_latency_20260630_215813_display1_keepstdin.tar.gz`

关键结果：

- YOLO polygon 生效：
  - `polygon_records/person_records=15590/15590`
- Preview polygon 生效：
  - `source=polygon`
  - `max_polygon_fill=7`
  - `max_bbox_fill=0`
- G 链路稳定：
  - `Traceback=0`
  - `ValueError=0`
  - `[FAIL]=0`
- OBS-SRT：
  - `start ffmpeg=8`
  - `restart_backoff=0`
  - `Address already in use=0`
  - `Broken pipe=0`
- Global YOLO：
  - 合计约 `316.218 FPS`
  - worker 0:
    - `wall_fps=158.146`
    - `postprocess_ms_p50=0.485`
    - `total_ms_p50=17.383`
  - worker 1:
    - `wall_fps=158.072`
    - `postprocess_ms_p50=0.139`
    - `total_ms_p50=14.683`
- Event mask 延迟：
  - `lag_p50_frames=5`
  - `lag_p50_ms=125`
  - `current_mask_missing_ratio_avg=0.910`
  - `applied_ratio_avg=0.0`
  - `capture_to_record_ms_p50_avg=24.276`

## 阶段结论

1. 主预览人体 mask 显示链路已补齐。
2. YOLO polygon 已真实启用，不再只是 bbox fallback。
3. 开启 polygon 对 Global YOLO 总吞吐影响很小：
   - bbox fallback run：`316.239 FPS`
   - polygon run：`316.218 FPS`
4. polygon 会带来少量 postprocess/JSONL 开销：
   - worker 0 postprocess p50 从约 `0.161ms` 增至约 `0.485ms`
   - 但整体仍满足 8 路 40Hz 附近吞吐。
5. G 链路 timestamp rollback 崩溃已在 5 分钟实测中修复验证。
6. 事件 mask 过滤仍未真正生效，主因不是 polygon 缺失，而是事件侧 `current-only + wait_ms=0` 与 YOLO 结果天然落后约 5 帧不匹配。

## 潜在风险点

### 1. polygon 输出带来的长期性能风险

当前 5 分钟测试性能正常，但 polygon 会增加：

- JSONL 写入体积；
- Preview 解析和绘制开销；
- Global YOLO postprocess 时间；
- 状态/日志归档体积。

当前点数上限为 96，若人多、遮挡复杂或开启 `retina_masks=true`，开销可能进一步增加。

### 2. G 链路 timestamp rollback 修复是保护性修复

当前修复保证 worker 不因渲染时间戳回退退出，但它不是底层时间戳回退的根因修复。

可能影响：

- 回退帧会被 clamp 或跳过事件背景渲染；
- 极端情况下 event overlay 背景可能短时重复/空帧；
- 子弹检测主链路不应被这个异常杀死，但显示帧时间仍需后续监控。

建议后续把以下计数纳入状态窗口：

- `frame_gen_backward_clamp_count`
- `frame_gen_backward_retry_count`
- `frame_gen_backward_skip_count`

### 3. Event mask filter 仍然是当前最大未解决问题

当前过滤策略仍是：

- `event-human-mask-current-only`
- `event-human-mask-wait-ms=0.0`

实测中 event worker 当前 sync 通常领先最新 YOLO human_result 约 5 帧，即约 125ms。因此：

- `current_mask_missing_ratio_avg≈0.91`
- `applied_ratio_avg=0.0`

这意味着 polygon 已经可用，但事件侧严格当前帧匹配导致 mask 仍无法应用。

### 4. BEV display slot ID 与 Global Person gid 可能不完全一致

为保证 BEV 窗口显示稳定，BEV overlay 新增了独立 display slot ID。它是“显示稳定 ID”，不一定永远等于 Global Person ID server 输出的 `global_id`。

风险：

- 调试时需要区分 `display_player_id` 和 `source_global_id`。
- 如果 Global Person ID 源端持续错误跳号，BEV display slot 可以缓和画面观感，但不能替代源端追踪修复。
- reid gate 过大时，两个近距离人员可能被显示层错误续接；过小时，短时丢失后仍可能新建 display slot。

建议实测时重点观察：

- `global_person_overlay.display_slot_count`
- `global_person_overlay.display_draw_count`
- `global_person_overlay.display_relinked_this_poll`
- `global_person_overlay.display_created_this_poll`

### 5. shutdown 尾态状态不能代表运行中状态

`system_status_latest.json` 在 runner 结束后可能进入 `stopping` 状态，Preview `drawn=false` 或 worker broken 可能只是 shutdown 尾态。

判断运行期状态应优先使用：

- `status_summary.csv`
- `status_snapshot.jsonl`
- 运行中采样的 `system_status_latest.json`

### 6. launcher profile 与运行时配置仍需防覆盖

项目中 runtime config 会由 launcher 生成。后续判断某项配置是否生效时，需要同时检查：

- launch profile；
- launcher 实际命令；
- `_runtime_config`；
- worker status/latest；
- 归档中的 `runner.out` 和 launcher logs。

## 回滚点

远端同步前已备份：

- `sync_ipc/remote_backups/polygon_gfix_pre_20260630_215657`

可回滚文件：

- `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py`
- `global_yolo_infer_server.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`
- `tools/analyze_human_mask_latency.py`
- `global_person_id_server.py`
- `global_bev_fusion_renderer.py`

快速关闭 polygon 的配置回滚：

```json
"global_yolo_include_mask_polygons": false
```

如需保留显示层但降低开销，可继续使用 Preview 的 `bbox_fallback`。

## 下一步优化方向

### 方向 A：修复事件 mask filter 的时序匹配策略

目标：

- 让事件侧可以使用“最近可用的人体 mask”，而不是只接受当前 sync。

候选方案：

- nearest sync 匹配：允许使用当前帧前后一定窗口内的人体结果；
- history/carry window：允许使用最近 N 帧历史 mask；
- 按 wall time 匹配：用 MVS soft trigger 时间线做事件切片与 YOLO record 的时间对齐；
- 保留 `current-only` 作为严格模式开关，新增 tolerant 模式用于实战。

建议初始参数：

- `max_lag_frames=6~8`
- `max_lag_ms=150~200`
- `max_future_frames=0~1`

预期效果：

- `applied_ratio` 从 0 提升到可观水平；
- `current_mask_missing_ratio` 明显下降；
- 事件误删风险可通过 lag/window 指标监控。

### 方向 B：完善 Event worker 状态与可观测性

新增状态字段：

- 当前 sync；
- 最新 human_result sync；
- 使用的 mask sync；
- lag frames/ms；
- polygon/bbox mask 来源；
- filter applied/drop ratio；
- timestamp rollback clamp/retry/skip 计数；
- no-current-sync、empty-events、empty-current-mask、current-frame-expired 分类计数。

目标：

- 状态窗口能直接说明为什么 mask 没用上，而不是只看到黑屏或 stale。

### 方向 C：降低 polygon 的长期成本

候选方案：

- 按相机/人数量动态降低 polygon 点数；
- 只给事件过滤链路输出 polygon，Preview 可按需读取；
- 对静态/相近 polygon 做缓存；
- 将 polygon rasterization 移到事件 worker 或 GPU；
- 在高负载时自动降级到 bbox mask。

### 方向 D：继续验证 G 链路底层 timestamp 回退来源

当前已避免崩溃，但建议后续分析：

- native HAL FIFO 是否偶发输出非单调事件时间；
- event trigger map 是否存在重复/回退；
- pre-spatter guard reset 是否影响 frame generator 时间；
- OnDemandFrameGenerationAlgorithm 是否应在每次 source reset 后重建实例，而不是只 reset。

### 方向 E：更长时间稳定性测试

### 方向 F：Global Person ID 与 BEV ID 稳定性验证

建议新增一轮 5-10 分钟验证：

- 观察 PyOpenGL CUDA Preview 与 Global BEV Fusion 中同一人员 ID 是否稳定；
- 对比 `global_person_latest.json` 的 `global_id` 与 BEV `display_player_id`；
- 统计 display relink/create 频率；
- 评估 `reid_gate_m` 是否需要从 1.2m 收紧到 0.8-1.0m；
- 若源端 `global_id` 仍频繁跳变，再继续修 Global Person ID 的关联策略，而不是只依赖 BEV 显示层。

建议下一轮：

- 10 分钟 polygon + G fix 全系统测试；
- 同时采集 CPU hotspot；
- 对比 bbox fallback 与 polygon 模式的：
  - Global YOLO FPS；
  - JSONL 写入体积；
  - Preview render FPS；
  - event worker CPU；
  - applied/missing ratio。

## 当前阶段是否适合进入下一步

适合进入下一步，但下一步重点应从“显示和 polygon 输出”转向“事件侧 mask 时序匹配”。

本阶段已经证明：

- polygon 输出可行；
- Preview polygon 显示可行；
- G 链路崩溃已被抑制；
- 当前 mask 过滤未生效不是因为没有 polygon，而是因为时序匹配策略过严。
- BEV ID 显示稳定性已加入阶段修复，但仍需要远端窗口观察和日志统计验证。
