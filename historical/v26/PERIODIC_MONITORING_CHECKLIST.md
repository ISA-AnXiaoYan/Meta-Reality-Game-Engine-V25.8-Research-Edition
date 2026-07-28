# 项目周期性监控检查文档

本文档用于现场运行、版本迭代、性能复测后的周期性检查。目标不是替代日志分析，而是把高频 BUG 和性能恶化信号整理成一套固定巡检流程，方便快速判断当前系统是否处于健康、降级或异常状态。

## 1. 检查周期

| 场景 | 建议频率 | 主要目的 |
| --- | --- | --- |
| 每次启动后 | 启动后 1-3 分钟 | 确认各子进程、共享内存、状态窗口接入正常 |
| 短测 | 10 分钟 | 确认 FPS、延迟、事件链路、YOLO 链路没有快速恶化 |
| 长测 | 20-30 分钟 | 确认队列、OBS/SRT、Global BEV、事件 overlay 没有累积性退化 |
| 版本修改后 | 每次代码同步后 | 对比修改前后性能与错误日志 |
| 现场异常后 | 立即 | 固化日志，避免进程退出后丢失证据 |

## 2. 优先查看的文件

| 类别 | 文件/目录 | 用途 |
| --- | --- | --- |
| 总状态 | `sync_ipc/status_logs/*/status_latest.json` | 当前状态窗口数据源 |
| 状态趋势 | `sync_ipc/status_logs/*/status_summary.csv` | FPS、延迟、队列、掉帧趋势 |
| 状态页签 | `sync_ipc/status_logs/*/status_tabs.jsonl` | 状态窗口各页签明细 |
| 进程热点 | `cpu_hotspots_*/top_processes.txt` | 快速定位 CPU 热点进程 |
| 线程热点 | `cpu_hotspots_*/threads/*_threads.txt` | 定位单进程内热点线程 |
| PID 采样 | `cpu_hotspots_*/pidstat.txt` | 区分持续高负载和瞬时尖峰 |
| GPU 状态 | `cpu_hotspots_*/gpu.txt` | GPU 利用率、显存、视频编码状态 |
| 子进程日志 | `sync_ipc/launcher_logs/*.log` | 各模块启动/异常原因 |
| Global YOLO | `sync_ipc/global_yolo_status.json`、`sync_ipc/global_yolo_worker_*_status.json`、`sync_ipc/global_yolo_worker_*_perf.csv` | YOLO 主链路性能 |
| Global Person ID | `sync_ipc/global_person_status.json`、`sync_ipc/global_person_latest.json` | 跨相机 ID 融合状态 |
| Global BEV | `sync_ipc/global_bev_status.json`、`/dev/shm/global_bev_latest.json` | BEV 渲染、输出、控制状态 |
| Event Overlay | `sync_ipc/event_overlay_overview_status.json` | 独立事件 overlay 窗口真实状态 |

## 3. 高频 BUG 清单

### 3.1 可见光画面黑屏或不更新

**常见表现**
- `PyOpenGL CUDA Preview` 窗口黑屏。
- 窗口显示 `DRAW OK`，但可见光画面为空。
- 某一路或多路 MVS frame_id 长时间不变化。
- `/dev/shm/mvs_raw_latest_*.mmap` 存在，但 age 很大。

**高概率原因**
- CUDA/PBO 后端未 ready。
- 预览进程读到旧 mmap，MVS 写端已重建共享内存。
- MVS worker 未持续写帧。
- 状态窗口读取旧字段，误判真实渲染状态。

**重点指标**
- `renderer.video_backend.state`
- 每路 `frame_id` 是否持续递增。
- 每路 `frame_age_ms` / shm mtime。
- `sync_ipc/launcher_logs/gl_cuda.log` 中是否有 `cuda_pbo_ready`、`compile_kernel`、`nvcc`、`PTX` 相关错误。

**处置顺序**
1. 先确认 MVS 进程是否在跑。
2. 再确认 `/dev/shm/mvs_raw_latest_*.mmap` age 是否新鲜。
3. 再看 Preview 后端是否 `cuda_pbo_ready`。
4. 若 frame_id 卡住但 MVS 仍活跃，优先检查 reader reopen/watchdog 逻辑。

### 3.2 事件 overlay 画面缺失或事件点不可见

**常见表现**
- `Event Overlay Overview` 窗口空白。
- 状态窗口显示事件 stale/not_initialized。
- 开启 4px 膨胀后 FPS 下降或事件点仍不可见。

**高概率原因**
- 主 Preview 仍读旧 overlay 状态，没有接入独立 `event_overlay_overview_status.json`。
- 事件 overlay shm 没有更新。
- GL shader 膨胀未 ready，或 fallback 到 CPU 稀疏膨胀造成负载升高。
- 同步模式为 `nearest_mvs` 时，MVS frame_id 或 timestamp 不更新导致事件被判 stale。

**重点指标**
- `event_overlay_overview_status.json` 中每路 `drawn`、`stale`、`shader_ready`、`effective_backend`。
- `event_overlay_preview_dilate_backend` 是否为 `gl_shader`。
- `display_nonzero_pixels`、`overlay_nonzero_pixels`。
- 事件 worker 输出 FPS 与 `event_viz_build_max_fps`。

**处置顺序**
1. 以 `event_overlay_overview_status.json` 为准，不优先相信主 Preview 的旧 overlay 字段。
2. 确认膨胀后端是 GPU shader。
3. 检查事件 shm 是否新鲜。
4. 若事件新鲜但显示 stale，检查同步基准 MVS 是否卡帧。

### 3.3 Global YOLO FPS 未达标或波动大

**常见表现**
- 总 FPS 低于 320。
- 单路低于 40 FPS。
- 平均 FPS 看起来接近目标，但 tail latency 大。
- partial batch 增多，batch 等齐率下降。

**高概率原因**
- batch collect wait 太短或不均衡。
- 某个 worker 的相机组更重，例如 `B,D,F,H` 等齐率低。
- `gpu_fast` 后处理被绕开，启用了 mask polygon/retina mask 等 CPU 重路径。
- merge publisher 或 legacy jsonl 写入积压。
- GPU 同时被 Global BEV、Preview、OBS 编码占用。

**重点指标**
- `global_yolo_worker_*_perf.csv`
- `global_yolo_worker_*_status.json`
- `global_yolo_status.json`
- 每 worker FPS、每路 FPS、batch size、partial batch、predict_ms、postprocess_ms。
- `status_summary.csv` 中 YOLO capture-to-result 延迟。
- V24-Z 异步后处理链路还要看 `published_fps_total`、`inferred_fps_total`、`async_record_postprocess_enabled`、`postprocess_queue_depth`、`postprocess_dropped_batches`、`polygon_contract_ok`、`published_persons_with_polygon_ratio`。
- 状态窗口中的 `global_yolo_*` / `yolo_*` 指标应与 `sync_ipc/global_yolo_status.json` 对齐；若 `status_key_ok=false`，先修状态映射，不要直接下性能结论。

**预警阈值**
- 总 YOLO FPS 长时间低于 300：黄色预警。
- 总 YOLO FPS 长时间低于 280：红色预警。
- 单路 FPS 低于 38：黄色预警。
- 单路 FPS 低于 35：红色预警。
- batch 等齐率明显下降：优先检查 microbatch collect wait。
- V24-Z 队列深度持续不回落或达到队列上限：黄色预警，优先确认 polygon 后处理是否仍在追帧。
- `postprocess_dropped_batches > 0` 或 `polygon_contract_ok=false`：红色预警，不能作为有效验收结果。
- 低人框密度冒烟只能证明链路和状态接入正常，不能替代高负载复测；高负载复测需记录每 worker 人框数、`predict_ms`、polygon/postprocess 耗时和 batch 间隔。

### 3.4 Global Person ID 坐标偏移或 ID 重复

**常见表现**
- 人物移动动态正确，但物理坐标整体偏移。
- 同一个人出现多个 ID。
- 速度显示异常，如几十 m/s。
- ID 在相机重叠区域频繁跳变。

**高概率原因**
- 坐标变换模式不对。
- 顶部俯视相机仍使用了透视视角下的脚底点逻辑。
- bbox center / 相机中心权重 / 重叠区域聚类参数不匹配现场。
- `calib_true/field_calib_camera*.json` 与运行时配置路径不一致。

**重点指标**
- `global_person_status.json`
- `global_person_latest.json`
- 坐标变换模式、重复抑制次数、越界数量、异常速度数量。
- 每个 track 的来源相机数和最近更新时间。

**处置顺序**
1. 先确认 `calib_true` 是否被实际子进程引用。
2. 再检查坐标变换模式。
3. 再调整聚类门限、重复抑制距离和最大速度限制。
4. 最后才调整几何阈值。

### 3.5 Global BEV 画面错位、颜色异常或 FPS 下降

**常见表现**
- 初始画面方向不对，需要按快捷键后才正常。
- 足球场地面拼接正常，但动态人体/掩体在拼接区域有重影。
- BEV render FPS 低于 30，output FPS 低于 20。
- 开启 overlay 后 BEV 明显拖慢其他链路。

**高概率原因**
- runtime state 未保存或启动未读取上次正确配置。
- 草坪颜色匹配、增益、gamma 参数不适合当前灯光。
- 每像素 8 路反向采样成本高。
- PBO readback 阻塞。
- 新增事件/弹道 overlay 与主 warp pass 混在一起。

**重点指标**
- `global_bev_status.json`
- render FPS、output FPS、warp_ms、overlay_ms、readback_ms、present_ms。
- `debug_mode`、`texture_y_flip`、`field_transform_mode`、turf match 状态。

**预警阈值**
- display FPS 低于 25：黄色预警。
- display FPS 低于 20：红色预警。
- output FPS 低于 15：黄色预警。
- readback 耗时持续升高：优先检查 PBO 异步路径。

### 3.6 Hit Judge 命中漏判或误判

**常见表现**
- 有事件/弹道，但命中结果为空。
- `no_human` 计数上升。
- 命中结果与画面人体位置不一致。
- 某些相机单独正常，多相机融合后异常。

**高概率原因**
- 事件时间与 human_result 时间对齐失败。
- YOLO 输出是 `carry_last` 弱结果，不是真实检测。
- event human mask 路径读取了旧 JSONL。
- MVS/event sync root 被 launcher 生成配置覆盖。

**重点指标**
- `hit_no_human_count`
- `capture_to_result_ms`
- `human_carry` / `infer_source=carry_last`
- `event_human_mask_jsonl_template`
- `_runtime_config/*.json`
- hit judge 子进程命令行。

**处置顺序**
1. 先检查生成的 runtime config。
2. 再检查实际子进程命令行。
3. 再看 human_result 是否新鲜。
4. 再调 mask buffer / sync grace / 几何阈值。

### 3.7 OBS/SRT 反复启动或拖慢系统

**常见表现**
- `ffmpeg` 进程反复出现/退出。
- CPU hotspot 中 OBS/SRT 相关进程占用升高。
- 直播输出异常同时 Preview/BEV FPS 下降。

**高概率原因**
- SRT listener/推流端重连。
- ffmpeg 参数导致重复启动。
- 输出帧率或码率过高。
- 录像/直播和 BEV readback 争抢 GPU/编码资源。

**重点指标**
- `sync_ipc/launcher_logs/*obs*`
- `cpu_hotspots_*/top_processes.txt`
- `gpu.txt` 中编码器占用。
- ffmpeg 启动次数。

## 4. 性能恶化指标总表

| 模块 | 指标 | 正常 | 黄色预警 | 红色预警 |
| --- | --- | --- | --- | --- |
| MVS | trigger FPS | 约 40 | < 38 | < 35 |
| MVS | frame age | < 100ms | > 250ms | > 1000ms |
| Preview | video backend | `cuda_pbo_ready` | 重试中 | error |
| Event | overlay backend | `gl_shader` | auto fallback | CPU 膨胀高负载/失败 |
| Event | overlay stale | false | 单路 stale | 多路 stale |
| Global YOLO | total FPS | >= 320 | < 300 | < 280 |
| Global YOLO | per-cam FPS | >= 40 | < 38 | < 35 |
| Global YOLO | partial batch | 低且稳定 | 持续升高 | 大量 partial |
| Global YOLO V24-Z | postprocess queue | 0 或短时回落 | 持续 > 1 | 达上限或持续增长 |
| Global YOLO V24-Z | polygon contract | true 且 ratio=1.0 | ratio 下降 | false 或 dropped > 0 |
| Global YOLO V24-Z4 | input buffer aligned full ratio | 上升且稳定 | fallback 升高 | 持续 fallback/partial |
| Global YOLO V24-Z4 | visual/exposure spread p95 | 在配置预算内 | 接近预算 | 超预算且 FPS 下降 |
| Global Person ID | abnormal speed | 少量瞬时 | 持续出现 | 大量出现 |
| Global BEV | display FPS | >= 30 | < 25 | < 20 |
| Global BEV | output FPS | >= 20 | < 15 | < 10 |
| Global BEV | event display budget overrun | <= 0.20 and rotating | > 0.20 or repeated same tail cams | fixed tail cams or stale visuals |
| Global BEV | MVS upload budget overrun | <= 0.20 and rotating | > 0.20 or held cams > 2 | fixed held cams or visible MVS stall |
| Event OBS-SRT | restart circuit breaker | closed or occasional retry | circuit open on one unused route; check `failure_window_count` | repeated circuit open on active route or `failure_window_count` keeps rising |
| Hit Judge | no_human | 低 | 持续升高 | 大量漏匹配 |
| 系统 | 单进程 CPU | 可解释 | 单核 > 90% | 多进程持续 > 90% |
| 系统 | GPU | 有余量 | > 85% 且 FPS 下降 | 接近满载且卡顿 |

## 5. 固定巡检流程

### 5.1 启动后 1-3 分钟

1. 看状态窗口总览是否为 `正常` 或可接受的 `降级`。
2. 看 8 路 MVS frame_id 是否递增。
3. 看 Global YOLO 双 worker 是否都启动。
4. 看 Event Overlay Overview 是否 8 路 drawn。
5. 看 Global BEV 是否有画面，render/output FPS 是否正常。
6. 看 `launcher_logs` 是否有 Traceback、CUDA、OpenGL、mmap、permission 错误。

### 5.2 10 分钟短测

1. 保存 `status_logs`。
2. 保存 `cpu_hotspots_*.tar.gz`。
3. 统计 Global YOLO 总 FPS、单路 FPS、batch 等齐率。
4. V24-Z 链路额外统计 `published_fps_total`、`inferred_fps_total`、队列深度、丢批数、polygon 合约和每 worker 人框密度。
5. 统计 Preview/BEV/Event Overlay FPS。
6. 检查 OBS/SRT 是否反复启动。
7. 检查异常速度、重复 ID、no_human 是否上升。

### 5.3 20-30 分钟长测

1. 对比前 5 分钟和最后 5 分钟 FPS。
2. 检查队列积压是否随时间增加。
3. 检查 shm age 是否出现周期性变老。
4. 检查 CPU hotspot 是否从可解释进程扩散到更多进程。
5. 检查 GPU 利用率是否上升但 FPS 下降。
6. 检查状态窗口是否仍有未连接指标。

## 6. 异常判断优先级

遇到异常时按这个顺序判断：

1. **进程是否活着**：先看 process map。
2. **数据是否新鲜**：再看 shm age、sidecar age、frame_id。
3. **状态是否接对**：不要只信旧 status 字段，优先看新 sidecar。
4. **链路是否积压**：看队列、batch、capture-to-result。
5. **GPU/CPU 是否瓶颈**：看 hotspot、pidstat、nvidia-smi。
6. **算法参数是否不适配**：最后才调整阈值、坐标模式、同步窗口。

## 7. 常用结论模板

### 正常

```text
系统状态正常：MVS 8 路 40Hz 更新，Global YOLO 双 worker 正常，Event Overlay GPU 膨胀正常，Global BEV display/output FPS 达标，未发现持续 CPU/GPU 热点。
```

### 降级但可运行

```text
系统降级运行：主链路仍可工作，但存在单路 stale / YOLO FPS 低于目标 / BEV output FPS 下降 / 某些状态指标未接入。建议保留日志并继续观察，不建议直接调算法阈值。
```

### 需要中断排查

```text
系统异常：存在多路 MVS 不更新、CUDA/PBO 后端 error、Global YOLO worker 缺失、事件 overlay 多路 stale、或 hit judge 输入断流。建议停止本轮测试，保存 status_logs、launcher_logs、cpu_hotspots 后再修复。
```

## 8. V25 Shadow 服务检查

当 disposable profile 开启 `v25_live_shadow_enable` 时，检查：

- `system_status_latest.json.categories.v25_shadow_services.state`
- `services.visualization.input_age_ms`
- `services.combat_engine.input_connected`
- `services.game_bridge.input_connected`
- 三个服务均为 `official_result=false`
- `official_writer_count=0`
- `production_network_enabled=false`

若当前 profile 未开启，类别应为 `DISABLED` 且 `affects_overall=false`。不能把
Shadow 状态展示成正式命中或正式游戏发送结果。

## 9. 后续文档维护规则

链路文档维护以 `CHAIN_DOCUMENTATION_STANDARD.md` 为准。事件侧更新必须同步
`EVENT_SIDE_CHAIN.md`，MVS 相机侧更新必须同步 `MVS_CAMERA_CHAIN.md`，并在对应
Revision Log 中记录变更。

1. 每发现一个高频 BUG，补充到第 3 节。
2. 每新增一个核心 sidecar/status 文件，补充到第 2 节。
3. 每次性能目标变化，更新第 4 节阈值。
4. 每次链路重构后，检查状态窗口是否仍存在旧字段引用。
5. 文档中的阈值以现场实测为准，不以单次峰值为准。
