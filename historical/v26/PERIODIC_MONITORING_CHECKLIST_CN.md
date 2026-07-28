# 项目周期性监控检查清单

更新时间：2026-07-09

本文档是 `PERIODIC_MONITORING_CHECKLIST.md` 的中文可读版，用于现场运行、版本迭代和性能复测后的固定巡检。目标不是替代日志分析，而是快速判断系统处于健康、降级还是异常状态。

## 检查频率

| 场景 | 建议频率 | 目的 |
| --- | --- | --- |
| 每次启动后 | 启动后 1-3 分钟 | 确认子进程、共享内存、状态窗口正常 |
| 短测 | 5-10 分钟 | 确认 FPS、延迟、Event、YOLO 没有快速恶化 |
| 长测 | 20-30 分钟 | 确认队列、OBS/SRT、BEV、overlay 没有累积退化 |
| 版本修改后 | 每次代码同步后 | 对比修改前后性能与错误日志 |
| 现场异常后 | 立即 | 固化证据，避免进程退出后丢失信息 |

## 优先查看文件

| 类别 | 文件/目录 | 用途 |
| --- | --- | --- |
| 总状态 | `sync_ipc/status_logs/*/status_latest.json` | 状态窗口数据源 |
| 状态趋势 | `sync_ipc/status_logs/*/status_summary.csv` | FPS、延迟、队列趋势 |
| 状态页签 | `sync_ipc/status_logs/*/status_tabs.jsonl` | 状态窗口各页明细 |
| 进程热点 | `cpu_hotspots_*/top_processes.txt` | 定位 CPU 热点进程 |
| 线程热点 | `cpu_hotspots_*/threads/*_threads.txt` | 定位单进程内部热点线程 |
| PID 采样 | `cpu_hotspots_*/pidstat.txt` | 区分持续高负载和瞬时尖峰 |
| GPU 状态 | `cpu_hotspots_*/gpu.txt` | GPU 利用率、显存、视频编码状态 |
| 子进程日志 | `sync_ipc/launcher_logs/*.log` | 各模块启动/异常原因 |
| Global YOLO | `sync_ipc/global_yolo_status.json`、`global_yolo_worker_*_status.json` | YOLO 主链路性能 |
| Global Person ID | `sync_ipc/global_person_status.json`、`global_person_latest.json` | 跨相机 ID 融合状态 |
| Global BEV | `sync_ipc/global_bev_status.json` | BEV 渲染、输出、控制状态 |
| Event Overlay | `sync_ipc/event_overlay_overview_status.json` | 事件 overlay 窗口真实状态 |
| Hit Judge | `sync_ipc/hit_judge_status.json` | 命中链路 authority、候选、过滤状态 |
| Game Bridge | `sync_ipc/game_event_bridge_status.json`、`game_event_bridge_sent.jsonl` | 对外发送状态 |

## 高频问题巡检

### 可见光黑屏或不更新

先看：

- MVS worker 是否存活。
- `/dev/shm/mvs_raw_latest_*.mmap` age 是否新鲜。
- 每路 `frame_id` 是否递增。
- Preview 是否 `cuda_pbo_ready`。
- Global BEV 是否因为 upload budget hold last frame。

### Event overlay 缺失或不同步

先看：

- Event worker 是否存活。
- Native HAL broker 是否有数据。
- `event_layer.presentation_worker` 是否 running。
- `display_mismatch_cam_count`。
- `event_delta_ms_by_cam`。
- 是否出现某路 outlier，例如 50-80ms warn 或 >80ms hard。

### YOLO 性能下降

先看：

- `inferred_fps_total`
- `published_fps_total`
- `postprocess_queue_depth`
- `postprocess_lag_ms`
- `postprocess_backpressure_ms`
- polygon contract 是否正常。
- 是否打开了重诊断 JSONL。

### Global Person ID 不稳定

先看：

- `track_count`
- `created_tracks`
- `id_switch_suspects`
- `global_person_target`
- 相机交接区出生/死亡规则是否启用。
- anchor/foot_pixel 选点策略是否符合当前版本。

### 命中误判或漏检

先看：

- `final_hit_authority`
- `strong_gate_enabled`
- `strong_reasoning_enabled`
- `human_noise_filter_mode`
- `human_noise_track_start_bbox_expand_px`
- `pseudo_hit_total`
- `pseudo_hit_human_noise_suppressed`
- `hit_candidate_all.jsonl`
- `pseudo_hit_all.jsonl`

## 远端测试后固定动作

每轮远端测试结束后：

1. 确认 `launcher_rc=124` 或合理退出码。
2. 看 `runtime_overall_state`。
3. 看 `runtime_broken_links` / `runtime_degraded_links`。
4. 看 `status_key_ok`。
5. 拉取 `.tar.gz` 归档。
6. 记录 run label、run dir、archive path。
7. 对比上一轮主要指标。

## 性能健康参考

以下只是经验参考，具体以当前节点目标为准：

- Global YOLO 总 FPS：约 270-320fps。
- Global BEV render FPS：目标约 30fps，短时 28-30fps 可接受。
- Event layer GUI 绘制耗时：应低于预算，大多数 tick 不应阻塞 GUI。
- `event_human_mask_stats_*.jsonl` 默认不应增长。
- `runtime_broken_links` 应为空。
- `status_key_ok` 应为 true。

## 证据保存

现场反馈必须尽量绑定到具体时间和 run：

- 真实射击时间。
- 是否命中。
- 是否有人体移动。
- 是否出现误判。
- 状态窗口观察到的异常。
- run label 和归档路径。

不要只记录“好像漏检”或“感觉卡了”。最好记录绝对时间，例如 `2026-07-09 15:23:10`。

## V25 Shadow 服务检查

当 disposable profile 开启 `v25_live_shadow_enable` 时，检查：

- `system_status_latest.json.categories.v25_shadow_services.state`
- `services.visualization.input_age_ms`
- `services.combat_engine.input_connected`
- `services.game_bridge.input_connected`
- 三个服务均为 `official_result=false`
- `official_writer_count=0`
- `production_network_enabled=false`

未开启时，该类别必须为 `DISABLED` 且 `affects_overall=false`。Shadow 结果不得显示成正式命中或正式游戏发送结果。
