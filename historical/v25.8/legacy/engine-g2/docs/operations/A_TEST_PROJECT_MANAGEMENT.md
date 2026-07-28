# A 测项目代码管理台账

本工作区从 `workspace_v16b_30min_stable_20260701_213825` 开始作为 A 测基线。A 测阶段只在本目录内推进，不回写旧 `项目代码` 目录，避免旧运行产物和历史补丁干扰。

## A0.0 基线

来源:

- V16-B 30 分钟稳定性实测后的源码快照。
- 已排除 `.git`、`.ssh`、`sync_ipc`、`_sandbox`、缓存和本地备份目录。

基线说明:

- `V16B_WORKSPACE_README.md`
- `OPTIMIZATION_HISTORY_20260630_20260701.md`

## A0.1 状态面板 key 修复

状态: 已完成本地静态验证；第一次 10 分钟远端实测发现 YOLO FPS 语义问题并已修正，等待第二次 10 分钟复测。
范围:

- `pyopengl_cuda_preview_renderer.py`
- `global_yolo_infer_server.py`
- `A0_1_STATUS_PANEL_KEYS_AND_STARTUP.md`
- `run_a01_10min_test.sh`

修复摘要:

- 状态面板和兼容 CSV 同时识别 `global_yolo_worker_*_status.json`。
- `categories.global_yolo` 改为优先从 Global YOLO merge/worker status 派生 FPS、batch、predict/total latency。
- `global_yolo_infer_server.py` 的 `last_batch` 增加 `predict_fps_total` 和 `total_fps_total`。
- 复测后修正: `categories.global_yolo.infer_fps` 改为基于 `stats.frames_inferred` 的增量真实吞吐，batch capacity 只作为 `capacity_fps/fps_source` 辅助显示。
- 复测后修正: `categories.global_bev.pbo_ready` 改为读取 `global_bev_status.json.readback.pbo_ready`，并显示 `pbo_mode/pbo_ready_count`。
- `categories.event.per_camera` 的 `event_rate_mevs/loop_fps/guard_state` 改为 Event worker status 优先。
- `categories.event.active_count` 使用 telemetry active 与 fresh worker 数的最大值，避免真实 worker 正常但面板显示 0。
- `status_summary.csv` 的 Global YOLO 关键字段回填统一状态，避免旧 telemetry 误置 0。

验证:

- `py_compile` 通过。
- `launch_profiles/627_event_overlay_bev.json` JSON 解析通过。

后续 A0.1 远端验收点:

- 状态窗口“全局YOLO”显示真实 worker_count、infer_fps、batch、predict_ms、total_ms。
- 状态窗口“事件相机链路”每路显示真实 loop_fps、event_rate_mevs、guard_state。
- `status_summary.csv` 中 Global YOLO 指标不再为旧 telemetry 的 0。

### A0.1 最终实测结论

状态: 通过。

最终通过归档:

- 远端: `sync_ipc/full_system_tests/a01_10min_20260701_230112_display1_keepstdin.tar.gz`
- 本地: `C:\Users\axyis\Documents\判定开发\remote_tests\a01_10min_20260701_230112_display1_keepstdin.tar.gz`

关键结论:

- `launcher_rc=124`，10 分钟 timeout 正常结束。
- Global YOLO 真实吞吐 p50 约 `319.7 fps`，worker perf CSV 交叉验证总吞吐约 `317.9 fps`。
- `yolo_infer_fps` 已从 batch capacity 口径修正为真实吞吐口径。
- `yolo_capacity_fps` 保留为容量参考，p50 约 `414.9 fps`。
- `yolo_fps_source` 主要为 `frames_inferred_delta`。
- Global BEV `pbo_ready=true`，`pbo_mode=pbo_3`，render p50 约 `30.31 fps`。
- MVS 8 路稳定 active，单路 p50 约 `39.88 fps`。
- Event 8 路稳定 active，启动阶段短暂 BROKEN 属拉起过程。

说明:

- 第三轮后段因环境剧烈变化触发大量事件/噪声弹道可视化，导致 Preview render 短时降到约 `50-70 fps`。
- 该现象不判定为 A0.1 状态 key 修复失败，记录为后续“可视化压力保护/事件噪声抑制”风险项。

新增管理文档:

- `A_TEST_REMOTE_RUNBOOK.md`
- `A_TEST_BACKLOG_RISK_REGISTER.md`

## V17-A Diagnostic Log Throttle

Status: implemented locally, pending remote restart validation.

Scope:

- Disable `event_human_mask_stats_*.jsonl` by default.
- Add event-worker JSONL write throttle and flush interval guard.
- Keep human-mask summary counters in status JSON while removing high-volume per-slice diagnostic writes from normal runs.

Version note:

- See `../versions/NODE_VERSION_20260702_V17_A_DIAG_LOG_THROTTLE.md`.
