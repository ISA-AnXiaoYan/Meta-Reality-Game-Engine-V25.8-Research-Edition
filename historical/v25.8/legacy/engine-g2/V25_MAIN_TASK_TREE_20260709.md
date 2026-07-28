# V25 Main Task Tree: Source-Mode Replay And Startup Control

更新时间：2026-07-09

本文档记录 V25 主线任务树。它以当前 `v24-a-bev-event-sync-truth`
工作树为基线，吸收 V25-B External Trigger replay、OBS-SRT 修复、以及
H 路 Event Overlay false positive 修复后的实测结果。

## 当前基线结论

1. 生产默认链路仍为 `event_source_mode=live_hal`，实时取流链路不应因离线
   replay 工作而退化。
2. Event RAW replay 的第一原则是 External Trigger First：正式回放必须优先使用
   RAW 内 External Trigger 做同步证据；缺失 External Trigger 时只能标记 dataset
   degraded，不能静默合成真实同步。
3. H 路问题已定位为显示/状态链路 false positive，而不是硬件或 Event worker
   真断链：
   - `Event Overlay Overview` 已支持从 legacy
     `event_overlay_latest_{camera}` fallback 到 display-only
     `event_overlay_masked_latest_{camera}`。
   - 低事件量相机如果 `event_worker_{camera}_status.json` 心跳缺失，可用新鲜的
     `event_overlay_{camera}_latest.json` 作为 worker active 旁证。
   - 最终远端 smoke
     `event_h_worker_infer_final_smoke_20260709_171733_display1_keepstdin`
     结果为 `runtime_overall_state=OK`，`runtime_broken_links=[]`，
     `runtime_degraded_links=[]`。
4. 测试报告层已能纠正 overview 与 `system_status_latest.json` 最后一拍采样不同步
   导致的 `event:H:overlay_stale` 误报；状态窗口后续仍应继续减少这种最后一拍错觉。
5. 仍未收口的风险：Global BEV MVS texture upload budget overrun、个别 Event-to-BEV
   visual sync outlier、以及外部 Game Bridge TCP 连接状态。这些不阻塞 V25 source-mode
   架构，但要在每轮远端实测中记录。

## V25-A：启动控制面板 + Source Mode Profile 生成

目标：提供一个启动前控制面板，用于选择实时/离线源模式，并生成可复现的 launch
profile 或 profile overlay。V25 阶段不要求 broker 热切换，源模式在启动前确定。

状态：V25-A1 已实现最小产品化入口。

已完成：

- 新增 `remote_ops/v25_replay_control_panel.py`，支持 CLI 与 PySide6 GUI fallback：
  - 校验 `dataset_dir`；
  - 选择 `live/event_replay/mvs_replay/full_replay`；
  - 选择 `realtime/fast`；
  - 写入 `sync_ipc/v25_replay_control.json`；
  - 写入 `sync_ipc/v25_replay_status.json`；
  - 生成 `sync_ipc/v25_replay_launch_command.sh`。
- `tools/remote_test.ps1` 新增 `v25-replay-control` 动作，避免现场为了刷新启动
  控制状态而手写 SSH/Powershell 复杂命令。
- `tools/remote_test.ps1 check-dataset-manifest` 新增 `-PrintResolvedManifest`，
  后续抽查 resolved manifest 不再手写 SSH/Powershell 引号嵌套命令。

计划：

1. 新增启动控制面板，至少包含：
   - `event_source_mode`: `live_hal` / `file_replay`
   - `mvs_source_mode`: `live_camera` / `file_replay`
   - Event RAW dataset folder
   - MVS frame dataset folder
   - replay timing: `realtime` / `fast`
   - 是否允许 debug-only missing External Trigger 降级
2. 控制面板只生成 profile/overlay，不直接修改生产基准 profile。
3. 生成的 profile/启动参数/控制状态必须写入 `sync_ipc/` 和本次 run 目录，
   供复盘时确认实际启动参数。
4. 启动前校验：
   - `file_replay` 模式必须有 dataset folder 或显式 raw/frame mapping。
   - 正式验收默认 `allow_missing_ext_trigger=false`。
   - Event/MVS replay 不允许混用来自不同 session 的未声明数据。

验收：

- 通过状态窗口或启动面板生成一个 replay profile。
- `launch_fusion_system.py --launch-profile <generated>` 可打印/启动正确 child args。
- 生成的 profile、profile overlay、source-mode 摘要被归档到 run 目录。

## V25-B：Event File FIFO Broker

目标：增加独立 Event File FIFO Broker，让 Prophesee RAW 离线文件写入现有
`sync_ipc/native_hal_fifos/event_fifo_{camera}.evb` 合同，使下游 Event worker、
tracking、pseudo hit、final hit、Preview、Global BEV、Game Bridge 不需要改 IPC。

状态：主线实现已启动，远端生产 live smoke 已验证未破坏实时链路；file replay 完整验收
仍待补齐。

已完成：

- 新增 `event_source_mode=live_hal|file_replay`。
- 新增 `event_file_fifo_broker` 构建目标与启动路径。
- 新增 `tools/check_event_raw_ext_trigger.py` 用于检查 RAW 内 External Trigger。
- Native HAL broker 状态增加 trigger-in 观测字段。
- H 路 false positive 修复后，Event Overlay 状态不再误把 masked display 流缺省
  当作 Event worker 断链。

下一步：

1. 远端重新构建 C++ broker。
2. 用当前 live RAW 录制样本跑 `check_event_raw_ext_trigger.py --update-manifest`。
3. 启动 `event_source_mode=file_replay` shadow run，确认 FIFO freshness、Event worker
   consume、overview、Global BEV event layer 均正常。
4. 对比 live run 与 file replay run 的事件计数、trigger track、sync index 轨迹。

验收：

- A-H RAW 都有 `valid_for_replay_sync=true`。
- `file_replay` 模式下 A-H FIFO fresh，Event worker 不需要代码分支即可消费。
- 缺失 External Trigger 的 RAW 默认 fail-fast。

## V25-C：MVS File Frame Broker

目标：增加可见光 MVS 离线数据源，使 MVS 录制帧能够回放到现有 MVS shm/sidecar 合同，
供 Global YOLO、Preview、Global BEV、Global Person ID 复盘。

状态：待实现。

建议路线：

1. 新增独立 MVS File Frame Broker，不改 MVS worker 实时取流路径。
2. 输出合同对齐现有：
   - `/dev/shm/mvs_raw_latest_{camera}.mmap`
   - `/dev/shm/mvs_latest_{camera}.mmap`
   - `sync_ipc/mvs_latest_{camera}.json`
   - 必要的 raw-ring/history metadata
3. replay timing 由 dataset manifest 中的 MVS frame timestamp、sync_index、trigger
   timeline 决定。
4. 第一版优先支持 headless/high-speed 回归；需要 GUI 观测时再走 realtime 模式。

验收：

- `mvs_source_mode=file_replay` 时 Global YOLO 能读取 replay shm。
- Global BEV MVS 背景、Preview 可见光、Global Person ID 坐标更新正常。
- MVS replay run 与原录制 run 的 frame_id/sync_index/age 关系可解释。

## V25-D：Dataset Manifest 与自动读取

目标：一个数据集文件夹即可被启动控制面板或 launcher 自动识别全部数据类型。

状态：V25-D1/D2 已实现基础 resolved manifest 契约。

Manifest 最小字段：

```json
{
  "schema_version": 1,
  "dataset_id": "dataset_YYYYMMDD_HHMMSS",
  "created_wall_us": 0,
  "source_version": "",
  "scenario": "",
  "data_types": {
    "event_raw": {},
    "mvs_raw": {},
    "human_polygon": {},
    "event_tracks": {},
    "trigger_timeline": {},
    "system_status": {}
  },
  "sync": {
    "root": "mvs_soft_trigger",
    "event_external_trigger_required": true
  },
  "profiles": {
    "record_profile": "",
    "recommended_replay_profile": ""
  }
}
```

验收：

- 选择 dataset folder 后无需手填 A-H raw/frame 路径。
- manifest 能明确该数据集能用于哪些验证：hit logic、Global Person ID、位置更新、
  Event/MVS sync、或仅 evidence replay。
- manifest 中记录测试说明、现场反馈、人工标记、缺失数据与降级原因。

当前实现：

- `remote_ops/check_v25_dataset_manifest.py` 保留旧的默认检查报告，同时新增：
  - `--print-resolved-manifest`
  - `--write-resolved-manifest [path]`
  - `--dataset-label`
  - `--scenario-name`
  - `--people-count`
  - `--shooting unknown|none|present`
  - `--hit-events`
  - `--miss-events`
  - `--manual-label-file`
  - `--operator-notes`
- resolved manifest 默认落到所选 dataset 的
  `replay_manifest_resolved.json`，schema kind 为
  `v25_dataset_resolved_manifest`。
- resolved manifest 中稳定记录：
  - `dataset_id/session_id`
  - `source_roots`
  - `capabilities`
  - `sync`
  - Event/MVS/cross-sync 文件路径
  - `recommended_launch`
  - `scenario`
  - `degraded_reasons`

## V25-E：Launcher/Profile 接入

目标：launcher 用统一 source-mode 字段决定实时或离线数据源，并将实际 child process
args 写入 run 归档。

状态：V25-E1/E2 已实现 source-mode effective 与报告收口。

计划：

1. 扩展 `launch_fusion_system.py`：
   - 接收 `--event-source-mode`
   - 接收 `--mvs-source-mode`
   - 接收 `--dataset-dir`
   - 从 manifest 展开 Event/MVS broker 参数
2. 扩展 `multi_camera_launcher.py`：
   - Event `file_replay` 启动 `event_file_fifo_broker`
   - MVS `file_replay` 启动 MVS File Frame Broker
   - live 与 replay broker 互斥，避免双写同一 shm/FIFO
3. 生成 `_runtime_config/source_mode_effective.json`，记录最终生效源模式。
4. 状态窗口显示 source-mode、dataset id、manifest 校验状态、broker state。

当前实现：

- `launch_fusion_system.py` 新增 `--dataset-dir`。
- 当 `event_source_mode=file_replay` 或 `mvs_source_mode=file_replay` 时，launcher 会从
  dataset manifest/checker 自动补齐：
  - `event_file_fifo_broker_raw_files`
  - `mvs_file_frame_broker_dataset_dir`
- launcher 写出：
  - `sync_ipc/source_mode_effective.json`
  - `sync_ipc/_runtime_config/source_mode_effective.json`
  - `sync_ipc/v25_resolved_manifest_latest.json`
  - `sync_ipc/_runtime_config/v25_resolved_manifest.json`
- `source_mode_effective.json` 明确记录 live/file replay 互斥事实，避免把“profile
  请求值”误判为“运行中实际生效值”。
- `remote_ops/run_fullsys_test.sh` 会把上述 source-mode effective 与 resolved
  manifest 文件归档到 run `status/` 目录。
- `remote_ops/check_fullsys_run.py` 的 full/compact report 已增加
  `source_mode_effective` 与 `v25_resolved_manifest` 摘要字段。
- 状态窗口与 `sync_ipc/system_status_latest.json` 已增加
  `categories.source_mode_replay`：
  - 中文窗口页签为“启动/回放源”；
  - 总览显示实际 `event_source_mode_effective`、`mvs_source_mode_effective`、
    dataset/session、`usable_for_full_replay_startup` 和
    `usable_for_full_replay_sync`；
  - 控制页同步显示 `v25_replay_control.json`、`v25_replay_status.json`、
    `source_mode_effective.json` 与 resolved manifest 的存在性和路径；
  - `remote_ops/check_fullsys_run.py` 的 `status_key_audit` 已把该类别纳入
    必查项，避免窗口仍读旧链路或缺 key 时误收口。
- `remote_ops/run_fullsys_test.sh` 与 `remote_ops/start_latest_system.sh`
  会把 V25 source/replay 状态文件归档到 run `status/` 目录，包括
  `v25_replay_control.json`、`v25_replay_status.json` 与
  `v25_replay_launch_command.sh`。
- `remote_ops/preflight_clean_runtime.sh` 会把旧
  `source_mode_effective.json`、`v25_resolved_manifest_latest.json` 以及
  `_runtime_config` 中的 source-mode/resolved-manifest 运行态文件移入
  `preflight_stale_status/`，防止上一轮 replay/live 残留污染下一轮报告。

V25-A/D/E 产品化补丁验收：

- 本地：
  - 新增/修改 Python 文件 `py_compile` 通过。
  - `launch_fusion_system.py --help` 已暴露 `--dataset-dir`。
  - `tools/remote_test.ps1 check` 通过。
- 远端工具链：
  - `deploy-worktree` 已部署当前工作树，远端备份：
    `sync_ipc/remote_backups/git_branch_deploy_before_20260709_224757.tar.gz`。
  - `v25-replay-control -ReplayControlMode live` 已写出
    `sync_ipc/v25_replay_control.json`、`v25_replay_status.json`、
    `v25_replay_launch_command.sh`。
  - `check-dataset-manifest -PrintResolvedManifest` 验证 scenario 写入：
    `dataset_label=v25_a1_productized_replay`，
    `operator_notes=D1_E1_A1_E2_D2_acceptance`。
- live smoke：
  - run：`v25_a1_product_live_smoke_20260709_224845_display1_keepstdin`
  - assessment：PASS
  - `runtime_overall_state=OK`
  - `source_mode_effective`: `event=live_hal`, `mvs=live_camera`
  - Global BEV render FPS 约 `30.23`
  - Global YOLO published FPS 约 `291.07`
- full replay smoke：
  - run：`v25_a1_product_full_replay_smoke_20260709_225216_display1_keepstdin`
  - assessment：PASS
  - `runtime_overall_state=OK`
  - `source_mode_effective`: `event=file_replay`, `mvs=file_replay`
  - `event_file_fifo_broker_raw_camera_count=8`
  - `mvs_file_frame_broker_enabled=true`
  - resolved manifest `full_replay_sync=true`
  - Global BEV render FPS 约 `31.11`
  - Global YOLO published FPS 约 `237.98`
  - MVS file broker 8 路发布，`error_count=0`
- source-mode 状态窗口 key 收口 smoke：
  - run：`v25_status_window_link_smoke_20260709_231245_display1_keepstdin`
  - assessment：WARN，但 `runtime_overall_state=OK`、`status_key_ok=true`
  - `source_mode_effective`: `event=live_hal`, `mvs=live_camera`
  - full summary 已确认 `status_key_audit` 包含并通过
    `system_status.categories.source_mode_replay.*`
  - WARN 来源为 BEV 性能/同步既有风险：render FPS 约 `25.98`、1 路
    Event sync outlier、MVS texture upload 有 4 路被预算 hold；不阻塞本次
    source-mode 状态 key 收口。

本轮坑与工作模式修正：

- 尝试通过 SMB 直接读取远端 resolved manifest 时遇到权限拒绝。
- 尝试手写 SSH/Python 一行命令时再次复现 PowerShell/SSH 引号剥离，导致 Python
  字符串被破坏。
- 修正：后续 resolved manifest 抽查统一使用
  `tools/remote_test.ps1 check-dataset-manifest -PrintResolvedManifest ...`。
  不再手写复杂 SSH 变量、引号或 heredoc 命令。

验收：

- live profile 启动保持现有实时链路性能。
- file replay profile 启动时不碰真实相机硬件。
- 同一 run 的 source-mode 证据能从 run 目录独立复盘。

## V25-F：远端测试工具链固化

目标：继续把远端复杂操作固化在 `tools/remote_test.ps1` 与 `remote_ops/` 中，避免
PowerShell/SSH 引号剥离、状态文件半写读取、以及旧进程污染。

状态：已有稳定主流程；本轮已通过 H 路 smoke 验证 report-level overview reconciliation。

计划：

1. 增加 V25 专用动作或参数：
   - `smoke-live`
   - `smoke-event-replay`
   - `smoke-mvs-replay`
   - `smoke-full-replay`
   - `build-native-brokers`
   - `check-dataset-manifest`
   - `v25-replay-control`
2. `run_fullsys_test.sh` 继续归档：
   - per-camera event worker/status/overlay sidecars
   - Event file broker logs
   - MVS file broker logs
   - generated source-mode profile
   - dataset manifest 与校验报告
   - `source_mode_effective.json`
   - `v25_resolved_manifest_latest.json`
3. SMB JSON 检查脚本使用 retry reader，避免读取 `system_status_latest.json` 半写内容。
4. Summary 中保留 `runtime_overview_reconciled_links`，但不能用它掩盖 worker/sync 真故障。

验收：

- 现场测试不再需要手写复杂 SSH 一行命令。
- replay smoke 一条命令能完成构建、启动、summary、归档。
- 报告能区分“状态采样误报”和“真实链路断开”。

## V25-F current delta: report-level worker-sidecar reconciliation

Status: implemented and remotely validated.

This delta extends `check_fullsys_run.py` with
`runtime_worker_reconciled_links`. It only applies to archived
`event:{camera}:worker` false positives caused by shutdown sampling. The
report may clear the broken link only when the same run contains:

1. a fresh/drawn Event Overlay Overview row for that camera;
2. a worker-owned `event_overlay_{camera}_latest.json` or
   `event_viz_{camera}_latest.json` published inside the run start/end window,
   or Event worker `EventPerf` log samples from the same run when shutdown
   sampling failed to archive the sidecar.

This is a report-only guard. It does not change Event worker behavior, FIFO
contracts, hit authority, masked overlay output, or BEV rendering. It must not
be used to hide real worker/sync failures.

Remote validation:

- `v25e_report_live_recheck_20260709_204517_display1_keepstdin`
- `runtime_overall_state=OK`
- `runtime_broken_links=[]`
- `runtime_degraded_links=[]`
- live source request was `--event-source-mode live_hal` and
  `mvs_source_mode_requested=live_camera`.
- `mvs_file_broker_stats` is now skipped in live reports with
  `skipped_reason=mvs_source_mode_not_file_replay`, avoiding stale replay
  broker pollution.
- C-H worker false positives were reconciled from fresh masked overlay rows and
  Event worker `EventPerf` samples from the run log.

## V25-E current delta: full replay pacing and display-sync diagnostics

Status: implemented and remotely validated for startup/full replay smoke.

The first full replay smoke with
`v25e_full_record_anchor_20260709_200415_display1_keepstdin_full` proved that
the dataset can be launched with Event RAW + MVS AVI/CSV and that the recorded
`event_trigger_map_*.jsonl` anchor is present. It also exposed an important
runtime distinction:

1. `event_worker_sync` is the Event producer head. In file replay it can run
   ahead when the Event RAW broker keeps real-time pace but MVS video decode and
   shm publication fall below 40 fps.
2. `global_bev.event_layer.presentation_worker` is the BEV display-selected
   event bundle. This can still be synced because BEV selects from the event
   history ring by target sync.

To prevent downstream hit/mask/tracking replay from drifting with the producer
head, Event worker now has `event_replay_mvs_pacing`:

- enabled by default;
- active only when direct-index file replay sync is in use;
- reads `sync_ipc/trigger_status.json`;
- allows Event to run at most 2 MVS ticks ahead by default;
- waits up to 250 ms per slice, then timeout-releases to avoid deadlock;
- writes pacing diagnostics into each `event_worker_{camera}_status.json`.

`check_fullsys_run.py` now reports both:

- `event_worker_sync`: latest Event producer head, useful for diagnosing replay
  drift and FIFO backpressure;
- `event_display_sync`: BEV actual displayed event bundle, useful for visual
  sync acceptance.

Acceptance for V25 full replay must consider both fields. Producer head can be
ahead temporarily, but it should not grow without bound after pacing is active;
BEV display acceptance should use `event_display_sync.synced=true` or
presentation-worker `bundle_target_delta_ticks` within tolerance.

Remote validation:

- Dataset:
  `recordings/v25e_full_record_anchor_20260709_200415_display1_keepstdin_full`
- Full replay run:
  `v25e_report_full_replay_recheck_20260709_204938_display1_keepstdin`
- `runtime_overall_state=OK`
- `runtime_broken_links=[]`
- `runtime_degraded_links=[]`
- Event file broker: 8/8 cameras opened, 8/8 `valid_for_replay_sync=true`.
- MVS file broker: 8/8 cameras published, `total_frames_published=44016`.
- `event_worker_sync.locked_cam_count=8`
- `event_worker_sync.direct_index_cam_count=8`
- Event producer head stayed bounded relative to BEV delayed target:
  `mapped_sync_delta_min_to_bev=39`,
  `mapped_sync_delta_max_to_bev=45`.
- `event_display_sync.synced=true`
- `event_display_sync.bundle_lookup_state=exact`
- `event_display_sync.bundle_target_delta_ticks=0`
- `event_display_sync.bundle_hit_ratio=0.9984`

Remaining risk:

- File replay broker reports high `write_errors_total`/`exceptions_total` after
  timeout shutdown/backpressure. The runtime stayed OK and all replay sync
  fields passed, but a later V25-F toolchain hardening pass should split
  broker errors into run-phase vs shutdown-phase counts so reports do not look
  scarier than the actual data path.
- In live smoke, C-H were healthy in EventPerf and masked overview, but their
  final `event_worker_sync.cameras.*` entries could still be null when the
  worker status JSON was not captured in the final runtime snapshot. This is
  treated as report-field completeness risk, not a data-path failure. A later
  V25-F pass should archive/merge EventPerf-derived per-camera evidence into a
  dedicated diagnostics field instead of leaving the sync table sparse.

## V25-E/F retrospective: pitfalls and work-mode fixes

Date: 2026-07-09.

This round achieved the V25-E/F smoke-level goal, but exposed several recurring
pitfalls that must shape the next work mode:

1. PowerShell direct script execution can fail with local execution policy.
   Always call remote tests through:
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/remote_test.ps1 ...`.
   Do not run `.\tools\remote_test.ps1 ...` as the default automated path.
2. Do not write complex ad hoc SSH commands for remote testing. Keep logic in
   `tools/remote_test.ps1` and `remote_ops/`; use uploaded temporary Bash
   scripts so `$`, pipes, quotes, and heredocs are not stripped by Windows
   PowerShell.
3. Source-mode must be read before interpreting broker stats. In live mode,
   stale MVS file-broker status files can exist from a previous replay run.
   Reports must skip them unless `mvs_source_mode_requested=file_replay`.
4. Event producer head and displayed BEV event bundle are different clocks.
   `event_worker_sync` diagnoses producer drift/backpressure; visual acceptance
   uses `event_display_sync`.
5. Shutdown-time status sampling can mark healthy Event workers broken. A
   broken worker link may only be reconciled when the same run has fresh/drawn
   overlay evidence plus worker-owned sidecar or EventPerf evidence. This is
   report-only and must not hide real sync or data-path failures.
6. Dataset readiness has two levels. `usable_for_full_replay_startup=true`
   means the system can launch and consume the data; `usable_for_full_replay_sync=true`
   requires a recorded common sync anchor such as `event_trigger_map_{camera}.jsonl`.
7. File replay broker `write_errors` need phase-aware interpretation. Current
   high counts after timeout are not enough to fail an otherwise OK run, but
   V25-F should split run-phase and shutdown-phase errors in a later pass.

Target assessment for this round:

| Target | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Live production path not degraded | Passed | `v25e_report_live_recheck_20260709_204517_display1_keepstdin`, runtime OK, no broken/degraded links | Pacing bypasses live HAL; stale MVS file broker stats skipped. |
| Full Event+MVS replay startup | Passed | `v25e_report_full_replay_recheck_20260709_204938_display1_keepstdin`, runtime OK | Dataset auto-expanded to Event RAW + MVS AVI/CSV. |
| Full replay sync/display smoke | Passed | 8/8 Event direct-index workers, `event_display_sync.synced=true`, exact bundle, delta 0 | Smoke-level visual sync accepted. |
| Producer drift bounded | Passed for smoke | `mapped_sync_delta_min_to_bev=39`, `mapped_sync_delta_max_to_bev=45` | Delta is relative to delayed BEV target; no unbounded hundreds/thousands tick drift after pacing. |
| Report source-mode correctness | Passed | live report skips MVS file broker with `mvs_source_mode_not_file_replay` | Prevents old replay files polluting live reports. |
| Report completeness | Partially passed | C-H worker false positives reconciled with EventPerf | Sync table can still be sparse when final worker JSON is not captured; keep as V25-F follow-up. |
| Whole V25 product closure | Not yet | V25-A startup control panel and broader source-mode UI remain outside this smoke | This round closes V25-E/F smoke-level risk, not the entire V25 branch. |

## V25-G：文档更新与节点收口

目标：V25 每个子阶段进入下一阶段前必须更新 living docs 和节点记录。

状态：进行中。

必须同步维护：

- `EVENT_SIDE_CHAIN.md` / `EVENT_SIDE_CHAIN_CN.md`
- `MVS_CAMERA_CHAIN.md` / `MVS_CAMERA_CHAIN_CN.md`
- `REMOTE_TEST_WORKFLOW.md` / `REMOTE_TEST_WORKFLOW_CN.md`
- `EVIDENCE_REPLAY_DATASET_V1.md` / `EVIDENCE_REPLAY_DATASET_V1_CN.md`
- 本文档 `V25_MAIN_TASK_TREE_20260709.md`

节点收口标准：

1. 本地 compile/syntax/profile 校验通过。
2. 远端 live smoke 不降级。
3. 对应 replay smoke 通过。
4. `system_status_latest.json` 和 `check_fullsys_run.py summary` 都能显示新 source-mode
   与 broker 状态。
5. 文档记录 run label、验收结论、风险与下一步。

## V25-A2：远端桌面启动控制面板与受管一键启动

更新时间：2026-07-10。

状态：主线实现和 dry-run 验证完成，首次现场按钮启动待确认。

- 远端桌面新增 `MRG-Startup-Control-Panel.desktop`。
- 面板默认使用 `live`，仅打开窗口不会启动系统。
- 新增“一键启动整套系统”，调用既有 `start_latest_system.sh`，保留清理、锁、stdin 保活、状态采样和归档。
- source-mode 参数改用 `launch_extra_argv` 结构化数组传递，避免 shell 引号和包含空格路径被拆分。
- 启动状态写入 `sync_ipc/v25_panel_launch_status.json`。
- 新增 `check-startup-panel` dry-run 和 `restart-startup-panel` 标准工具入口。

验证证据与剩余风险见 `docs/versions/NODE_VERSION_20260710_V25_STARTUP_CONTROL_PANEL.md`。

## 推荐推进顺序

1. 先完成 V25-F 的 `build-native-brokers` 与 replay smoke 工具链。
2. 补齐 V25-B file replay 远端 shadow 验收。
3. 做 V25-D manifest v1，让后续 V25-C/V25-E 不再手写路径。
4. 实现 V25-C MVS File Frame Broker。
5. 做 V25-A 启动控制面板，并让它只生成 profile/overlay。
6. 完成 V25-E launcher/profile 自动接入。
7. 用 live、event-only replay、mvs-only replay、full replay 四类 run 做横向对比后，
   进入 V25-G 节点收口。
