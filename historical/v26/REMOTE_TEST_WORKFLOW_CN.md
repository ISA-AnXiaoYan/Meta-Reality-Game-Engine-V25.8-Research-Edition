# 远端测试工作流

更新时间：2026-07-09

本文档是 `REMOTE_TEST_WORKFLOW.md` 的中文伴随文档。目标是让现场测试和开发同事用稳定流程启动、停止、评估和归档远端系统，避免 PowerShell/SSH 引号、管道、`$()`、heredoc 被本地剥离或错误解释。

## 核心原则

复杂逻辑放在远端 Linux 主机的 `remote_ops/` 内。Windows 本地只用 `tools/remote_test.ps1` 做薄触发器。

`tools/remote_test.ps1` 会把远端命令写成临时 Bash 脚本上传到 `/tmp`，再通过 SSH 执行 `bash <script>`。这样能绕开以下常见问题：

- PowerShell 把 `$变量`、`$()` 提前展开。
- 管道 `|` 被本地或远端 shell 拆错。
- 引号嵌套导致远端看到的命令和本地写的不一致。
- heredoc 在跨平台命令里被截断或转义。

远端桌面入口也统一通过工具链安装，不要手工编辑 `.desktop`：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 install-desktop-shortcuts
```

该命令会安装“一键启动”“一键停止”和“元现实游戏引擎 - 启动控制面板”。控制面板默认使用实时源；打开窗口不会直接启动系统，只有点击“一键启动整套系统”后，才会写入 V25 source-mode 控制状态并调用受管启动脚本。需要从本地验证远端 GUI 时使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 open-startup-panel
```

控制面板日志位于 `sync_ipc/v25_startup_control_panel.log`。
一键启动状态位于 `sync_ipc/v25_panel_launch_status.json`，必须结合其中的 `state`、`launcher_pid`、`run_dir`、`launch_extra_argv` 和 `ready_returncode` 判断是否真正就绪。

## 常用命令

所有命令从 Windows 本地 Git 仓库根目录执行：

```powershell
.\tools\remote_test.ps1 sync-ops
.\tools\remote_test.ps1 deploy
.\tools\remote_test.ps1 deploy-worktree
.\tools\remote_test.ps1 validate
.\tools\remote_test.ps1 list-datasets -DatasetLimit 20
.\tools\remote_test.ps1 check-dataset-manifest -DatasetDir recordings/<session> -DatasetLabel battle_1v1 -ExperimentNote "现场说明"
.\tools\remote_test.ps1 check-dataset-manifest -DatasetDir recordings/<session> -PrintResolvedManifest
.\tools\remote_test.ps1 v25-replay-control -DatasetDir recordings/<session> -ReplayControlMode full_replay -ReplayTiming realtime
.\tools\remote_test.ps1 smoke-mvs-broker -DatasetDir recordings/20260708_181936_full -MaxFrames 160
.\tools\remote_test.ps1 smoke-mvs-replay -DatasetDir recordings/20260708_181936_full -Duration 90 -DeployWorktree -Wait -SummaryAfter

.\tools\remote_test.ps1 smoke -Duration 370 -Label smoke_remote_ops
.\tools\remote_test.ps1 smoke -Duration 370 -Label full_record_smoke -FullRecording
.\tools\remote_test.ps1 stress -Duration 1800 -Label stress_remote_ops

.\tools\remote_test.ps1 summary -Label smoke_remote_ops
.\tools\remote_test.ps1 assess -Label smoke_remote_ops -AssessmentNode V24-D
.\tools\remote_test.ps1 fetch -Label smoke_remote_ops
.\tools\remote_test.ps1 monitor -Label desktop_latest

.\tools\remote_test.ps1 stop
.\tools\remote_test.ps1 stop -Label stress_remote_ops
.\tools\remote_test.ps1 start-latest -Label desktop_latest
.\tools\remote_test.ps1 stop-latest -Label desktop_latest

.\tools\remote_test.ps1 hot-update -BboxStartExpandPx 48 -HumanNoiseMode soft -AuthorityMode strong_reasoning
```

如果本机 PowerShell 禁止脚本执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 validate
```

## 默认远端环境

| 项 | 默认值 |
| --- | --- |
| Host | `ysxq@192.168.31.115` |
| Remote project | `~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627` |
| Launch profile | `launch_profiles/627_event_overlay_bev.json` |
| Python | `~/miniconda3/envs/ids_recv/bin/python3` |

## 启动模式

### 短测 smoke

用于 5-10 分钟功能检查和回归：

```powershell
.\tools\remote_test.ps1 smoke -Duration 370 -Label v24_smoke
```

### 等待结束并输出 summary

```powershell
.\tools\remote_test.ps1 smoke -Duration 370 -Label v24_smoke -Wait -SummaryAfter
```

### 长时间压力测试

```powershell
.\tools\remote_test.ps1 stress -Duration 1800 -Label v24_stress
```

### 固定桌面长效运行

```powershell
.\tools\remote_test.ps1 start-latest -Label desktop_latest
.\tools\remote_test.ps1 monitor -Label desktop_latest
.\tools\remote_test.ps1 stop-latest -Label desktop_latest
```

## 部署规则

`deploy` 上传当前 Git 分支快照，不包含未提交改动：

```powershell
.\tools\remote_test.ps1 deploy
```

`deploy-worktree` 上传当前工作树中已修改和未跟踪的文件，适合文档补丁、热修补验证：

```powershell
.\tools\remote_test.ps1 deploy-worktree
```

注意：`deploy` 只覆盖归档里存在的文件，不会删除远端历史残留的新文件。如果某个版本新增过临时工具文件，回退时要显式确认远端残留已清理。

## Runner 会检查什么

每次标准 run 会归档以下关键证据：

- `launcher_rc.txt`，其中 `124` 表示 timeout 到时结束，通常正常。
- `runtime_overall_state`、`runtime_broken_links`、`runtime_degraded_links`。
- 状态窗口 key audit。
- Global BEV render/output FPS、event layer 状态。
- Global YOLO FPS、polygon contract、input buffer/audit。
- Event Overlay Overview 控制和状态。
- `event_human_mask_stats_*.jsonl` 前后大小，确认诊断日志是否失控增长。
- `launcher_logs/` 和各服务 `*_status.json`。
- 远端 run 目录和 `.tar.gz` 归档。

## V25 Source-Mode / Replay 测试口径

V25 的启动模式、Event replay、MVS replay 测试仍必须走 `tools/remote_test.ps1`
和 `remote_ops/`。不要从 PowerShell 手写复杂 SSH 一行命令做 replay 或启动诊断。

source-mode 测试每轮至少归档并比对：

- 通过 `list-datasets` 输出确认所选 dataset folder 的数据类型。
- 本轮生成的 launch profile 或 profile overlay。
- 实际生效的 source-mode 摘要，包括 `event_source_mode`、`mvs_source_mode`，
  以及 run 归档中的 `source_mode_effective.json`。
- dataset manifest、manifest 校验报告，以及 resolved manifest：
  `replay_manifest_resolved.json` 或 run 归档中的
  `v25_resolved_manifest_latest.json`。
- Event file broker 日志、A-H FIFO freshness。
- Event broker stats JSONL 摘要，至少确认 `source_mode`、`sync_policy`、
  A-H `valid_for_replay_sync`、`triggers_seen` 和 `frames_written`。
- MVS file broker 日志、A-H frame/shm freshness。
- MVS file broker stats 摘要，至少确认 `source_mode=file_replay`、`cameras_published=8`、
  `error_count=0`、每路 `frames_published` 持续增长。
- `check_fullsys_run.py summary` 输出中的 `runtime_overview_reconciled_links`。
- `check_fullsys_run.py summary` 输出中的 `runtime_worker_reconciled_links`。

`runtime_overview_reconciled_links` 只用于说明显示/状态采样 false positive，例如
`system_status_latest.json` 最后一拍报 stale，但更晚的 Event Overlay Overview 已证明
该路 fresh/drawn。它不能掩盖真实的 worker、sync、MVS、YOLO、hit judge 或 bridge 故障。

`runtime_worker_reconciled_links` 比 overview 修正更严格，只允许解释停机采样导致的
worker false positive，例如运行结束归档时 `event:H:worker` 被最后一拍 `stopping` 状态污染。
只有同一个归档 run 同时存在 fresh/drawn 的 Event Overlay Overview 行，以及本次 run
start/end 时间窗口内发布的 worker-owned sidecar（如 `event_overlay_H_latest.json` 或
`event_viz_H_latest.json`）时，才允许清除该 worker broken link。没有新鲜 worker sidecar
时不能用它掩盖真实 worker 断链。

V25-B broker stats 注意事项：`write_errors` / `exceptions` 非零时必须结合每路
`last_error` 判断。正常 timeout 停机也可能记录 FIFO 写入被 shutdown 打断；后续应把
这类 shutdown write interrupted 与运行中 FIFO 写失败分开统计，不能只看 aggregate
`errors` 一个数字决定通过/失败。

V25-C MVS replay 注意事项：

- `smoke-mvs-broker` 只验证 standalone broker 能把数据集 MVS 帧写入 shm/sidecar。
- `smoke-mvs-replay` 才验证 full system 启动路径。该模式默认追加
  `--mvs-file-frame-broker-loop`，避免短数据集播完后 broker 退出导致 launcher 停机。
- `mvs_source_mode=file_replay` 时 live MVS split worker 不应启动，避免和 broker 双写
  `/dev/shm/mvs_latest_{camera}.mmap`。
- run 目录中的 `source_mode_request.json`、`launch_command.sh`、`test_report.json`
  是判断实际启动参数的优先证据。
- MVS-only replay smoke 当前仍会启动 live Event；如果现场没有事件 sparse/trigger 输出，
  `runtime_overall_state` 可能因 Event worker 标为 `BROKEN`。此时 V25-C 验收应优先看
  `mvs_file_broker_stats`、Global YOLO input alignment、Global BEV MVS alignment。
  full Event+MVS replay 的统一通过/失败口径留到 V25-E。

V25-D/E dataset manifest 注意事项：

- 历史数据可能被拆在两个同名目录中：`recordings/<session>` 保存 MVS AVI/CSV，
  `raw_records/<session>` 保存 Event RAW。`check_v25_dataset_manifest.py` 会自动合并同名
  sibling bundle，因此现场只需要传其中一个目录。
- `check-dataset-manifest -DatasetDir <dataset>` 会运行 V25 checker，并在所选
  dataset 目录写入 `replay_manifest_resolved.json`。现场可用 `-DatasetLabel` 和
  `-ExperimentNote` 把测试说明、人工标记说明、现场反馈写入 resolved manifest。
- 需要在 Windows 端查看 resolved manifest 时，加 `-PrintResolvedManifest`，不要再手写
  SSH/Powershell 的 Python 一行命令或 heredoc。
- `v25-replay-control -DatasetDir <dataset> -ReplayControlMode full_replay`
  会刷新：
  - `sync_ipc/v25_replay_control.json`
  - `sync_ipc/v25_replay_status.json`
  - `sync_ipc/v25_replay_launch_command.sh`
- `smoke-full-replay -DatasetDir <dataset>` 现在会把统一 `--dataset-dir` 也传给
  `launch_fusion_system.py`，同时保留显式 Event/MVS broker 参数。判定实际源模式时，
  优先看 `source_mode_effective.json`，不要只看 profile 预期值。
- 真实状态窗口必须显示同一套运行事实：
  `system_status_latest.json.categories.source_mode_replay` 和窗口页签
  “启动/回放源”都必须能看到实际 source mode。V25 source-mode 节点收口前必须确认：
  - `status_key_audit.ok=true`；
  - `system_status.categories.source_mode_replay.event_source_mode_effective`；
  - `system_status.categories.source_mode_replay.mvs_source_mode_effective`；
  - `source_mode_effective_exists=true`；
  - 任一 file replay run 都应有 `v25_resolved_manifest_exists=true`。
- run 归档必须保留 `source_mode_effective.json`、
  `v25_resolved_manifest_latest.json`、`v25_replay_control.json`、
  `v25_replay_status.json`、`v25_replay_launch_command.sh`。不要只凭 profile
  判断实际运行模式。
- `smoke-full-replay -DatasetDir recordings/<session>` 会由 manifest checker 自动展开：
  Event RAW 使用 `raw_records/<session>` 中的 A-H RAW，MVS broker 使用真正包含 MVS 文件的
  `recordings/<session>`，避免把 raw_records 目录误传给 MVS broker。
- 正式 Event/full replay 默认要求 Event RAW 已验证 External Trigger。只有临时诊断时才使用
  `-AllowMissingExtTrigger`，这类 run 不能作为同步验收通过证据。
- 如果 manifest checker 显示 `event_external_trigger_valid=false`，需要先对对应 RAW 执行
  External Trigger 检查并更新 manifest，再启动正式 full replay。
- 当前 V25-D/E 的 full replay 已能自动展开 split bundle，并能启动 Event file broker
  与 MVS file broker；但 BEV 事件层与 MVS 背景的严格同步还需要 dataset manifest
  记录“录制时 Event trigger index 到 MVS program_sync_index 的共同锚点”。仅有
  Event RAW External Trigger 和 MVS frame index 还不够。
- manifest checker 现在区分两个层级：
  - `usable_for_full_replay_startup=true`：Event/MVS 离线源足够启动并被主链路消费。
  - `usable_for_full_replay_sync=true`：除了能启动，还具备严格同步验收所需的共同锚点。
    若该字段为 `false` 且 `degraded_reasons` 包含 `full_replay_sync_anchor_missing`，
    只能做启动/吞吐/链路兼容性测试，不能做 BEV Event/MVS 严格同步验收。
- 新 full recording 结束时，`remote_ops/run_fullsys_test.sh` 会把
  `sync_ipc/event_trigger_map_*.jsonl`、`event_sync_map_ring_*.json`、
  `global_trigger_timeline*.json*`、`sync_start.flag` 等同步锚点复制到
  `raw_records/<session>/replay_sync/` 和 `recordings/<session>/replay_sync/`。
  后续 checker 会递归发现这些 `event_trigger_map` 文件，用于判定
  `usable_for_full_replay_sync`。
- 如果 full replay summary 中 `event_synced_cam_count=0`，先看
  `event_worker_{camera}_status.json`：
  - `ext_trigger.selected_edge_count > 0` 且 `event_sync.map_count > 0` 表示 Event worker
    已消费 RAW trigger。
  - 如果 overlay `program_sync_index` 与 Global BEV `target_sync_id` 相差数百帧，
    这是 replay 数据集缺少共同同步锚点或 Event/MVS replay 有效速率不一致，不应归因为
    Event RAW 硬件数据缺失。
  - `summary` / `monitor` 输出中的 `event_worker_sync` 是快速诊断入口：重点看
    `bev_target_sync_id`、`mapped_sync_min/max`、`mapped_sync_delta_min/max_to_bev`、
    `direct_index_cam_count`。如果 mapped sync 比 BEV target 大几百到几千 tick，
    应优先判定为数据集共同同步锚点缺失或回放速率/循环漂移，不要继续调 BEV 显示阈值。
- `event_sync_direct_index_map=true` 是 file replay 的受控实验路径，只在
  `sync_start.flag.source_mode=mvs_file_replay` 下启用；它不能替代正式数据集中的
  原始 trigger-to-MVS 对齐记录。

当前 V25 参考 smoke：

```text
event_h_worker_infer_final_smoke_20260709_171733_display1_keepstdin
runtime_overall_state=OK
runtime_broken_links=[]
runtime_degraded_links=[]
```

这轮只证明 H 路 fallback/inferred-active 状态修复没有破坏实时
`event_source_mode=live_hal` 生产链路；它不代表 Event RAW file replay 或 MVS file replay
已经完成验收。V25-A 到 V25-G 的后续推进以 `V25_MAIN_TASK_TREE_20260709.md` 为准。

V25-E 当前参考 run：

```text
v25e_report_live_recheck_20260709_204517_display1_keepstdin
runtime_overall_state=OK
runtime_broken_links=[]
runtime_degraded_links=[]
mvs_file_broker_stats.skipped_reason=mvs_source_mode_not_file_replay
```

这轮用于验证 live 模式不会被旧 MVS file broker 状态污染。读取报告时必须先看
`source_mode_request`，再解释 broker stats。

```text
v25e_report_full_replay_recheck_20260709_204938_display1_keepstdin
dataset=recordings/v25e_full_record_anchor_20260709_200415_display1_keepstdin_full
runtime_overall_state=OK
event_worker_sync.locked_cam_count=8
event_worker_sync.direct_index_cam_count=8
event_display_sync.synced=true
event_display_sync.bundle_lookup_state=exact
event_display_sync.bundle_target_delta_ticks=0
```

full replay 报告必须同时看两个字段：

- `event_worker_sync`：Event worker producer head，用来判断 direct-index 锁定、
  replay drift 和 FIFO backpressure。
- `event_display_sync`：Global BEV 实际显示的事件 bundle，用来做视觉同步验收。

不能把 producer head 与 BEV 已显示 bundle 混为一谈。

## V25-E/F 本轮踩坑与改进后的工作模式

本轮暴露的问题和后续固定做法如下：

| 坑位 | 影响 | 后续固定做法 |
| --- | --- | --- |
| PowerShell 直接执行 `.ps1` 被 ExecutionPolicy 拦截 | 本地脚本无法启动，容易让测试中断 | 默认使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 ...` |
| 手写复杂 SSH/PowerShell 命令 | `$`、管道、引号、heredoc 容易被本地剥离或转义错 | 复杂逻辑必须沉到 `remote_ops/` 或由 `tools/remote_test.ps1` 上传临时 Bash 脚本 |
| live 报告读取到上一次 replay 的 MVS file broker 状态 | live run 被旧状态污染，误判 degraded/broken | 先读 `source_mode_request`；live 模式下 `mvs_file_broker_stats.skipped_reason=mvs_source_mode_not_file_replay` 是正常结果 |
| 把 Event producer head 当成 BEV 实际显示同步 | 误判 full replay 不同步或错调 BEV 阈值 | `event_worker_sync` 看 producer drift；`event_display_sync` 才是 BEV 视觉验收口径 |
| 停机采样导致 C-H worker status JSON 缺失 | 健康 worker 被误报 broken | 只允许用同一 run 的 fresh masked overlay + worker-owned sidecar/EventPerf 证据做报告层 reconciliation；不能掩盖真实断链 |
| 只看“能启动”就认为 full replay 同步通过 | 离线数据集缺共同锚点时会产生假同步结论 | 区分 `usable_for_full_replay_startup` 和 `usable_for_full_replay_sync`；严格同步必须有 `event_trigger_map_{camera}.jsonl` 等共同锚点 |
| replay broker timeout 后 `write_errors` 偏高 | 报告看起来吓人，但未必是运行期数据丢失 | 结合 runtime OK、per-camera valid、display sync 判断；后续 V25-F 要把运行期/停机期错误拆分 |

每轮 V25 远端测试结束后，必须把以下信息写入节点文档：

1. 测试命令和 run label。
2. `source_mode_request`。
3. live/replay 的 broker stats 是否属于本轮 source-mode。
4. `event_worker_sync` 与 `event_display_sync` 的判读结论。
5. 是否存在 report-level reconciliation，以及使用了什么证据。
6. 哪些风险属于工具链/报告口径，哪些风险属于真实数据链路。

## 结束和拉取

停止最新 run：

```powershell
.\tools\remote_test.ps1 stop
```

停止指定 label：

```powershell
.\tools\remote_test.ps1 stop -Label v24_stress
```

拉取归档：

```powershell
.\tools\remote_test.ps1 fetch -Label v24_stress
```

默认本地归档目录为：

```text
C:\dev\mrg_remote_tests
```

## 常见风险

- 刚启动 1 分钟内 Event worker、YOLO 或 polygon 可能处于 warm-up，短暂 broken 不一定是真故障。需要复查 live status。
- 如果 `status_freshness.global_bev_from_this_run=false`，不能用旧状态判断当前 run。
- 如果 `event_human_mask_stats_*.jsonl` 持续增长，优先检查诊断日志限流是否被打开。
- 如果出现 PowerShell/SSH 引号问题，不要继续堆复杂一行命令，改进 `remote_ops/` 脚本。
