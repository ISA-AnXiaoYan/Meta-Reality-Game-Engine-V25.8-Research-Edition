# Evidence Replay Dataset v1

## 目标

A1 的数据集用于快速版本回归，复盘 Global Person ID、位置更新、pseudo/final hit、伤害输出和游戏事件桥接链路。

它不用于验证 YOLO/Event 原始感知性能。polygon、弹道轨迹、命中 debug 等重数据只在状态窗口或远端脚本发出录制 start 后写入，系统空闲时不额外复制这些重数据。

## 录制入口

### 状态窗口

在 Sync Diagnostics 状态窗口的人员 ID 数据集页面使用：

- `Evidence Replay Start`
- `Evidence Replay Stop`

Evidence Replay Start 会写入：

- `dataset_kind=evidence_replay`
- `include_mask_polygons=true`
- `include_bullet_trajectory=true`
- `include_hit_debug=true`

### 全量录制入口

状态窗口新增：

- `开始全量录制`
- `停止全量录制`

全量录制会用同一个 `session_id` 同时启动两条链路：

- `sync_ipc/record_control.json`: `targets=["event_raw","mvs_raw"]`
- `sync_ipc/person_id_dataset_control.json`: `dataset_kind=evidence_replay`

Evidence Replay Dataset 会强制保留：

- YOLO 后的人体 polygon: `human/human_result_polygon_<camera>.jsonl`
- 事件侧弹道点/事件轨迹: `bullets/*.window.jsonl`
- pseudo/final/shadow/reject/hit debug: `hit_judge/*.window.jsonl`

传感器 RAW 和结构化 evidence 分别写入不同根目录，但共享同一个
`session_id`，用于后续复盘关联。

旧的 Person ID 数据集按钮仍保留，默认 `dataset_kind=person_id`，不会保存 polygon/弹道/命中 debug 重数据。

### 远端自动测试

示例：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 smoke `
  -Label a1_evidence_smoke `
  -Duration 370 `
  -PersonDataset mixed `
  -EvidenceReplayDataset `
  -Wait `
  -SummaryAfter
```

可选数据集类型：

- `negative`
- `positive`
- `mixed`
- `battle_1v1`

`-EvidenceReplayDataset` 会把远端环境变量转成 recorder 控制字段：

- `PERSON_ID_DATASET_KIND=evidence_replay`
- `PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS=1`
- `PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY=1`
- `PERSON_ID_DATASET_INCLUDE_HIT_DEBUG=1`

全量录制远端入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 smoke `
  -Label full_record_smoke `
  -Duration 370 `
  -FullRecording `
  -Wait `
  -SummaryAfter
```

`-FullRecording` 会同时启动 Event RAW、MVS RAW、Evidence Replay Dataset，
并用同一个 `${RUN_ID}_full` 作为 session id。

## 输出目录

Evidence Replay 数据集写入：

```text
recordings/evidence_replay_datasets/<session_id>/
```

全量录制时传感器 RAW 写入：

```text
recordings/<session_id>/
raw_records/
```

主要文件：

```text
manifest.json
summary.json
README.md
status_samples.jsonl
operator_annotations.jsonl
human/human_result_polygon_<camera>.jsonl
global/global_person_tracks_baseline.jsonl
bullets/overlay_bullet_point_all.window.jsonl
bullets/overlay_bullet_event_all.window.jsonl
hit_judge/pseudo_hit_all.window.jsonl
hit_judge/hit_candidate_all.window.jsonl
hit_judge/hit_candidate_shadow_strong_reasoning_all.window.jsonl
hit_judge/hit_reject_debug_all.window.jsonl
hit_judge/damage_attribution_events.window.jsonl
hit_judge/game_event_bridge_sent.window.jsonl
index/timeline_index.json
index/camera_index.json
index/bullet_track_index.json
```

## 当前验证

2026-07-04 本地探针验证通过：

- start 前旧行不会进入数据集。
- start 后 human polygon、global track、operator annotation、bullet、pseudo/final hit、reject debug、damage、bridge sent 均可写入。
- polygon 字段保留，mask bitmap/RLE 等重 bitmap 字段仍过滤。
- stop 后生成 `timeline_index.json`、`camera_index.json`、`bullet_track_index.json`。

2026-07-04 远端验证通过：

- 远端编译通过。
- 远端控制文件可写入 `dataset_kind=evidence_replay` 和 include 开关。
- 远端短录制生成 `recordings/evidence_replay_datasets/20260704_132518_evidence_mixed_a1_smoke_evidence`。
- 短录制后系统 monitor 为 OK，Global BEV render 约 30fps。

## 风险点

- Evidence Replay 保存 polygon 和多条 debug JSONL，长时间录制会明显增加磁盘写入和数据量；只应在需要制作数据集时打开。
- 当前 A1 是 evidence 采集格式，后续 Windows 本地可视化播放器还需要按该目录结构读取。
- 无射击 smoke 中 `bullet_point_rows/pseudo_hit_rows/final_hit_rows` 可能为 0，这是正常情况。
- PowerShell/SSH 命令中不要使用远端 `$变量`，容易被本地 PowerShell 提前剥离；优先使用 `tools/remote_test.ps1` 封装入口。
