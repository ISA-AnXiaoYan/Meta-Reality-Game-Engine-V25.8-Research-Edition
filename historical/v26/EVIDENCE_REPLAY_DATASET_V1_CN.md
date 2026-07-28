# Evidence Replay Dataset v1

更新时间：2026-07-09

本文档是 `EVIDENCE_REPLAY_DATASET_V1.md` 的中文可读版。原文存在历史编码乱码，本文重新整理当前数据集录制和回放契约。

## 目标

Evidence Replay Dataset 用于快速版本回归，复盘以下逻辑：

- Global Person ID。
- 人员位置更新服务。
- pseudo hit / final hit。
- 伤害输出和 Game Event Bridge。
- 命中判定参数横向对比。

它不用于验证 YOLO/Event 原始感知性能。也就是说，回放数据可以测试“后处理逻辑是否一致”，但不能证明现场摄像头、事件相机、YOLO 推理速度一定正常。

## 录制原则

性能敏感数据只在明确启动录制时写入：

- YOLO 后 polygon。
- 事件侧弹道轨迹。
- pseudo/final/shadow/reject/hit debug。
- 人员 ID 轨迹。

系统空闲或普通实测时，不应额外复制这些重数据，避免影响主链路性能。

## 状态窗口入口

在 Sync Diagnostics 状态诊断窗口的数据集页面：

- `Evidence Replay Start`
- `Evidence Replay Stop`

启动 Evidence Replay 后应写入：

- `dataset_kind=evidence_replay`
- `include_mask_polygons=true`
- `include_bullet_trajectory=true`
- `include_hit_debug=true`

## 全量录制入口

全量录制应使用同一个 `session_id` 同时启动：

- `sync_ipc/record_control.json`
- `sync_ipc/person_id_dataset_control.json`

当前 V24-Z4 回退边界下，全量录制包含：

- Event sensor/HAL raw。
- MVS raw。
- YOLO 后 polygon/person evidence。
- Event bullet trajectory。
- pseudo/final/hit debug。

当前不包含：

- `event_postmask_events`
- `event_slices_post_polygon_mask.*`

这两个 post-mask 派生事件流在 V24-Z5/V24-Z6 方向已暂停，未来若重启，必须以事件侧 mask raster 为默认 evidence source 并重新远端验收。

## Event RAW External Trigger 要求

从 V25-B-EXTTRIG 开始，录制出的 Event RAW 默认应包含 External Trigger Events。当前项目 live_full 路径中，事件相机 `enable_ext_trigger_in=true`，C++ Native HAL broker 会启用 `I_TriggerIn::Main`，RAW 录制走 `I_EventsStream::log_raw_data()`，因此后续 Event RAW 回放应以 RAW 内 External Trigger 作为同步基准。

数据集收口时需要执行：

```powershell
python tools\check_event_raw_ext_trigger.py <dataset_session_or_raw_dir> --update-manifest
```

验真结果需要写入 `manifest.json`：

```json
{
  "event_raw_expected_ext_trigger": true,
  "event_raw_ext_trigger_verified": true,
  "event_raw_ext_trigger_count": {
    "A": 12032
  },
  "event_raw_ext_trigger_check": {
    "cameras": {
      "A": {
        "period_check_mode": "trigger_id_polarity_group",
        "selected_trigger_track": {
          "id": 0,
          "p": 1,
          "median_period_us": 25000
        },
        "valid_for_replay_sync": true
      }
    }
  }
}
```

注意：Prophesee RAW 里通常会同时记录 External Trigger 上升沿和下降沿。数据集验收必须按 `id + polarity` 分组检查同一边沿流的周期，不能把所有边沿混在一起用相邻间隔判断；混合统计接近 80Hz 是正常现象，不代表 MVS 40Hz 触发异常。

如果某路 RAW 没有 External Trigger，该数据集不能作为真实同步 full replay 验收样本。调试时可以显式打开 synthetic/missing-trigger 降级，但 manifest 和状态窗口必须标记为 degraded，不能混入正式横向对比结果。

## 远端自动测试示例

录制 Evidence Replay：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 smoke `
  -Label a1_evidence_smoke `
  -Duration 370 `
  -PersonDataset mixed `
  -EvidenceReplayDataset `
  -Wait `
  -SummaryAfter
```

全量录制：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 smoke `
  -Label full_record_smoke `
  -Duration 370 `
  -FullRecording `
  -Wait `
  -SummaryAfter
```

可选数据集类型：

- `negative`
- `positive`
- `mixed`
- `battle_1v1`

## 数据集目录结构

一个录制 session 应集中在一个按时间区分的文件夹中。调取回放数据时，只选择该文件夹，程序应自动识别里面的数据类型。

建议结构：

```text
dataset_session/
  manifest.json
  README.md
  human/
    human_result_polygon_A.jsonl
    ...
  bullets/
    *.window.jsonl
  hit_judge/
    pseudo_hit_*.window.jsonl
    hit_candidate_*.window.jsonl
    reject_*.window.jsonl
  raw/
    event_raw/
    mvs_raw/
  status/
    system_status_latest.json
    global_yolo_status.json
    global_person_status.json
    hit_judge_status.json
```

## manifest 必须说明

`manifest.json` 或 README 至少写清：

- 数据集类型：negative/positive/mixed/battle_1v1。
- 是否有真实射击。
- 是否有真实命中。
- 是否有人体移动、交叉、贴近。
- 录制开始/结束时间。
- session_id。
- 远端 run_dir。
- 使用的 branch/commit/profile。
- 是否包含 polygon。
- 是否包含 bullet trajectory。
- 是否包含 hit debug。
- 已知现场标记或人工反馈。

## V25-D source-mode manifest 扩展

从 V25 开始，数据集 manifest 还需要服务于启动前 source-mode 选择。选择一个数据集文件夹后，启动控制面板或 launcher 应自动读取 Event/MVS/trigger/polygon/轨迹等数据类型，不再要求现场手填 A-H 路径。

建议新增或补齐字段：

```json
{
  "source_modes": {
    "event_source_mode": "file_replay",
    "mvs_source_mode": "file_replay",
    "replay_timing": "realtime"
  },
  "sync": {
    "root": "mvs_soft_trigger",
    "event_external_trigger_required": true,
    "event_external_trigger_valid": false,
    "period_check_mode": "trigger_id_polarity_group"
  },
  "paths": {
    "event_raw": {},
    "mvs_raw": {},
    "human_polygon": {},
    "event_tracks": {},
    "trigger_timeline": {}
  },
  "validation": {
    "usable_for_event_file_replay": false,
    "usable_for_mvs_file_replay": false,
    "usable_for_full_replay": false,
    "degraded_reasons": []
  }
}
```

正式 replay 验收不能把缺失 External Trigger 的 Event RAW 自动合成为真实同步数据。需要调试时可以显式降级，但 manifest 必须写入 `degraded_reasons`，并且横向对比报告要把它排除出正式验收样本。

### V25-D manifest checker 输出要求

`remote_ops/check_v25_dataset_manifest.py` 是当前启动前数据集校验入口。它需要把旧 flat manifest 和新结构化 manifest 统一整理成以下字段：

- `source_modes.event_source_mode`
- `source_modes.mvs_source_mode`
- `source_modes.replay_timing`
- `sync.root`
- `sync.event_external_trigger_required`
- `sync.event_external_trigger_valid`
- `validation.usable_for_event_file_replay`
- `validation.usable_for_mvs_file_replay`
- `validation.usable_for_full_replay`

如果旧数据集只写了 `event_raw_ext_trigger_verified=true` 而没有 `sync.root`，checker 可以按当前 V25 约定临时推断为 `mvs_soft_trigger`，但必须在 `warnings` 里提示。新录数据集应直接在 manifest 中写入 `sync.root=mvs_soft_trigger`，避免后续 full replay 与横向对比时产生歧义。

## 回放验证目标

Evidence Replay 用于横向比较：

- same input 下 final hit 数量是否变化。
- false positive 是否下降。
- missed hit 是否下降。
- Global Person ID 的 created_tracks、id_switch_suspects、track_count 是否改善。
- 命中事件是否稳定绑定到 global ID。

不要用 Evidence Replay 判断：

- YOLO 实时 FPS。
- Event HAL 取流稳定性。
- USB/相机硬件链路。
- GPU 渲染压力。
- OBS/SRT 推流稳定性。
