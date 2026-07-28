# V25-B-EXTTRIG 节点说明：Event RAW External Trigger First Replay

更新时间：2026-07-09

## 2026-07-09 补充：回到 V25 主线任务树

V25-B-EXTTRIG 不是孤立节点，后续推进回到
`V25_MAIN_TASK_TREE_20260709.md` 中维护的 V25-A 到 V25-G 主线：

- V25-A：启动控制面板 + source mode profile 生成。
- V25-B：Event File FIFO Broker。
- V25-C：MVS File Frame Broker。
- V25-D：Dataset manifest 与自动读取。
- V25-E：Launcher/Profile 接入。
- V25-F：远端测试工具链固化。
- V25-G：文档更新与节点收口。

本轮 H 路问题修复后的参考 smoke 为
`event_h_worker_infer_final_smoke_20260709_171733_display1_keepstdin`：
`runtime_overall_state=OK`、`runtime_broken_links=[]`、
`runtime_degraded_links=[]`。该结果证明实时 `event_source_mode=live_hal`
生产链路未因 V25-B/H 路 fallback 修复而降级，但不代表 Event RAW file replay
或 MVS file replay 已完成验收。

## 目标

本节点把事件相机离线 RAW 回放路线改为 External Trigger First：

- 实时生产链路默认不变，继续 `event_source_mode=live_hal`。
- 离线事件 RAW 回放通过启动前 profile 指定 `event_source_mode=file_replay`。
- 新增 `event_file_fifo_broker` 读取 Prophesee RAW，并写入现有 `sync_ipc/native_hal_fifos/event_fifo_{camera}.evb`。
- 下游 Event worker、Spatter、CppLineMotionFilter、pseudo hit、final hit、Preview、Global BEV、Game Bridge 不需要改 IPC 合同。
- 录制 RAW 缺少 External Trigger 时默认 fail-fast，不能静默合成同步并用于真实同步验收。

## 当前代码依据

当前生产 profile 和代码路径下，Event RAW 理论上应包含 External Trigger：

- `launch_profiles/627_event_overlay_bev.json` 默认 `event_native_hal_broker_enable=true`。
- `ids_8cam_fusion_config_627_runtime.json` 每路事件相机 `enable_ext_trigger_in=true`。
- `multi_camera_launcher.py` 会把 `enable_ext_trigger_in` 转换为 C++ broker 参数 `--enable-trigger-in`。
- `hal_event_fifo_broker.cpp` 启动时调用 `I_TriggerIn::enable(Main)`。
- RAW 录制走 `I_EventsStream::log_raw_data(path)`。

## 新增/修改文件

- `native_event_bridge/hal_event_fifo_broker.cpp`
  - 增加 trigger 观测字段。
- `native_event_bridge/event_file_fifo_broker.cpp`
  - 新增 Event RAW file replay broker。
- `native_event_bridge/CMakeLists.txt`
  - 增加 `event_file_fifo_broker` 构建目标。
- `tools/check_event_raw_ext_trigger.py`
  - 新增 Event RAW External Trigger 验真工具。
- `launch_fusion_system.py`
  - 增加 `--event-source-mode` 与 file broker 参数。
- `multi_camera_launcher.py`
  - `event_source_mode=file_replay` 时启动 `event_file_fifo_broker`。
- `launch_profiles/627_event_overlay_bev.json`
  - 增加默认 source mode 配置，默认仍为 `live_hal`。
- `EVENT_SIDE_CHAIN.md`
  - 增加修订记录。
- `EVIDENCE_REPLAY_DATASET_V1_CN.md`
  - 增加 Event RAW External Trigger 验真要求。

## 新增状态字段

`sync_ipc/event_native_hal_broker_stats.jsonl` 的每路 camera 里新增：

```json
{
  "trigger_in_requested": true,
  "trigger_in_enable_attempted": true,
  "trigger_in_enable_ok": true,
  "ext_trigger_decoder_available": true,
  "first_ext_trigger_ts_us": 123,
  "last_ext_trigger_ts_us": 456,
  "last_ext_trigger_wall_us": 1780000000000000,
  "last_ext_trigger_age_ms": 18.2
}
```

## Event RAW 验真命令

远端或本地具备 Prophesee SDK/C++ broker 构建产物后执行：

```bash
python3 tools/check_event_raw_ext_trigger.py /path/to/dataset_or_raw_dir --update-manifest
```

输出会包含：

- 每路 `trigger_count`。
- `first_trigger_ts_us`。
- `last_trigger_ts_us`。
- `median_period_us`。
- `trigger_tracks`，按 External Trigger `id + polarity` 分组统计。
- `selected_trigger_track`，用于正式周期验收的边沿流。
- `max_gap_us`。
- `valid_for_replay_sync`。

默认期望 40Hz MVS 软触发，对应同一 `id + polarity` 边沿流周期约 `25000us`。不要直接用所有 External Trigger 边沿混合后的相邻间隔做验收；RAW 中通常会同时记录上升/下降沿，混合后总量接近 80Hz，`median_period_us` 可能表现为脉宽或高低电平间隔。

## file replay 启动参数

示例：

```bash
python3 launch_fusion_system.py \
  --launch-profile launch_profiles/627_event_overlay_bev.json \
  --event-source-mode file_replay \
  --event-file-fifo-broker-raw-files A=/data/A.raw,B=/data/B.raw \
  --event-file-fifo-broker-replay-timing realtime
```

重要参数：

```json
{
  "event_source_mode": "live_hal | file_replay",
  "event_file_fifo_broker_raw_files": "A=/data/A.raw,B=/data/B.raw",
  "event_file_fifo_broker_replay_timing": "realtime | fast",
  "event_file_fifo_broker_allow_missing_ext_trigger": false
}
```

`event_file_fifo_broker_allow_missing_ext_trigger=true` 只允许调试。正式数据集和横向验收不得使用该降级结果作为真实同步结论。

## 验收顺序

1. 远端构建 `native_event_bridge/build/event_file_fifo_broker`。
2. 录制 2-3 分钟 live_full Event RAW。
3. 执行 `tools/check_event_raw_ext_trigger.py`，确认 A-H 均有 External Trigger。
   - 验收看 `period_check_mode=trigger_id_polarity_group`、`selected_trigger_track.median_period_us≈25000`、`valid_for_replay_sync=true`。
4. 启动 `event_source_mode=file_replay` shadow，确认 FIFO 文件生成且 Event worker 能消费。
5. 再进入 full replay，与 MVS replay 对齐。
6. 切回 `live_hal`，确认实时生产链路性能和状态不下降。

## 风险点

- 本节点本地只完成 Python 语法检查和 diff 检查；Windows 本机没有 CMake，C++ 编译必须在远端 Linux/Metavision 环境验证。
- 第一版 file broker 依赖 SDK Driver 的 CD / External Trigger callback 顺序。若发现 trigger-only frame 过多，需要改为更严格的统一时间合并队列。
- Event RAW 文件开头不一定从 trigger index 0 开始，后续 full replay 必须使用首个 External Trigger 做归一化，不要假设从 0 开始。
- External Trigger 验收必须按 `id + polarity` 分组，同一路 RAW 可能同时包含两条 40Hz 边沿流；把两条边沿流混算会误判为 80Hz 或不规则周期。
- 如果某路 RAW 缺 trigger，应标记 dataset degraded，而不是自动 synthetic。
