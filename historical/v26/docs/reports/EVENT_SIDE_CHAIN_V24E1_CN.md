# V24-E1 事件侧显示策略中文记录

日期：2026-07-07

本节点记录 `event_visual_mask_slice_us` / `event_visual_mask_accumulation_us` 热切换，以及 Global BEV 事件层同步判定改为只看事件窗口结束点的策略。该策略必须在后续链路更新中保留。

## Visual Mask 时间窗口

启动默认值：

| Key | 默认值 |
| --- | --- |
| `event_visual_mask_slice_us` | `33333` |
| `event_visual_mask_accumulation_us` | `33333` |

状态窗口“控制按钮 / 运行参数”页新增两个控件：

| 控件 | 写入字段 |
| --- | --- |
| `visual slice ms` | `event_visual_mask_slice_us` / `event_visual_mask_slice_ms` |
| `visual acc ms` | `event_visual_mask_accumulation_us` / `event_visual_mask_accumulation_ms` |

控制文件：

```text
sync_ipc/event_mask_control.json
```

C++ Native HAL broker 运行中轮询该控制文件，只更新显示专用 visual mask 的 slice/accumulation 参数，不重启相机，不改变 4ms raw FIFO，不改变弹道追踪和命中判定。

## BEV 事件层同步策略

必须保留的策略：

```text
Global BEV event layer display sync timestamp = event_window_end_us
```

说明：

- `event_overlay_masked_latest_{camera}` 和 sparse history 是累计窗口显示结果。
- 运营侧实际看到的是窗口结束时刻发布出的视觉帧。
- 如果用 `event_window_center_us` 做 MVS sync 映射，会天然引入半个累计窗口偏差，在 20ms BEV 显示同步门槛附近制造假 outlier。
- 因此 Global BEV 事件层候选排序、sync-ring 映射、delta 判定都必须使用 `event_window_end_us`。
- `event_window_start_us` 和 `event_window_center_us` 只作为诊断字段保留。

## 状态字段要求

必须能在状态中看到：

```text
global_bev_status.json.event_layer.sync_timestamp_policy = event_window_end_us
event_layer.per_camera.*.event_layer_sync_timestamp_policy = event_window_end_us
```

HAL broker visual mask overlay 状态需要能看到：

- 当前生效 `slice_us`
- 当前生效 `accumulation_us`
- `control_path`
- `control_seq`
- `control_apply_count`
- `control_error`

## 远端实测记录

2026-07-07 已完成三轮远端验证：

| 测试标签 | 结论 |
| --- | --- |
| `v24e1_visual_mask_endts_20260707_205453_display1_keepstdin` | 10 分钟主测正常结束，`launcher_rc=124`，全系统运行状态 OK，BEV 事件层策略为 `event_window_end_us`。 |
| `v24e1_visualmask_hotseq_rebuild_20260707_211629_display1_keepstdin` | 热切换专项测试：运行中从 33.333ms 切到 25ms，再切回 33.333ms；8 路 HAL visual mask 均记录 `control_apply_count=2`。 |
| `v24e1_status_link_confirm_20260707_212537_display1_keepstdin` | 状态链路确认：`system_status_latest.json` 已能看到 `global_bev.event_layer.sync_timestamp_policy=event_window_end_us`，每路 HAL visual mask 已能看到 `slice_us / accumulation_us / control_seq / control_apply_count / sync_timestamp_policy`。 |

实测中修复过一个状态字段 bug：C++ broker 原先用 32 位整数解析 `event_mask_control.json.seq`，会让 `visual_mask_control_seq` 溢出为负数；现在改为 64 位解析。这个 bug 不影响 visual mask timing 生效，但会误导热切换状态判断。

剩余风险：严格 20ms BEV 事件层显示同步门槛下，A/G 等相机偶发 outlier 仍存在。这属于 BEV 显示同步鲁棒性问题，不是 raw Event 取流、弹道跟踪或 final hit 判定变更。

## 边界

本改动只影响显示链路：

- 不改变 Event HAL 取流频率。
- 不改变 Event worker raw tracking。
- 不改变 pseudo hit / final hit。
- 不改变 Game Bridge 输出。
- 不改变命中判定 100ms 同步门槛。
