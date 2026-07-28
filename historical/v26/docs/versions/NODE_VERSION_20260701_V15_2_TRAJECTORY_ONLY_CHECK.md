# V15.2 Trajectory-Only 节点检查版

日期：2026-07-01

## 节点定位

本节点是 V15.2 的阶段性检查版，不定义为最终验收版。

核心目标是确认 V15 后端出现子弹轨迹数据中断/堵塞后，新的修复方向能否把弹道候选从人体 mask 强依赖链路中解耦出来，并保证全系统能够稳定启动、运行和归档。

本节点的判断边界：

- 已验证：配置生效、Event worker 启动稳定、Event 到 hit_judge 候选链路恢复、远端 5 分钟全系统流程正常结束。
- 未验证：真实射击命中召回率、误检率最终水平、长时间压力下候选质量、现场复杂人体运动下的误检抑制。

## 本阶段主要变化

### 1. hit_judge 最终判定门槛切换

最终判定模式从依赖 mask / trigger / person sync 的综合证据，切换为：

```text
hit_judge_final_mode = trajectory_only
trajectory_only_final_enabled = true
```

当前 trajectory-only 门槛：

```text
min_track_points = 5
min_track_path_len_px = 80.0
min_duration_ms = 15.0
min_straightness = 0.85
max_line_error_px = 12.0
min_speed_px_per_ms = 0.6
min_approach_len_px = 45.0
min_point_segment_len_px = 1.0
allow_unknown_line_error = true
min_score = 0.38
```

目的：

- raw tracking 继续产生更稳定的弹道候选。
- final hit 不再把人体 mask / trigger / person sync 作为必须证据。
- 人体 mask 从“判定阻断条件”降级为显示/分析/诊断信息，避免 mask 延迟或缺失直接掐断弹道链路。

### 2. Event human mask 改为 visual_only

运行配置中：

```text
event_human_mask_mode = visual_only
event_track_input_policy = raw
event_human_mask_buffer_release_policy = strict_same_sync_drop
event_tsf1_pub_max_fps = 83.3333333333
```

含义：

- Event worker 的 STC / spatter / line_filter 输入使用 raw event。
- 人体 mask 仍参与可视化、状态、统计、诊断。
- `visual_only` 下 mask 不再丢弃 raw tracking 输入，因此状态中的 drop 为 0 是预期结果。
- strict same-sync buffer 仍服务于可视化 mask 对齐，不再阻塞 raw tracking。

### 3. V15.2 启动阻断修复

本轮远端实测前后发现并修复了两个直接 bug：

1. CLI 参数 choices 漏配。

   现象：

   ```text
   argument --event-human-mask-mode: invalid choice: 'visual_only' (choose from 'hard_events')
   ```

   修复：

   `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py` 的 `--event-human-mask-mode` 增加：

   ```text
   visual_only / visual-only
   stats_only / stats-only
   observe / soft
   off / none / disabled / disable
   ```

2. EventViz mainloop 统计字段未初始化。

   现象：

   ```text
   KeyError: 'buffer_release_viz_only_pushed'
   ```

   修复：

   - 初始化 `event_viz_mainloop_stats['buffer_release_viz_only_pushed'] = 0`。
   - 递增处改为 `get(..., 0) + 1`，防止后续合并遗漏初始化再次导致 worker 退出。

## 远端实测记录

有效实测归档：

```text
sync_ipc/full_system_tests/fullsys_10min_20260701_164420_display1_keepstdin.tar.gz
```

测试时长：

```text
310s
```

启动结果：

```text
launcher_rc = 124
```

说明：`124` 是 timeout 到时结束，属于本测试流程的正常退出。

远端修改前备份：

```text
sync_ipc/code_backups/v15_2_before_5min_20260701_162753
```

无效尝试记录：

```text
fullsys_10min_20260701_162935_display1_keepstdin
原因：正式 profile 实际读取 ids_8cam_fusion_config_627_runtime.json，首次只改了 ids_8cam_fusion_config.json，hit_judge 仍是旧模式。

fullsys_10min_20260701_163407_display1_keepstdin
原因：配置已生效，但 Event worker CLI 不接受 visual_only。

fullsys_10min_20260701_163812_display1_keepstdin
原因：CLI 已修复，但 visual_only buffer release 统计字段 KeyError 导致 Event worker 退出。
```

有效 run：

```text
fullsys_10min_20260701_164420_display1_keepstdin
```

## 有效 run 关键结果

### Event worker

运行中观察：

```text
A-H 共 8 路 Event worker 均 state=active
loop_fps 约 230-270Hz
cpp=True
overlay=True
mask=True
sync lock=LOCKED
```

Event 日志关键错误计数：

```text
Traceback = 0
KeyError = 0
invalid choice = 0
exit code 1 = 0
```

这说明前面两个启动阻断已经在有效 run 中消失。

### hit_judge

最终状态：

```text
hit_judge_final_mode = trajectory_only
trajectory_only_final_enabled = true
hit_candidate_total = 38
```

阶段观察：

```text
hit_candidate_total: 26 -> 37 -> 38
```

这说明 Event -> hit_judge 的候选链路已经恢复，不再是前一阶段的候选断流状态。

主要 reject 分布：

```text
point_track_path_too_short
point_track_points_too_small
point_trajectory_only_track_points_too_small
point_trajectory_only_path_len_too_short
point_segment_too_short
point_duplicate_hit_ignored
```

这符合无明确真实射击输入时，短轨迹/弱轨迹噪声被过滤掉的预期。

### MVS / sync

运行中观察：

```text
MVS A-H 基本维持 40Hz
trigger_ok 持续增长
sync lock=LOCKED
```

局部日志中存在个别 `trig_delta_us` 瞬时偏高，但未观察到系统级断链或 launcher 中断。

### OBS-SRT

检查结果：

```text
restart_backoff = 0
Address already in use = 0
Broken pipe = 0
```

OBS 每路有一次 ffmpeg start，结束时正常 close。未发现反复重启问题。

## 当前结论

V15.2 检查版通过“链路恢复与稳定启动”检查。

可以确认：

- 正式运行配置已经切到 `trajectory_only`。
- Event human mask 已经降级为 `visual_only`。
- raw tracking 不再被 mask buffer / mask 缺失直接阻断。
- Event worker 不再因 `visual_only` 参数或统计字段异常退出。
- 远端 5 分钟全系统 run 能正常结束和归档。
- Event -> hit_judge 候选链路恢复，最终累计候选不再是 0。

不能确认：

- 真实射击不漏检。
- 人体移动噪声不会误检。
- 长时间运行后 display path / overlay bullet point 不会再次后期中断。
- `trajectory_only` 阈值已经是最终最优。

## 剩余风险

### 1. trajectory_only 可能降低误检抑制证据

当前 final hit 不再强依赖 mask / trigger / person sync，优点是避免同步链路和 mask 延迟掐断真实弹道，风险是人体移动噪声如果形成足够长、足够直的轨迹，也可能进入最终候选。

后续必须用真实射击 + 人体运动干扰数据验证。

### 2. 状态输出仍有口径差异

runner 中 `[HIT] recv=0 cand=0` 是窗口统计输出；`hit_judge_status.json` 中 `hit_candidate_total=38` 是累计状态。

这容易让现场判断产生误读。建议后续把 hit_judge perf log 增加累计字段，或在状态窗口中明确区分：

```text
window_recv / window_cand
total_recv / total_candidate
```

### 3. display path 仍需继续诊断

V15 之前已经观察到后期弹道不显示，原因更接近 `line_filter.get_draw_paths(False)` 后期不再产出 display trajectory points，而不是 Preview / Global BEV 绘制崩溃。

V15.2 本轮确认 raw tracking 和 hit_judge 候选恢复，但还不能证明长时间 display path 不会再断。

### 4. 本次不是命中率验收

本次 5 分钟 run 主要用于启动阻断修复和链路恢复验证。如果现场没有真实射击或真实射击样本不足，不应用它评估漏检率。

### 5. 正式 profile 配置源仍是高风险点

本轮再次确认正式启动读取：

```text
launch_profiles/627_event_overlay_bev.json
config = ids_8cam_fusion_config_627_runtime.json
```

后续所有参数修改必须同时确认：

- 源配置文件
- launcher 生成的 `_runtime_config`
- 子进程实际命令行

只改 `ids_8cam_fusion_config.json` 可能不会影响正式 run。

## 下一阶段建议

### A. 真实射击验收

使用当前 V15.2 检查版进行 10-20 分钟真实射击测试。

重点记录：

- 人工真实射击时间点。
- 命中是否漏检。
- 是否出现人体移动误检。
- `hit_candidate_total` 与实际命中数量的关系。
- `trajectory_only` reject reason 分布。

### B. display path 长时间稳定性

继续观察：

```text
draw_paths_count
draw_path_points_count
display_eligible_count
active_track_count
active_but_not_display_count
last_overlay_bullet_point_age_ms
bullet_sender_queue_size
bullet_sender_dropped
```

目标是确认：

- late run 是否仍会出现 overlay bullet point 长时间不更新。
- 如果出现，是 raw tracking 没产出、line_filter 不显示、sender 没发送，还是 preview/BEV TTL 清空。

### C. trajectory_only 阈值分层

如果出现误检，优先不要回退到 mask 强门槛，而是分层调高轨迹质量证据：

```text
min_track_path_len_px
min_duration_ms
min_straightness
max_line_error_px
min_approach_len_px
duplicate suppression
```

建议一次只改一个主阈值，避免同时改变导致难以归因。

### D. 状态窗口口径修正

把 `window` 与 `total` 统计拆开显示，降低现场误判。

建议优先加：

```text
hit_judge.window_recv
hit_judge.window_candidate
hit_judge.total_recv
hit_judge.total_candidate
event.last_overlay_bullet_point_age_ms
event.display_eligible_count
```

## 节点判定

本节点状态：

```text
V15.2 Trajectory-Only Check: 通过链路检查，未通过最终验收
```

建议作为下一轮真实射击测试的基线版本。
