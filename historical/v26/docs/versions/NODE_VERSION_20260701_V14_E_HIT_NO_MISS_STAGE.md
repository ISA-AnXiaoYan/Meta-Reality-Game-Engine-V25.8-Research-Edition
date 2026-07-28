# V14-E 真实命中阶段节点

日期：2026-07-01

## 阶段结论

V14-E 的主要目标是修复事件弹道命中链路中的漏检问题，核心改动包括：

- `hit_judge_server.py` 增加 visual point late-human retry。
- 事件追踪输入从人体 mask 可视化过滤链路中解耦。
- hit reject debug/status 进入常态化输出。
- Global BEV event bullet overlay 支持独立 `flip_y` 变换。

本轮远端真实运动实测中，现场观察结论为：

- 真实命中检测没有再出现漏检。
- 测试提前结束的原因不是漏检，而是 `D/E/F/H` 出现事件弹道检测失效，导致继续射击验证意义下降。
- 当前主要剩余问题变为人体移动产生短轨迹噪声弹道，引起误检。

因此，本节点记录为：

- V14-E 漏检修复方向：通过阶段验收。
- 系统整体终验：不通过，需要进入 V15 继续优化误检和 DEFH 检测失效。

## 远端实测归档

远端目录：

`sync_ipc/full_system_tests/fullsys_v14e_20min_20260701_115018_display1_keepstdin`

远端归档：

`sync_ipc/full_system_tests/fullsys_v14e_20min_20260701_115018_display1_keepstdin.tar.gz`

归档内新增：

- `operator_observation_v14e.json`
- `analysis_v14e_operator.json`
- `hit_candidate_all.jsonl`
- `v14_debug/hit_reject_debug_all.jsonl`

启动结果：

- `launcher_rc=0`
- 手动提前停止，runner 正常收尾归档。

## 关键数据

命中候选总数：

`116`

按相机分布：

```text
A 33
B 29
C 28
D 17
E 1
F 1
G 7
H 0
```

候选几何：

```text
point_in_mask: 108
segment_intersect_mask: 8
```

误检风险量化：

```text
track_path_len < 35px: 105 / 116
track_path_len < 20px: 90 / 116
track_point_count <= 2: 86 / 116
track_duration_ms <= 8ms: 81 / 116
track_path_len p50: 9.584px
```

这个分布说明当前误检主要来自人体移动或人体边缘事件形成的短轨迹片段。这些片段因为进入人体 mask，容易通过 `point_in_mask` 得分直接成为 hit candidate。

## 已确认有效

- V14-E late-human retry 没有阻塞命中输出。
- `event_track_input_policy=raw` 后，事件弹道追踪不会再被人体 mask 可视化过滤直接错杀。
- Global BEV event bullet overlay 在实测中实际绘制，`flip_y` 生效。
- Global YOLO 双 worker 运行稳定，legacy YOLO 明确跳过。

## 剩余风险

1. `D/E/F/H` 事件弹道检测失效。

   现象：
   - D 早期仍有 17 个候选，后段变弱。
   - E/F 各只有 1 个候选。
   - H 没有候选。
   - D/E/F/H 后段 MVS pending queue 约 22-23，有少量 drop。

   初步判断：
   - 不是 hit judge 单点问题。
   - 更可能发生在 event worker 取流、sync map、buffer release、spatter/line filter 输入质量或相机事件量侧。

2. 人体移动噪声误检。

   现象：
   - 多数候选轨迹过短，但进入人体 mask 后得分过高。
   - 需要恢复 visual point hit 的最小轨迹质量门槛。

3. 测试归档脚本此前未默认复制 `hit_candidate_all.jsonl`。

   已在 V15 方向中列为必须修复，避免后续复盘缺关键候选文件。

## V15 优化方向

V15-A：误检抑制

- `visual_point_hit_min_track_points` 从默认 2 提高到 3。
- `visual_point_hit_min_track_path_len_px` 从 0 提高到 20px 起步。
- 启用 `visual_point_hit_body_motion_reject_enable`。
- 目标：人体移动短轨迹误检明显下降，真实命中不恢复漏检。

V15-B：DEFH 检测失效诊断

- 测试归档必须包含：
  - `hit_candidate_all.jsonl`
  - `hit_candidate_*.jsonl`
  - `hit_reject_debug_all.jsonl`
  - `event_loop_profile_*.jsonl`
  - `event_worker_*_status.json`
  - `event_human_mask_stats_*.jsonl`
  - `event_sync_health_*.json`
- 按相机输出候选、reject、raw event rate、track_event_count、mapped_mvs_sync_index、buffer age 的时间线。

V15 验收建议：

- 真实命中不漏检。
- 误检候选显著低于 V14-E。
- D/E/F/H 至少能说明是相机事件输入不足、同步停滞、buffer 等待、line filter 断链，还是 hit judge 后端拒绝。
