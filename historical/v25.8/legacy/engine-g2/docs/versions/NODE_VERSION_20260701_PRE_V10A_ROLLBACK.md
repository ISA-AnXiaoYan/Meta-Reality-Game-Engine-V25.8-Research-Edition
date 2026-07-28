# Node Version: Pre V10-A Rollback

日期：2026-07-01

## 节点结论

项目已回退到 V10-A 实施前备份节点：

- 来源备份：`sync_ipc/code_backups/v10a_20260630_224333`
- 本地 staging：`remote_tests/v10a_20260630_224333_pre_restore`
- 本地回退前备份：`项目代码/_local_backup_before_pre_v10a_restore_20260701_0010`

本节点放弃后续两条实测后未达目标的路线：

- V10-A：以 Global YOLO fast mmap / 当前 sync 严格匹配为核心的两帧滞后尝试。
- bboxfast：Global YOLO 内 bbox-only 快速发布线程。

## 当时的优化方向选择回顾

当时的主问题是 Event worker 使用当前 MVS/event sync 过滤事件，但人体 mask 结果经 YOLO 生产后经常落后当前 sync，多数情况下被 `max_lag_frames=2` 策略判定为 stale。

当时比较过的方向：

- Event 侧 buffer/等待：低侵入，可提高命中同帧 mask 的机会，但会引入事件显示/弹道链路延迟。
- Global YOLO fast mmap：绕过 legacy JSONL 轮询和文件覆盖问题，目标是让 Event worker 更早拿到最新人体结果。
- Global YOLO bbox-only 快发：进一步绕过 polygon 生成和完整后处理，只先发布人体框。
- 调整 YOLO 微批/partial batch：更可能触及根因，但会影响全局 YOLO 吞吐和多相机同步节奏。
- 放宽 Event 侧 lag 策略：能提高应用率，但会牺牲“当前帧硬过滤”的同步严格性。

V10-A 当时被选择，是因为它风险相对可控，保留当前帧硬过滤语义，同时尝试减少 JSONL/发布路径延迟。但后续实测证明，fast mmap 和 bbox-only 快发虽然能接通并减少发布链路不确定性，却没有把整体 lag 从约 7 帧压到 2 帧，说明主瓶颈不在发布格式，而在 YOLO 结果生产节奏与 Event 当前 sync 的相位差。

## 后续建议

如果继续追求两帧内 mask，应优先回到方向选择层重新决策：

- 要严格两帧：重点做 YOLO 微批/调度/取帧相位优化。
- 要稳定过滤效果：考虑 Event 侧有界等待或放宽 lag，但明确接受延迟/同步严格性代价。
- 要先保系统稳定：保持当前 Pre V10-A 节点，暂不继续 mask 两帧优化。
