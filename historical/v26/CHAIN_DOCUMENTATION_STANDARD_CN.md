# 链路文档维护标准

更新时间：2026-07-09

本文档是 `CHAIN_DOCUMENTATION_STANDARD.md` 的中文伴随文档。它规定：凡是会改变运行时行为或操作员理解的链路变化，都必须同步更新对应链路文档和修订记录，否则不能收口版本。

## 适用范围

规则适用于所有 production、shadow、可视化、replay、测试链路，只要变化会影响运行行为、状态解释或现场判断。

当前必须维护的 living chain documents：

| 链路 | 必须维护文档 | 负责范围 |
| --- | --- | --- |
| Event side | `EVENT_SIDE_CHAIN.md` / `EVENT_SIDE_CHAIN_CN.md` | Event HAL、Event worker、event overlay、bullet point、pseudo hit、hit judge 输入输出 |
| MVS camera side | `MVS_CAMERA_CHAIN.md` / `MVS_CAMERA_CHAIN_CN.md` | MVS trigger、MVS worker、MVS shm、YOLO/Preview/BEV 消费者 |
| Full system architecture | `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704.md` / `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704_CN.md` 或后续替代文档 | 跨链路拓扑、模块边界、契约 |
| Remote test workflow | `REMOTE_TEST_WORKFLOW.md` / `REMOTE_TEST_WORKFLOW_CN.md` | 启动、停止、部署、测试、summary、归档 |
| Git workflow | `PROJECT_GIT_WORKFLOW.md` / `PROJECT_GIT_WORKFLOW_CN.md` | 版本收口和仓库规则 |
| Periodic monitoring checklist | `PERIODIC_MONITORING_CHECKLIST.md` / `PERIODIC_MONITORING_CHECKLIST_CN.md` | 周期巡检、热点定位、测试后证据复盘 |

## 必须更新的内容

当一个改动触碰链路时，开发者必须更新：

1. 受影响的链路文档。
2. 链路文档的 revision log。
3. 状态窗口或 `system_status_latest.json` 中变化的 status/control key。
4. 收口版本时的 node version 文档。
5. 对应中文伴随文档，若该文档存在。

即使改动很小，也适用这条规则。例如：

- 默认周期从 16ms 改到 33.33ms。
- 某个 key 改名或含义变化。
- authority 从 `strong_gate` 改成 `strong_reasoning`。
- 输出从共享 append JSONL 改到 run-scoped output。

## 什么算链路变化

以下都必须更新链路文档：

- 进程启动顺序或 launch profile 默认值变化。
- runtime config 生成逻辑或 child process args 变化。
- 输入源、输出文件、shared memory 名称、UDP/TCP 端口、FIFO 路径、sidecar JSON 路径变化。
- status key、control key、字段语义、状态窗口分类变化。
- authority 边界变化，例如 `strong_gate`、`strong_reasoning`、`manual_only`、`publish`、`shadow`、`display_only`。
- 数据周期、频率、缓冲、丢弃策略、stale threshold、sync source 变化。
- 影响 hit、person ID、tracking、Preview、BEV、Game Bridge 的过滤、evidence、dedupe、fallback 策略变化。
- 远端测试流程或数据集录制行为变化。

## 链路文档模板

每个链路文档至少包含：

1. Purpose and authority boundary。
2. Current production path。
3. Data-flow diagram 或有序数据流。
4. Inputs、outputs、state files、control files。
5. Runtime profile defaults 和 hot-update controls。
6. Status-window 和 system-status keys。
7. Health checks 和 common failure points。
8. Revision log。
9. Open risks 和 next validation needs。
10. 本轮踩坑复盘，以及对应的工作模式修正。

## 修订记录字段

每条 revision log 至少包含：

| 字段 | 必须写什么 |
| --- | --- |
| Date | 绝对日期，例如 `2026-07-09` |
| Version or node | 分支、节点文档、或短变更名 |
| Changed chain | Event、MVS、Hit Judge、Global BEV、Game Bridge 等 |
| Change summary | 运行时行为改了什么 |
| Contract impact | 对上下游、状态窗口、操作员判断有什么影响 |
| Validation | 本地检查、远端 run、数据集回放、或尚未验证 |
| Risks | 未关闭风险、回退条件、后续验证需求 |

## 收口禁止项

出现以下情况，不允许收口版本：

- 代码改了链路，但 living doc 没更新。
- 状态窗口显示字段变了，但没有写兼容/迁移说明。
- 远端运行用的是 dirty checkout，却当成正式版本证据。
- 测试归档路径、run label、profile、commit 没写清。
- 把 runtime output、录制数据、密钥、热点采样误提交到 Git。
- 远端测试中暴露的坑只停留在聊天记录，没有写入节点文档和流程文档。

## 复盘规则

每轮远端验证结束后，必须在当前节点文档里写一段短复盘，回答：

1. 本轮开发、部署、启动、停止、summary、现场观察中哪里失败或险些失败。
2. 问题属于工具链/报告口径、旧 runtime 残留、source-mode 混淆、真实数据链路、
   性能压力，还是现场操作流程。
3. 下一轮用什么固定工作模式或工具修改避免复发。
4. 本轮版本目标是通过、部分通过，还是失败。结论必须引用 run label 和决定性 status 字段。

如果这些答案只存在聊天里，没有落到项目文档，就不能作为版本交接或 Git 收口依据。
