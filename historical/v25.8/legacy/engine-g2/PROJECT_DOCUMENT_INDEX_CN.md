# 项目主要文档中文索引

更新时间：2026-07-10

本文档给新同事一个可读入口。原则是：中文解释业务和操作语义，英文文件名、JSON key、状态 key、控制 key 保持不翻译，方便和代码、脚本、日志直接对应。

## 建议阅读顺序

| 顺序 | 中文文档 | 原始/配套文档 | 适合对象 | 先读目的 |
| --- | --- | --- | --- | --- |
| 1 | `PROJECT_GIT_WORKFLOW_CN.md` | `PROJECT_GIT_WORKFLOW.md` | 所有开发人员 | 明确仓库、分支、提交、远端部署和禁止提交内容 |
| 1.1 | `governance/README_CN.md` | `governance/*.json` | 所有开发人员 | 理解 canonical 入口、文件责任域和“历史债务只减不增”护栏 |
| 2 | `REMOTE_TEST_WORKFLOW_CN.md` | `REMOTE_TEST_WORKFLOW.md` | 现场测试/开发 | 学会稳定启动、停止、拉取、评估远端实测 |
| 3 | `V25_MAIN_TASK_TREE_20260709.md` | 无 | V25 开发/测试 | 跟踪启动控制面板、Event/MVS 离线 broker、manifest、launcher/profile 和测试工具链的主线计划 |
| 3.1 | `docs/versions/NODE_VERSION_20260709_V25_ADE_STATUS_WINDOW_CLOSURE.md` | 无 | V25 收口复盘 | 记录 V25-A/D/E source-mode 状态窗口 key 对齐、归档补齐和节点风险 |
| 3.2 | `docs/versions/NODE_VERSION_20260710_V25_STARTUP_CONTROL_PANEL.md` | 无 | V25 启动控制面板 | 记录远端桌面入口、受管一键启动、dry-run 验证边界与回退方式 |
| 3.3 | `docs/versions/NODE_VERSION_20260710_G0_G2_REPOSITORY_GOVERNANCE.md` | `governance/*.json` | 仓库治理 | 记录 G0-G2 护栏、目录整理、量化成果和未关闭债务 |
| 4 | `CHAIN_DOCUMENTATION_STANDARD_CN.md` | `CHAIN_DOCUMENTATION_STANDARD.md` | 所有会改链路的人 | 明确版本收口时必须更新哪些链路文档 |
| 5 | `MVS_CAMERA_CHAIN_CN.md` | `MVS_CAMERA_CHAIN.md` | 感知/同步/可视化开发 | 理解可见光 MVS 链路、40Hz trigger 和下游消费者 |
| 6 | `EVENT_SIDE_CHAIN_CN.md` | `EVENT_SIDE_CHAIN.md` | 事件相机/命中开发 | 理解 Event HAL、Event worker、弹道、mask、hit evidence |
| 7 | `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704_CN.md` | `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704.md` | 架构/模块化开发 | 理解多进程模块边界、数据契约和控制面 |
| 8 | `GLOBAL_YOLO_C3_ARCHITECTURE_CN.md` | `GLOBAL_YOLO_C3_ARCHITECTURE.md` | YOLO/人员定位开发 | 理解 Global YOLO 输入、输出、polygon、anchor 和性能边界 |
| 9 | `EVIDENCE_REPLAY_DATASET_V1_CN.md` | `EVIDENCE_REPLAY_DATASET_V1.md` | 数据集/回归测试开发 | 理解 Evidence Replay 能复盘什么、不能验证什么 |
| 10 | `GAME_EVENT_BRIDGE_HANDOFF_20260703_CN.md` | `GAME_EVENT_BRIDGE_HANDOFF_20260703.md` | 游戏进程对接开发 | 理解向游戏管理系统发送 presence/hit 的契约 |
| 11 | `PERIODIC_MONITORING_CHECKLIST_CN.md` | `PERIODIC_MONITORING_CHECKLIST.md` | 现场测试/运维 | 建立启动后、短测、长测、异常后的巡检顺序 |

## 当前必须长期维护的 living docs

这些文档不是临时报告，而是项目运行契约。只要对应链路变化，就必须同步更新：

- `EVENT_SIDE_CHAIN.md` / `EVENT_SIDE_CHAIN_CN.md`
- `MVS_CAMERA_CHAIN.md` / `MVS_CAMERA_CHAIN_CN.md`
- `REMOTE_TEST_WORKFLOW.md` / `REMOTE_TEST_WORKFLOW_CN.md`
- `PROJECT_GIT_WORKFLOW.md` / `PROJECT_GIT_WORKFLOW_CN.md`
- `CHAIN_DOCUMENTATION_STANDARD.md` / `CHAIN_DOCUMENTATION_STANDARD_CN.md`
- `PERIODIC_MONITORING_CHECKLIST.md` / `PERIODIC_MONITORING_CHECKLIST_CN.md`
- `V25_MAIN_TASK_TREE_20260709.md`，直到 V25-G 节点收口前作为 source-mode/replay 主线跟踪文档
- 跨链路架构文档：`MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704.md` / `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704_CN.md` 或后续替代文档

## 历史文档目录

- `docs/versions/`：节点版本与阶段版本记录。
- `docs/reports/`：测试报告、横向比较和历史问题复盘。
- `docs/architecture/`：尚未升级为 living contract 的阶段架构设计。
- `docs/operations/`：阶段测试计划和项目管理记录。
- `archive/`：不参与运行的配置与源码快照。

## 版本收口时的文档规则

一个节点版本不能只靠“代码跑了”收口。必须同时满足：

1. 本地仓库 root、branch、remote、status 已确认。
2. 远端测试使用 `tools/remote_test.ps1` 或 `remote_ops/` 标准流程。
3. 运行时证据、归档目录、测试标签和风险点写入节点文档。
4. 改到哪条链路，就更新对应 living doc 的修订记录。
5. 状态窗口 key、控制 key、JSON 字段含义有变化时，必须写清兼容关系。

## 给新同事的注意点

- 不要只看 launch profile 推断真实运行状态。`launch_fusion_system.py` 会生成 runtime config 和实际 child process args。
- 不要把 `sync_ipc/`、录制数据、远端归档、CPU hotspot、`.ssh/`、`__pycache__/` 提交到 Git。
- 不要手写复杂 PowerShell/SSH 一行命令。远端实测优先走 `tools/remote_test.ps1`。
- UI 可以显示中文分类，但 JSON key 必须保持稳定英文。
- `stable_track_id`、`global_runtime_id`、`global_id_int`、`source_yolo_id` 等名字不能混用。具体含义要看对应链路文档。
