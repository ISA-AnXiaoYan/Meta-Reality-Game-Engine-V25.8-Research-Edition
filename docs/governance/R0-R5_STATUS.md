# R0–R5 execution status

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本文件记录 `Meta-Reality-Game-Engine-V25.8-Research-Edition` 的公开仓库推进状态。

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0 审批冻结 | 进行中 | 用户已授权推进至 R6 前；逐项 CSV 终态仍留给 R6 |
| R1 空仓骨架 | 完成 | `pyproject.toml`、目录、许可证和 SPDX 规则 |
| R2 Apache 区域 | 完成 | `src/mrge/contracts.py`、`src/replay/` |
| R3 AGPL 研究引擎 | 完成 | `src/mrge/engine/`、`mrge` CLI |
| R4 研究 Adapter/Perception | 完成（合成边界） | synthetic Adapter；没有真实硬件或模型 |
| R5 Legacy 语义对齐 | 研究版基线完成 | `RESEARCH_RELEASE_MANIFEST.json` 与候选语义字段 |
| R6 Rename 验收 | 未执行 | 需要最终 SPDX、SBOM、secret、绝对路径扫描和 CSV 终态 |

## 权威边界

所有推理输出都是 `candidate`。本仓库不提供 `official_result`、生产判定、现场有效性或生产切换声明。
真实硬件 Adapter、真实数据、模型权重、供应商 SDK、网络写入和运行时归档均不在首发范围。
