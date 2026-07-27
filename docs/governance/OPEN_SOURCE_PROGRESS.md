# Open-source progress

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本账跟踪公开仓库的开源计划，不把代码已提交、CI 通过、研究资格和生产权威混为一谈。

| 计划阶段 | 当前状态 | 已公开证据 | 尚未声称 |
|---|---|---|---|
| G0 发布授权/基线 | 完成 | `RENAME_APPROVAL.json`、单公开仓库、候选标签 | 不代表法律复核完成 |
| G1 权利/供应商/数据清理 | 研究范围完成 | SPDX、来源清单、无真实数据/权重/SDK、Secret 扫描 | 不代表第三方独立版权审计完成 |
| G2 架构抽取 | 合成研究链完成 | FakeBackend、2/8 路合成、Protocols、NullAdapter/NullBridge | 不代表真实硬件或现场链路完成 |
| G3 可复现研究包 | 公开 CPU 路径完成 | 两个样例、确定性生成器、故障注入、五类研究报告 | 不代表实时性能或命中质量 |
| G4 治理/安全/发布材料 | 完成（仓库范围） | CI、SBOM、REUSE、许可证、贡献/安全/限制文档 | 不代表外部法律意见或漏洞扫描覆盖历史全部对象 |
| G5 公开前候选验证 | 待独立执行 | 本地 R6 checker 和候选材料 | 不把本地 PASS 当作独立评估 PASS |
| G6 正式发布 | 待用户/维护者执行 | Git 标签和 Release Notes 已准备 | 未宣称正式生产发布 |

## 当前公开能力

当前 `main` 可运行合成输入、JSONL Replay、显式故障注入、V26 公共 schema、研究报告和 CPU-only 质量检查。所有输出都保留 `official_result=false` 或 `research-only`/`candidate`/`shadow` 权威边界。

## 下一步门

1. 在不同于开发机的干净环境执行 README、样例、报告和扫描器。
2. 对候选提交、契约、样例和 SBOM 重新计算摘要并记录。
3. 完成独立版权/许可证复核后，再决定是否创建新的正式 Release。
