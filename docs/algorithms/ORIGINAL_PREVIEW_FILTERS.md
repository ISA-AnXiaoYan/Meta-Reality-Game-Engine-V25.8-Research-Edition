# Original preview and filter algorithms

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本目录中的实现是基于项目历史行为和公开契约的 clean-room 研究实现，不复制供应商 SDK、模型、真实数据或私有运行文件。

当前公开：

- 热像素过滤和明确 reject 统计；
- 邻近 Event cluster；
- Polygon 点数/维度/stale 检查；
- fail-open 的候选门结果；
- 稀疏 Event 预览点物化，不创建完整 raster。

所有输出都属于预览、诊断或候选事实。它们不能替代 RAW、不能生成 `official_result`，也不能直接改变生产 Judge 权威。

后续算法迁移必须补充：来源 commit、clean-room 处理方式、参数 Schema、失败语义、合成样例、确定性摘要和适用条件。
