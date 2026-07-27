# Contributing

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

感谢参与 MRGE 研究版建设。提交代码前请确认：

1. 不提交真实硬件 SDK、真实数据、模型权重、凭证或绝对路径。
2. 新增源码带有正确的 SPDX 标识；Contracts/Replay 与 Engine 的许可证边界不能混用。
3. 所有跨模块字段变化同步更新契约、Replay 样例、测试和 `SOURCE_PROVENANCE.json`。
4. 候选、Shadow、qualified、`authority_ready` 和生产权威必须分开描述。
5. 本地运行 `python tools/r6_release_check.py` 和 `python -m pytest -q`。

提交说明应包含变更范围、验证命令、证据路径和未解决边界。不要把局部 smoke/replay 结果描述为现场终验或生产就绪。
