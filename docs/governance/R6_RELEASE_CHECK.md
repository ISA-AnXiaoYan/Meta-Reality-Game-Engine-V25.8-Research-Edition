# R6 release check

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

R6 是 Rename 验收与研究版候选标签门，不是生产资格门。

检查脚本：

```powershell
python tools/r6_release_check.py
```

必须同时满足：

- Rename 审批记录为 `APPROVED`，覆盖 68 个映射项；
- 根目录只有一个 `pyproject.toml`；
- 代码和配置文件有 SPDX 标识；
- 无绝对路径、密钥、真实数据、模型权重或真实硬件 Adapter；
- `RESEARCH_RELEASE_MANIFEST.json` 继续声明 `qualification_claimed=false`、`production_cutover=false`；
- 研究版 CLI 和 JSONL 回放在 CPU 环境可运行。

本次候选标签只代表公开研究包边界已冻结，不代表现场有效性、生产判定或最终资格通过。
