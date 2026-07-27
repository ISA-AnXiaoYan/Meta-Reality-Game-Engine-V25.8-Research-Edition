# Meta-Reality-Game-Engine V25.8 Research Edition

研究版元现实游戏引擎：以可复现的合成输入、契约和回放为首发边界。

本仓库当前推进至 R5。它不包含真实硬件 Adapter、真实数据、模型权重、供应商 SDK 或生产判定权威。

## 快速开始

```powershell
python -m pip install -e .
mrge validate
mrge simulate --frames 3
mrge replay --input examples/replay.jsonl
```

目录许可证边界见 [`docs/governance/RENAME.md`](docs/governance/RENAME.md)。Contracts 与 Replay 使用 Apache-2.0；其余研究引擎代码使用 AGPL-3.0-only。每个源文件均带 SPDX 标识。

## 当前状态

这是 V25.8 的研究版发布候选工作区，不是生产系统，也不代表现场有效性、资格通过或生产切换。
