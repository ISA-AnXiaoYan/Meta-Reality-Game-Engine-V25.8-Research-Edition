# Meta-Reality Game Engine — Research Edition

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

> 面向 BB 弹、水弹、软弹等实体对战玩法的开源研究底座：把多相机感知、空间几何、事件分析与回放组织为可追溯的人体命中研究证据。

Meta-Reality Game Engine（MRGE）用于研究“弹体是否可能接触人体、可能命中谁、发生于何时何地”的可复核链路。它可作为自动计分、命中提示、赛事回放、争议复核和线下元现实玩法的研发基础。

**研究用途声明：** 本仓库不是认证裁判系统、现场安全系统或生产 Authority。候选结果、回放结果和历史记录均不构成正式比赛判定。

## 从这里开始

| 你要做什么 | 入口 |
| --- | --- |
| 5 分钟运行合成示例 | [快速开始](docs/GETTING_STARTED.md) |
| 开发或提交代码 | [开发与贡献](docs/DEVELOPMENT.md) / [CONTRIBUTING.md](CONTRIBUTING.md) |
| 理解当前研究包 | [当前研究架构](docs/architecture/CURRENT_RESEARCH.md) |
| 阅读 V26 历史实现 | [V26 历史架构](docs/architecture/V26_HISTORICAL_ARCHITECTURE.md) |
| 浏览 V26 知识图谱 | [图谱说明与启动脚本](historical/knowledge-graph/README.md) |
| 查看 V1.0 目标设计 | [V1.0 路线图](docs/architecture/V1_0_ROADMAP.md) |

~~~powershell
python -m pip install -e . pytest
mrge validate
mrge generate-sample --output .\out\sample.jsonl --cameras 2 --frames 6
mrge report --input .\out\sample.jsonl --output .\out\reports
~~~

这些命令只使用合成输入；不会启动相机、Event HAL、远端服务或现场进程。

## 版本与支持状态

| 名称 | 状态 | 用途 |
| --- | --- | --- |
| 研究包 `0.2.0.dev0` | 当前可开发 | 合成 Adapter、Contracts、Replay、预览、过滤和 Geometry Profile |
| `historical/v25.8/`、`historical/v26/` | 冻结参考 | 已授权公开的源码、研究记录、配置和部署材料；不作为受支持的启动入口 |
| V26 知识图谱 | 静态导航 | 帮助定位历史模块和关系；不证明运行、资格或 Authority |
| V1.0 架构 | 路线图 | `IMPLEMENTATION IN PROGRESS / NOT QUALIFIED`，不是本仓库已实现功能清单 |

版本规则见 [VERSIONING.md](docs/governance/VERSIONING.md)。稳定可复现版本由 Git tag 与 GitHub Release 表达；默认开发分支为 `main`。

## 许可证与数据边界

- `contracts/`、`replay/` 和 `src/replay/`：Apache-2.0。
- 研究引擎、工具、文档及未另行声明内容：AGPL-3.0-only。
- 历史文件优先遵循其文件内 SPDX 或明确声明；供应商 SDK、二进制、受限权重、凭据和未获授权内容不随仓库发布。
- 历史研究数据的公开可见性不自动授予训练、再识别或二次分发许可；使用前请遵循 [数据政策](docs/data/DATA_POLICY.md) 和 [历史数据目录](historical/DATA_CATALOG.md)。

完整路径映射见 [LICENSE-MAP.md](LICENSES/LICENSE-MAP.md)，第三方与归档边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。闭源或其他商业部署需求请阅读 [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)。

## 仓库结构

~~~text
src/mrge/       当前研究引擎
contracts/      可互操作契约（Apache-2.0）
replay/         回放格式与工具（Apache-2.0）
examples/       网络无关的合成样例
research/       可派生研究 Profile 与变体模板
historical/     冻结的 V25.8–V26 历史档案与 V26 知识图谱
docs/           上手、架构、治理、许可证与数据说明
~~~

历史档案的支持状态、排除项与来源分别见 [historical/README.md](historical/README.md)、[SUPPORT_STATUS.md](historical/SUPPORT_STATUS.md)、[EXCLUSIONS.md](historical/EXCLUSIONS.md) 和 [RELEASE_MANIFEST.json](historical/RELEASE_MANIFEST.json)。

## 贡献与治理

我们欢迎合成样例、可复现研究、契约完善、文档、测试和不依赖私有硬件的 Adapter 改进。请先阅读：

- [GOVERNANCE.md](GOVERNANCE.md)：决策、维护和支持边界；
- [ROADMAP.md](ROADMAP.md)：公开路线图；
- [CONTRIBUTING.md](CONTRIBUTING.md)：提交前检查与 CLA；
- [SECURITY.md](SECURITY.md)：安全报告路径；
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)：社区行为准则。

旧版 V25.8 导览保留在 [docs/README-V25.8-LEGACY.md](docs/README-V25.8-LEGACY.md)，仅作历史阅读参考；请以本 README 和 `main` 分支的文档为准。
