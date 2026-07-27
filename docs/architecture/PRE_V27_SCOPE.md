# Pre-V27 original-code scope

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本仓库现在将 V27 之前形成的原创研究代码纳入公开审计范围，但“纳入审计范围”不等于“已获准复制”。机器可读来源见 `provenance/PRE_V27_CUTOFF_MANIFEST.json`，逐文件分类见 `provenance/ORIGINALITY_CLASSIFICATION.csv`。

## 公开目标

允许其他开发者基于同一套原创预览、过滤、几何、回放和候选判定代码，针对不同传感器布局、坐标锚点和游戏规则建立支线研发。未生产启用的 Shadow、Dry-run、失败和撤回实现可以保留，但必须进入研究变体目录并带有状态说明。

## 迁移顺序

1. 先冻结来源 checkout、commit 和文件 hash。
2. 再逐文件证明原创性、许可证和第三方依赖边界。
3. 先公开 Contracts/Replay/Geometry，再公开 Preview/Filters/Fusion，最后公开实验变体。
4. 任何真实 SDK、数据、权重、配置和生产网络写入继续留在私有边界。

## 权威边界

预览、过滤、World State、Polygon、IR-ID 和 Hit Candidate 都是可观察研究事实或候选事实；它们不能被命名、展示或自动升级为 `official_result`、Final Verdict、生产 Judge 或游戏权威。

## 当前状态

G7 已冻结候选来源和排除项；G8 尚未完成逐文件原创性/许可证复核。在 G8 通过前，不得从历史 checkout 批量复制源码到公开主线。

当前 hash-only inventory 共 2,857 条：1,311 条 V27 reference-only、1,536 条待原创性/许可证复核、10 条因 proprietary/private/vendor/secret 命名被阻断。inventory 不包含源码内容，也不代表任何条目已经获准迁移。
