# 项目文档目录

长期运行契约在完成下一阶段迁移前仍保留在仓库根目录，并由 `PROJECT_DOCUMENT_INDEX_CN.md` 统一索引。

本目录保存不会被运行时加载的历史和阶段性文档：

- `versions/`：节点版本、阶段版本和版本 manifest。
- `reports/`：实测报告、横向比较、故障复盘和历史 handoff。
- `architecture/`：阶段架构方案，不自动代表当前生产契约。
- `operations/`：阶段测试计划和项目管理记录。

当前 IR-ID 主动 LED 最小现场实测入口：

- 中文：[operations/IR_ID_V26D_MINIMUM_FIELD_RUN.zh-CN.md](operations/IR_ID_V26D_MINIMUM_FIELD_RUN.zh-CN.md)
- English: [operations/IR_ID_V26D_MINIMUM_FIELD_RUN.en.md](operations/IR_ID_V26D_MINIMUM_FIELD_RUN.en.md)

移动历史文档后必须更新根目录 living docs、`PROJECT_DOCUMENT_INDEX_CN.md` 以及文档内部引用。生产链路变化仍必须按 `CHAIN_DOCUMENTATION_STANDARD_CN.md` 更新对应 living docs。
