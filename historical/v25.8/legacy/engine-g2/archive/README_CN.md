# 历史快照目录

本目录只保存不参与运行时解析的历史快照。

- `config_snapshots/`：历史配置快照。
- `source_snapshots/`：历史源码快照。
- `legacy_entrypoints/`：为后续 G3 迁移预留；迁入前必须先建立 canonical 入口和兼容包装。

运行脚本、launch profile 和远端部署工具不得直接引用 `archive/`。需要回退时应通过 Git commit/tag 恢复，不应把这里的快照重新变成第二份可独立维护的生产源码。
