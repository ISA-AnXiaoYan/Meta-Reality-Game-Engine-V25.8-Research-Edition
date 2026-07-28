# 仓库治理契约

本目录保存项目文件管理的机器可读真相。治理原则是先阻止污染继续增加，再逐步偿还历史债务。

## 文件

- `entrypoints.json`：生产、可选服务和运维入口的唯一 canonical 路径。
- `file_ownership.json`：目录和文件模式对应的责任域与分类。
- `hygiene_baseline.json`：当前历史债务上限；已有问题只能减少，不能增加。

## 规则

1. 新服务不得直接增加到仓库根目录。
2. legacy alias 只能作为兼容入口，不允许独立增加业务逻辑。
3. 运行数据、模型、录像、归档、密钥和临时构建产物不得进入 Git。
4. 新代码不得增加 `/home/ysxq/` 等机器绝对路径。
5. 新增或迁移入口时必须同步更新 `entrypoints.json`。
6. 文件责任域变化时必须同步更新 `file_ownership.json`。

执行检查：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\dev_check.ps1
```

或单独运行：

```bash
python3 tools/check_repo_hygiene.py
```
