# 项目 Git 工作流

更新时间：2026-07-09

本文档是 `PROJECT_GIT_WORKFLOW.md` 的中文伴随文档，用于固定 A 测阶段代码管理规则，避免错目录提交、中文路径/引号问题、运行数据污染 Git、以及远端版本不明确。

## 标准仓库

| 项 | 值 |
| --- | --- |
| GitHub remote | `https://github.com/ISA-AnXiaoYan/Meta-Reality-Game-Engine.git` |
| 推荐开发目录 | `C:\dev\meta-reality-game-engine` |
| 旧/过渡目录 | `C:\Users\axyis\Documents\判定开发\项目代码` |

后续开发优先使用 ASCII 路径 `C:\dev\meta-reality-game-engine`。旧中文路径只作为过渡副本和历史参考。

## 提交前 root 检查

每次 staging 前必须执行：

```powershell
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git status --short --branch
```

确认 root 是项目源码目录，不是外层归档目录、日志目录或旧工作区。不要从 `C:\Users\axyis\Documents\判定开发` 外层直接 `git add`。

## 节点版本收口检查清单

每个 node/version commit 前：

1. 确认真实 repo root、branch、remote、status。
2. 确认运行输出没有被 staged：
   - `sync_ipc/`
   - `recordings/`
   - `desktop_runs/`
   - `*.tar.gz`
   - `cpu_hotspots*`
   - `.ssh/`
   - `__pycache__/`
3. 新增或更新节点文档，写清：
   - 分支
   - launch profile
   - 远端 run label
   - run directory
   - archive path
   - 验证摘要
   - 未关闭风险
4. 更新所有受影响的 living chain document，并追加修订记录。
5. 如果状态窗口 key 或 control key 变化，必须写清兼容/迁移计划。
6. 对改动文件运行相应语法或 profile 检查。
7. 如果跑过远端测试，执行 `tools/remote_test.ps1 assess` 并记录结果路径。
8. 从 `C:\dev\meta-reality-game-engine` 提交。
9. 推送当前分支和必要 node tag。

不要把远端机器的 dirty Git checkout 当成节点版本证据。节点版本的源码真相是本地 commit，并且要推送到 `origin`。

## 链路文档规则

链路文档是项目契约，不是可选报告。

- 事件侧变化必须更新 `EVENT_SIDE_CHAIN.md` / `EVENT_SIDE_CHAIN_CN.md`。
- MVS、相机、trigger、frame source 变化必须更新 `MVS_CAMERA_CHAIN.md` / `MVS_CAMERA_CHAIN_CN.md`。
- 跨链路架构变化必须更新模块化架构文档。
- 文档维护格式以 `CHAIN_DOCUMENTATION_STANDARD.md` / `CHAIN_DOCUMENTATION_STANDARD_CN.md` 为准。

如果运行时行为已经改变，但相关链路文档还描述旧路径，不允许收口版本。

## 应该提交什么

可以提交：

- 源码。
- launch profile 和配置模板。
- `remote_ops/` 远端运维脚本。
- `tools/` 中的开发/测试工具。
- living docs、节点文档、测试说明。

不应提交：

- `sync_ipc/` 运行时状态。
- 录制数据和 replay dataset 大文件。
- 远端归档 `.tar.gz`。
- CPU hotspot 采样结果。
- 密钥、证书、`.ssh/`。
- `__pycache__/`、临时构建产物。

## 回退规则

回退不是只删代码，还要写清楚“当前契约是什么”。

回退某个未完成方向时：

1. 恢复或删除对应代码/工具。
2. 删除远端可能残留的新增工具文件。
3. 更新 living docs 修订记录。
4. 新增 rollback node 文档，记录撤回原因和未来重启条件。
5. 远端 smoke 确认没有残留 key、文件、产物。

## 仓库治理护栏

以下文件是机器可读的仓库真相：

- `governance/entrypoints.json`：每个运行角色的唯一 canonical 入口。
- `governance/file_ownership.json`：文件模式、责任域和分类。
- `governance/hygiene_baseline.json`：允许逐步减少、禁止继续增加的历史债务。

每次提交前必须执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\dev_check.ps1
```

检查会阻止新增根目录膨胀、零字节文件、重复源码组、机器绝对路径，以及未被 `.gitignore` 覆盖的运行产物。已有 warning 是待迁移债务，不代表允许继续新增。
