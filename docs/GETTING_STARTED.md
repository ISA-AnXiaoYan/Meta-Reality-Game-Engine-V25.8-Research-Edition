# 快速开始

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本指南只运行公开的合成输入，不会连接真实相机、Event 设备、远端主机或游戏服务。

## 前置条件

- Python 3.10 或更高版本
- Git
- 可选：Node.js 20+，仅在需要打开 V26 静态知识图谱时使用

## 安装与自检

~~~powershell
git clone https://github.com/ISA-AnXiaoYan/Meta-Reality-Game-Engine-Research-Edition.git
Set-Location Meta-Reality-Game-Engine-Research-Edition
python -m pip install -e . pytest
mrge validate
python -m pytest -q
~~~

预期的 mrge validate 输出包含 authority: research-only。

## 生成并分析一个双相机合成样例

~~~powershell
New-Item -ItemType Directory -Force .\out | Out-Null
mrge generate-sample --output .\out\sample.jsonl --cameras 2 --frames 6
mrge report --input .\out\sample.jsonl --output .\out\reports
Get-ChildItem .\out\reports
~~~

输出是研究报告和候选级事实，official_result 始终为 false。不要把它解释为人体命中、比赛计分或真实设备质量结论。

## 回放与故障注入

~~~powershell
mrge replay --input .\examples\quickstart_2cam_synthetic.jsonl --fault duplicate_frame
~~~

这用于检查 Replay 在重复帧、乱序帧、过期 Polygon 等研究故障下的行为；并不重跑真实 YOLO 或 Event 模型。

## V26 知识图谱

图谱是冻结历史代码的静态导航工具。Windows 下可执行：

~~~powershell
.\tools\open_v26_knowledge_graph.ps1
~~~

macOS/Linux 使用 ./tools/open_v26_knowledge_graph.sh。脚本只复制公开 JSON 并启动一个仅监听本机回环地址的查看器；完整边界和依赖说明见 [图谱 README](../historical/knowledge-graph/README.md)。

## 下一步

- 修改几何假设：[Geometry Profile](architecture/GEOMETRY_PROFILES.md)。
- 派生研究支线：[研究变体约定](../research/variants/README.md)。
- 阅读历史实现：[V26 历史架构](architecture/V26_HISTORICAL_ARCHITECTURE.md)。
