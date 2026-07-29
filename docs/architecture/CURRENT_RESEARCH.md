# 当前研究包架构

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

当前公开、受支持的研究表面是合成输入、公开契约、Replay、过滤、几何与报告。它不直接拥有真实硬件、赛事状态或生产 Authority。

~~~mermaid
flowchart LR
    S["Synthetic Adapter"] --> P["Research Pipeline"]
    P --> C["Polygon / Geometry Contracts"]
    P --> F["Preview and Filters"]
    C --> R["Replay JSONL"]
    F --> R
    R --> O["Research Reports / official_result = false"]
~~~

开发者可以替换场地几何、相机布局假设和合成输入，但必须把输入、时间基准、锚点、输出事实和消费者版本化。YOLO、人体轮廓、弹道或事件候选都不是最终比赛判定。
