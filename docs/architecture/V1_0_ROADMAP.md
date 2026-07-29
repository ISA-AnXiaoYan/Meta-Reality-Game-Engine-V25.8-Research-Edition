# V1.0 目标架构路线图

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

这是正在推进的目标设计摘要，状态为 IMPLEMENTATION IN PROGRESS / NOT QUALIFIED。它不是 V26 已实现功能清单，也不构成生产切换、真实比赛或裁决 Authority 声明。

~~~mermaid
flowchart TB
    G["Execution Governance / Admission / Lease / Trust"] --> S["Controlled Sensor Ingress / MVS / Event ownership"]
    S --> P["GPU Perception and Fusion / Polygon / Track / Trajectory"]
    P --> E["Evidence Facts / identity-neutral first"]
    E --> L["Canonical Ledger / Sealed Replay"]
    L --> R["Post-match Review / read-only shadow assistance"]
    G -. "no payload ownership" .-> P
~~~

可复用原则：治理平面只管理准入、资源租约和信任域；它不读取图像/Event 负载，也不生成业务裁决。身份、命中存在性、归因和比赛结果必须在独立的来源、资格和 Authority 门之后推进。

公开研究仓库目前只覆盖其中可公开复现的 Contracts、Replay、Synthetic、Preview、Filter 与 Geometry 基础。真实硬件插件、密钥、现场配置和任何生产 Authority 仍在独立受控边界之外。
