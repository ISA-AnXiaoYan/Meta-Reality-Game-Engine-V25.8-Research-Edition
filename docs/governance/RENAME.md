# Rename and public-boundary guide

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本文件定义 V25.8 研究版向“Pre‑V27 原创代码研究底座”扩展后的公开命名、目录和权威边界。它不授权复制未通过逐文件原创性/许可证复核的历史源码。

## 已批准的公开命名

| 旧概念或历史实现 | 公开名称/目录 | 状态 |
|---|---|---|
| Global YOLO / 人体 Polygon | `mrge.perception`、`contracts/polygon_gate.schema.json` | Contract/研究接口已公开；真实后端仍是可选插件 |
| Event 热像素、cluster、stale 筛选 | `mrge.filters.event`、`mrge.filters.polygon` | clean-room 研究实现已公开 |
| `sparse_gl` Event 显示经验 | `mrge.preview.sparse_event` | clean-room 稀疏点预览已公开；无完整 raster/Authority 写权 |
| BEV/场地坐标假设 | `mrge.geometry`、`contracts/geometry_profile.schema.json` | profile 驱动，不再由 camera index 推断 |
| 脚定位点 | `anchor_mode=foot_point` | 公开 geometry profile |
| 四周朝中心传感器布局 | `sensor_layout=four_sides_inward` | 公开研究变体/独立 profile |
| V26 Recording Bundle | `contracts/recording_bundle.schema.json` | 研究契约；不带真实 RAW/归档 |
| V26 Polygon Gate | `contracts/polygon_gate.schema.json` | `candidate`，不改变 Hit Judge 权威 |
| V26 IR-ID / Player State | `contracts/ir_id.schema.json` | `shadow`，不构成玩家主身份 |
| 历史 Shadow/Dry-run/失败实验 | `research/variants/*` | 必须包含 `STATUS.md` 和不可宣称项 |
| 历史来源与审计 | `provenance/*` | hash-only inventory；未审计源码不得批量复制 |

## 当前公开目录

```text
contracts/                 Apache-2.0：稳定 Schema/事实合同
replay/                    Apache-2.0：JSONL 回放与故障注入
src/mrge/
  adapters/                AGPL：Fake/File/Null 边界
  perception/              AGPL：研究后端接口
  filters/                 AGPL：Event/Polygon 过滤
  preview/                 AGPL：只读稀疏预览
  geometry/                AGPL：GeometryProfile、Pose、变换
  engine/                  AGPL：候选管线与研究报告
research/
  geometry_profiles/       Apache-2.0：布局/锚点 profile
  variants/                AGPL：条件化支线与实验状态
docs/architecture/         AGPL：架构、范围、下版预告
docs/algorithms/           AGPL：算法语义和限制
provenance/                AGPL：来源、hash、候选检查
```

## 不能在 Rename 中改变的语义

- `HitCandidate` 不能改名为 `official_result`、Final Verdict 或生产命中。
- `candidate`、`shadow`、`research-only`、qualified、`authority_ready`、production cutover 必须分开记录。
- 不能用 `frame_id-1`、最大 seen、wall-time 最近邻或本地序号伪造同步、身份或几何绑定。
- 预览、过滤、Recorder、UI 和研究报告没有 Authority WAL/签名/生产网络写权。
- AIRY、IR code、视觉 track、相机编号和模型意见不能直接成为 `participant_uid`、Hit Existence 或 Relation Authority。
- 真实 SDK、设备 handle、模型权重、真实数据、私有 profile、凭证和运行归档继续排除。

## 与下一版本架构的复用关系

当前仓库只复用 V27 架构中已经适合作为公开研究底座的模式：

1. profile 驱动的几何和校准边界；
2. 稀疏 Event 点预览而非完整 Event raster；
3. 预览/过滤与 Authority 解耦；
4. 身份中立的几何事实优先于玩家归因；
5. 可选能力（例如 AIRY）默认非 Authority、可降级；
6. 来源、合同、回放和候选检查先于真实硬件接入。

不会在本仓库重复实现或复制：真实 Event HAL、供应商 SDK、生产 Authority WAL、比赛 Judge、Game Manager、真实模型执行、私有设备管理或现场部署脚本。

## 下一版本预告：从研究底座走向完整 MRGE 架构

> 状态：架构预告，不是功能承诺、资格结论或生产发布。

下一版本将以 V27 完整体架构为设计输入，在当前公开的 Replay、Preview、Filter、Geometry 和研究支线之上扩展，重点是复用已验证的结构原则，而不是再造一套平行系统。

### V27 完整目标架构图

> 这张图是开发复用的**目标结构图**：它说明模块归属、只读方向和禁止跨越的 Authority 边界；并不把图中模块表述为当前仓库已经实现、资格通过或可用于生产。

```mermaid
flowchart TB
    GM["Game Manager\n比赛、玩家与装备主身份"]
    OP["Operator Console\n启动、受控停止、状态"]
    SELF["Startup Self-Test Window\n只读 Gate 投影"]
    CFG["Hardware Profile Editor\nMVS / Event 候选配置"]
    LIFE["Lifecycle Control Service\n意图、状态、监督"]
    SYNCCTRL["Sync Authority Controller\nReceipt / Term / Fence / WAL"]

    subgraph GOV["执行治理面 EGP：只治理执行元数据"]
        ACTION["Action Registry\n注册动作与副作用类型"]
        DAG["Capability DAG\n依赖与 Gate"]
        ADMIT["Admission Controller\n默认拒绝"]
        LEASE["Resource Lease Manager\nHardware / GPU / Schema"]
        TRUST["Trust Domain Registry\nDEV / QUAL / PROD"]
        RUNNER["Approved Runner\n唯一受批准执行器"]
        GWAL["Governance Ledger\nAttempt / Admission / Lease / Closeout"]

        ACTION --> ADMIT
        DAG --> ADMIT
        TRUST --> ADMIT
        ADMIT --> GWAL
        ADMIT --> LEASE
        LEASE --> GWAL
        LEASE --> RUNNER
    end

    subgraph SENSOR["传感与受控传输层"]
        MVS["MVS A-H\nBayer RAW"]
        FRAME["Owned Frame Ring"]
        EVENTHOST["EventOwnershipDomain\n首选单一 EventHalHost 管理 A-H"]
        EADAPTER["Logical Adapter A-H\n独立线程、Ring、Frontier"]
        ERING["Owned Event Batch Ring"]
        EING["EventIngressActor\n每路 Ring 单消费者"]
        TRIGGER["Raw Trigger Edge Fact"]
        WINDOW["Owned Event Window Store"]
        RAWARCHIVE["Event Raw Archive Writer\nProfile 受控的独立副本"]

        MVS --> FRAME
        EVENTHOST --> EADAPTER --> ERING
        ERING --> EING
        EING --> TRIGGER
        EING --> WINDOW
        EING --> RAWARCHIVE
    end

    subgraph AIRY["可选 AIRY 几何辅助：默认关闭、非 Authority"]
        AIRYHOST["AirySourceHost\n私有 SDK 与单一设备 Owner"]
        CLOCK["Airy Native Time Bridge\n点级 window_ref"]
        AIRYOBS["AiryGeometryObservationFact\nparticipant_uid = null"]
        ASSIST["TrackingAssistFact\nNON_AUTHORITY"]
        AIRYPROJ["NonAuthorityMotionTrackProjection"]
        AIRYLEDGER["Optional Capability Ledger"]
        AIRYSEAL["OptionalCapabilitySeal"]

        AIRYHOST --> CLOCK --> AIRYOBS --> ASSIST --> AIRYPROJ
        AIRYOBS --> AIRYLEDGER
        ASSIST --> AIRYLEDGER
        AIRYPROJ --> AIRYLEDGER --> AIRYSEAL
    end

    subgraph AUTH["同步权威闭环"]
        AWAL["Single-writer Authority WAL"]
        READY["A-H Ready Barrier"]
        PREAMBLE["Probe / Fixed Train / Coded Preamble"]
        RECEIPT["AuthorityReceipt"]
        BIND["Trigger Binding"]
        COHORT["A-H Cohort"]
        GAP["GapFact"]
        TERM["TerminalReason Registry"]
        EPOCHSEAL["AuthorityEpochSeal"]

        SYNCCTRL --> AWAL --> READY --> PREAMBLE --> RECEIPT
        TRIGGER --> BIND --> COHORT
        RECEIPT --> BIND
        GAP --> TERM --> EPOCHSEAL
        COHORT --> EPOCHSEAL
    end

    subgraph GPU["GPU 感知与融合数据面"]
        H2D["Pinned H2D Upload"]
        ISP["CUDA ISP"]
        YOLO["4× TensorRT Worker\nAB / CD / EF / GH，768×768 FP16"]
        SEG["GPU Mask / Polygon Primitive"]
        VTRACK["Visual Runtime Track Builder"]
        EGUARD["EventPixelDefectGuard"]
        EVTGPU["GPU Event Denoise\nSparse Point Projection"]
        TRAJ["Event Ballistic Detection"]
        WARP["Event-centric Warp"]
        FUSION["CUDA Fusion Compositor"]
        SURFACE["GPU Presentation Surface"]

        FRAME --> H2D --> ISP --> YOLO --> SEG --> VTRACK
        WINDOW --> EGUARD --> EVTGPU --> TRAJ
        ISP --> WARP
        SEG --> WARP
        EVTGPU --> WARP
        TRAJ --> WARP
        WARP --> FUSION --> SURFACE
    end

    subgraph ID["身份、绑定与无身份几何层"]
        PID["ParticipantIdentityFact\nGame Manager 唯一主身份"]
        EQUIP["EquipmentIdentityAssignmentFact"]
        IROBS["IRMarkerObservationFact\n观测，不是玩家身份"]
        IDBIND["PlayerTrackBindingFact\n有效区间与 disposition"]
        VGEOM["VisualTrackGeometryFact\n无身份 World / BEV 几何"]
        PLOC["VisualBaselineLocationFact / Series"]
        PPOLY["PlayerPolygonBindingFact"]

        GM --> PID --> EQUIP
        EING --> IROBS
        EQUIP --> IDBIND
        IROBS --> IDBIND
        VTRACK --> IDBIND
        VTRACK --> VGEOM
        VGEOM --> PLOC
        IDBIND --> PLOC
        IDBIND --> PPOLY
        SEG --> PPOLY
        VTRACK -. "只读基线参考" .-> AIRYPROJ
    end

    subgraph EVIDENCE["语义事实、证据与潜在判定层"]
        POLY["Polygon Fact"]
        BALL["BallisticTrajectoryFact"]
        HEB["HitEvidenceBundle\n仅身份中立引用；未来 Gate"]
        HIT["HitExistenceFact\n未来独立批准的 Producer"]
        REL["RelationFact\nsource / target 归因；未来 Gate"]
        CHAIN["PlayerEvidenceChainIndex\n只索引，不创造事实"]
        LEDGER["Canonical Ledger"]

        SEG --> POLY
        TRAJ --> BALL
        POLY --> HEB
        BALL --> HEB
        VGEOM --> HEB
        HEB -. "仅已批准 Hit Producer" .-> HIT
        HIT -. "命中存在与归因分离" .-> REL
        IDBIND -. "有效 source / target 区间" .-> REL
        PPOLY -. "仅用于归因" .-> REL
        POLY --> LEDGER
        BALL --> LEDGER
        PID --> LEDGER
        EQUIP --> LEDGER
        IROBS --> LEDGER
        IDBIND --> LEDGER
        PLOC --> LEDGER
        PPOLY --> LEDGER
        HEB --> LEDGER
        BIND --> LEDGER
        RECEIPT --> LEDGER
        EPOCHSEAL --> LEDGER
        PID --> CHAIN
        IDBIND --> CHAIN
        PLOC --> CHAIN
        PPOLY --> CHAIN
        BALL --> CHAIN
        HIT --> CHAIN
        REL --> CHAIN
        CHAIN --> LEDGER
    end

    subgraph PRESENT["展示、编码、记录层"]
        GL["OpenGL Local Preview\n严格只读图层白名单"]
        NVENC["Hardware Encode Plane"]
        WEB["HTML5 / WebRTC"]
        SRT["Main + A-H SRT"]
        MATCH["Match Evidence Lite"]
        MATCHSEAL["MatchEvidenceSeal"]

        SURFACE --> GL
        SURFACE --> NVENC
        NVENC --> WEB
        NVENC --> SRT
        NVENC --> MATCH
        RAWARCHIVE --> MATCH
        LEDGER --> MATCH --> MATCHSEAL
    end

    subgraph REPLAY["标准比赛回放：不重跑算法或判定"]
        NVDEC["NVDEC MVS Decode"]
        ERAW["Event RAW Decode"]
        RFUSE["GPU Replay Fusion"]

        MATCH --> NVDEC --> RFUSE
        MATCH --> ERAW --> RFUSE
        MATCH --> RFUSE
    end

    subgraph REVIEW["独立赛后模型分析与人工复核"]
        PACK["PostMatch EvidencePack Builder\n命中复盘 / 规则违规复盘"]
        OUTBOX["Durable Analysis Outbox"]
        REQ["Provider Payload Adapter\n无网络、无密钥"]
        EGRESS["Policy Egress Gateway\n唯一网络与密钥 Owner"]
        CLOUD["Remote API Only\nDeepSeek-V4-Flash 候选"]
        RESP["Provider Response Adapter\n无网络、无密钥"]
        VERIFY["Local Schema / Ref / Policy Verifier"]
        HOPINION["LLMHitOpinionFact\nSHADOW_ONLY"]
        VOPINION["LLMViolationOpinionFact\nSHADOW_ONLY"]
        ALEDGER["Analysis Ledger"]
        CASE["PostMatchReviewCase\n争议 / 规则审计 / 人工请求"]
        WORKBENCH["Post-Match Review Workbench\nblind-first、独立角色"]
        HUMAN["HumanArbitrationDecisionFact\nREVIEW_ONLY"]
        ASEAL["MatchAnalysisSeal"]

        LEDGER -->|"只读引用"| PACK
        AIRYLEDGER -->|"可选 NON_AUTHORITY 引用"| PACK
        MATCH -->|"已封存或临时只读输入"| PACK
        PACK --> OUTBOX --> REQ --> EGRESS --> CLOUD
        CLOUD --> EGRESS --> RESP --> VERIFY
        VERIFY --> HOPINION --> ALEDGER
        VERIFY --> VOPINION --> ALEDGER
        MATCH --> CASE
        HOPINION --> CASE
        VOPINION --> CASE
        CASE --> WORKBENCH --> HUMAN --> ALEDGER
        ALEDGER --> ASEAL
    end

    GM --> LIFE
    OP -->|"幂等 ControlCommand"| LIFE
    CFG -->|"ConfigApplyRequest"| LIFE
    LIFE -->|"ExecutionRequest；无设备句柄"| ADMIT
    RUNNER -->|"Admission + Hardware Lease + Fence"| SENSOR
    RUNNER -->|"Admission + GPU Lease"| GPU
    RUNNER -->|"仅请求 Receipt；无签名密钥"| SYNCCTRL
    LIFE -->|"SelfTestRunFact / Status"| SELF
    GWAL -. "只读治理状态" .-> SELF
    SENSOR -. "Ready / Readback" .-> SELF
    AUTH -. "Gate / Receipt" .-> SELF
    GL -->|"只读展示"| OP
```

#### 复用与禁止连接

| 结构 | 开发者应复用的边界 | 明确禁止的捷径 |
|---|---|---|
| Execution Governance Plane | Action Registry → Admission → Lease → Approved Runner 的执行顺序 | 读取图像/Event 负载、写 Authority WAL、持有设备句柄或签名密钥 |
| Event 接入 | 单一物理 Owner、A-H 独立 Adapter/Ring/Frontier、Raw→Sanitized 分流 | 多 Owner 争抢设备、让 Raw CD 直接进入算法、合并多路 Source Frontier |
| 感知与预览 | 四 Worker 感知、Authority No-Loss 与 Preview Latest-Wins 分离 | 为显示或编码丢弃事实、由 Preview 反压 Authority |
| 身份与几何 | `participant_uid` 由 Game Manager 签发；几何先无身份，再在有效窗口绑定 | 用 IR code、track、AIRY、相机编号或 LLM 直接生成玩家身份 |
| Hit 与 Relation | `HitEvidenceBundle → HitExistence → Relation` 的存在/归因拆分 | 以玩家接近、身份或模型意见跳过 Hit existence |
| AIRY | 默认 `DISABLED`，可选 `SHADOW/AUGMENT`，独立 Optional Seal | 改写基线位置、身份绑定、Hit/Relation、AuthorityReceipt 或 Base qualification |
| 回放与复核 | 标准 Replay 只消费已记录内容；模型/人工只读封存证据 | 标准回放重跑模型/算法，或让 LLM/人工改写比赛状态 |

当前公开仓库实现的是 Contracts、Replay、合成 Adapter、Preview/Filter、GeometryProfile、候选结果和研究报告。图中的真实 A-H 设备、YOLO Engine、Event Ownership Domain、Seal、玩家绑定、Hit/Relation 和赛后分析均是下一版本架构方向，需分别经过来源、硬件、资格、Release Manifest、Admission、Lease、Trust Domain 和 Authority Gate。

### 阶段性架构图

```mermaid
flowchart LR
  P0["阶段 0：公开研究底座<br/>Contracts / Replay / Synthetic / Preview / Filter / Geometry"]
  P1["阶段 1：离线集成<br/>Recorded Facts / 4-worker Pattern / Event Ownership Contract / Evidence Bundle"]
  P2["阶段 2：受治理能力<br/>Admission / Lease / Trust / Private Hardware Plugins"]
  P3["阶段 3：封存证据与受限裁判<br/>Seals / Binding / Hit Existence / Relation / Post-match Review"]
  P0 --> P1 --> P2 --> P3
  O["Optional AIRY<br/>DISABLED / SHADOW / AUGMENT"] -. "non-blocking and NON_AUTHORITY" .-> P1
  O -. "separate qualification only" .-> P2
  X["Production Cutover"] -. "never implied by prior stages" .-> P3
```

| 阶段 | 可做什么 | 明确不能做什么 |
|---|---|---|
| 阶段 0 | 公开原创研究代码、合成样例、回放、过滤、预览和 GeometryProfile | 连接真实设备、写生产 Authority、使用真实数据/权重 |
| 阶段 1 | 离线重放已记录事实，验证四 Worker/单 Owner/证据 bundle 的合同与守恒 | 把离线 PASS 当作硬件资格或真实命中质量 |
| 阶段 2 | 在 Admission、Lease、Trust Domain 和独立插件边界下接入硬件 | 复用开发目录、私有 profile 或绕过 readback/校准/资格 Gate |
| 阶段 3 | 处理已封存证据上的绑定、Hit/Relation 与赛后审查 | 自动改写比赛结果、HP、Game State 或生产 Authority |

| 方向 | 复用原则 | 当前公开基础 |
|---|---|---|
| 多相机感知 | 采用 AB/CD/EF/GH 四个 Global YOLO Worker 的分组模式；完整 Polygon 与发布背压分开处理 | FakeBackend、Polygon Contract、Replay |
| Event 所有权 | 每台 Event 设备只能有一个物理 Owner；公共核心只保留 Adapter SPI/事实合同 | Null/Fake Adapter、Event Filter |
| 稀疏预览 | 只传输可绘制 Event 点，不传完整 raster；Preview 不反压事实/Authority | `mrge.preview.sparse_event` |
| 几何支线 | 以 `GeometryProfile`、`CalibrationRef`、anchor/profile 取代硬编码场地坐标 | foot-point、four-sides-inward profile |
| 身份与命中 | 先保留身份中立的 Polygon/Visual Geometry/Trajectory，再在独立 Gate 后做玩家绑定与 Relation | `candidate`/`shadow` Contract 边界 |
| 证据与回放 | Replay 读取已记录事实；标准回放不重跑真实 YOLO/Event/模型 | JSONL、Recording Bundle、研究报告 |
| 可选增强 | AIRY 作为默认关闭、可缺省、NON_AUTHORITY 的运动几何支持 | 研究支线与 Optional Capability 约束 |
| 治理 | 执行治理平面只管理 Admission/Lease/Capability，不读取图像/Event 负载，也不写业务 Authority | provenance、检查器和 Release Manifest 模式 |

### 当前不会重复建设的部分

- 不复制真实 Event HAL、相机 SDK、私有设备发现、生产启动脚本或现场配置；
- 不另建一个混合 UI/Authority/生命周期的超级进程；
- 不以 Preview、IR code、AIRY、LLM 或模型意见生成玩家身份、Hit、Relation、HP 或游戏状态；
- 不在标准 Replay 中重新跑真实模型或上传真实视频/Event RAW；
- 不绕过 Release Manifest、Lease、Trust Domain、校准与资格 Gate。

### 分阶段预告

1. **公开研究层**：补齐 profile、事实 Schema、Synthetic/Replay、过滤和只读预览。
2. **离线集成层**：以记录事实验证四 Worker、Event 所有权、证据 bundle 与独立 GeometryProfile。
3. **受治理能力层**：在独立 Admission/Lease/Trust Gate 下接入真实硬件和可选插件；不复用开发目录或私有配置冒充资格。
4. **后续裁判/赛后分析层**：Hit Existence、Relation、人工复核和模型意见仅在封存证据与独立 Authority 后作为受限能力推进。

完整架构参考：V27 Clean Runtime 的《MRGE 1.0 架构规范》。该规范是设计输入和边界来源；本公开预告只提炼其可公开复用的结构，不复制其私有 Adapter、真实硬件配置或生产授权语义。

当前仓库的 `pre-v27-original-audit.1` 仍只是本地候选检查通过的公开研究审计标签。独立环境复核、第三方许可证审计、真实硬件 Gate、资格、Authority Ready 和正式 Release 仍需分别完成。

## Pre‑V27 迁移规则

每个候选文件必须先出现在 `PRE_V27_FILE_INVENTORY.csv`，并完成以下终态之一：

- `ORIGINAL_PUBLIC`：可按目标许可证迁移；
- `ORIGINAL_DECOUPLE`：重写 SDK/路径/进程耦合后迁移；
- `ORIGINAL_EXPERIMENTAL` 或 `ORIGINAL_VARIANT`：迁入 `research/variants/` 并带状态；
- `THIRD_PARTY_BOUND`、`PRIVATE_DATA_BOUND`、`UNKNOWN`：不进入公开源码。

## 文档状态

本文已按当前 Pre‑V27 扩展内容更新，待用户审阅后才提交和推送。它不改变既有 `pre-v27-original-audit.1` 标签。
