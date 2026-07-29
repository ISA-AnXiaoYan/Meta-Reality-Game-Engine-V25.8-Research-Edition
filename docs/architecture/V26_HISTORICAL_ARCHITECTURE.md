# V26 历史实现架构

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本图根据 historical/v26/ 中已公开的源码、配置、工具和静态知识图谱整理，用于定位迁移和研究切入点。它描述历史结构，不表示这些服务现在可安全启动或已通过现场资格。

~~~mermaid
flowchart LR
    subgraph I["历史输入"]
      MVS["MVS / Recording Bundle"]
      EVT["Event Window / Trigger"]
    end
    subgraph P["感知与融合"]
      YOLO["YOLO Segmentation / Mask / Polygon"]
      TRACK["Person / Runtime Track"]
      BEV["Homography / BEV / Geometry"]
      FUSION["Multi-view Preview / Fusion"]
    end
    subgraph J["事件与证据研究"]
      FILTER["Event Filter"]
      TRAJ["Trajectory Candidate"]
      JUDGE["Historical Hit Judge / Evidence"]
    end
    subgraph O["输出"]
      GAME["Game Bridge"]
      REPLAY["Replay / Reports / Records"]
    end
    MVS --> YOLO --> TRACK --> BEV --> FUSION
    EVT --> FILTER --> TRAJ --> JUDGE
    YOLO --> JUDGE
    BEV --> JUDGE
    JUDGE --> GAME
    MVS --> REPLAY
    EVT --> REPLAY
    FUSION --> REPLAY
    JUDGE --> REPLAY
~~~

供应商 SDK/HAL、模型权重、凭据和未获授权材料不随仓库发布。请先读 [历史档案支持状态](../../historical/SUPPORT_STATUS.md)，再使用 [V26 知识图谱](../../historical/knowledge-graph/README.md) 定位源码；不要直接执行历史启动脚本。
