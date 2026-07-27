# V26 public interfaces

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本批次公开 V26 后续能力的研究接口，不复制真实运行时、真实数据或供应商代码。

| 接口 | 公开文件 | 语义 |
|---|---|---|
| Recording Bundle | `contracts/recording_bundle.schema.json` | Event/MVS/evidence/manifest/finalize 的证据容器边界 |
| Polygon Gate | `contracts/polygon_gate.schema.json` | pass/fail/fail-open/unknown 的候选门状态 |
| IR-ID | `contracts/ir_id.schema.json` | 玩家身份候选、confirmed/shadow 状态和 marker 信息 |
| Hardware boundary | `src/mrge/adapters/null_adapter.py` | 显式表示没有打开真实设备 |

这些接口不改变 Hit Judge 的生产权威。Polygon Gate 的 `authority` 固定为 `candidate`；IR-ID 的 `authority` 固定为 `shadow`；Recording Bundle 的 `authority` 固定为 `research-only`。
