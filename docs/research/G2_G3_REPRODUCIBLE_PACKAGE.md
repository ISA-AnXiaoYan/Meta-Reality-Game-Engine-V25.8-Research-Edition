# G2/G3 可复现研究包

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

本批次把公开仓库从“目录骨架”推进到可重复运行的最小研究包：输入事实经过合成 Adapter 后，每帧产生一个明确的终态；输出始终带有 `official_result=false`，不能被解释为生产判定。

## 已公开能力

- `FakeBackend`：不依赖 GPU、相机、供应商 SDK 或模型权重的确定性后端。
- `mrge simulate`：生成候选结果，每个输入帧对应一个 `candidate_emitted` 终态。
- `mrge generate-sample`：从零生成 JSONL 合成样例。
- `tools/build_synthetic_sample.py`：以固定字段和排序从零重建样例，便于哈希比较。
- `examples/quickstart_2cam_synthetic.jsonl`：2 路合成快速样例。
- `examples/fullchain_8cam_replay.jsonl`：8 路回放链样例。
- `mrge replay`：重放 JSONL，同时明确 `terminal_state=replayed` 和 `official_result=false`。
- Replay fault injection：显式覆盖缺帧、重复帧、乱序、过期 Polygon 和 Event 空流。
- `ResultEnvelope`：把候选结果、终态和权威布尔值放在同一个稳定契约中。

## 当前未声称

- 未实现真实 YOLO 权重、真实相机 Adapter 或现场部署。
- 未实现正式 Hit Judge、`official_result` 或生产 Game Manager 权威。
- 未将历史真实数据复制进公开仓库。

## 本地验证

```powershell
python -m pip install -e . --no-deps
mrge validate
mrge simulate --frames 2
mrge generate-sample --output .\tmp-sample.jsonl
mrge replay --input .\tmp-sample.jsonl
```

生成的样例只用于研究和回放测试，不应作为现场效果或资格证明。
