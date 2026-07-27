# Known limitations

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

- 当前 FakeBackend 和合成样例不代表真实 YOLO、相机质量、命中率或误检率。
- `HitCandidate`、Polygon Gate 和 IR-ID schema 都不是生产最终判定或游戏权威。
- 未提供真实硬件 Adapter、供应商 SDK、真实数据和模型权重。
- 当前 CLI 是研究/回放工具，不提供高可用服务、实时延迟保证或现场安全保证。
- pytest 在 CI 中安装；本地未安装测试依赖时只能运行编译、契约和 CLI 直接检查。
- V26-A/B/C/F 能力仍需独立资格、现场验证和权威切换门，不能由本仓库的 PASS 字段替代。
