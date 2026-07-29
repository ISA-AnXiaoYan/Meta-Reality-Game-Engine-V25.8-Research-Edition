# 开发说明

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

## 支持的开发表面

| 区域 | 可以修改 | 必须保持的边界 |
| --- | --- | --- |
| src/mrge/ | 研究引擎、合成 Adapter、过滤、预览、几何 | 不生成生产 Authority 或正式裁决 |
| contracts/、replay/ | 公开格式与兼容实现 | 先更新 schema、样例和测试 |
| examples/ | 网络无关的合成输入 | 不引入真实个人数据、凭据或受限权重 |
| research/ | Profile 与实验变体 | 明确状态和适用假设 |
| historical/ | 默认不改动 | 仅维护者可做纠错、来源补充或许可证修复 |

## 本地检查

~~~powershell
python -m pip install -e . pytest
python -m pytest -q
python tools/check_markdown_links.py
python tools/check_governance.py
python tools/r6_release_check.py
~~~

提交信息应说明：变更范围、验证命令、关联契约、是否涉及历史归档，以及任何未解决的证据边界。

## 兼容性规则

1. 新增跨模块字段时，同时更新 JSON Schema、Replay 样例、测试和变更日志。
2. candidate、shadow、qualified、authority_ready 和生产结果是不同状态，不得互相替代。
3. 不提交供应商 SDK、真实设备句柄、密钥、令牌、绝对路径、受限模型权重或未经明确授权的数据。
4. 引擎与工具使用 AGPL-3.0-only；Contracts/Replay 使用 Apache-2.0。新增文件遵循 [许可证映射](../LICENSES/LICENSE-MAP.md)。

贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
