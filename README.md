# ArknightsStoryJson

《明日方舟》剧情 JSON 兼容分发仓库。

本仓库不再从 050644zf/ArknightsStoryJson 或 Kengxxiao/ArknightsGameData
构建剧情包，而是从
[arknights-data-pipeline](https://github.com/3aKHP/arknights-data-pipeline)
（AKDP）工厂仓库的 GitHub Release 直接投影 `zh_CN.zip`。

## 资产

每个 Release 包含：

| 资产 | 说明 |
|------|------|
| `zh_CN.zip` | zh_CN 剧情包（含 story JSON 和 LLM 摘要） |
| `manifest.json` | 溯源、指标、摘要覆盖率和 SHA-256 校验 |

兼容仓把 AKDP 产出的摘要当作不可变输入。不在同步时补摘要、重新摘要或
调用 LLM。摘要覆盖率（`summary_coverage`）记录在 manifest 中，取值为
`complete`、`partial` 或 `missing`。

## 发布流程

`.github/workflows/sync-and-release.yml` 定时检查 AKDP 最新非 draft Release：

1. 下载工厂 `manifest.json` 和 `zh_CN.zip`
2. 通过 `scripts/akdp_source.py` 验证字节 SHA-256/size 与工厂清单一致
3. 运行 `scripts/release_gate.py` 做契约、回归、完整性和摘要覆盖率校验
4. 以 `akdp-<versionId>-v1` 标签创建 draft Release，回下校验后公开

兼容仓只投影原始字节，不解压、不重压缩、不注入数据、不调用 LLM。

## 测试

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py
```
