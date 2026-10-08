# 策略研究与多因子系统

负责多因子组合、成本账本、历史研究、组合选优与内部验证，保留既有策略研究入口。

从 [crypto-quant-research-platform](https://github.com/stellanhou/crypto-quant-research-platform) 拆出，保留相关文件历史及 2026-10-08 本地未提交的研究改动。原仓库继续保存完整历史、数据工具和统一控制台。

## 安装与使用

```bash
uv sync --frozen --all-extras
.venv/bin/multifactor-research strategy-research --help
make test
```

Python 版本由 `.python-version` 固定。每个仓库使用自己的 `.venv`，两者沿用 `crypto_quant` 包路径，不应装入同一个环境。历史文档中的 `python -m crypto_quant.cli strategy-research ...` 在本仓库仍可使用。

## 数据与两系统衔接

配套仓库：[因子挖掘](https://github.com/stellanhou/factor-mining)。

完整行情库和历史运行结果保留在原工作区 `/Users/stellan/量化投资/market_data/`、`/Users/stellan/量化投资/experiments/`。仓库只附带下述固定工程测试样例。命令的 `--db`、`--run`、`--baseline` 等路径应显式指向所需文件；合同中的相对路径按原有合同规则解析，移到新目录前需核对。新运行结果默认写入本仓库的 `experiments/`。

因子挖掘生成 FM-v6 JSON 创意卡，多因子系统通过合同中的卡片路径读取。历史卡片及归档按原有相对位置保留。拆仓不启动或恢复研究任务。

共享公式、数据访问、模型调用及证据工具按实际依赖保存在各仓库中，不要求安装另一仓库；`multifactor-research` 中的 `research/factor_mining/` 仅保留所需支持模块。统一网页控制台继续从原仓库运行。

模型凭据通过环境变量或各仓库本地 `.env` 提供；本次不复制凭据。仅提交空配置模板。

## 文档

- [模块说明](src/crypto_quant/research/strategy_research/README.md)
- [文档索引](docs/README.md)
- [迁移记录](docs/repository-split.md)

证据读取回归测试所需的一份固定工程快照（压缩后约 2.3 MB）保存在 `tests/fixtures/`，其来源与边界见测试样例说明；完整历史结果仍留在原目录。
