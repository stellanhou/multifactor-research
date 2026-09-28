# 真实划分的策略研究合同

RSI和基差均为research用途：A+B开发 `[2022-08-01,2025-08-01)`，原C内部验证 `[2025-08-01,2026-08-01)`，UTC。完整策略最终前向验证尚未启动。

先核对行情，不调用模型、不计算收益：

```sh
PYTHONPATH=src .venv/bin/python -m crypto_quant.cli research-data-policy
PYTHONPATH=src .venv/bin/python -m crypto_quant.cli strategy-research check-data --contract examples/strategy_research/research/rsi.contract.json --db market_data/crypto_quant.sqlite --stage development --output experiments/my-rsi-development-audit
```

`--stage validation`显式审计原C；审计不评价收益，但会记录读取并保存快照。输出目录必须全新。替换合同为basis.contract.json可检查基差路线。

2026-09-22检查：按用户授权直接插值原库小时价格后，RSI两段数据可读取；基差价格网格已完整，BTC资金费事件的结算标记价仍有开发7条、验证3条缺失，暂不能完成该路线完整回测。不得以小时插值标记价格悄悄冒充资金费事件的结算价。

真实模型研究命令（本次接口改造没有执行）：

```sh
PYTHONPATH=src .venv/bin/python -m crypto_quant.cli strategy-research run --contract examples/strategy_research/research/rsi.contract.json --idea experiments/idea_pool/community--rsi-reversal--20260922.json --db market_data/crypto_quant.sqlite --output experiments/my-rsi-research
```

模型可以设计及优化，retain后才用validate显式启动内部验证，不自动调用模拟账户。示例保留原门槛作为明确输入，不代表这些门槛已经构成统计显著性或模拟账户准入标准；正式投入研究前按用途确认。

原库存在事后线性插值，回测可用于研发诊断，但不能声称完整因果重放。参见[小时行情插值处理记录](../../../docs/data/小时行情插值处理记录.md)。旧工程合同、旧运行与已失败验证保持原样。
