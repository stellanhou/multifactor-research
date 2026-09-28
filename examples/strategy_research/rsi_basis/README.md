# RSI与现货—永续基差工程示例

从项目根目录运行；每次选择新的输出目录：

```sh
PYTHONPATH=src .venv/bin/python -m crypto_quant.cli strategy-research run --contract examples/strategy_research/rsi_basis/rsi.contract.json --idea experiments/idea_pool/community--rsi-reversal--20260922.json --replay examples/strategy_research/rsi_basis/rsi.replay.json --db market_data/crypto_quant.sqlite --output experiments/strategy_research/my-rsi-check

PYTHONPATH=src .venv/bin/python -m crypto_quant.cli strategy-research run --contract examples/strategy_research/rsi_basis/basis.contract.json --idea experiments/idea_pool/community--spot-perp-basis--20260922.json --replay examples/strategy_research/rsi_basis/basis.replay.json --db market_data/crypto_quant.sqlite --output experiments/strategy_research/my-basis-check
```

固定回复只用于验证程序链路，保留决定不是策略有效性结论。去掉--replay会调用真实模型；正式研究前需确定独立数据区间和准入门槛。run不读取B段。

RSI使用2026年4月；基差使用2024年3月。原2026年4月基差样本没有正基差开仓条件，保留零交易结果后补充有正基差的样本用于验收成交路径。阈值为工程样例，未按收益优化。完整记录见experiments/strategy_research/rsi_basis_20260922/verification.md。
