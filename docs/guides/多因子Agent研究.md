# 多因子 Agent 研究使用说明

本流程把 Agent 限定在候选审阅、组合假设和有限子集选择。它不计算因子、不产生订单，也不计算盈亏；选出的因子子集会交给与传统基线相同的确定性账户程序回放。

## 开始前

先完成[多因子传统基线](多因子传统基线.md)，得到一个完整的基线运行目录。Agent 会读取其中冻结的创意卡、因子面板、共同行情样本、基线指标与数据说明。当前示例基线目录为 `experiments/strategy_research/multifactor_v1/multifactor-engineering-20261002/`。

示例候选为三张已有 FM-v6 卡片；它们都实际通过了24小时 B 检验，方向均为 `-1`。三张卡的原研究使用固定30币横截面，首版基线使用10币工程币池，故这里重新计算的因子和账户结果只代表本合同口径。

Agent session 文件冻结 schema、运行编号、最大实验数和上下文长度：`examples/strategy_research/multifactor/agent.session.json`。示例 `experiment_budget=1`；一项实验最多触发三次模型调用：初始审阅、组合设计、执行后审阅。Agent 可以在设计阶段选择停止，此时不再运行 Agent 臂实验；预先声明的固定对照臂仍按预算执行。

## 运行真实模型

在仓库根目录运行：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research agent-run \
  --baseline experiments/strategy_research/multifactor_v1/multifactor-engineering-20261002 \
  --session examples/strategy_research/multifactor/agent.session.json \
  --output experiments/strategy_research/multifactor_agent \
  --provider codex \
  --model gpt-6-luna \
  --reasoning-effort max \
  --codex-bin '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex' \
  --timeout-seconds 300
```

此命令为本机 GPT-6 Luna 账户目录显式指定桌面 app 内的 Codex 可执行文件。2026-10-02只读目录检查中，SDK捆绑 runtime `0.154.0` 没有列出 `gpt-6-luna`；桌面 app runtime `0.159.0-alpha.12.1` 返回该模型，并列出 `low`、`medium`、`high`、`xhigh`、`max` reasoning effort。命令使用 `--model gpt-6-luna --reasoning-effort max`，不修改仓库内其他研究入口的全局默认模型。GPT-6 Luna 的官方模型说明也列出了 `max` reasoning effort：[GPT-6 Luna 模型说明](https://developers.openai.com/api/docs/models/gpt-6-luna)。

运行前应先确认 `--baseline` 指向已完成的 P1 运行目录。Agent 运行会使用 session 中的新 `run_id` 创建产物；更换输入或重新开展一次运行时，应使用新 session `run_id` 和新输出目录，不覆盖旧结果。

## Agent 做什么

流程固定为“审阅基线 → 设计组合 → 确定性回放 → 审阅回放结果”。首轮审阅只能引用提供的证据记录。设计角色可以提出 `stop`，或选择冻结候选池的2至N−1张卡；当前三张候选时只能选两张。允许复算历史基线已有的子集，但同一子集在本次会话只执行一次。选中后，程序对方向统一并标准化后的因子等权合成，不接受模型给公式加权、改因子、改交易规则或改成本。

同一多空名额、总敞口、单币权重、24小时调仓周期、手续费、滑点、资金费、保证金规则与 P1 基线账户执行器会用于 Agent 子集。所有信号使用基线已保存的共同有效样本。若 Agent 停止，则不再产生 Agent 臂的子集回放；报告分别列出两臂实际使用的次数。

比较报告还会计算一个事先确定顺序的固定子集作为基准臂，并将它限制在与 Agent 相同的最大实验预算内。固定子集定义及运行结果不会送进模型上下文；模型只看到 P1 基线证据和当前允许引用的记录。所有组合由同一确定性账户程序计算，模型只负责证据审阅和候选选择。

## 固定回复回放

`--replay` 使用保存的工程回复运行相同角色与确定性执行路径，不调用模型服务：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research agent-run \
  --baseline experiments/strategy_research/multifactor_v1/multifactor-engineering-20261002 \
  --session examples/strategy_research/multifactor/agent.session.json \
  --output experiments/strategy_research/multifactor_agent_replay \
  --replay examples/strategy_research/multifactor/agent.replay.json
```

固定回放只核对 Agent 接口、输入边界和确定性账户执行。它不验证真实 Codex runtime、GPT-6 Luna 的模型回复，也不证明 Agent 提升了组合表现。真实模型运行与固定回复回放必须使用不同的输出目录。

## 结果如何解释

P1 基线结果和这些候选卡的历史检验结果在本次 Agent 研究前已经暴露，并参与了候选与流程设计。因此，Agent 子集、固定顺序子集和 P1 等权组合的差异属于历史工程比较，不能称为独立、无偏的 Agent 增量证据。固定子集对照能核验相同预算和账户程序下的结果差异，不能恢复已经发生的数据与结果暴露。

该工程区间为2025-09-01至2025-09-15，位于原 C 日期范围内；原库经过事后插值，且没有可靠的全字段逐行修复掩码。所有收益、资金费和成本结果均须按 `retrospective_or_unknown` 的因果来源解释。程序成功执行表示研究流程完成，不代表策略有效、独立验证通过或可投入模拟／实盘。
