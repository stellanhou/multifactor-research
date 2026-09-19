# 创意卡驱动的策略研究最小闭环

输入创意卡与显式研究合同，依次执行设计、确定性回测、评估、决策。优化任务返回设计环节，产生新版本；保留的一个版本冻结后才读取验证段。验证后不再调用模型或返回开发循环。

本模块是主策略研究入口，与因子挖掘共享模型传输、只追加记录和JSON工具。因子挖掘的A/B统计准入不被修改。

## 本次实现范围

- 唯一可执行策略能力：现有BTC/ETH现货强弱轮动；小时收盘信号、下一小时开盘成交、先卖后买、现金或单资产多头。
- 可改参数：收益观察窗口、两币收益差阈值、强者最低窗口收益。每次修改只允许动一个已声明参数，其余规则固定。完整规则由程序附在策略定义中。
- 数据只读本地 `MarketDataStore`；检查每小时完整网格、OHLC、成交额和收盘可用时间，不填补、不联网。
- 复用 `generate_relative_strength_targets` 和 `run_relative_strength_backtest`。基准同样经过扣费回测；费用压力情景乘以合同指定倍数。
- 开发段版本共享完全相同的数据和时间网格；保存全部尝试与父版本关系。修改任务预先声明收益改善幅度与回撤容忍值，程序进行描述性配对比较。
- 参数、日期、成本、门槛由调用方提供；Agent不能越界。保留、暂停、淘汰、数据不足、不支持的创意和技术失败都有保存状态。
- 不启动Paper或Demo。真实模型研究质量、正式策略参数和准入门槛仍需另行确认。

## 文件职责

| 文件 | 职责 |
|---|---|
| contracts.py | 研究合同与模型输出边界；固定的可执行规则 |
| engine.py | 本地数据检查、快照、已有回测引擎适配、基准与压力测试、程序判断 |
| workflow.py | 角色衔接、版本与修改任务、证据引用核验、冻结与验证隔离 |
| cli.py | run/status入口；固定回复回放或显式真实模型配置 |

三个模型角色使用相同 `JsonModel.complete` 接口。请求和完整回复逐次存档；不静默裁剪上下文，不自动重试或伪造替代分析。超过合同的输入字节上限直接留痕报错。首版传递全部开发记录及数值摘要，逐小时完整数值另存CSV；模型不能声称逐行读过未进入上下文的CSV。

## 工程回放

项目根目录执行：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research run \
  --contract examples/strategy_research/contract.json \
  --idea examples/strategy_research/idea.json \
  --db market_data/crypto_quant.sqlite \
  --replay examples/strategy_research/replay.json
```

示例运行使用本地2026年4月开发数据、5月验证数据。固定回复演示v1→窗口修改任务→v2→保留v1→冻结验证。v1、v2均真实回测；保留v1是预先写好的工程场景，不是模型对结果的真实研究判断。日期、参数、门槛均是工程输入，不得用作正式研究已批准的证据。

回放强制 `purpose=engineering`，从不调用外部模型服务。`engineering_complete` 表示流程完成，数值条件是否满足另看 `validation.passed_declared_checks`。示例日期存在既往研究使用，验证仅隔离于本次运行，不宣称从未使用的样本外证据。

输出在 `experiments/strategy_research/<run_id>/`。运行目录已存在会拒绝覆盖；需要新的工程回放时指定另一个 `--output`，保留旧证据。验证失败不能通过重复运行伪装首次验证。

```sh
.venv/bin/python -m crypto_quant.cli strategy-research status \
  --run experiments/strategy_research/btc-eth-engineering-replay
```

`status` 读取终态，不提供后台调度或中断恢复。暂停条件和中断记录保存后，由后续明确的新任务承接；不得直接覆盖旧目录重新读取验证段。

## 证据与边界

运行根目录保存创意、合同、全部项目Python代码指纹、开发/验证数据快照、模型请求与回复，以及只追加研究记录。每个版本保存逐小时目标、净值、成交费用汇总和已结束交易CSV。冻结记录绑定选定版本、开发证据及合同；验证读取标记在数据读取之前写入，失败也保留。验证预热区间与开发区间重叠的观测必须完全一致。

每段从独立现金账户开始，预热只计算信号；期末按市价估值，不强制平仓。最大回撤包含初始本金，费用已进入净值。报告中的年化夏普是小时收益描述统计，不是HAC推断。

验证条件包括净收益、相对扣费基准的收益、回撤、交易发生小时数及成本压力结果。它们是显式合同检查，尚未包含正式研究的多重检验校正、完整Walk-Forward、组合相关性与容量审查；`retained_for_review` 只表示可以继续审查，不等于进入Paper。

暂不支持任意模型生成Python、其他策略家族、动态历史币池、多个最终版本共用验证段筛选或自动执行。超出能力的创意应返回 `unsupported`。

## 验证

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_strategy_research.py' -v
```

测试覆盖真实SQLite到回测的双版本循环、下一小时执行和因果性、费用及基准、开发/验证隔离、冻结前后数据缺口、证据引用、参数越界、失败修改、重复尝试、暂停/淘汰、上下文溢出、数据变化与不可覆盖记录。

当前在线入口默认通过官方Codex Python SDK使用ChatGPT订阅，模型为`gpt-5.6-luna`、推理强度为`max`。统一环境安装及登录见[项目说明](../../../../README.md)；`--replay`保持离线。
