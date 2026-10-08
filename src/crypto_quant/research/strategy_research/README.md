# 研究任务驱动的策略研究入口

输入创意卡与显式研究合同，依次执行设计、计算语义核对、确定性回测、评估、决策。优化任务返回设计环节，产生新版本。`run`只处理开发段；保留候选后停止。后续显式执行`validate`才冻结候选并读取验证段，验证不调用模型或返回开发循环。

本模块是主策略研究入口，与因子挖掘共享模型传输、只追加记录和JSON工具。因子挖掘的A/B统计准入不被修改。

## 按数据提出思路，再检查执行能力（2026-09-22）

探索范围不再由BTC/ETH现货模板限定。设计角色收到开发段本地数据目录（字段含义、单位和各来源的币种）、当前执行能力目录，以及原始创意卡。目录只查询开发段，不读取验证段；币种出现不等于字段完整。

设计优先返回`action=propose`。`parameters`包含`family`、`symbols`、`required_fields`、`required_capabilities`、`definition`；六项`calculation_meaning`完整描述原始思路、买卖、仓位和风险规则。不能表达为已实现参数时`definition=null`，原始思路仍保存，不要求改成现货轮动。已有`action=design`的执行定义和固定回放仍可使用。

程序先保存`v*-proposal`，再生成`v*-capability-check`：无开发段有效字段值为`missing_data`；未实现的币种、字段接入、交易行为或未配置执行路线为`missing_execution_capability`。两类缺口分别保存，允许同时出现。已有路线但未配置相应合同会明确要求该路线合同，不会擅自改变日期、成本或参数范围。通过初检后才加载精确回测数据、检查预热和缺口，并进入原有计算核对、回测、评估、决策循环。精确数据检查失败保存`v*-data-gap`与停止原因。

本次不新增交易市场或执行器，不自动修复缺失值，不启动补数或实现新的策略类型。资金费累计等衍生字段的目录统计只表示源事件存在，完整窗口由已有读取器核验。新提案示例回放为`examples/strategy_research/factor_rules/exploration.replay.json`，与同目录合同、创意卡配合使用。

## 因子组合与规则策略（2026-09-22）

`task.strategy_family=factor_rule_spot` 允许策略 Agent 自主设计完整策略，不再只挑选轮动参数。因子挖掘评价信号预测能力；本入口评价组合、买卖和仓位规则在实际成交成本后的表现。

输入研究问题和创意卡，设计角色在 `parameters` 输出：

- `signals`：命名信号列表，每项为 `name` 和 `expression`。公式复用现有因子算子；空列表表示直接在规则里使用原始字段。创意卡可以携带因子公式供设计参考，但本入口不自动扫描因子库。
- `score`：数值组合评分，可引用信号名。例如 `add(momentum,mul(0.1,activity))`。
- `entry` / `exit`：独立的数值表达式，大于0触发。退出优先；已有目标持仓在未触发退出时仍有候选资格，新持仓须触发入场。比较可用 `sub`、`min`、`max`、`sign` 表达。
- `allocation`：`max_positions`、`gross_exposure`、`max_asset_weight`，分别控制最多持仓数、总目标仓位与单币上限。按评分降序选择，并列按币名升序；选中资产等权且受单币上限约束，剩余为现金。
- `rebalance_hours`：正整数；从研究起点前一根信号bar开始计时，仅在调仓时点检查入场/退出并重建目标，下一小时执行。两次调仓之间不成交；预定成交小时为合成数据时延至下一真实小时。是否成交仍受既有最小交易比例影响。

第一版支持 BTC/ETH 现货小时 OHLCV、成交额，即 `spot_open/high/low/close/volume/quote_volume`。只使用现有因果算子，不执行模型生成的 Python。所有公式的最短初始化历史需求不得超过合同 `parameter_space.warmup_hours`；在调仓时点规则值缺失会明确失败。公式不能使用未来收益标签或未接通的衍生品字段。交易仍是下一小时开盘、先卖后买、现货做多/现金；沿用已声明的缺口处理。

优化角色可以针对 `signals`、`score`、`entry`、`exit`、`allocation` 或 `rebalance_hours` 中的一个组成部分提出修改；设计角色给出新定义，其他部分保持原样。`min_return_improvement` 必须为正数收益率差，`max_drawdown_increase` 为非负回撤幅度差，0.01表示1个百分点。程序记录配对结果，不把失败的修改自动视为进步。

每版的定义记录保存原始策略与展开执行计划；`rule-values.csv` 保存组合评分和入场/退出数值，原有目标、成交与净值文件仍保存。验证冻结完整定义并使用同一执行路径。

示例位于 `examples/strategy_research/factor_rules/`：`multi-factor.replay.json` 检查多信号组合，`breakout.replay.json` 检查突破/均线退出规则。以下不带 `--replay` 的命令使用真实模型自主设计；固定回放另加对应 `--replay` 参数：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research run \
  --contract examples/strategy_research/factor_rules/contract.json \
  --idea examples/strategy_research/factor_rules/idea.json \
  --db market_data/crypto_quant.sqlite
```

示例日期及门槛沿用工程验收用途；每次运行须使用新目录。已有 `btc_eth_spot_relative_strength` 合同继续执行原轮动流程。两条路线复用同一个研究循环，没有新增一套 Agent 调度。

## RSI与现货—永续基差（2026-09-22）

`factor_rule_spot` 可用 `ts_rsi(spot_close,14)`：以前14个连续变化的上涨/下跌均值初始化，之后Wilder递推。仅无下跌为100，仅无上涨为0，两者皆0约定50；缺失后重新初始化。表达式的lookback表示最短种子历史，递归仍依赖面板初始化后的全部过去值，因此合同必须固定热身起点；示例预热168小时。公式严格低于30后穿到高于30入场，高于50退出，等于阈值时不触发。RSI平滑参考[TradingView定义](https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/)。

`spot_perp_basis` 使用单个BTCUSDT或ETHUSDT的现货、USDT永续成交价、标记价和原始资金费事件。合同固定`symbol`与`margin_guard_ratio`，模型只从合同列表选择`entry_basis`、`exit_basis`、`capital_fraction`。基差为永续收盘/现货收盘−1；达到正入场阈值开现货多头和等数量永续空头，降到退出阈值平仓，下一小时开盘双腿成交。每轮持仓数量固定；样本结束按市值估值，不强制平仓。基准是零收益现金。

现货购买扣现金，永续空头不计入卖出本金，另锁定开仓永续名义价值100%的现金作为抵押；`capital_fraction`为现货占总权益比例，必须小于0.5，实际现金不足时报错。两腿各计费用和滑点；费用压力测试不放大资金费。资金费现金流=空头数量×事件标记价×原始费率，保留原始毫秒时间。整点同刻先结算旧持仓再交易，整点之后事件结算新持仓，属于明确的执行时序假设。结算定义参考[Binance资金费说明](https://www.binance.com/en/support/faq/detail/360033525031)。

保证金检查使用开盘标记价与小时最高标记价，小时内先扣负资金费、不提前计入正资金费，触发合同风险阈值时停止并保存部分账本。这个小时级保护机制不复现交易所历史维持保证金档位或逐笔清算成交。三组小时行情和资金费必须完整，缺失不代填；本路线`imputed_hours=[]`且`min_trade_fraction=0`。

产物包括`ledger.csv`（现金、抵押、双腿数量、市值、损益、费用）、`funding-payments.csv`、`closed-trades.csv`；基础和压力场景各保存一份。沿用同一设计、核对、评估、优化和独立验证流程，冻结文件包括资金费原始快照。

可运行合同及固定回复：`examples/strategy_research/rsi_basis/`。工程验证见`experiments/strategy_research/rsi_basis_20260922/verification.md`。真实行情回放只验证执行能力；模型自主研究和正式B验证尚未启动。

## 研究任务与计算核对（2026-09-19）

合同必须显式包含 `task`，记录研究类型 `research_type`、策略家族 `strategy_family`、研究问题 `question`、交付层级 `deliverable`。可执行组合为 `strategy_backtest` / (`btc_eth_spot_relative_strength`、`factor_rule_spot` 或 `spot_perp_basis`) / `strategy_candidate`。其他组合返回 `unsupported`，不读取行情或调用模型。各阶段路线见[研究系统扩展Plan](../../../../docs/plans/研究系统扩展Plan.md)。

设计角色还须给出 `calculation_meaning`：输入、信号、成交、仓位、成本、参数六项具体含义。程序依据固定回测适配器和所选参数生成 `execution_plan`，计算角色比较原定义与执行计划，说明各项一致性、具体差异或缺少的依据，并检查假说及修改任务是否落实。

2026-09-22更新：不再要求模型逐字抄写原文。程序在 `sources` 中为六项判断绑定 `record_id` 和 `field_path`，指向该版本定义记录中的 `calculation_meaning.<维度>`、`execution_plan.<维度>`；保存的原记录就是证据来源。模型返回 `matches` 与 `reason`，并返回假说判断及原因；`true` 表示一致，`false` 表示不一致，`null` 表示无法判断。总判断规则为：任一false则false，否则任一null则null，否则true。模型审核意见不构成运行正确性的证明。

程序仍需解析可执行参数、支持字段、预热和成交时序，否则无法计算回测。模型顶层回复、设计含义和计算审核中的额外说明字段不参与解析；原始回复完整留档。版本、计划哈希、记录引用和优化任务标签不再作为停止条件。

设计、计算、评估、决策四角色仍需输出可解析的JSON，字段类型和执行所需的参数格式错误会反馈给同一角色，最多纠正三次。请求、原始回复和 `.validation-error.json` 均独立保存；第四次仍无法解析则保存 `error`。`context_bytes`保留在旧合同结构中，但不再限制模型请求。

计算角色的语义判断继续保存为研究记录；`false`或`null`不再中止回测，也不强制返回设计修改。回测仍使用程序编译的执行计划。关键计算行为由程序测试覆盖，包含RSI严格30/50边界、下一小时成交、费用、仓位和时间因果性。

每个版本保存 `*-definition.json` 和 `*-calculation.json`，下游评估与决策收到这些已保存记录。验证使用选中版本的可执行参数，不重新调用计算角色。新运行不再保存或比较运行环境版本；旧运行的`environment.json`留在原目录。

## 开发与策略内部验证（2026-09-22更新）

`run`的保留终态为`retained_for_validation`，保存`development-complete.json`：选中版本、合同和规则。该命令不会读取内部验证段。

示例正式研究合同仍使用A+B开发、原C内部验证，最终测试为模拟账户前向测试。策略入口现在接受合同显式指定的其他历史区间；结果中的实际日期与既往数据用途需据实阅读。以下验证段均指本任务的内部验证段，不指因子B。

`validate --run <目录> --db <数据库>`使用已保存的选中策略计算内部验证段。运行环境、旧证据哈希和开发快照差异不再阻止验证；同一运行可重复验证。首次结果保存在根目录，后续结果保存在`validation-attempts/attempt-xxxx/`，`status`展示最新一次。没有可执行选中策略的运行仍无法验证。

用途记录保存在data-usage.json，并传给设计与评估上下文；验证访问记录也包含用途。

验证结果单独写入`validation-result.json`与`validation-report.md`，保留开发阶段`result.json`和`report.md`不变。`status`在开发状态下附加`validation_stage`展示独立验证结果。旧运行没有新检查点，不能用这个入口补跑验证。

## 显式小时缺口处理

合同必须包含`imputed_hours`列表，空列表保持严格完整网格检查。只有列出的缺失小时可以合成OHLC，取该小时之前最近两根完整真实K线逐字段均值；排除提前结束或已经合成的K线，不使用未来价格。成交量、成交额、成交笔数和主动买入量设为0；合成时间标记`synthetic=true`。原始数据库不变。

紧邻已授权缺口之前、实际结束时间位于原小时内的短K线保留原值并标记`shortened=true`；信号仍等待名义小时结束。其他缺口与时间异常仍停止。合成小时内策略、基准和费用压力回测均禁止买卖，持仓延续，以合成收盘价作估算估值；下一真实小时按最新目标执行。合成数据及短K线会影响滚动信号与估值，属于明确披露的数据处理假设。

旧运行目录和旧合同保留作历史证据。新运行使用新增 `task` 的合同和四角色回复格式，旧三角色回放不能直接执行。

## 原轮动路线的实现范围

- 唯一可执行策略能力：现有BTC/ETH现货强弱轮动；小时收盘信号、下一小时开盘成交、先卖后买、现金或单资产多头。
- 可改参数：收益观察窗口、两币收益差阈值、强者最低窗口收益。每次修改只允许动一个已声明参数，其余规则固定。完整规则由程序附在策略定义中。
- 数据只读本地 `MarketDataStore`；检查每小时完整网格、OHLC、成交额和收盘可用时间，仅按上述显式合同合成已授权小时，不联网取行情。
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

四个模型角色使用相同 `JsonModel.complete` 接口。请求和完整回复逐次存档；不自动重试模型服务或伪造替代分析。请求按角色携带当前版本、父版本和所需开发证据；旧实验仅传结果摘要，完整记录保留在 `development_records/`，请求中的 `record_index` 列出原记录ID与哈希。开发数据目录只传给设计角色一次。超过合同的输入字节上限仍留痕报错；逐小时完整数值另存CSV，模型不能声称逐行读过未进入上下文的CSV。

## 工程回放

项目根目录执行：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research run \
  --contract examples/strategy_research/contract.json \
  --idea examples/strategy_research/idea.json \
  --db market_data/crypto_quant.sqlite \
  --replay examples/strategy_research/replay.json
```

示例运行使用本地2026年4月开发数据，5月仅为合同预留验证区间。固定回复演示v1→窗口修改任务→v2→保留v1后停止。若需检查工程验证入口，另行执行：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research validate \
  --run experiments/strategy_research/btc-eth-engineering-replay \
  --db market_data/crypto_quant.sqlite
```

v1、v2均真实回测；保留v1是预先写好的工程场景，不是模型对结果的真实研究判断。日期、参数、门槛均是工程输入，不得用作正式研究已批准的证据。

本轮验证见[第一阶段核查记录](/Users/stellan/量化投资/experiments/strategy_research/phase1_20260919_inputs/verification.md)。本地回放两版各完成设计、计算核对、评估、决策，共8次固定回复。用户授权后另以真实模型完成四角色首版研究，模型根据开发段亏损与落后基准选择淘汰，没有加载B。真实模型尚未覆盖修改与冻结验证路径，这两条路径由工程回放覆盖。

回放强制 `purpose=engineering`，从不调用外部模型服务。单独验证后的`engineering_complete`表示验证程序完成，数值条件是否满足另看`validation.passed_declared_checks`。示例日期存在既往研究使用，验证仅隔离于本次运行，不宣称从未使用的样本外证据。

输出在 `experiments/strategy_research/<run_id>/`。运行目录已存在会拒绝覆盖；需要新的工程回放时指定另一个 `--output`，保留旧证据。重复验证单独留档，其结果应按重复使用过的历史区间解释。

```sh
.venv/bin/python -m crypto_quant.cli strategy-research status \
  --run experiments/strategy_research/btc-eth-engineering-replay
```

`status` 读取最新开发状态及独立验证状态。运行每完成一个开发节点便保存检查点；开发阶段模型调用失败或进程中断后，可从有效检查点继续：

```sh
.venv/bin/python -m crypto_quant.cli strategy-research resume \
  --run experiments/strategy_research/<run_id> \
  --db market_data/crypto_quant.sqlite
```

原运行若使用固定回复回放，`resume` 也须带原 `--replay` 文件；已收到的回复会跳过。恢复不重复已完成节点，原错误结果和模型调用记录保留，后续结果另存 `resume-*.result.json`。若中断节点已留下未检查点化的定义、实验记录或部分回测文件，恢复仍会停止，因为直接重放会与已有文件冲突。旧运行没有检查点不能从此入口继续；内部验证可再次执行`validate`。

## 证据与边界

运行根目录保存创意、合同、开发/验证数据快照、模型请求与回复，以及研究记录。每个版本保存逐小时目标、净值、成交费用汇总和已结束交易CSV。重复验证的结果分目录保存；不同尝试可能使用不同的历史数据状态，应分别阅读。

每段从独立现金账户开始，预热只计算信号；期末按市价估值，不强制平仓。最大回撤包含初始本金，费用已进入净值。报告中的年化夏普是小时收益描述统计，不是HAC推断。

验证条件包括净收益、相对扣费基准的收益、回撤、交易发生小时数及成本压力结果。它们是显式合同检查，尚未包含正式研究的多重检验校正、完整Walk-Forward、组合相关性与容量审查；`retained_for_review` 只表示可以继续审查，不等于进入Paper。

暂不支持任意模型生成Python、上述三种能力以外的策略家族、动态历史币池、多个最终版本共用验证段筛选或自动执行。超出能力的创意应返回 `unsupported`。

## 验证

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_strategy_research.py' -v
```

测试覆盖真实SQLite到回测的双版本循环、下一小时执行和因果性、费用及基准、参数越界、重复验证，以及旧证据和语义判断不再阻断计算的行为。

当前在线入口默认使用小米Token Plan的`mimo-v2.6-flash`，深度思考开启；本地`.env`配置`MIMO_API_KEY`。MiMo当前没有独立的思考强度档位。追加`--provider codex`可显式选择ChatGPT订阅Luna/max。双接口配置见[项目说明](../../../../README.md)；`--replay`保持离线。

真实日期合同与无模型覆盖检查示例见[research示例](../../../../examples/strategy_research/research/README.md)。小时行情插值是用户授权的原库事后处理，局限见[处理记录](../../../../docs/data/小时行情插值处理记录.md)。
