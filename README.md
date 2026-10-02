# 加密货币量化研究项目

## 本地控制台

在项目根目录运行 `.venv/bin/python -m crypto_quant.web_console.server`，然后打开 `http://127.0.0.1:8766/`。左侧按数据、因子挖掘、创意卡池、策略研究、市场扫描、模拟执行和运行记录组织常用流程；完整命令保存在“高级工具”。数据页展示已保存的现货、永续、资金费、持仓和独立清算数据盘点，标明盘点日期与覆盖边界；采集计划位于“高级工具”。因子页按研究来源列出每个已保存候选的名称、公式、含义和实际研究阶段，能切换查看历史 Goal 与独立运行；因子目标直接填写问题与成果数量。创意卡池只读取已写入 `experiments/idea_pool/` 的卡片，按来源展示社区、市场扫描及通过准入的因子创意，并保留待研究状态与原始记录入口。策略研究可先检查开发数据，再从可用方案和匹配的创意卡中选择，页面为每次运行生成新编号。模拟执行先选环境和会话，运行前展示本次填写的设置。长任务可在页面查看输出或中断；从命令行或其他对话启动的因子 Goal 也会在最近任务和因子页自动更新状态、阶段、入池数量与错误。已有策略研究和因子目标也会列入运行记录。控制台只监听本机地址，使用已有环境变量中的模型和测试网配置。

统一运行环境：**Python 3.12.13**，项目根目录 `.venv`。数据采集、因子计算、回测、研究Agent和测试均使用这个环境。

## 项目目录

输入位于 `market_data/` 和 `examples/` 的研究合同；`src/crypto_quant/` 负责处理；运行结果及证据保存在 `experiments/`。命令入口为 `src/crypto_quant/cli.py`，四个因子角色位于 `src/crypto_quant/research/factor_mining/`。文档索引见[项目文档](docs/README.md)，检查代码位于 `tests/`，辅助脚本位于 `scripts/`。

Git 保存源码、测试、合同示例和文档；`market_data/`、`experiments/`、`scratch/` 为本地数据及运行空间。FM-v6 是因子挖掘的唯一执行主线，正式入口使用 [V6 研究合同](examples/factor_mining/research.contract.json)。旧版本迁移、压缩和重跑工具已退出源码与默认测试；当前 [操作说明](docs/guides/因子挖掘Agent使用说明.md) 不依赖旧 Goal。

## 数据说明

[研究数据划分与使用规范](docs/data/研究数据划分与使用规范.md)：因子 A/B、策略内部开发与验证、最终测试的分工，以及既有数据暴露与工程验证记录。

[小时行情插值处理记录](docs/data/小时行情插值处理记录.md)：2026-09-22获准直接补写的现货／标记价格、方法和研究限制。

[本地数据总览](docs/data/本地数据总览.md)：已有数据类别、字段含义、文件位置、ABC覆盖，以及三类清算数据的使用状态与缺口。

## 安装与检查

```bash
uv sync --frozen --all-extras
.venv/bin/python --version
make test
```

`.python-version`固定解释器，`pyproject.toml`声明依赖，`uv.lock`锁定完整版本。`requirements-lock.txt`是从同一锁文件导出的清单。修改依赖后更新锁及导出文件；日常运行使用`--frozen`，避免自动升级。

```bash
uv lock
uv export --all-extras --frozen --no-emit-project --format requirements-txt --output-file requirements-lock.txt
```

现有数据环境的 NumPy 2.0.2、Pandas 2.3.3、Matplotlib 3.9.4、PyArrow 21.0.0 版本保持一致。原根目录Python 3.13环境保存在`.venv-before-codex`；原`market_data/.venv-liquidations`保留作迁移前环境，日常命令统一改用`.venv/bin/python`。移动后的旧环境不作为可直接运行的入口。

## LangChain 与 LangGraph 研究运行层

因子挖掘、策略研究和市场扫描共用LangChain消息链；MiMo由ChatOpenAI执行，Codex订阅及工程回放通过受控传输适配器进入同一链。静态角色输出使用严格LangChain解析器，动态因子Schema继续按原契约校验；受控原文读取使用StructuredTool，模型不能越过记录ID和数据阶段白名单。

LangGraph负责以下角色交接和循环：

- 因子：构想→逐候选计算/修复→评估→优化→下一轮或完成；证据分页、三次格式纠正也使用图分支。
- 策略：设计→能力/数据检查→语义审核→回测→评估→决策；语义修正和策略优化显式返回设计。
- Goal：选择任务→探索→准备冻结→B验证→成果登记；缺数据进入等待，达到原目标才结束。
- 扫描单轮模式：获取行情→检测异动→解释并保存创意卡；dry-run跳过模型调用。持续扫描由行情流驱动，见下文。

CLI命令名保持不变。LangGraph不另建第二套检查点数据库。自然循环使用显式运行配置，避免框架默认步数上限变成研究停止规则。因子挖掘仍保留其数据和统计准入规则；策略研究的版本、历史证据及重复验证拦截已移除，具体边界见模块说明。

研究流程运行时命令行会显示当前阶段／角色、模型调用尝试、重试等待和耗时；耗时步骤每30秒显示一次“进行中”。进度事件逐行保存为JSONL，启动时会打印日志绝对路径。因子和策略运行的日志位于运行目录旁的`<run_id>.progress.jsonl`；Goal主日志位于Goal目录旁，子运行日志位于Goal目录的`run_progress/`；市场扫描的单轮模式日志位于创意库同级的`scan_runs/<scan_id>/progress.jsonl`，持续模式的异动记录位于`scan_events/`。在另一个终端用`tail -f`加启动时打印的日志路径即可持续查看。日志不包含模型请求、回复或密钥，也不参与研究证据冻结；原始请求和回复仍按原有证据目录保存。

迁移前运行保留作历史证据。代码变更不会单独阻止续跑，恢复仍须通过合同、数据和已保存证据检查。新流程验收见[重构验收](experiments/langchain_agent_refactor_20260923/verification.md)，范围和步骤见[重构Plan](docs/plans/LangChain重构Plan.md)。

## 实时市场扫描

`scan-live` 默认持续运行：启动时从币安公开REST接口取得5分钟历史基线，然后订阅30币现货及U本位永续的1分钟K线。每根分钟K线收盘后计算滚动5分钟价格收益和永续成交额；OI的新5分钟观测每分钟查询一次，已结算资金费率每5分钟检查一次。上述指标任一触发就写入`experiments/scan_events/`。断线后的分钟数据从REST补齐，数据超过2分钟未更新时不参与当次判断。同币种5分钟内持续触发的记录合为一个事件，AI在后台根据首次触发事实生成一张创意卡；后续触发保留在事件记录中。AI失败仍保留事件。

```bash
# 实时观察两个币，保存事件但不调用模型
.venv/bin/python -m crypto_quant.cli scan-live --symbols BTCUSDT,ETHUSDT --dry-run
# 原有1小时单轮扫描仍可用于对照
.venv/bin/python -m crypto_quant.cli scan-live --once --dry-run
```

默认本地只读接口为`http://127.0.0.1:8765/status`和`http://127.0.0.1:8765/events?limit=100`，可用`--port`修改。正式模式的事件位于`experiments/scan_events/`，`--dry-run`事件位于`experiments/scan_events_dry_run/`，创意卡继续写入`experiments/idea_pool/`。按`Ctrl-C`停止。启动需要先补齐历史数据；5分钟时限从行情流正常运行且数据新鲜时计算。分位数仍沿用现有1%/99%规则，5分钟口径的阈值还需历史回放校准。

## Agent模型

默认使用小米 Token Plan 中国区接口 `https://token-plan-cn.xiaomimimo.com/v1`，模型 `mimo-v2.6-flash`，深度思考开启。在项目根目录 `.env` 填写 `MIMO_API_KEY=...`，或设置同名环境变量；环境变量优先。密钥不进入研究记录，也不提交 Git。MiMo 当前只区分思考开启/关闭，`enabled` 是此接口可用的最高设置。

所有在线入口共用 `--provider mimo|codex` 选择接口。`--model` 未指定时按接口选择默认模型；`--thinking enabled|disabled` 只用于小米，`--reasoning-effort` 只用于 Codex。默认单次超时300秒，可用 `--timeout-seconds` 调整。接口失败不会自动换模型或换供应商。

```bash
# 默认小米模型列表
.venv/bin/python -m crypto_quant.cli factor-mine models
# 保留原有 ChatGPT 订阅 Luna 入口
.venv/bin/python -m crypto_quant.cli factor-mine models --provider codex
# 同样可在 strategy-research run、factor-mine explore、scan-live 后追加：
# --provider codex --model gpt-5.6-luna --reasoning-effort max
```

Luna 仍通过官方 Codex Python SDK（`openai-codex==0.154.0`）和 ChatGPT 订阅登录，默认推理强度 `max`，原有会话隔离与工具限制保留。小米使用 Chat Completions 协议，直接传入完整消息；原始回复、用量及接口配置留档。小米支持合同声明的输出上限，Luna 仍要求 `output_tokens=null`。扫描 `--dry-run` 与策略 `--replay` 不调用模型。Goal 恢复严格使用已保存的接口配置，不随当前默认模型切换。

各角色的回复仍需通过 Pydantic 解析字段、类型及执行参数。各流程最多纠正三次（加上首次回复，最多四次回复尝试）；超过次数停止并留档。额外说明字段不执行，交易定义仍由策略编译器检查。扫描格式错误不再生成占位创意卡；请求、原始回复及校验错误保存到创意库同级 `scan_model_calls/`。因子挖掘的数据及证据检查保持独立；策略研究的具体停止条件见模块说明。

小米调用通过 LangChain `ChatOpenAI`（`langchain-openai==1.6.3`）和 OpenAI SDK 执行。SDK对连接中断、超时及可重试HTTP错误最多自动重试2次（含首次共3次）；耗尽后停止，不叠加上层网络重试。响应中途断开同样进入SDK重试边界。原始响应、HTTP错误及接收中断记录经密钥脱敏后保存，模型、深度思考和JSON输出约定保持不变；原有Pydantic回复纠正继续负责格式错误。接口设置记录库版本及重试次数，旧接口运行不能冒充新接口原地续跑。

小米参数参考：[官方 Chat Completions 文档](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)。

历史报告和检查点保留。因子与策略研究均不再因代码指纹变化而拒绝续跑；策略研究也不再比较Python和依赖库版本。已保存输入或结果发生变化时，应依据各次记录判断其可比性。

详细操作见[因子挖掘Agent使用说明](docs/guides/因子挖掘Agent使用说明.md)和[策略研究说明](src/crypto_quant/research/strategy_research/README.md)。官方接口参考：[Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)、[认证](https://learn.chatgpt.com/docs/auth)。

策略研究新增 `factor_rule_spot`：由 Agent 定义命名信号、组合评分、入场/退出、仓位与调仓间隔，程序执行含成本回测；支持价量多因子和规则策略。首版输入仍为 BTC/ETH 小时现货，例子见 `examples/strategy_research/factor_rules/`。因子挖掘继续独立评价信号预测能力，原轮动合同仍可运行。

策略研究已接通 Wilder RSI 算子与 `spot_perp_basis` 等数量现货多/永续空双腿账本（含原始资金费、双腿成本和小时保证金保护）；示例见 `examples/strategy_research/rsi_basis/`，能力范围和工程验证见策略研究模块说明。

`research-data-policy` 可查看默认研究日期；`strategy-research check-data --help` 可查看数据检查入口。策略研究接受合同指定的其他历史区间，真实区间合同示例见 `examples/strategy_research/research/`。采集、旧单项研究与数据诊断命令仍是通用工具，其结果不自动获得正式研究资格。
