# 加密货币量化研究项目

统一运行环境：**Python 3.12.13**，项目根目录 `.venv`。数据采集、因子计算、回测、研究Agent和测试均使用这个环境。

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

## Agent模型

使用官方 **Codex Python SDK**（`openai-codex==0.154.0`），其依赖自带匹配的Codex运行时。项目只接受ChatGPT订阅登录；认证及凭据刷新由官方组件完成。

```bash
codex login
.venv/bin/python -m crypto_quant.cli factor-mine models
.venv/bin/python -m crypto_quant.cli factor-mine explore --help
.venv/bin/python -m crypto_quant.cli strategy-research run --help
.venv/bin/python -m crypto_quant.cli scan-live --help
```

默认模型`gpt-5.6-luna`、推理强度`max`，每次调用默认等待300秒。通过`--model`、`--reasoning-effort`、`--timeout-seconds`显式调整。因子挖掘、Goal、策略研究、市场扫描解读及历史候选重评共享该适配器。扫描`--dry-run`及策略`--replay`仍然不调用模型。

每次角色调用均使用独立临时会话和受限工具配置；研究证据由现有程序传入。Codex运行时仍会携带全局`AGENTS.md`规则，项目文档和个人记忆已禁用。失败不会自动切换到付费API或其他模型。因子及策略研究记录请求、结果和用量，SDK版本附在原始回复中。因子合同需设置`output_tokens=null`，SDK不支持指定单次输出token上限。

历史运行绑定代码与Python/NumPy/Pandas指纹，不能修改旧合同后原地续跑。旧报告和检查点保留；新环境的正式研究应使用新的运行ID。此次模型与环境迁移本身不表示因子通过B段检验。

详细操作见[因子挖掘Agent使用说明](因子挖掘Agent使用说明.md)和[策略研究说明](src/crypto_quant/research/strategy_research/README.md)。官方接口参考：[Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)、[认证](https://learn.chatgpt.com/docs/auth)。
