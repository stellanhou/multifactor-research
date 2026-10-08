# 跨仓库与证据读取测试样例

`fm_v6_workflow_card.json` 由原因子挖掘 `_idea_card` 导出。因子仓库验证相同输入仍生成此格式，多因子仓库验证此格式可被消费；测试不要求安装另一系统。

`multifactor-engineering-20261002.tar.gz` 来自原仓库本地 `experiments/strategy_research/multifactor_v1/multifactor-engineering-20261002`，purpose=engineering、agent_used=false。原证据读取与篡改检测测试硬编码依赖此未跟踪目录，因此拆仓时将这一固定测试样例纳入版本控制。测试在临时目录解包，仅重定位 result.root，保留卡片、输入、账本及全部原有断言。此文件仅用于工程回归，不作为新的研究或策略成绩。

压缩文件 SHA-256：`ccf51d19ad93669256e94502c64d148b4d6db30c1883a0a388a74f9c46cfe359`。
