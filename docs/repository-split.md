# 仓库拆分记录

- 日期：2026-10-08。
- 源仓库：`stellanhou/crypto-quant-research-platform`。
- 源分支：`codex/factor-mining-next`。
- 源提交：`7f5ddbfb3ce1041ea50157a753c1ecb091a8ec73`。
- 新仓库：`stellanhou/multifactor-research`，私有，默认分支 `main`。
- 历史提取仅保留本仓库相关文件，提交身份及时间保留，过滤后的提交哈希改变。完整提交保留在原仓库。
- 在提取历史后复制当前工作区选定文件，纳入已存在的未提交改动，原工作区及暂存区不变。
- 未迁移市场数据、运行结果、虚拟环境、凭据、统一控制台和其他无关业务。
- 独立安装、命令帮助及本仓库测试用于验证拆分。

共享模块仍保留原路径，以维持证据中的源码路径及现有调用契约。此后各仓库分别维护；跨仓库交付使用 FM-v6 创意卡和显式数据路径。

证据读取回归测试所需的一份固定工程快照（压缩后约 2.3 MB）保存在 `tests/fixtures/`，其来源与边界见测试样例说明；完整历史结果仍留在原目录。

## 拆分验收

- 独立虚拟环境执行 `uv sync --frozen --all-extras` 成功。
- 本仓库完整测试：333 passed，90 subtests passed（283.61 秒）；14 条 Matplotlib/Pyparsing 弃用提示，无失败。
- `multifactor-research strategy-research --help` 和 `research-data-policy` 成功。
- 隔离模式确认包从本仓库加载，无因子挖掘 Goal 模块；所有源码内部导入闭合。
- 相对原工作区的运行代码变化仅为独立命令入口及支持包初始化；研究计算代码逐字节一致。
- 原工作区暂存区和未提交文件列表保持不变。
