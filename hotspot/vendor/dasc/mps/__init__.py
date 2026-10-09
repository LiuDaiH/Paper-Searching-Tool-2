"""dasc.mps —— Streit-Garg 线：MPS（Matroidal Prerequisite Systems）的实现。

arXiv 2607.08151，Streit & Garg, *Approximation Algorithms for Matroidal
Prerequisite Systems*, to appear APPROX 2026, cs.DS/cs.DM。

**学术定位**（见 `论文趋势挖掘/计划/Streit-Garg线-MPS工程实现计划.md`）：
给这套刚成形的理论框架提供**第一个真实实例 + 第一个实证评测**，并
服务移动云杯的论文趋势分析查询工具的「关键路径/必备先序」类回答
（`移动云杯/热点挖掘系统-方案设计.md` §「与大创互鉴」）。

本包是**纯算法核心**，不含任何 LLM / 网络 / 导师库写入。输入是
(𝕌, ℳ, ↝) 的抽象（论文：元素；主题簇：matroid；引用缩点：poset），
输出是可行词/近似比。工具与数据组装在 `dasc/mps/notes`、`_out/mps`。

模块
----
- :mod:`matroid`：matroid 抽象（rank/span/独立盘）—— 语义可替换 + partition 独立集
- :mod:`poset`：poset 抽象（理想/主理想/偏序），引用图缩点
- :mod:`core`：MPS 本体 —— 可行词语言 / Δ / conn / gtest(=r∘ideal) / closure
- :mod:`algorithms`：四个近似算法（Δ-, (1+conn)-, (2+conn)-, 随机子模）
- :mod:`objective`：目标函数（可加权重 / 子模覆盖）
- :mod:`instance`：从 rl_prep + 移动云杯簇组装真实实例
"""
