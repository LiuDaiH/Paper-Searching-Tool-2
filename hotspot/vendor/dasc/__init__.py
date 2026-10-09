"""dasc —— 依赖感知的序贯覆盖最大化（Dependency-Aware Sequential Coverage）。

大创「方向 1：RL 学检索与探索策略」的算法侧实现。**本包对导师库
`rl_acdemical_paper/` 只读，永不写入。**

为什么这个包在导师库**外面**
------------------------------

`科研阅读路径/规范/语料库验收要求.md` §C 的原话：「新项目位置落在库外
`obsidian\\dasc\\`，**不放在导师库里面**，否则『永不写入』只剩口头承诺」。

放在里面的话，`sys.path` 会把导师库根目录带进来，`import rllib` 变成随手一挥；
`__pycache__` 会长在同一棵树上；而"我没写它"这句话就再也没有机制背书了。

⚠️ 闸① 必须在**任何** import 之前装上
--------------------------------------

导师库的两个已知泄漏点之一是「`import rllib` 会写 `.pyc` 进库内」。
`sys.dont_write_bytecode = True` 只在**赋值之后**才生效——如果先有人
`import rllib` 再 `import dasc`，闸就晚了，而**晚装的闸不会报错**，
它只是失效。所以它放在本文件顶层：`import dasc` 这一下就把闸装上，
早于任何子模块。真正兜住"晚装"那种情况的是 `readonly.py` 的闸③（运行期断言）。

四道闸的完整说明见 :mod:`dasc.readonly`。
"""

from __future__ import annotations

import sys

# ── 闸① 禁用字节码写入 ──────────────────────────────────────────────────
# 必须在任何可能 import 导师库的代码之前执行。见模块 docstring。
sys.dont_write_bytecode = True


# ── 兜住「控制台编码崩掉整个进程」─────────────────────────────────────────
# 中文 Windows 的控制台/管道默认是 **cp936(GBK)**。汉字在里面没问题，但
# `⇒`(U+21D2)、`⚠`(U+26A0) 这类符号**不在 GBK 里** ⇒ `print` 当场抛
# `UnicodeEncodeError`。而它是在**打印的半路**炸的：前面几行结论已经出来、
# 后面几行永远不出来，退出码变成 1 —— **一份自检的结论被一个字符吃掉了**。
#
# 2026-09-20 实测（当天的自检份数；**份数会变，别照抄这个数**，见 run_all.discover）：
#   同一份 `python -m dasc.checks.check_coverage`，前者 `exit=1`、后者 `exit=0`；
#   当天那份清单里**五份**在默认编码下崩（readonly / clusterers / llm 三份不崩，
#   因为它们的输出碰巧没用到非 GBK 字符）。
# ⚠️ 更要紧的是**它之前是怎么被瞒住的**：那轮"全绿"是在
#   `PYTHONIOENCODING=utf-8` 下跑的 —— **绿的成立条件没人写下来**。
#
# 修法分两种情形，判据是 `isatty()`：
#   · **接着控制台**（人在终端里看）⇒ **不动编码**（动了汉字在 GBK 控制台上
#     会变乱码），只把错误策略放宽成 `replace` ⇒ 非 GBK 字符退化成 `?`。
#   · **被重定向/管道**（落文件、被别的东西读）⇒ 直接改成 UTF-8 ⇒ 汉字与
#     符号都原样保留，下游按 UTF-8 读就对了。Git Bash(mintty) 交互也算这种。
# 两种情形都带 `errors="replace"`：**宁可少一个符号，不可少一条结论。**
# 本包所有入口（`checks/*`、`llm/pilot.py`、`cluster/clusterers.py` 等）都先
# import 本模块，所以放在这里一处就够。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(
            encoding=None if _stream.isatty() else "utf-8",
            errors="replace",
        )
    except (AttributeError, ValueError):       # 被重定向成非文本流 / 已 detach
        pass

__all__ = ["readonly", "corpora", "ratelimit"]
