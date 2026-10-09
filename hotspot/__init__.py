"""hotspot —— 本地论文库研究方向热点挖掘（移动云杯）。

语料 = 导师论文库 `rl_acdemical_paper/`，**只读，永不写入**。
复用 `obsidian/dasc/` 的既有设施，**不复制代码**（避免两份漂移）。

为什么本包在导师库外面、也在 dasc 外面
--------------------------------------

`dasc/` 是「方向 1」的算法侧实现，本包是移动云杯这条新方向。两条方向
各自的 README 要说清各自的事，所以拆成两个目录、两个包。

但**设施是同一套**：四道闸只读网关、窄接口、LLM 客户端、聚类，全在
`dasc/` 里已经落地并自检过。这里只做 `sys.path` 接线，不抄代码。

⚠️ 闸① 必须在**任何** import 之前装上
--------------------------------------

与 `dasc/__init__.py` 同一条理由：`sys.dont_write_bytecode = True` 只在
**赋值之后**生效。晚装的闸不报错，它只是失效。所以它放在本文件顶层，
早于任何子模块、也早于 `import dasc`。

四道闸的完整说明见 `dasc.readonly`。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 闸① 禁用字节码写入 ──────────────────────────────────────────────────
# 必须在任何可能 import 导师库的代码之前执行。见模块 docstring。
sys.dont_write_bytecode = True


# ── 兜住「控制台编码崩掉整个进程」─────────────────────────────────────────
# 与 `dasc/__init__.py` 同一处理，长理由在那边（此处不重复）：
#   · 接着控制台 ⇒ **不动编码**，只放宽 `errors="replace"`
#   · 被重定向/管道 ⇒ 换 UTF-8
# 两种情形都带 `errors="replace"`：**宁可少一个符号，不可少一条结论。**
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(
            encoding=None if _stream.isatty() else "utf-8",
            errors="replace",
        )
    except (AttributeError, ValueError):       # 被重定向成非文本流 / 已 detach
        pass


# ── 接线：把 vendor 的 dasc（自包含副本）放 sys.path 最前，让 `import dasc`
# 打到本仓库 `hotspot/vendor/dasc/` 而不是外部 obsidian/dasc。工具/ 一并放上
# （md2vectors.py 是库根散件，dasc/cluster/embed.py 也是这么引的）。
# 只有一套环境，不做可配置抽象（方案 §不做清单）。
_VENDOR = Path(__file__).resolve().parent / "vendor"
_OBSIDIAN = Path(r"D:\大学编程学习\obsidian")

for _p in (_OBSIDIAN / "工具", _OBSIDIAN, _VENDOR):   # 后插的 vendor 在最前 → 优先命中自包含 dasc
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
del _p, _VENDOR


# ── 本项目的落点 ────────────────────────────────────────────────────────
#: 本项目根（`移动云杯/`）
HOTSPOT_ROOT: Path = Path(__file__).resolve().parents[1]

#: 产物目录。可全量重建，不入库。
WORKSPACE: Path = HOTSPOT_ROOT / "workspace"

#: dasc 自包含副本（vendor 化，`import dasc` 指向这里）
DASC_ROOT: Path = Path(__file__).resolve().parent / "vendor" / "dasc"

#: 导师论文库（只读）。访问一律走 `dasc.readonly.LibraryView`。
LIBRARY_ROOT: Path = _OBSIDIAN / "rl_acdemical_paper"

__all__ = [
    "HOTSPOT_ROOT",
    "WORKSPACE",
    "DASC_ROOT",
    "LIBRARY_ROOT",
]
