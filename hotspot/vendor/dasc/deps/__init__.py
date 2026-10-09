"""依赖图：从源码包抽引用、打冻结快照。

两块，都在本包内：

- :mod:`dasc.deps.latexcite` —— 双抽取器（`.bbl` / 内联 / `.bib` 三路），
  移植自 `科研阅读路径/probe/t3_citation_probe.py`。
- :mod:`dasc.deps.snapshot` —— B5 冻结快照 + 内容哈希。

⚠️ **不在这里 import 导师库的任何东西。** T3 探针是用 `importlib` 直接加载
导师库里的 `rllib/extract/latex.py` 的，那会写 `.pyc` 进导师库（泄漏点②）。
移植时**刻意不抄那一处**，见 :mod:`dasc.deps.latexcite` 的模块 docstring。

⚠️ **主库（3447 个 `.src`）还没跑**——移动硬盘 `G:` 没插。
本包现在只在本地 160 个 `.src` 上验过。等插盘再指向主库。
"""

from __future__ import annotations

__all__: list[str] = []
