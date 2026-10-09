"""聚类：摘要 → 向量 → 簇。**这一层的产物是中间物，不是交付物。**

对外只暴露 :func:`embed_pool`（向量化）与 :func:`prepare_offline`（离线开关）。
簇划分在 :mod:`dasc.cluster.clusterers`——**刻意与向量化分开**：
前者要几分钟 CPU，后者是纯 numpy，分开才能只重跑变的那一半。
"""

from __future__ import annotations

from .embed import OUT_DIR, build_docs, embed_pool, prepare_offline

__all__ = ["OUT_DIR", "build_docs", "embed_pool", "prepare_offline"]
