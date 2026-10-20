"""L4 向量检索 —— **numpy 暴力余弦**（方案 §3.4：13,119×1024 fp32=54 MB，毫秒级，不引向量库）。

供两条路用：L1 方向选择器的向量召回、L4 的 LOCAL/GLOBAL 向量候选。
"""

from __future__ import annotations

import numpy as np

from ..config import load_config, paths_from
from ..structure.embed import load_all

__all__ = ["load_index", "cosine_topk", "query_topk"]

#: 进程内向量索引缓存：key = (workspace, 分片文件名/大小/mtime)。
_INDEX_CACHE: dict[tuple, tuple[np.ndarray, list[str]]] = {}


def load_index(cfg=None, *, kind: str = "meeting"):
    """加载全库向量索引。返回 `(X, ids)`，`X` 已 L2 归一化。"""
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    subdir = "arxiv_vectors" if kind == "arxiv" else "vectors"
    files = sorted((ws / subdir).glob("vec_*.npz"))
    key = (
        str(ws), subdir,
        tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files),
    )
    hit = _INDEX_CACHE.get(key)
    if hit is not None:
        return hit
    X, ids = load_all(ws / subdir)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if len(_INDEX_CACHE) >= 4:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)), None)
    _INDEX_CACHE[key] = (X, ids)
    return X, ids


def cosine_topk(query_vec: np.ndarray, X: np.ndarray, ids: list[str], k: int = 50
                ) -> list[tuple[str, float]]:
    """`query_vec`（dims=1024）对全库余弦，返回 top-k `[(id, score)]`。"""
    q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    scores = X @ q
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    return [(ids[i], float(scores[i])) for i in top]


def query_topk(query_vec: np.ndarray, k: int = 50, *, cfg=None
               ) -> tuple[list[tuple[str, float]], dict]:
    """便捷版：即载即查。返回 `(top_k, meta{index_n, index_dim})`。"""
    X, ids = load_index(cfg)
    hits = cosine_topk(query_vec, X, ids, k=k)
    return hits, {"index_n": len(ids), "index_dim": X.shape[1]}
