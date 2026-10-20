"""Hybrid retrieval: BM25 + dense vectors + RRF + optional rerank/MMR.

默认不依赖外部模型：BM25 与 dense 的排名做 RRF；配置 `rerank_model` 后
才对 top-N 调 cross-encoder。任何一步失败都退回融合排序，不阻塞问答。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..config import load_config
from ..structure.embed import embed_query
from .bm25 import BM25Index
from .vector import load_index

__all__ = ["HybridHit", "HybridResult", "rrf_fuse", "retrieve"]


@dataclass
class HybridHit:
    paper_id: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rerank_score: Optional[float] = None


@dataclass
class HybridResult:
    hits: list[HybridHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def rrf_fuse(
    rankings: dict[str, Iterable[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion。输入每个通道的 `(id, score)` 有序列表。"""
    weights = weights or {}
    fused: dict[str, float] = {}
    for source, rows in rankings.items():
        w = float(weights.get(source, 1.0))
        for rank, (pid, _score) in enumerate(rows, start=1):
            fused[pid] = fused.get(pid, 0.0) + w / (k + rank)
    return fused


def _top_from_scores(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max(0, k)]


def _dense_scores(query: str, *, cfg, index, model: str) -> list[tuple[str, float]]:
    if not query.strip():
        return []
    X, ids = index
    qv = embed_query(query, model=model)
    if X.shape[0] == 0 or qv.shape[0] != X.shape[1]:
        return []
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    scores = Xn @ qv
    k = int((cfg.get("retrieval") or {}).get("top_k_vector") or 50)
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]
    return [(ids[int(i)], float(scores[int(i)])) for i in top]


def _bm25_scores(
    bm: BM25Index, bm_ids: list[str], keywords: list[str], k: int
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for q in keywords:
        if not q.strip():
            continue
        for i, score in bm.search(q, k=k):
            pid = bm_ids[i]
            scores[pid] = max(scores.get(pid, 0.0), float(score))
    return _top_from_scores(scores, k)


def _mmr(
    ranked: list[HybridHit],
    *,
    index,
    query_vec: np.ndarray,
    k: int,
    lam: float,
) -> list[HybridHit]:
    """轻量 MMR：相关性 + 与已选论文的多样性。"""
    if not ranked or k <= 0 or index is None:
        return ranked[:k]
    X, ids = index
    row = {pid: i for i, pid in enumerate(ids)}
    selected: list[HybridHit] = []
    pool = list(ranked)
    while pool and len(selected) < k:
        best_i, best_v = 0, -1e9
        for i, hit in enumerate(pool):
            j = row.get(hit.paper_id)
            if j is None:
                score = hit.score
            else:
                rel = float(np.dot(X[j], query_vec))
                if selected:
                    sims = [
                        float(np.dot(X[j], X[row[h.paper_id]]))
                        for h in selected if h.paper_id in row
                    ]
                    div = max(sims) if sims else 0.0
                else:
                    div = 0.0
                score = lam * rel - (1.0 - lam) * div
            if score > best_v:
                best_i, best_v = i, score
        selected.append(pool.pop(best_i))
    return selected


def _cross_encoder_rerank(
    query: str,
    hits: list[HybridHit],
    *,
    corpus,
    model: str,
    top_n: int,
) -> Optional[list[HybridHit]]:
    """可选 cross-encoder；模型不可用/加载失败返回 None。"""
    if not model or not hits:
        return None
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        ce = CrossEncoder(model, device="cpu")
        pairs = []
        for h in hits[:top_n]:
            p = corpus.get(h.paper_id)
            text = f"{p.title}\n{p.abstract}" if p is not None else h.paper_id
            pairs.append((query, text[:2000]))
        scores = ce.predict(pairs)
        for h, score in zip(hits[:top_n], scores):
            h.rerank_score = float(score)
        return sorted(hits, key=lambda h: (
            h.rerank_score if h.rerank_score is not None else h.score
        ), reverse=True)
    except Exception:  # noqa: BLE001 —— 模型缺失/下载失败不阻塞
        return None


def retrieve(
    corpus,
    query: str,
    *,
    cfg=None,
    keywords: Optional[list[str]] = None,
    bm25=None,
    index=None,
    top_k: int = 300,
) -> HybridResult:
    """执行 BM25 + dense + RRF + 可选 rerank/MMR。"""
    cfg = cfg or load_config()
    ret = cfg.get("retrieval") or {}
    model = str((cfg.get("embed") or {}).get("model") or "BAAI/bge-m3")
    keywords = keywords or [query]
    notes: list[str] = []

    if index is None:
        from ..corpus import arXivCorpus  # noqa: PLC0415

        kind = "arxiv" if isinstance(corpus, arXivCorpus) else "meeting"
        index = load_index(cfg, kind=kind)
    if bm25 is None:
        from ..config import paths_from  # noqa: PLC0415
        from .bm25 import build_or_load  # noqa: PLC0415

        name = "bm25_cache_arxiv.pkl" if type(corpus).__name__ == "arXivCorpus" else "bm25_cache.pkl"
        bm25 = build_or_load(corpus, paths_from(cfg)["workspace"] / name)
    bm, bm_ids = bm25

    bm_k = int(ret.get("hybrid_bm25_k") or 300)
    vec_k = int(ret.get("hybrid_vector_k") or ret.get("top_k_vector") or 100)
    bm_rows = _bm25_scores(bm, bm_ids, keywords, bm_k)
    vec_rows = _dense_scores(
        " ".join(keywords) if keywords else query,
        cfg=cfg, index=index, model=model,
    )[:vec_k]
    fused = rrf_fuse(
        {"bm25": bm_rows, "vector": vec_rows},
        k=int(ret.get("rrf_k") or 60),
        weights={"bm25": 1.0, "vector": float(ret.get("vector_weight") or 1.0)},
    )
    bm_map = {pid: (i + 1, score) for i, (pid, score) in enumerate(bm_rows)}
    vec_map = {pid: (i + 1, score) for i, (pid, score) in enumerate(vec_rows)}
    hits = [
        HybridHit(
            paper_id=pid, score=score,
            bm25_score=bm_map.get(pid, (None, 0.0))[1],
            vector_score=vec_map.get(pid, (None, 0.0))[1],
            bm25_rank=bm_map.get(pid, (None, 0.0))[0],
            vector_rank=vec_map.get(pid, (None, 0.0))[0],
        )
        for pid, score in fused.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)

    rerank_top = int(ret.get("rerank_top_n") or 0)
    rerank_model = str(ret.get("rerank_model") or "")
    if rerank_top > 0 and rerank_model:
        reranked = _cross_encoder_rerank(
            query, hits, corpus=corpus, model=rerank_model, top_n=rerank_top
        )
        if reranked is None:
            notes.append("cross-encoder 不可用，回退 RRF 排序")
        else:
            hits = reranked

    # 可选 MMR；默认 lambda=0.75，偏相关性。
    mmr_k = min(int(ret.get("hybrid_top_k") or top_k), top_k)
    lam = float(ret.get("mmr_lambda") or 0.75)
    if hits and mmr_k > 0 and bool(ret.get("mmr_enabled", True)):
        qv = embed_query(query, model=model)
        hits = _mmr(hits, index=index, query_vec=qv, k=mmr_k, lam=lam)
    else:
        hits = hits[:mmr_k or top_k]

    return HybridResult(
        hits=hits[:top_k],
        notes=notes,
        stats={
            "bm25_candidates": len(bm_rows),
            "vector_candidates": len(vec_rows),
            "fused": len(fused),
            "reranked": bool(rerank_model),
            "returned": len(hits[:top_k]),
        },
    )
