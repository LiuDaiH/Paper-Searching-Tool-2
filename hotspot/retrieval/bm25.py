"""BM25 词法检索（hybrid 的词法侧）—— 2026-09-28。

替换 `MergedCorpus.search` 的**全词 AND 子串匹配**：BM25 按相关度打分排序，
部分命中也召回（按分），长度归一，与向量检索构成 **词 × 义 hybrid**。

自写 `BM25Index`（纯标准库，不依赖外部包）；索引一次离线构建、缓存到
`workspace/bm25_cache.pkl`（对应 `corpus.list_papers` 顺序），可重建。
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str | None) -> list[str]:
    return _WORD.findall((text or "").lower())


class BM25Index:
    """经典 BM25（Okapi）打分器。"""

    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        for d in docs:
            toks = _tok(d)
            self.doc_len.append(len(toks))
            c = Counter(toks)
            self.tf.append(c)
            for w in c:
                self.df[w] += 1
        self.N = len(docs)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

    def _idf(self, w: str) -> float:
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) if n else 0.0

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        q = {w: self._idf(w) for w in set(_tok(query)) if self.df.get(w, 0)}
        if not q:
            return []
        avg = max(self.avgdl, 1e-9)
        scores: list[float] = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_len[i]
            tf = self.tf[i]
            s = 0.0
            for w, idf in q.items():
                f = tf.get(w, 0)
                if f:
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
            scores[i] = s
        top = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i]) for i in top if scores[i] > 0]


def build_or_load(corpus, cache_path: Path):
    """构建或加载 BM25 索引 + 对应 paper_id 列表（缓存，可重建）。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    papers = corpus.list_papers(limit=None)
    ids = [p.paper_id for p in papers]
    docs = [f"{p.title or ''} {p.abstract or ''}" for p in papers]
    idx = BM25Index(docs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump((idx, ids), fh)
    return idx, ids


def search(bm: BM25Index, ids: list[str], query: str, k: int = 50) -> list[str]:
    """查询 → top-k paper_id（score>0 才返回）。"""
    return [ids[i] for i, _ in bm.search(query, k=k)]
