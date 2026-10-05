"""引用层「共同来源」：基于聚合引用图 `paper_refs.json` 算两主题的共同被引 arXiv。

依赖 `scripts/agg_refs.py` 的产物 `workspace/depgraph/paper_refs.json`：
- `paper_refs`: {paper_id → [被引 arxiv_ids]}
- `ref_to_papers`: {arxiv_id → [引用它的 paper_ids]}

思路（2026-09-30）：
两主题各自论文章 → 各自的被引 arXiv 集合 → **共同被引**（两主题都引用的 arXiv）＝
两主题共享的骨干/根 → 若该 arXiv 在语料里（`arxiv_id` 命中），反推出其来源会议/标题。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

__all__ = ["load_paper_refs", "arxiv_lookup", "norm_arxiv", "co_cited_core"]


def _default_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workspace" / "depgraph" / "paper_refs.json"


@lru_cache(maxsize=1)
def load_paper_refs(path: str | Path | None = None) -> dict:
    """惰性加载聚合引用图（缓存一次）。缺失时返回空结构。"""
    p = Path(path) if path else _default_path()
    if not p.exists():
        return {"paper_refs": {}, "ref_to_papers": {}, "stats": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"paper_refs": {}, "ref_to_papers": {}, "stats": {}}


def norm_arxiv(a: str) -> str:
    """arXiv 号归一化（无空格小写；去版本后缀）——新旧式都收。"""
    a = a.strip().lower()
    return re.sub(r"v\d+$", "", a)


@lru_cache(maxsize=1)
def arxiv_lookup(corpus) -> dict[str, object]:
    """语料里 `arxiv_id` → Paper 的映射（供反推来源会议/标题）。"""
    d: dict[str, object] = {}
    for p in corpus.list_papers():
        if p.arxiv_id:
            d.setdefault(norm_arxiv(p.arxiv_id), p)
    return d


def co_cited_core(groups, prefs: dict | None = None, corpus=None,
                  top: int = 8, min_common: int = 1) -> list[dict]:
    """两（或多）主题论文组 → 共同被引 arXiv 列表（按共引频次降序）。

    - `groups`: 每个元素是一组 `paper_id`（来自检索/聚类）。
    - 返回条目含 arXiv、被引频次、语料归属（标题/会议）。
    """
    prefs = prefs or load_paper_refs()
    paper_refs = prefs.get("paper_refs") or {}
    group_sets: list[set] = []
    for g in groups:
        s: set[str] = set()
        for pid in g:
            for a in paper_refs.get(pid, []):
                s.add(a)
        group_sets.append(s)
    if len(group_sets) < 2 or not all(group_sets):
        return []

    common = set.intersection(*group_sets) if group_sets else set()
    if len(common) < min_common:
        return []

    counts: dict[str, int] = {}
    union_pids: set[str] = set().union(*groups)
    for pid in union_pids:
        for a in paper_refs.get(pid, []):
            if a in common:
                counts[a] = counts.get(a, 0) + 1

    al = arxiv_lookup(corpus) if corpus is not None else {}
    out: list[dict] = []
    for a in common:
        p = al.get(norm_arxiv(a))
        out.append({
            "arxiv_id": a,
            "cited_by": counts.get(a, 0),
            "in_corpus": bool(p),
            "title": getattr(p, "title", "") if p else "",
            "venue": getattr(p, "venue", "") if p else "",
        })
    out.sort(key=lambda x: (-x["cited_by"], -x["in_corpus"]))
    return out[:top]
