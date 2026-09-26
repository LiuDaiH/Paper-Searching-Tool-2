"""24+25 合并语料源（方案 A，2026-09-24）。

读 G 盘两份 `papers.jsonl`（`G:\\rl_data24\\资源` + `G:\\rl_data\\资源`），并成
一套 22,747 篇的 `MergedCorpus`，接口与 `dasc.Corpus` 兼容（`list_papers` / `get`
/ `search` / `stats` + no-op `lib`），供 direction / embed / cluster / ui 使用。

**保留 `year` 分组能力**：每篇 `Paper.year`（2024/2025）都在；`list_papers(year=...)`
可按年筛；簇的跨会三态默认为**跨年混合**，但 `VenueProfile.counts_by_year` 保留
分年计数（见 `structure/venue_matrix.py`）。主口径仍跨会、不做跨年，24 仅辅助验证。

**为什么不复制 dasc**：继续复用 dasc 的聚类/LLM 等设施；这里只做一个"合并+读 jsonl"
的薄适配器，字段映射到现有 `Paper` 契约。
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .config import load_config

__all__ = ["Paper", "MergedCorpus", "open_corpus"]

#: 英文子串检索用（大小写不敏感，按词匹配以略提升召回）
_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Paper:
    """与 dasc 的 Paper 契约对齐的最小实现（保留 `year`）。"""

    paper_id: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    arxiv_id: str = ""
    is_rl: bool = False
    subfields: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.paper_id


def _load_source(path: Path) -> list[Paper]:
    out: list[Paper] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rl = r.get("rl") or {}
            sub = tuple(s for s in (rl.get("subfield") or []) if s)
            out.append(Paper(
                paper_id=r.get("paper_id") or "",
                title=r.get("title") or "",
                abstract=r.get("abstract") or "",
                year=r.get("year"),
                venue=r.get("venue") or "",
                arxiv_id=r.get("arxiv_id") or "",
                is_rl=bool(rl.get("is_rl")),
                subfields=sub,
            ))
    return out


class MergedCorpus:
    """把 24+25 两份 jsonl 并成一套语料。`paper_id` 并集（交集为 0，去重兜底）。"""

    def __init__(self, sources: Iterable[str | Path] = ()):
        if not sources:
            cfg = load_config()
            sources = (cfg.get("corpus") or {}).get("sources") or []
        by_id: dict[str, Paper] = {}
        for s in sources:
            for p in _load_source(Path(s)):
                by_id.setdefault(p.paper_id, p)  # 先 24 后 25；冲突保留先来
        self._by_id = by_id
        self._papers = list(by_id.values())

    # ---- 查询 ---------------------------------------------------------- #

    def list_papers(self, limit: Optional[int] = None, pool: str | None = None,
                    rl_only: bool = False, year: int | None = None) -> list[Paper]:
        out = self._papers
        if rl_only:
            out = [p for p in out if p.is_rl]
        if year is not None:
            out = [p for p in out if p.year == year]
        if pool:
            key = _norm(pool)
            out = [p for p in out if any(_norm(s) == key for s in p.subfields)]
        if limit is not None:
            out = out[:limit]
        return out

    def get(self, paper_id: str) -> Paper | None:
        return self._by_id.get(paper_id)

    def search(self, query: str, k: int = 50) -> list[str]:
        """轻重：标题/摘要大小写不敏感子串匹配（22,747 篇上每问 O(n)，够用）。"""
        terms = [t for t in _WORD.findall(query.lower()) if t]
        if not terms:
            return []  # 无检索词（如纯中文方向）不匹配任何篇，而不是全命中
        hits: list[str] = []
        for p in self._papers:
            hay = (p.title + " " + (p.abstract or "")).lower()
            if all(t in hay for t in terms):
                hits.append(p.paper_id)
        return hits[:k]

    def stats(self) -> dict:
        return {
            "papers": len(self._papers),
            "is_rl": sum(1 for p in self._papers if p.is_rl),
            "pools": {s for p in self._papers for s in p.subfields},
        }

    # ---- 兼容 `with c.lib`（no-op，读 jsonl 无写面） ---------------------- #

    @contextmanager
    def lib(self) -> Iterator[None]:
        yield


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def open_corpus(cfg=None) -> MergedCorpus:
    """从 config 的 `corpus.sources` 打开合并语料。"""
    sources = ((cfg or load_config()).get("corpus") or {}).get("sources") or []
    return MergedCorpus(sources)
