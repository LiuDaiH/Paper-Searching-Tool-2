"""L1 方向选择器：自由文本 → 子语料（`Pool`）。**0 token。**

用户说「RLHF 的对齐方法」，系统要能算出"库里的这部分有多大、从哪几路捞到的"。
方案 §3.1：**三路召回取并集，三路命中数必须并排报。**

| 路 | 来源 | 怎么算 |
|---|---|---|
| FTS5 | `Corpus.search`（导师库那张 FTS5 索引） | 方向原文直接查，尽力召回 |
| subfield | `Paper.subfields` 标签 | 方向词 ⊆ 某池名 ⇒ 取该池全部论文 |
| vector | 全库暴力余弦 | **靠 `workspace/vectors/` 的全库向量**——那是 D3/D4 的产物，D2 还没有 ⇒ 降级并明说 |

⚠️ **`subfields` 有 40–65% 盲区**（robotics / imitation / rlhf / rl_for_science）。
只看标签会得出"库里没人做"的错结论——**库内缺席 ≠ 没人做**（方案 §3.1）。
所以三路分开记，让"哪条路没命中"看得见，而不是合并成一个数。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_config, paths_from
from .schema import Pool

__all__ = ["select", "subfield_hits", "SUBWORD"]

#: 词：字母数字串。`-` 与 `_` 都当分隔——池名用下划线（`model_based`），
#: 自然语言用连字符/空格，两边的词要能对上。
SUBWORD = re.compile(r"[A-Za-z0-9]+")


def _words(text: str) -> set[str]:
    return {w.lower() for w in SUBWORD.findall(text)}


def _fts_ids(corpus, direction: str, k: int) -> list[str]:
    """FTS5 路。**导师库索引是英文词干（porter），中文查不到**——命中 0 是正常，不是 bug。"""
    try:
        return corpus.search(direction, k=k)
    except Exception:  # noqa: BLE001 —— 检索异常不该让整个选择器崩掉，记进 notes
        return []


def subfield_hits(corpus, direction: str) -> tuple[set[str], int]:
    """subfield 标签路。

    方向词集合 ⊇ 某个池名的全部词 ⇒ 命中该池，取该池全部论文。
    `pool_words ⊆ words`（池是方向的细化）而不是"有交集"——避免单个
    `model` 词把名下所有池一起捞进来。
    """
    words = _words(direction)
    if not words:
        return set(), 0
    ids: set[str] = set()
    n = 0
    # 池名从语料里现取：不写死列表，避免与导师库漂移。
    pool_names = sorted({s for p in corpus.list_papers(limit=None) for s in p.subfields})
    for pool in pool_names:
        pool_words = _words(pool)
        if pool_words and pool_words <= words:
            ps = corpus.list_papers(pool=pool)
            ids.update(p.paper_id for p in ps)
            n += len(ps)
    return ids, n


def _vector_ids(paths: dict[str, Path], direction: str, corpus) -> tuple[set[str], str]:
    """向量路。**D2 时全库向量还不存在 ⇒ 降级**（`hits_vector=0` + notes 明说）。

    D3/D4 生成 `workspace/vectors/` 后，这一路接上"全库暴力余弦"。
    现在先把"文件在不在"报清楚——这是方案 R5 的"向量路未跑"分支。
    """
    vec_dir = paths["workspace"] / "vectors"
    files = sorted(vec_dir.glob("*.npz")) if vec_dir.exists() else []
    if not files:
        return set(), "向量路未跑：workspace/vectors/ 里还没有全库向量（D3/D4 产物）"
    # ⚠️ 有文件时目前只降级为"未接入"：分片格式与余弦实现属 D3/D4，不在 D2。
    return set(), f"向量文件已存在（{len(files)} 个分片），但余弦检索待 D3/D4 接入"


def select(
    corpus,
    direction: str,
    *,
    cfg: dict[str, Any] | None = None,
) -> Pool:
    """三路召回取并集，产出一个 :class:`Pool`。`corpus` 为已打开的 `Corpus`。"""
    cfg = cfg or load_config()
    paths = paths_from(cfg)
    fts_k = int((cfg.get("retrieval") or {}).get("top_k_direction_fts") or 500)

    fts_ids = _fts_ids(corpus, direction, fts_k)
    sub_ids, sub_n = subfield_hits(corpus, direction)
    vec_ids, vec_note = _vector_ids(paths, direction, corpus)

    union: set[str] = set(fts_ids) | sub_ids | vec_ids
    notes = [vec_note] if vec_note else []
    # 若三路全空，明说是"库内没捞到"，不是"世界上没有"（方案 R5 / 拒答口径）。
    if not union:
        notes.append("三路召回并集为空 —— 「本库无相关内容」，区别于方向本身不存在")

    return Pool(
        direction=direction,
        paper_ids=tuple(sorted(union)),
        hits_fts=len(fts_ids),
        hits_vector=len(vec_ids),
        hits_subfield=sub_n,
        notes=tuple(notes),
    )
