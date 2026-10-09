"""把池子里的论文摘要变成向量。**复用库根 `工具/md2vectors.py`，一行模型代码都不重写。**

为什么是"复用"而不是"再写一个"
--------------------------------

库根 `工具/md2vectors.py`（**在导师库外面**，所以 import 它不踩闸②）已经把
bge-m3 在这台机器上的坑全踩完了，三条都写在它的常量里：

| 常量 | 值 | 为什么 |
|---|---|---|
| `HF_HOME` | `D:\\hf_cache` | **C 盘满会导致 Python 段错误**，权重 2.2 GB 不能放 C |
| `MAX_SEQ_LENGTH` | 1024 | bge-m3 是 XLM-R-large（568M），8k 上限在 CPU 上每篇要几分钟 |
| `BATCH_SIZE` | 4 | MX330 只有 2 GB 显存，装不下 fp32 的 2.3 GB 权重 ⇒ 纯 CPU |

⚠️ **这三个值一个都别改。** 改了不是"调参"，是把那几小时的踩坑重走一遍。

⚠️ 为什么要自己设离线开关，而不是让 `md2vectors.main()` 去设
--------------------------------------------------------------

它的离线判定写在 `main()` 里，**不在 `embed()` 里**。直接调 `embed()` 的话
`HF_HUB_OFFLINE` / `HF_ENDPOINT` 全都没设——而 huggingface.co 从这里**不可达**。
所以 :func:`prepare_offline` 把那段逻辑抄了过来（抄的是行为，不是常量，
所以没有 `check_ratelimit_sync` 那种漂移风险；但**它仍是抄的**，
`md2vectors.py` 哪天改了那段，这里要跟着改）。

向量**不是交付物**
------------------

`科研阅读路径/规范/语料库结构与向量化.md` §6 的原话：「向量化不是本项目的交付物」、
「向量检索在这里的作用是辅助检索/去重/聚类，**不是主指标**」。
所以本模块的产物落在 `dasc/_out/`（已 gitignore），是**中间物**——
它喂给聚类，聚类喂给贡献标签，链子的尽头是覆盖度，不是向量。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

#: 库根。`md2vectors.py` 2026-09-20 从库根搬到了 `工具/`，所以这里指 `工具/` 而不是根。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "工具"))

import md2vectors  # noqa: E402  —— 库根 `工具/` 下的那个；在导师库外面

from dasc.corpora import Corpus  # noqa: E402

__all__ = ["prepare_offline", "build_docs", "embed_pool", "OUT_DIR"]

#: 产物落点。**在 `dasc/` 自己这边**，不进导师库，也不入库（见 `.gitignore`）。
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "_out"

MODEL = md2vectors.DEFAULT_MODEL


def prepare_offline(model: str = MODEL) -> tuple[bool, str]:
    """按 `md2vectors.main()` 的逻辑设离线开关。**必须在构造模型之前调。**

    返回 ``(权重在不在本机, 说明)``。**"在不在"是要报出来的**——
    不在就意味着这次会走网络，而网络这件事本库的做法是一律明说，
    不悄悄依赖。见 `vault-curl-empty-body` 那条：这个环境的抓取路子本来就窄。
    """
    cached = md2vectors.model_cache_exists(model)
    if cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    where = os.environ.get("HF_HOME") or "(默认)"
    return cached, (
        f"权重已在本机，离线跑（HF_HOME={where}）"
        if cached
        else f"⚠️ 权重不在 {where}，这次要联网（走 hf-mirror）"
    )


def build_docs(papers, *, with_title: bool = True) -> tuple[list[str], list[str]]:
    """把论文变成待嵌入的文本。返回 ``(ids, docs)``。

    **标题 + 摘要**，不是光摘要。标题是最高信号的一行（十来个 token），
    丢掉它没有好处；`语料库验收要求` 的 B4 也记着"标题匹配是主力"。

    ⚠️ 摘要为空时**不要把标题当成摘要混过去**——那种条目会被聚成一个
    看着正常的簇，而它其实是"我们没拿到摘要"。本库栽过这个方向。
    这里照实记数并报出来。
    """
    ids: list[str] = []
    docs: list[str] = []
    for p in papers:
        body = (p.abstract or "").strip()
        if with_title:
            head = (p.title or "").strip()
            text = f"{head}\n\n{body}" if head and body else (head or body)
        else:
            text = body
        ids.append(p.paper_id)
        docs.append(text)
    return ids, docs


def embed_pool(
    pool: str = "rlhf",
    *,
    limit: int | None = None,
    out: Path | None = None,
) -> dict[str, object]:
    """把一个池子向量化并落盘。返回一份**实测**报告（不是"预计"）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with Corpus() as c:
        papers = c.list_papers(pool=pool, limit=limit)
        empty = [p.paper_id for p in papers if not (p.abstract or "").strip()]
        ids, docs = build_docs(papers)

    cached, note = prepare_offline()
    print(f"模型：{MODEL}")
    print(f"离线：{note}")
    print(f"池子：{pool}  篇数：{len(papers)}  摘要为空：{len(empty)}  上限：{limit}")
    print(f"设备：CPU（MX330 装不下 fp32 的 2.3 GB 权重）"
          f"  MAX_SEQ_LENGTH={md2vectors.MAX_SEQ_LENGTH}  BATCH_SIZE={md2vectors.BATCH_SIZE}")
    print()
    if empty:
        print(f"⚠️ 有 {len(empty)} 篇摘要为空，它们的向量代表的是标题而不是摘要：{empty[:5]}")
        print()

    t0 = time.perf_counter()
    entries, dim = md2vectors.embed(ids, docs, MODEL)
    dt = time.perf_counter() - t0

    vectors = np.array([entries[i]["embedding"] for i in ids], dtype=np.float32)
    per = dt / max(len(ids), 1)
    print()
    print(f"完成：{vectors.shape} 维={dim}  用时 {dt:.1f}s（{per:.2f}s/篇）")
    if limit:
        print(f"→ 按此速率推算全池（586 篇）：约 {per * 586 / 60:.1f} 分钟")

    out = out or (OUT_DIR / f"{pool}_vectors.npz")
    np.savez_compressed(
        out,
        vectors=vectors,
        ids=np.array(ids),
        model=np.array([MODEL]),
        dim=np.array([dim]),
        with_title=np.array([True]),
    )
    print(f"已落盘：{out}（{out.stat().st_size / 1024:.0f} KB，**中间物、不入库**）")
    return {
        "pool": pool,
        "n": len(ids),
        "dim": dim,
        "seconds": dt,
        "sec_per_paper": per,
        "empty_abstract": len(empty),
        "cached_weights": cached,
        "out": str(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="把某个池的摘要向量化（bge-m3，CPU，离线）")
    ap.add_argument("--pool", default="rlhf")
    ap.add_argument("--limit", type=int, default=None,
                    help="先拿 n 篇试跑计时（计划要求：先 50 篇，再决定全量）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    embed_pool(a.pool, limit=a.limit, out=Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
