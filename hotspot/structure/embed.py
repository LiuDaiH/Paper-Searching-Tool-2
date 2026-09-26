from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import hotspot  # noqa: F401 —— 装闸① + 接线 `工具/`
import md2vectors  # noqa: E402 —— 库根 `工具/` 下的；在导师库外面，import 不踩闸②

from dasc.cluster.embed import build_docs, prepare_offline  # noqa: E402  —— 复用，不重写

__all__ = ["embed_corpus", "chunk_path", "load_all", "DEFAULT_OUT"]


#: 默认落点：`workspace/vectors/`（config `paths.workspace` 下）。可全量重建，不入库。
def DEFAULT_OUT() -> Path:
    return hotspot.WORKSPACE / "vectors"


def chunk_path(out_dir: Path, batch: int) -> Path:
    """第 `batch` 块（0 起）的落盘名。**名字由块号定，续跑据此跳过。**"""
    return out_dir / f"vec_{batch:04d}.npz"


def embed_corpus(
    corpus,
    *,
    out_dir: Path | None = None,
    chunk_size: int = 500,
    limit: int | None = None,
    model: str = md2vectors.DEFAULT_MODEL,
) -> dict:
    """把全库（或 `limit` 篇）摘要向量化，**分块落盘、可续跑**。

    复用 `md2vectors.embed`（CPU，bge-m3，离线）。与 `dasc.cluster.embed.embed_pool`
    的区别就在续跑：**每 `chunk_size` 篇落一个分片，已完成的分片不重算**——
    全库 ~8.5h，这是 R1 的唯一对策。

    返回一份**实测**进度报告（不是"预计"）。
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT()

    papers = corpus.list_papers(limit=limit)
    ids, docs = build_docs(papers)
    n = len(ids)
    if n == 0:
        return {"n": 0, "done": 0, "skipped": 0, "note": "没有论文可向量化"}

    cached, note = prepare_offline(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (n + chunk_size - 1) // chunk_size
    done = 0       # 本次新算的篇数
    skipped = 0    # 已存在分片里的篇数（续跑省下的）
    t0 = time.perf_counter()
    total_sec: dict[int, float] = {}

    for b in range(n_batches):
        lo, hi = b * chunk_size, min((b + 1) * chunk_size, n)
        out_path = chunk_path(out_dir, b)
        if out_path.exists():
            skipped += hi - lo
            continue
        chunk_ids = ids[lo:hi]
        chunk_docs = docs[lo:hi]
        tb = time.perf_counter()
        entries, dim = md2vectors.embed(chunk_ids, chunk_docs, model)
        vectors = np.array([entries[pid]["embedding"] for pid in chunk_ids], dtype=np.float32)
        np.savez_compressed(
            out_path,
            vectors=vectors,
            ids=np.array(chunk_ids),
            model=np.array([model]),
            dim=np.array([dim]),
            with_title=np.array([True]),
        )
        total_sec[b] = time.perf_counter() - tb
        done += hi - lo
        print(f"  分片 {b + 1}/{n_batches} 实际写入（{lo}–{hi}，{hi - lo} 篇）"
              f"  用时 {total_sec[b]:.1f}s")

    dt = time.perf_counter() - t0
    per = dt / max(done, 1)
    return {
        "n": n,
        "done": done,
        "skipped": skipped,
        "batches": n_batches,
        "chunk_size": chunk_size,
        "dim": int(np.load(chunk_path(out_dir, 0))["dim"][0]) if done and n_batches else None,
        "seconds": dt,
        "sec_per_paper": per,
        "cached_weights": cached,
        "note": note,
    }


def load_all(out_dir: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """读回全库向量（多个分片合并）。`(vectors, ids)`，顺序即分片顺序。"""
    if out_dir is None:
        out_dir = DEFAULT_OUT()
    files = sorted(out_dir.glob("vec_*.npz"))
    if not files:
        raise FileNotFoundError(f"{out_dir} 下没有 vec_*.npz —— 先跑分块向量化")
    vecs, all_ids = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        vecs.append(z["vectors"])
        all_ids.extend(str(x) for x in z["ids"])
    return np.concatenate(vecs, axis=0), all_ids


def main() -> int:
    ap = argparse.ArgumentParser(
        description="全库摘要分块向量化（bge-m3，CPU，离线，可续跑）")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="先拿 n 篇试跑计时（探针 1：--limit 50）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from ..corpus import open_corpus

    c = open_corpus()
    with c.lib():
        rep = embed_corpus(
            c,
            out_dir=Path(a.out) if a.out else None,
            chunk_size=a.chunk_size,
            limit=a.limit,
        )

    print()
    print(f"全量 {rep['n']} 篇 → 新算 {rep['done']} 篇 / 跳过已存在 {rep['skipped']} 篇"
          f"（{rep['batches']} 个分片 × {rep['chunk_size']}）")
    print(f"离线：{rep['note']}")
    if rep["sec_per_paper"] and rep["done"]:
        print(f"实测 {rep['sec_per_paper']:.2f} s/篇，本批用时 {rep['seconds']:.1f}s"
              f" → 推算全库约 {rep['sec_per_paper'] * 13119 / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
