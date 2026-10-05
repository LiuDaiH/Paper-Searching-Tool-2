"""每周新增 arXiv 增量处理管线（用户核心诉求：以后新增的怎么处理，不折腾历史）。

读 `config.corpus.arxiv_source` 的 `papers.jsonl`（append 制）。用「水位」
`workspace/arxiv_rl/watermark.json` 记录上次已处理的全量 paper_id 集合，流程：
  1. 检测新增 = 当前全集 − 上次水位
  2. 增量向量化：**只对新增论文** embedding，落 `arxiv_vectors/vec_delta_*.npz`
     （按已向量化 id 去重；历史分片完全不碰，不跑全量）
  3. 增量引用边：`fetch_arxiv_refs(only=新增 arxiv_id)`，并入 `cite_graph_arxiv.pkl`
  4. 更新发布物：`export_arxiv_weekly` 重新打包（新增自动并入 vectors/meta/topics/manifest）
  5. 水位前移 = 当前全集

- `--init`：首次把「当前全库」设为水位基线，**跳过历史**，之后只处理每周新增。
- 全程幂等、断点可重入；每周/每日调一次即完成增量入库 + 发布。
- 红线不变；数据缺失明确报错，不假装。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))   # scripts/ 本身
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 项目根

import numpy as np

from hotspot.config import load_config, paths_from
from hotspot.corpus import open_arxiv_corpus

__all__ = ["update_arxiv_weekly", "main"]


def _watermark_path(cfg) -> Path:
    return paths_from(cfg)["workspace"] / "arxiv_rl" / "watermark.json"


def _load_wm(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def _embed_delta(c, new_ids, out_dir: Path, model: str, chunk_size: int = 500) -> int:
    """只对新增论文 embedding，追加到 `vec_delta_*.npz`；按已向量化 id 去重。"""
    from dasc.cluster.embed import build_docs  # noqa: E402
    import md2vectors  # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from hotspot.structure.embed import load_all
        _, existing = load_all(out_dir)
        existing = set(existing)
    except FileNotFoundError:
        existing = set()
    todo = [pid for pid in new_ids if pid not in existing]
    if not todo:
        print("  （新增均已向量化，跳过 embedding）")
        return 0
    seq = len(list(out_dir.glob("vec_delta_*.npz")))
    total = 0
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        papers = [p for pid in chunk if (p := c.get(pid))]
        if not papers:
            continue
        ids2, docs = build_docs(papers)
        entries, dim = md2vectors.embed(ids2, docs, model)
        vecs = np.array([entries[pid]["embedding"] for pid in ids2], dtype=np.float32)
        np.savez_compressed(out_dir / f"vec_delta_{seq:04d}.npz",
                            vectors=vecs, ids=np.array(ids2), model=np.array([model]),
                            dim=np.array([dim]), with_title=np.array([True]))
        seq += 1
        total += len(ids2)
        print(f"  delta 分片 {len(ids2)} 篇（新增 embedding）")
    return total


def update_arxiv_weekly(cfg, *, init: bool = False, limit: int | None = None,
                        delay: float = 1.6, skip_refs: bool = False,
                        skip_export: bool = False, publish_dir: str | None = None) -> dict:
    ws = paths_from(cfg)["workspace"]
    rl_dir = ws / "arxiv_rl"
    rl_dir.mkdir(parents=True, exist_ok=True)
    wm_path = _watermark_path(cfg)
    wm = _load_wm(wm_path)
    processed = set(wm.get("processed") or [])

    c = open_arxiv_corpus(cfg)
    all_ids = [p.paper_id for p in c.list_papers()]
    all_set = set(all_ids)

    if init:
        wm["processed"] = list(all_ids)
        wm_path.write_text(json.dumps(wm, ensure_ascii=False), encoding="utf-8")
        print(f"--init：水位 = 当前全库 {len(all_ids)} 篇（历史跳过；未来只处理新增）")
        if not skip_export:
            from export_arxiv_weekly import export_arxiv_weekly
            export_arxiv_weekly(cfg, publish_dir=publish_dir)
        return {"init": True, "n": len(all_ids)}

    new_ids = [pid for pid in all_ids if pid not in processed]
    if limit is not None:
        new_ids = new_ids[:limit]
    print(f"增量：上次处理 {len(processed)} / 当前 {len(all_ids)} → 新增 {len(new_ids)}")

    # 1) 增量向量化（只对新增，历史不碰）
    em = cfg.get("embed") or {}
    model = em.get("model") or "BAAI/bge-m3"
    n_emb = _embed_delta(c, new_ids, ws / "arxiv_vectors", model=model,
                         chunk_size=int(em.get("chunk_size") or 500))

    # 2) 增量引用边（只对新增 arxiv_id）
    if not skip_refs:
        from fetch_arxiv_refs import fetch_arxiv_refs
        new_arxiv = {p.arxiv_id for p in c.list_papers()
                     if p.paper_id in set(new_ids) and p.arxiv_id}
        fetch_arxiv_refs(cfg, limit=limit, delay=delay, only=new_arxiv)

    # 3) 更新发布物
    if not skip_export:
        from export_arxiv_weekly import export_arxiv_weekly
        export_arxiv_weekly(cfg, publish_dir=publish_dir)

    # 4) 水位前移 = 当前全集
    wm["processed"] = list(all_ids)
    wm_path.write_text(json.dumps(wm, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 增量完成；水位前移到 {len(all_ids)} 篇（embedding 新增 {n_emb}）")
    return {"new": len(new_ids), "n": len(all_ids), "embedding_new": n_emb}


def main() -> int:
    ap = argparse.ArgumentParser(description="每周新增 arXiv 增量处理管线")
    ap.add_argument("--init", action="store_true",
                    help="首次：把当前全库设为水位基线，跳过历史，之后只处理新增")
    ap.add_argument("--limit", type=int, default=None, help="试跑：只处理前 n 篇新增")
    ap.add_argument("--delay", type=float, default=1.6, help="引用拉取每篇节流秒数")
    ap.add_argument("--skip-refs", action="store_true", help="跳过引用拉取")
    ap.add_argument("--skip-export", action="store_true", help="跳过发布物打包")
    ap.add_argument("--publish-dir", default=None, help="发布物额外拷贝目录")
    a = ap.parse_args()
    update_arxiv_weekly(load_config(), init=a.init, limit=a.limit, delay=a.delay,
                        skip_refs=a.skip_refs, skip_export=a.skip_export,
                        publish_dir=a.publish_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
