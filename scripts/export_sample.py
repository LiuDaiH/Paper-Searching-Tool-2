"""B 档：生成「样例数据集」—— 从全量 24+25 会议语料与 arXiv 周报切一个可运行子集,
供别人零准备、不依赖G盘/原始语料地跑通检索/聚类（独立应用快照的 workspace 数据）。

产出：`workspace/sample/`（可打包进样例数据集的精简数据）
  sample_meeting.jsonl  sample_arxiv.jsonl   语料子集（字段与源一致）
  vec_meeting.npz  vec_arxiv.npz            向量子集（embed 兼容：vectors/ids/model/dim/with_title）
  bm25_meeting.pkl  bm25_arxiv.pkl          BM25 子缓存

选中约 `--meeting`（默认1200）/`--arxiv`（默认600）篇，覆盖 is_rl 与各年；
种子固定（--seed 42）保证可复现。只读全量、只写 sample，不改动现网 workspace 结构。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hotspot.config import load_config, paths_from
from hotspot.retrieval.bm25 import build_or_load as bm_build
from hotspot.structure.embed import load_all

__all__ = ["generate_sample", "main"]

MODEL = "BAAI/bge-m3"


def _choose(pids: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return rng.sample(sorted(set(pids)), min(n, len(set(pids))))


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_source_lines(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _slice_vectors(npz_dir: Path, chosen: list[str], out: Path) -> None:
    """从全量向量库按 paper_id 抽子集行，落成 embed 兼容的单个 npz。"""
    X, ids = load_all(npz_dir)          # 全量 (X, ids)，ids 为 paper_id
    idx = {pid: i for i, pid in enumerate(ids)}
    rows = [i for pid in chosen if (i := idx.get(pid)) is not None]
    if not rows:
        print(f"  ⚠️ 无向量可切：{npz_dir}")
        return
    sub = X[rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        vectors=sub.astype(np.float32),
                        ids=np.array([ids[i] for i in rows]),
                        model=np.array([MODEL]), dim=np.array([sub.shape[1]]),
                        with_title=np.array([True]))
    print(f"  向量子集 {sub.shape} → {out}")


def generate_sample(cfg=None, *, meeting: int = 1200, arxiv: int = 600,
                    seed: int = 42, out_dir: Path | None = None) -> Path:
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    m_root = out_dir or (ws / "sample")

    # 1) 全量语料行读入 + 选样
    m_src = [Path(s) for s in ((cfg.get("corpus") or {}).get("sources") or [])]
    a_src = (cfg.get("corpus") or {}).get("arxiv_source")
    m_rows = _read_source_lines(m_src)
    a_rows = _read_source_lines([Path(a_src)]) if a_src else []
    m_pids = [r["paper_id"] for r in m_rows if r.get("paper_id")]
    a_pids = [r["paper_id"] for r in a_rows if r.get("paper_id")]
    m_chosen = set(_choose(m_pids, meeting, seed))
    a_chosen = set(_choose(a_pids, arxiv, seed))
    m_sub = [r for r in m_rows if r["paper_id"] in m_chosen]
    a_sub = [r for r in a_rows if r["paper_id"] in a_chosen]

    _write_jsonl(m_sub, m_root / "sample_meeting.jsonl")
    _write_jsonl(a_sub, m_root / "sample_arxiv.jsonl")
    print(f"会议子集 {len(m_sub)} 篇 → {m_root/'sample_meeting.jsonl'}")
    print(f"arxiv 子集 {len(a_sub)} 篇 → {m_root/'sample_arxiv.jsonl'}")

    # 2) 向量子集
    print("[向量切片]")
    _slice_vectors(ws / "vectors", [r["paper_id"] for r in m_sub],
                   m_root / "vec_meeting.npz")
    _slice_vectors(ws / "arxiv_vectors", [r["paper_id"] for r in a_sub],
                   m_root / "vec_arxiv.npz")

    # 3) BM25 子缓存（对子集重建）
    print("[BM25 重建]")
    from hotspot.corpus import MergedCorpus
    mc = MergedCorpus([str(m_root / "sample_meeting.jsonl")])
    with mc.lib():
        bm_build(mc, m_root / "bm25_meeting.pkl")
    ac = MergedCorpus([str(m_root / "sample_arxiv.jsonl")])
    with ac.lib():
        bm_build(ac, m_root / "bm25_arxiv.pkl")
    print(f"BM25 子缓存 → {m_root/'bm25_meeting.pkl'}, {m_root/'bm25_arxiv.pkl'}")

    return m_root


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 B 档样例数据集（独立可运行的 workspace 数据）")
    ap.add_argument("--meeting", type=int, default=1200)
    ap.add_argument("--arxiv", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="输出目录（默认 workspace/sample）")
    a = ap.parse_args()
    out = Path(a.out) if a.out else None
    generate_sample(load_config(), meeting=a.meeting, arxiv=a.arxiv,
                    seed=a.seed, out_dir=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
