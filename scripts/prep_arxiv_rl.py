"""R2 周报 rl_prep 构建 —— 从 P3 的 `arxiv_vectors/` 整理成大创风格前置数据
（规划-周报接入大创算法.md）。

产出 `workspace/arxiv_rl/`：
  mobile_corpus_vectors_arxiv.npz   {vectors(4369×1024 fp32), ids}   ← 供 RF 特征/向量检索
  rl_prep/global_mean.npy           归一化向量的均值                 ← Vc = Vn - mean
  （`cite_graph_arxiv.pkl` 由 R1 拉到 `workspace/depgraph/`，训练时读取）

纯本地、可重建；缺 `arxiv_vectors` 时明确报错（先跑 P3 的 export_arxiv_weekly）。
⚠️ 若 arxiv_vectors 只覆盖子集（如 --limit 小样测过），R3 训练也只在子集上做——
   全量意义需先全量向量化（export_arxiv_weekly 不带 --limit，约 3h）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hotspot.config import load_config, paths_from
from hotspot.structure.embed import load_all

__all__ = ["build_rl_prep", "main"]


def build_rl_prep(cfg) -> Path:
    ws = paths_from(cfg)["workspace"]
    vec_dir = ws / "arxiv_vectors"
    out_dir = ws / "arxiv_rl"
    prep_dir = out_dir / "rl_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    V, ids = load_all(vec_dir)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    mean = Vn.mean(axis=0).astype(np.float32)

    np.savez_compressed(out_dir / "mobile_corpus_vectors_arxiv.npz",
                        vectors=V, ids=np.array(ids))
    np.save(prep_dir / "global_mean.npy", mean)
    print(f"周报 rl_prep 已写 → {out_dir}")
    print(f"  向量 {V.shape[0]} 篇 × {V.shape[1]}（ids {len(ids)}）; global_mean {mean.shape}")
    return out_dir


def main() -> int:
    build_rl_prep(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
