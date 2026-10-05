"""R3 周报未来被引 RF 重训 —— 仿 `dasc/rlr/export_future_rf`，改用周报数据
（规划-周报接入大创算法.md）。

读周报 `workspace/arxiv_rl/`：
  mobile_corpus_vectors_arxiv.npz + rl_prep/global_mean.npy（R2 产）
  cite_graph_arxiv.pkl（R1 产：`workspace/depgraph/`）
  论文 `year`（arXivCorpus）—— 时间轴批次

early/late 按 `year` 时间留出划分；未来标签 = 早批论文是否被晚批引用。
落 `workspace/arxiv_rl/rl_prep/future_rf_arxiv.pkl`：
  {"rf", "pca", "years", "early_ids", "nrev24"}
⚠️ 无未来引用信号（图空/太稀）→ 明确报"no future-cite signal"，不假装。
⚠️ 只在**已向量化子集**（arxiv_vectors 覆盖范围）内训练——全量需先全量向量化。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from hotspot.config import load_config, paths_from
from hotspot.corpus import open_arxiv_corpus

__all__ = ["train_future_rf_arxiv", "main"]

NTREES = 100
PCA_D = 20
SEED = 42


def _mean_dir_shift(V, vector_m, mean):
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    Vc = Vn - mean
    Vc = Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), 1e-12)
    return Vc


def train_future_rf_arxiv(cfg, *, pca_dim: int = PCA_D, ntrees: int = NTREES,
                          seed: int = SEED, early_until: int | None = None) -> Path:
    np.random.seed(seed)
    ws = paths_from(cfg)["workspace"]
    rl_dir = ws / "arxiv_rl"
    prep_dir = rl_dir / "rl_prep"
    vec_path = rl_dir / "mobile_corpus_vectors_arxiv.npz"
    graph_path = ws / "depgraph" / "cite_graph_arxiv.pkl"
    if not vec_path.exists() or not (prep_dir / "global_mean.npy").exists():
        raise FileNotFoundError(f"缺周报 rl_prep（{rl_dir}）——先跑 scripts/prep_arxiv_rl.py")
    if not graph_path.exists():
        raise FileNotFoundError(f"缺引用图（{graph_path}）——先跑 scripts/fetch_arxiv_refs.py")

    c = open_arxiv_corpus(cfg)
    # 统一标识符 = paper_id（与 arxiv_vectors 的 ids 及 R1 引用图对齐）
    years = {p.paper_id: p.year for p in c.list_papers()
             if p.paper_id and p.year is not None}

    z = np.load(vec_path, allow_pickle=False)
    V = z["vectors"].astype(np.float32)
    vec_ids = [str(x) for x in z["ids"]]
    mean = np.load(prep_dir / "global_mean.npy").astype(np.float32)
    idx = {ar: i for i, ar in enumerate(vec_ids)}

    with open(graph_path, "rb") as f:
        fwd = {k: set(v) for k, v in pickle.load(f)["forward"].items()}

    # early/late 时间留出
    in_years = [y for pid, y in years.items() if pid in idx]
    if not in_years:
        raise RuntimeError("语料内无 year 可作时间轴")
    if early_until is None:
        sy = sorted(in_years)
        early_until = int(np.percentile(sy, 60))
        while early_until >= sy[-1] and len(sy) > 1:
            early_until -= 1   # 保证晚批至少 1 篇
    early = [pid for pid, y in years.items() if y <= early_until and pid in idx]
    late = [pid for pid, y in years.items() if y > early_until and pid in idx]
    if not late:
        raise RuntimeError(f"无晚批论文（early_until={early_until}）——无法做时间留出")

    future = {e: 1 if any(s in set(late) for s in fwd.get(e, ())) else 0 for e in early}
    if sum(future.values()) == 0:
        raise RuntimeError("no future-cite signal（引用图内早批→晚批引用为 0）"
                           "——先补全引用图（fetch_arxiv_refs，或配 S2 key）")

    Vc = _mean_dir_shift(V, V, mean)
    early_idx = np.array([idx[e] for e in early])
    sub = np.array([idx[e] for e in early[::15]])
    if len(sub) < 2:
        raise RuntimeError("early 样本过少，无法 PCA——需更多已向量化论文或更大引用图")
    k = min(pca_dim, len(sub) - 1)
    pca = PCA(k).fit(Vc[sub])
    P = pca.transform(Vc)

    early_set = set(early)
    nrev, out = {}, {}
    for e in early:
        nrev[e] = sum(1 for s in fwd.get(e, ()) if years.get(s) == early_until)
        out[e] = sum(1 for s in fwd.get(e, ()) if s in early_set)

    X = np.column_stack([P[early_idx],
                         np.array([[nrev[e], out[e]] for e in early])])
    y = np.array([future[e] for e in early])
    clf = RandomForestClassifier(n_estimators=ntrees, random_state=seed, n_jobs=-1)
    clf.fit(X, y)

    model_path = prep_dir / "future_rf_arxiv.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"rf": clf, "pca": pca, "years": years,
                     "early_ids": early, "nrev24": nrev}, f, protocol=4)
    pos = int(y.sum())
    print(f"周报 RF 已导出 → {model_path}")
    print(f"  early_until={early_until} early={len(early)} late={len(late)} "
          f"future+={pos} ({100*float(np.mean(y)):.1f}%) dim={X.shape[1]}")
    if pos < 20:
        print("⚠️ 未来正样本过少（<20）—— 引用图不足，此模型不可靠；"
              "补全引用后重训（fetch_arxiv_refs 或配 S2 key）")
    return model_path


def main() -> int:
    ap = argparse.ArgumentParser(description="R3 周报未来被引 RF 重训")
    ap.add_argument("--early-until", type=int, default=None,
                    help="早批 year 上限（默认=年份中位），> 它为晚批")
    a = ap.parse_args()
    train_future_rf_arxiv(load_config(), early_until=a.early_until)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
