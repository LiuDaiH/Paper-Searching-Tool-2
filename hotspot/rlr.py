"""rlRAG 线接入移动云杯 —— 用**测试性能最好的算法**（RF(S+A) 先验）排序。

对应 `热点挖掘系统-方案设计.md` §「与大创互鉴」的 **RL/bandit 学检索策略**。

**为什么不用 RL，用 RF(S+A)**（2026-09-30 用户拍板 + 测试数据）：
`dasc/rlr` 的三阶段实测一致表明 **RL(GRPO) 追不上监督先验，RF(S+A) 是性能
最好的**（详见 `dasc/rlr/future.py`、`future_rl.py`）：
- `future_report.json`：**rf(S+A)** auc=0.7471 > lr 0.747 > heat 0.6804
- `future_rl_report_cover.json`：rawrf **8.72** > rl 8.18 > heat 6.82
- `future_rl_report_recall.json`：rawrf **0.1031** = rl 0.1031 > heat 0.0869
- `rl_report.json`：rl(cells 2.17) 全面远差于 greedy(9.25)

结论（`future_rl.py` docstring）：RL 需真 MDP 结构才有用；本语料「单点排序+去重」
RL 永远 >= heat 但追不上 RF。**所以本模块用 RF(S+A) 先验，不用 RL 本身。**

**为什么 RF(S+A) 是"该学/该引"排序的好信号**：它预测的是「一个 2024 论文会不会
被 2025 引用」= **时间留出的未来被引**。到移动云杯场景，就是对候选论文问
「它是不是那份会被后续工作吸收/引用的、值得读/该引的」——比纯语义/热度
多一层"未来价值"结构信号。且 RF 打分**不依赖 query**（单侧），对候选池整体排序。

**模型**：`dasc/_out/rlr/future_rf.pkl`（由 `dasc/rlr/export_future_rf.py` 导出）。
⚠️ 依赖 G 盘论文数据构建年份/标签的那步只在**导出**时做一次；推理只读
`future_rf.pkl` + `rl_prep`，不再碰 G 盘。数据缺 → 明确报错，绝不假装可排序。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = ["RLR_DIR", "FUTURE_RF", "RFRanker", "rank_candidates"]


RLR_DIR: Path = Path(r"D:\大学编程学习\obsidian\dasc\_out\rlr")
FUTURE_RF: Path = RLR_DIR / "future_rf.pkl"
#: 全库 bge-m3 向量（22,747×1024）—— 用 mmap 按需读，避免全量载入 OOM（8GB 机）。
_NDIR: Path = RLR_DIR.parent
VECS: Path = _NDIR / "mobile_corpus_vectors.npz"
VECS_MEAN: Path = _NDIR / "rl_prep" / "global_mean.npy"


class RFRanker:
    """RF 未来被引排序器（懒加载 `future_rf.pkl` + rl_prep 数据，纯只读）。

    打分 = `rf.predict_proba([P, nrev24, out24])[:, 1]`（未来被引概率）。
    提供 `rank(candidates)` → `[(cand, future_score)]` 降序；以及
    `score(cand)` 单篇打分。

    默认用会议库模型（`future_rf.pkl`）；传周报数据目录则用 `arxiv_ranker(cfg)`
    指向 `future_rf_arxiv.pkl` + 周报向量/图（R1–R3，规划-周报接入大创算法.md）。
    """

    def __init__(self, *, model=None, vecs=None, mean=None, graph=None):
        self._m = None
        self._p = {
            "model": Path(model) if model else FUTURE_RF,
            "vecs": Path(vecs) if vecs else VECS,
            "mean": Path(mean) if mean else VECS_MEAN,
            "graph": Path(graph) if graph else RLR_DIR.parent / "rl_prep" / "cite_graph.pkl",
        }

    def _load(self):
        if self._m is not None:
            return
        if not self._p["model"].exists():
            raise FileNotFoundError(
                f"RF 模型缺失（{self._p['model']}）——先跑对应训练脚本"
                f"（会议库: `python -m dasc.rlr.export_future_rf`；"
                f"周报: `python scripts/export_future_rf_arxiv.py`）")
        self._m = pickle.load(open(self._p["model"], "rb"))
        z = np.load(self._p["vecs"], mmap_mode="r")
        self._ids = [str(x) for x in z["ids"]]
        self._idx = {pid: i for i, pid in enumerate(self._ids)}
        self._V = z["vectors"]                  # mmap：按需读，不整载 93MB
        self._mean = np.load(self._p["mean"]).astype(np.float32)
        with open(self._p["graph"], "rb") as f:
            self._fwd = {k: set(v) for k, v in pickle.load(f)["forward"].items()}
        self._early_set = set(self._m["early_ids"])
        self._years = self._m["years"]

    def _vc(self, i: int):
        """第 i 篇的去流行中心加权向量（对齐 dasc features 的两次归一化语义）。"""
        row = np.asarray(self._V[i], dtype=np.float32).copy()
        row = row / max(float(np.linalg.norm(row)), 1e-12)
        row = row - self._mean
        return row / max(float(np.linalg.norm(row)), 1e-12)

    def _feat(self, cand: str) -> list:
        m = self._m
        if cand not in self._idx:
            return None
        i = self._idx[cand]
        P = m["pca"].transform(self._vc(i).reshape(1, -1))[0]
        nrev = sum(1 for s in self._fwd.get(cand, ()) if self._years.get(s) == 2024)
        out = sum(1 for t in self._fwd.get(cand, ()) if t in self._early_set)
        return list(P) + [nrev, out]

    def score(self, cand: str) -> float:
        self._load()
        f = self._feat(cand)
        if f is None:
            return 0.0
        return float(self._m["rf"].predict_proba([f])[0, 1])

    def rank(self, candidates: Iterable[str]) -> list[tuple[str, float]]:
        """按 RF 未来被引概率降序排候选（**批量向量化预测**）。不在语料的丢弃。"""
        self._load()
        m = self._m
        cands = list(dict.fromkeys(candidates))
        idxs = []
        out_cands = []
        for c in cands:
            if c in self._idx:
                idxs.append(self._idx[c])
                out_cands.append(c)
        if not idxs:
            return []
        # mmap 按需选行（只读用到的候选，不整载全库向量）
        Vsel = np.asarray(self._V[np.array(idxs, dtype=int)], dtype=np.float32)
        Vsel = Vsel / np.maximum(np.linalg.norm(Vsel, axis=1, keepdims=True), 1e-12)
        Vsel = Vsel - self._mean
        Vsel = Vsel / np.maximum(np.linalg.norm(Vsel, axis=1, keepdims=True), 1e-12)
        P = m["pca"].transform(Vsel)
        nrev = np.array([sum(1 for s in self._fwd.get(c, ())
                             if self._years.get(s) == 2024) for c in out_cands])
        out = np.array([sum(1 for t in self._fwd.get(c, ())
                            if t in self._early_set) for c in out_cands])
        X = np.column_stack([P, nrev[:, None], out[:, None]])
        probs = m["rf"].predict_proba(X)[:, 1]
        scored = sorted(zip(out_cands, probs), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in scored]


_ranker = RFRanker()


def arxiv_ranker(cfg=None) -> RFRanker:
    """周报 RF 排序器（R1–R3 产物）：`future_rf_arxiv.pkl` + 周报向量/图/mean。"""
    from .config import load_config, paths_from
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    rl = ws / "arxiv_rl"
    prep = rl / "rl_prep"
    return RFRanker(
        model=prep / "future_rf_arxiv.pkl",
        vecs=rl / "mobile_corpus_vectors_arxiv.npz",
        mean=prep / "global_mean.npy",
        graph=ws / "depgraph" / "cite_graph_arxiv.pkl",
    )


def rank_candidates(candidates: Iterable[str],
                    *, ranker: RFRanker | None = None) -> list[tuple[str, float]]:
    """便捷版：返回 `(cand, future_score)` 降序。"""
    r = ranker or _ranker
    return r.rank(candidates)
