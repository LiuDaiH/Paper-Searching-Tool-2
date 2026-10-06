"""Streit-Garg 线（MPS）接入移动云杯 —— 「关键路径/必备先序」回答能力。

对应 `热点挖掘系统-方案设计.md` §「与大创互鉴」的 **Matroidal 前置批**：
给定一个方向（检索到的论文集合 / 种子），用 MPS 判断**要读这个方向，
哪些子主题 / 论文是必须先覆盖的前提** —— 即输出一条按依赖排序的
「关键路径 / 必备先序」词，并带上它覆盖到的主题簇。

纯算法流：**无 LLM、无网络、无导师库写入**（复用 `dasc/mps/` 的算法核心与
`dasc/_out/rl_prep/` 的前置数据）。RLRAG 线已产出 cite_graph（引用边）与
coverage.pkl（paper2cluster + icf），本模块只做装配与采样，不重算。

⚠️ 数据缺失 = 明确报「未运行」而非假装空结果（本库签名病）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

from .config import load_config, paths_from

__all__ = ["compute_key_path", "KeyPathResult", "PREP_DIR"]


#: 单次 MPS 的计算时间预算（秒）：超时抛 MPSBudgetExceeded，由 compute_key_path 捕获降级。
#: 彻底根治卡死（算法内预算），而不是线程超时后仍在后台空转。
MPS_TIMEOUT = float(__import__("os").environ.get("MPS_TIMEOUT", "30"))


#: rlRAG 前置数据（`obsidian/dasc/_out/rl_prep/`）
PREP_DIR: Path = Path(r"D:\大学编程学习\obsidian\dasc\_out\rl_prep")


def _load_prep():
    import pickle
    if not (PREP_DIR / "cite_graph.pkl").exists() \
            or not (PREP_DIR / "coverage.pkl").exists():
        raise FileNotFoundError(
            f"rl_prep 前置数据缺失（{PREP_DIR}）——先跑 "
            f"`dasc/_out/rl_prep/rl_prep.py` 或补齐 cite_graph/coverage.pkl")
    cg = pickle.load(open(PREP_DIR / "cite_graph.pkl", "rb"))
    cov = pickle.load(open(PREP_DIR / "coverage.pkl", "rb"))
    return {"forward": cg["forward"], "reverse": cg["reverse"],
            "p2c": cov["paper2cluster"], "icf": cov["icf"],
            "noise": cov["noise"], "num_clusters": cov["num_clusters"]}


def _pick_seeds(pool_ids: Iterable[str], prep: dict, *, max_seeds: int = 8) -> list:
    """从方向集合里挑「有引用边且在簇中」的论文作种子（否则 MPS 无结构可挖）。"""
    seeds = []
    for pid in pool_ids:
        if pid in prep["forward"] and pid in prep["p2c"]:
            seeds.append(pid)
            if len(seeds) >= max_seeds:
                break
    return seeds


def compute_key_path(pool_ids: Iterable[str], *, cfg=None, algo: str = "mag_sub",
                     max_nodes: int = 250) -> "KeyPathResult":
    """对一个方向计算 MPS 关键路径。

    参数
    ----
    pool_ids: 检索到、属于该方向的论文 id（方向选择器 / 意图驱动聚类的输出）。
    algo: 'mag_sub'(默认) / 'scaffold_add' / 'mag_add' / 'rand_sub'。
    max_nodes: 前提闭包规模上限。

    返回 :class:`KeyPathResult`：可行词（按依赖排序的论文）、各词 member 的
    主题簇、覆盖到的簇集合与 ICF 和、Δ/conn、每步的**前提说明**。
    """
    from dasc.mps.instance import build_instance
    from dasc.mps.algorithms import (mag_submodular, scaffold_additive,
                                     mag_additive, random_submodular)
    from dasc.mps.core import MPSBudgetExceeded

    prep = _load_prep()
    pool_ids = list(dict.fromkeys(pool_ids))
    seeds = _pick_seeds(pool_ids, prep)
    if not seeds:
        # 该方向没有任何可建 MPS 的论文（无引用边或全不在簇中）—— 不是空，是没料
        return KeyPathResult(seed_count=0, note=(
            "方向内没有可建 MPS 的论文（需同时有引用边且已入主题簇）——"
            "不是'无此方向'，是依赖/主题结构在该子集上不可用。"))

    mps = build_instance(prep, seeds=seeds, max_nodes=max_nodes)
    # 预算：算法内部 gtest/cl/ideal 不断检查，超时即抛 MPSBudgetExceeded（可取消，不留后台线程）
    mps.budget_clock.deadline = time.monotonic() + MPS_TIMEOUT

    algorithms = {
        "mag_sub": lambda: mag_submodular(mps),
        "mag_add": lambda: mag_additive(mps, _icf_weights(mps)),
        "scaffold_add": lambda: scaffold_additive(mps, _icf_weights(mps)),
        "rand_sub": lambda: random_submodular(mps),
    }
    fn = algorithms.get(algo, algorithms["mag_sub"])
    try:
        r = fn()
    except MPSBudgetExceeded as e:
        # 超预算 → 诚实降级（不是空结果，是"这次没算完"）
        return KeyPathResult(seed_count=len(seeds), note=(
            f"MPS 计算超出时间预算（{MPS_TIMEOUT:.0f}s）——{e}；"
            "图过大/过稠，已跳过。可设 MPS_TIMEOUT 提高或缩小输入。"))

    if not mps.feasible(r.word):
        return KeyPathResult(seed_count=len(seeds), note=(
            f"算法输出未通过行可行性校验（{algo}）——数据口径问题，勿当结果。"))

    # 解析：每篇论文 -> 其主题簇（用于说明"先覆盖哪簇"）
    step_info = []
    covered: set = set()
    accum = 0.0
    for elem in r.word:
        cid = mps.matroid.cluster_of.get(elem)
        icf = float(mps.objective.icf.get(cid, 0.0)) if hasattr(mps.objective, "icf") else 0.0
        prev = mps.poset.prereq_of.get(elem, ())
        new_cover = cid not in covered
        if new_cover:
            accum += icf
            covered.add(cid)
        step_info.append({
            "paper": elem,
            "cluster": cid,
            "new_cover": new_cover,
            "icf": round(icf, 4),
            "k_prereqs": len(prev),
            "covered_so_far": round(accum, 4),
        })

    return KeyPathResult(
        seed_count=len(seeds),
        n_elements=len(mps.elems),
        n_edges=len(mps.poset.prereq_of),
        delta=mps.delta(),
        conn=mps.conn(),
        algo=algo,
        word=list(r.word),
        value=round(float(r.value), 4),
        covered_clusters=sorted((c for c in covered if c is not None),
                                key=str),
        steps=step_info,
        feasible=True,
        note="",
    )


def _icf_weights(mps):
    return {e: float(mps.objective.icf.get(mps.matroid.cluster_of.get(e), 0.0) or 0.0)
            for e in mps.elems}


class KeyPathResult:
    """关键路径结果（字典可 JSON 化）。"""

    def __init__(self, *, seed_count: int = 0, n_elements: int = 0,
                 n_edges: int = 0, delta: int = 0, conn: float = 0.0,
                 algo: str = "", word: list = (), value: float = 0.0,
                 covered_clusters: list = (), steps: list = (),
                 feasible: bool = False, note: str = ""):
        self.seed_count = seed_count
        self.n_elements = n_elements
        self.n_edges = n_edges
        self.delta = delta
        self.conn = conn
        self.algo = algo
        self.word = list(word)
        self.value = value
        self.covered_clusters = list(covered_clusters)
        self.steps = list(steps)
        self.feasible = feasible
        self.note = note

    def to_dict(self) -> dict:
        return {"seed_count": self.seed_count, "n_elements": self.n_elements,
                "n_edges": self.n_edges, "delta": self.delta,
                "conn": float(self.conn), "algo": self.algo,
                "word": self.word, "value": self.value,
                "covered_clusters": self.covered_clusters,
                "steps": self.steps, "feasible": self.feasible,
                "note": self.note}


def run_key_path_cli(pool_ids_text: str = "", *, algo: str = "mag_sub") -> int:
    """CLI 入口：给逗号分隔的 paper_id 或空（用代理种子）跑关键路径。"""
    if pool_ids_text.strip():
        pool = [x.strip() for x in pool_ids_text.split(",") if x.strip()]
    else:
        # 代理：尝试验证接入可用（无真实查询时）
        prep = _load_prep()
        pool = [k for k in prep["forward"].keys() if k in prep["p2c"]][:12]
    res = compute_key_path(pool, algo=algo)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pids", nargs="?", default="")
    ap.add_argument("--algo", default="mag_sub")
    a = ap.parse_args()
    raise SystemExit(run_key_path_cli(a.pids, algo=a.algo))
