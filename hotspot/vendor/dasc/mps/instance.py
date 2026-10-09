"""从真实语料组装 MPS 实例（语料实例化，阶段 3）。

数据源（`dasc/_out/rl_prep/`，rlRAG 线已产出）：
- `cite_graph.pkl`：`forward`（pid -> 被引 pid 集）、`reverse`。
- `coverage.pkl`：`paper2cluster`、`icf`、`noise`、`num_clusters`。

组装（对应 `Streit-Garg线-MPS工程实现计划.md` §阶段1 的建模决策）：
- **元素 𝕌**：查询种子（给定方向命中的论文）连同其**前提闭包**
  （被种子直接/间接引用的图内论文）组成的子图。这是「关键路径 / 必备先序」
  （mobile 云杯的 intent=relation）的自然载体：要知道某个方向，得先覆盖它的前提。
- **poset ↝**：`x` 引用了 `y` ⟹ `y ↝ x`（被引者是前提），再 SCC 缩点成 DAG。
- **matroid ℳ**：按 cluster 分组的 **partition matroid**（cap=1），
  span = 同主题簇覆盖，同簇论文语义可替换（语义来自 bge-m3 聚类）。
- **目标**：ClusterCover（子模，φ=Σ icf）或可加 ICF 分摊权重。

产物落 `dasc/_out/mps/`。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

from .core import MPS
from .matroid import PartitionMatroid
from .objective import Additive, ClusterCover
from .poset import scc_condense

__all__ = [
    "PREP_DIR", "OUT_DIR", "load_prep", "prereq_closure", "build_instance",
]


#: rlRAG 前置数据（dasc/_out/rl_prep/）
PREP_DIR: Path = Path(__file__).resolve().parents[1] / "_out" / "rl_prep"
#: 本线产物（dasc/_out/mps/）
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "_out" / "mps"


def load_prep(prep_dir: Path | None = None) -> dict:
    pd = Path(prep_dir or PREP_DIR)
    with open(pd / "cite_graph.pkl", "rb") as f:
        cg = pickle.load(f)
    with open(pd / "coverage.pkl", "rb") as f:
        cov = pickle.load(f)
    return {"forward": cg["forward"], "reverse": cg["reverse"],
            "p2c": cov["paper2cluster"], "icf": cov["icf"],
            "noise": cov["noise"], "num_clusters": cov["num_clusters"]}


def prereq_closure(seeds: Iterable[str], fwd: dict, *, max_nodes: int = 400) -> set:
    """种子 + 其前提闭包：沿「被引（前提）」反向扩展。

    `fwd[x]` = x 引用的论文；前提 = 被 x 引用者。要覆盖 x 得先覆盖其被引。
    """
    result: set = set()
    stack = list(seeds)
    while stack and len(result) < max_nodes:
        x = stack.pop()
        if x in result:
            continue
        result.add(x)
        for y in fwd.get(x, ()):
            if y not in result and len(result) < max_nodes:
                stack.append(y)
    return result


def build_instance(
    prep: dict,
    *,
    seeds: Iterable[str] | None = None,
    elems: Iterable[str] | None = None,
    obj: str = "cover",
    cap: int = 1,
    max_nodes: int = 400,
) -> MPS:
    """组装一个 MPS 实例。

    - 传 `seeds`：以种子前提闭包为元素（推荐；受 `max_nodes` 上限控制规模）。
    - 传 `elems`：直接用给定元素集（其内部边保留）。
    - 都不传：用全部有引用边的论文（较大，仅用于统计）。
    """
    fwd = prep["forward"]
    p2c = prep["p2c"]
    icf = prep["icf"]

    if elems is not None:
        node_set = set(elems)
    elif seeds is not None:
        node_set = prereq_closure(seeds, fwd, max_nodes=max_nodes)
    else:
        all_nodes = set(fwd.keys()) | set().union(*(v for v in fwd.values()))
        node_set = all_nodes

    # 边的方向：x 引用了 y → fwd 已表达。只保留两端都在 node_set 的边。
    edges: dict = {}
    for x, ys in fwd.items():
        if x not in node_set:
            continue
        keep = [y for y in ys if y in node_set]
        if keep:
            edges[x] = set(keep)

    nodes = sorted(node_set)
    poset, comp = scc_condense(nodes, edges, prereq_to_edges=True)

    co = {e: p2c.get(e) for e in nodes}
    # 去掉 matroid 环：rank 0 的字母（无簇 / 无自簇）也允许存在，但 span 恒自含
    mat = PartitionMatroid(elems=tuple(nodes), cluster_of=co, cap=cap)

    if obj == "cover":
        objective = ClusterCover(cluster_of=co, icf=icf, noise_token="noise")
    elif obj == "add":
        w = {e: float(icf.get(co.get(e), 0.0) or 0.0) for e in nodes}
        objective = Additive(w=w)
    else:
        raise ValueError(obj)

    return MPS(matroid=mat, poset=poset, objective=objective)


def save_index(instances: dict, out: Path = OUT_DIR):
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(
        json.dumps(instances, ensure_ascii=False, indent=2), encoding="utf-8")
