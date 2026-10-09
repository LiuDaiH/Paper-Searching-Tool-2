"""poset 抽象（前提偏序 ↝）与引用图缩点构建。

论文定义（main.tex §sec:preliminaries · Lattices and Order）：
- `(𝕌, ↝)` 是 poset，`y ↝ x` 表示「y 是 x 的前提」（先决）。
- **理想 (ideal)** `I ⊆ 𝕌` 是指：`z∈I ∧ y↝z ⟹ y∈I`（前提闭合的下闭集）。
- 由 `S` 生成的理想 `ideal(S) = {y | ∃z∈S, y ⊑ z}`（向下闭包）。
- **主理想 (principal ideal)** `ideal(y)`：单元素生成的理想。
- 论文中 `Δ = max_y r_M(ideal(y))` 是主理想的最大 matroid rank。

**本实现**：前提方向 = 「x 引用了 y ⟹ y↝x」（被引者是前提）。
引用图有环（66 节点互引）⇒ 先 **SCC 缩点成 DAG** 再取偏序与理想
（`大创执行步骤.md §7` 风险「依赖判定要容忍环」）。

SCC 的处理：同一强连通分量内的元素互相可达（互相是前提），在理想判定中
作为**一个整体**被纳入（`groups` 记录 元素→组代表）。跨 SCC 的边保留为
DAG 偏序 `prereq_of[x]`（x 的前提），并做传递闭包。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["Poset", "ideal_of", "scc_condense"]


@dataclass(frozen=True)
class Poset:
    """前提偏序。`prereq_of[x]` = y 的集合，满足 y ↝ x（y 是先决）。

    字段
    ----
    elems: 全部元素 id（与 matroid.ground 同序）。
    prereq_of: {x: set(y)} —— x 的（跨组）前提，**含传递闭包**。
    groups: {elem: group_rep} —— 同一 SCC 的元素映射到同一代表，
        理想判定时整组一并纳入（组内互相可达）。
    """

    elems: tuple[str, ...]
    prereq_of: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)

    def predecessors(self, x: str) -> frozenset:
        return frozenset(self.prereq_of.get(x, ()))


def _group_of(poset: Poset, x: str) -> str:
    return poset.groups.get(x, x)


def ideal_of(poset: Poset, S: Iterable[str]) -> frozenset:
    """S 生成的理想：所有 ≤ S 中某元素的元素（向下闭包）。

    组内（同 SCC）元素互相可达，因此只要有一个入栈，整组都纳入；再向下
    闭包到这些元素的跨组前提。
    """
    out: set = set()
    stack = list(S)
    seen: set = set()
    while stack:
        x = stack.pop()
        g = _group_of(poset, x)
        # 整组入栈（同 SCC 互相可达）
        members = [k for k, v in (poset.groups or {}).items() if v == g] or [x]
        for m in members:
            if m in seen:
                continue
            seen.add(m)
            out.add(m)
            for y in poset.prereq_of.get(m, ()):
                if y not in seen:
                    stack.append(y)
    return frozenset(out)


def scc_condense(
    nodes: Iterable[str],
    edges: dict,
    *,
    prereq_to_edges: bool = True,
) -> tuple[Poset, dict]:
    """把引用边缩点成 DAG 偏序。

    `edges`：邻接表。若 `prereq_to_edges=True`，则把输入边解释为
    「键引用了值」（值 = 键的前提）；否则相反。

    返回 `(poset, comp)`：
    - `poset`：缩点后的 poset（含 groups 与传递闭包）。
    - `comp`: {elem: scc_rep}。
    """
    nodes = list(dict.fromkeys(nodes))
    adj: dict = {n: set() for n in nodes}
    for a, bs in edges.items():
        if a not in adj:
            continue
        for b in bs:
            if b in adj and b != a:  # 忽略自环
                adj[a].add(b)

    # Tarjan SCC（iterative）
    index: dict = {}
    lowlink: dict = {}
    onstack: set = set()
    stack: list = []
    comp: dict = {}
    counter = [0]

    for start in nodes:
        if start in index:
            continue
        work = [(start, iter(adj.get(start, ())))]
        index[start] = lowlink[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        onstack.add(start)
        while work:
            v, it = work[-1]
            progressed = False
            for w in it:
                if w not in index:
                    index[w] = lowlink[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    onstack.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    progressed = True
                    break
                elif w in onstack:
                    lowlink[v] = min(lowlink[v], index[w])
            if progressed:
                continue
            work.pop()
            if work:
                p = work[-1][0]
                lowlink[p] = min(lowlink[p], lowlink[v])
            if lowlink[v] == index[v]:
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp[w] = v
                    if w == v:
                        break

    # 缩点边：跨 SCC 的入边（值→键），方向 = 键引用了值 ⟹ 值是键的前提
    prereq: dict = {n: set() for n in nodes}
    for a, bs in adj.items():
        ca = comp[a]
        for b in bs:
            if comp[b] != ca:
                prereq[a].add(b)

    # 传递闭包
    changed = True
    while changed:
        changed = False
        for a in nodes:
            out = set(prereq.get(a, ()))
            added = set()
            for b in out:
                add = prereq.get(b, ()) - out
                if add:
                    added |= add
            if added:
                prereq[a] = out | added
                changed = True

    poset = Poset(elems=tuple(nodes),
                  prereq_of={k: frozenset(v) for k, v in prereq.items()},
                  groups=dict(comp))
    return poset, comp
