"""目标函数：可加权重 / 单调子模覆盖。

论文（main.tex §sec:intro）考虑两类：
- **非负可加**：`f(X) = Σ_{y∈X} w(y)`，`w ≥ 0`。
- **单调子模**：满足递减回报 `(f/X)(z) ≥ (f/Y)(z)` for X⊆Y，且单调、归一化 `f(∅)=0`。

本实现的子模目标 = **覆盖单元（主题簇）的信息量覆盖**，对表 rlRAG：
`Φ(S) = Σ_{g ∈ covered(S)} icf(g)`。其中：

    covered(S) = S 成员所在的不同主题簇集合
    f(S)      = Σ_{c ∈ covered(S)} icf[c]

归一化：f(∅)=0。单调：加元素不减少覆盖。子模：加同属已覆盖簇的元素边际为 0，
加新簇的元素边际 = icf[新簇]，递减回报成立。

可加目标 `ICFSum` 则把每篇论文的权重设为**其所在簇的 icf 分摊**（支撑 for
Δ-scaffold 需要元素级权重），或由调用方直接给每篇权重。

两种目标都实现 :class:`Objective` 接口：`value(X)`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

__all__ = ["Objective", "Additive", "ClusterCover"]


class Objective:
    """目标函数接口：`value(X)` 对集合或可迭代求值。"""

    def value(self, X) -> float:
        raise NotImplementedError

    # 子模边际（供 MAG-sub 用）：(f/X)(z) = f(X∪{z}) - f(X)
    def marginal(self, X: Iterable, z) -> float:
        Xs = set(X)
        return self.value(Xs | {z}) - self.value(Xs)


@dataclass(frozen=True)
class Additive(Objective):
    """可加目标：f(X)=Σ w(y)。`w` 可为 dict 或 callable。"""

    w: object = None

    def _weight(self, y) -> float:
        if callable(self.w):
            return float(self.w(y))
        return float((self.w or {}).get(y, 0.0))

    def value(self, X) -> float:
        return sum(self._weight(y) for y in X)


@dataclass(frozen=True)
class ClusterCover(Objective):
    """单调子模覆盖目标：f(S)=Σ_{c∈covered(S)} icf[c]。

    `cluster_of`: {elem: cluster_id}（与 matroid 同源）。
    `icf`: {cluster_id: float}；未在簇中的元素覆盖 `noise_token`（默认 'noise'）。
    """

    cluster_of: dict
    icf: dict
    noise_token: str = "noise"

    def _clusters(self, X) -> set:
        out = set()
        for y in X:
            out.add(self.cluster_of.get(y, self.noise_token))
        return out

    def value(self, X) -> float:
        return sum(self.icf.get(c, 0.0) for c in self._clusters(X))
