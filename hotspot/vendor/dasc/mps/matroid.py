"""matroid 抽象与 partition matroid 实现。

论文一致的定义（main.tex §sec:preliminaries）：
- 独立集：matroid 的非空 set family 的成员
- **rank** `r(X)` = X 内最大独立集的规模
- **span** `σ(X)` = X 与「不增加 rank 的元素」之并：`σ(X)={y | r(X+y)=r(X)}`
  span 是 closure 算子（单调/幂等/扩展），编码**可替换性**（substitutability）。

**本实现的 matroid = partition matroid（按主题簇分组，每簇容量 = `cap`）**

- 独立集：每个簇至多取 `cap` 个元素。
- `rank(X)` = Σ_簇 min(cap, |X∩簇|)。
- `span(X)` = 所有「所在簇已被 X 触及」的元素 —— 即同簇元素**语义可替换**
  （span 相同），对表 rlRAG 的覆盖单元 `Φ(S)=Σ icf` —— 同一个主题簇内的
  论文覆盖同样的"功能"，选任意一篇都可代表该主题。

**为什么语义向量进了 matroid**：元素所属的「簇」本身正是 dasc 用 bge-m3
去流行余弦（rlRAG `features.py` 的 `dec`/`Vc`）聚类得到的——「同簇可替换」
这一 partition 结构承载了语义特征。额外的 `sim(e1,e2)` 提供去流行余弦相似度，
供 fill-in 等启发式选 'y' 时参考（不改变 matroid 结构）。

容量 `cap` 默认 1（每主题选一篇代表作 = 覆盖该主题一次，对应 ICF 覆盖）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = ["Matroid", "PartitionMatroid"]


class Matroid:
    """matroid 抽象基类。所有方法以元素 id 的 frozenset 为单位。"""

    ground: tuple[str, ...]

    def rank(self, X) -> int:
        raise NotImplementedError

    def span(self, X) -> frozenset:
        raise NotImplementedError

    def independent(self, X) -> bool:
        return self.rank(X) == len(X)

    def r_ground(self) -> int:
        return self.rank(self.ground)


@dataclass(frozen=True)
class PartitionMatroid(Matroid):
    """按簇分组的 partition matroid。`cap` 为每簇容量（默认 1）。

    Attributes
    ----------
    elems: 全部元素 id（按宇宙顺序）。
    cluster_of: {elem: cluster_id}。不在 kwargs 里的元素属于独立噪声簇（自成一簇）。
    cap: 每簇容量。
    sim: 可选，`sim(e1,e2)->float` 返回去流行余弦（仅供启发式参考，不参与结构）。
    """

    elems: tuple[str, ...] = ()
    cluster_of: dict | None = None
    cap: int = 1
    sim: object | None = None

    @property
    def ground(self) -> tuple[str, ...]:
        return self.elems

    def _clusters(self) -> dict:
        co = self.cluster_of or {}
        # 未指定簇的元素，各自独立成簇（rank 上等价于自由元素）
        out: dict = {}
        for e in self.elems:
            out.setdefault(co.get(e, ("_single", e)), set()).add(e)
        return out

    def rank(self, X) -> int:
        if not X:
            return 0
        clusters = self._clusters()
        byc: dict = {}
        for e in X:
            c = (self.cluster_of or {}).get(e, ("_single", e))
            byc[c] = byc.get(c, 0) + 1
        return sum(min(self.cap, n) for n in byc.values())

    def span(self, X) -> frozenset:
        if not X:
            return frozenset()
        co = self.cluster_of or {}
        touched = {co.get(e, ("_single", e)) for e in X}
        return frozenset(e for e in self.elems
                         if co.get(e, ("_single", e)) in touched)

    def elements_in_cluster(self, cluster_id) -> tuple[str, ...]:
        co = self.cluster_of or {}
        return tuple(e for e in self.elems if co.get(e) == cluster_id)
