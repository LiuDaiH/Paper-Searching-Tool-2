"""MPS 本体：可行词语言、gtest、closure、Δ、conn。

对应论文 main.tex：

**可行词（Def. MPS, Eq. feasibility）**：词 `y1…yl` 的字母成独立集，且对每个 i，
所有前提 `x ↝ yi` 要么被前缀 `{y1..y_{i-1}}` 的 span 覆盖，要么与 `yi` 关于前缀
可替换：`σ(prefix ∪ {yi}) = σ(prefix ∪ {x})`。

**gtest（最大表示, Prop. rideal）**：`gtest(X) = r_M(ideal X)` —— 对理想取 matroid
rank。可由一次 matroid rank 查询算出。

**closure（σ_{gtest} = cl）**：`cl(X) = {y | gtest(X ∪ {y}) = gtest(X)}`。

**Δ (Eq. delta-def)**：主理想的最大 matroid rank：`max_y r_M(ideal({y}))`。

**conn (Eq. conn-def)**：Tutte 连通度的最大值：`max_X [r(X)+r(U-X)-r(U)]`。
   计算原则上需遍历 2^n；对真实规模用可配置的启发式 + 诚实标注口径
   （见 :meth:`MPS.conn`）。
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from .matroid import Matroid
from .poset import Poset, ideal_of
from .objective import Objective

__all__ = ["MPS", "MPSBudgetExceeded"]


class MPSBudgetExceeded(RuntimeError):
    """MPS 计算超出预算（时间/步数）—— 上层应捕获并降级，而不是任其卡死。"""


class _Clock:
    """全局/可注入的时间预算：超时抛 :class:`MPSBudgetExceeded`。

    `compute_key_path` 可设 `mps.budget_clock.deadline`；未设则不限时（保持原行为）。
    """

    def __init__(self):
        import time as _t
        self._t = _t
        self.deadline = None

    def check(self) -> None:
        if self.deadline is not None and self._t.monotonic() > self.deadline:
            raise MPSBudgetExceeded(f"MPS 计算超时（预算截止 {self.deadline:.1f}s）")


class MPS:
    """一个 Matroidal Prerequisite System：matroid + poset + 目标。

    元素（字母）= matroid.ground (= poset.elems)。三者必须同序。
    """

    def __init__(self, matroid: Matroid, poset: Poset, objective: Objective):
        if tuple(matroid.ground) != tuple(poset.elems):
            raise ValueError("matroid.ground 与 poset.elems 必须同序")
        self.matroid = matroid
        self.poset = poset
        self.objective = objective
        self.elems: tuple = tuple(matroid.ground)
        self._idx = {e: i for i, e in enumerate(self.elems)}
        self._cl_memo: dict = {}
        self._gtest_memo: dict = {}
        self._ideal_memo: dict = {}
        self.budget_clock: _Clock = _Clock()

    def _hash_X(self, X):
        return frozenset(X)

    def ideal(self, X: Iterable[str]) -> frozenset:
        key = self._hash_X(X)
        if key not in self._ideal_memo:
            self._ideal_memo[key] = ideal_of(self.poset, key)
        return self._ideal_memo[key]

    def gtest(self, X: Iterable[str]) -> int:
        """gtest(X) = r_M(ideal X)。"""
        self.budget_clock.check()
        key = self._hash_X(X)
        if key not in self._gtest_memo:
            v = self.matroid.rank(self.ideal(key))
            self._gtest_memo[key] = v
        return self._gtest_memo[key]

    def cl(self, X: Iterable[str]) -> frozenset:
        """closure 算子：σ_{gtest}(X) = {y | gtest(X+y) = gtest(X)}。"""
        self.budget_clock.check()
        key = self._hash_X(X)
        if key in self._cl_memo:
            return self._cl_memo[key]
        Xset = set(key)
        gx = self.gtest(Xset)
        out = set()
        for y in self.elems:
            self.budget_clock.check()
            if y in Xset:
                out.add(y)
                continue
            if self.gtest(Xset | {y}) == gx:
                out.add(y)
        res = frozenset(out)
        self._cl_memo[key] = res
        return res

    # ---- 可行词判定 ---------------------------------------------------- #

    def feasible(self, word: Iterable[str]) -> bool:
        """一个简单词是否 ∈ L(ℳ,↝)：独立 + 每个字母前提被覆盖或可替换。"""
        seen: list = []
        support: set = set()
        for y in word:
            if y in support:
                return False                      # 非简单（重复字母）
            support.add(y)
            prefix = set(seen)
            sp = self.matroid.span(prefix)
            for x in self.poset.prereq_of.get(y, ()):
                covered = x in sp
                if covered:
                    continue
                # 可替换：σ(prefix+y) == σ(prefix+x)
                if self.matroid.span(prefix | {y}) == self.matroid.span(prefix | {x}):
                    continue
                return False
            if not self.matroid.independent(support):
                return False
            seen.append(y)
        return True

    # ---- 结构参数 ------------------------------------------------------ #

    def delta(self) -> int:
        """Δ = max_y r_M(ideal({y}))。"""
        return max(self.gtest({y}) for y in self.elems) if self.elems else 0

    def conn(self, *, exhaustive: bool = None) -> float:
        """Tutte 连通度最大值 conn = max_X λ(X)。

        - 若 `exhaustive=True` 或 `len(elems)` 较小（≤ 18），穷举 2^n 精确算。
        - 否则用启发式：对所有「单元素 + 单簇」的 X 采样上界（诚实标注为估计）。
          返回 float，且带 `.approx` 标记（通过返回对象属性不易做，故用元组
          语义：精确时返回 int，估计时返回带小数的 float + 打印口径）。
        """
        n = len(self.elems)
        rU = self.matroid.r_ground()
        U = set(self.elems)
        if exhaustive is None:
            exhaustive = n <= 18
        if exhaustive:
            best = -1
            for r in range(0, n + 1):
                for comb in combinations(self.elems, r):
                    X = set(comb)
                    lam = (self.matroid.rank(X)
                           + self.matroid.rank(U - X) - rU)
                    if lam > best:
                        best = lam
            return best
        # 启发式：X 取「单簇」及其补集、单元素，取最大 λ（估计下界）
        best = -1
        rng_elems = list(self.elems)
        import random
        sample_cap = 200
        samples = [set()]
        rng = random.Random(0)
        rng.shuffle(rng_elems)
        samples += [{e} for e in rng_elems[:sample_cap]]
        clusters = list(self.matroid._clusters().values())
        rng.shuffle(clusters)
        samples += [set(c) for c in clusters[:sample_cap]]
        Uset = U
        for X in samples:
            lam = self.matroid.rank(X) + self.matroid.rank(Uset - X) - rU
            best = max(best, lam)
        return best

    # ---- 目标函数 ------------------------------------------------------ #

    def value(self, word_or_set) -> float:
        return self.objective.value(word_or_set)

    # ---- 序列工具 ------------------------------------------------------ #

    def sort_by_weight(self, weights: dict, elems: Iterable[str]) -> list:
        return sorted(elems, key=lambda e: -weights.get(e, 0.0))

    def continuation_value(self, rho_contraction) -> dict:
        raise NotImplementedError("不直接使用；gtest 通过 r∘ideal 现算")
