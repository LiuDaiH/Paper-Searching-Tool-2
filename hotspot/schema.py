"""全链路数据契约。

**先写它**——所有模块的接口都从这里取，改这里等于改所有人的接口。
契约只增不改（沿用导师库铁律 5：`CanonicalPaper` schema 只增不改）。

分层对应（方案 §本项目分层）
----------------------------

===========  ==========================================
L1 语料层    :class:`Pool` —— 方向选择器的输出
L2 结构层    :class:`SpreadState` / :class:`VenueProfile` / :class:`ClusterMetric`
L3 压缩层    :class:`ClusterCard` —— LLM 只在这一层工作
L4 检索层    :class:`Context` —— 装配好的、顺序固定的上下文
L5 推理层    :class:`Claim` / :class:`Verdict` / :class:`Answer`
===========  ==========================================

⚠️ 一条贯穿全链的红线
---------------------

语料为 **24+25 两年、无引用网络**。但本项目**主口径选择对比跨会扩散、不做跨年**
（2026-09-22 拍板，见方案文件头修订框）。因此 :class:`SpreadState` **只有三态**，
没有第四态。任何地方都**不得**引入"退潮 / 下降 / 衰退"这类判断——那不是
"我们没算出来"，而是**本项目口径里不作跨年比较**（方案 §1.3）。

同理，`citations` 字段 **50 / 13,119 非空**，不是"引用少"，是"没去查过"
（导师库原话：否则 `citations: null` 会被读成"没人引用"）。本契约里
**不暴露任何以被引数为基础的字段**，免得有人顺手用上。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

#: 契约版本。**只增不改**；不兼容的改动必须换号并留迁移说明。
SCHEMA_VERSION = "1"


# ══════════════════════════════════════════════════════════════════════
# L1 语料层
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Pool:
    """方向选择器的输出：一个子语料。

    ⚠️ **三路命中数必须一起带着走**，因为它们不相等，而且差距本身就是结论。

    `subfields` 标签有 **40–65% 盲区**（robotics / imitation / rlhf /
    rl_for_science）。只看标签会得出"库里没人做"的错结论——而这正是导师库
    自己的头号病：**"库内缺席"被读成"没人做"**。所以这里把三路分开记，
    让"哪条路没命中"这件事**看得见**，而不是合并成一个数。

    Attributes:
        direction: 用户给的方向原文。
        paper_ids: 三路并集（去重后）。这是真正下游用的东西。
        hits_fts: FTS5/BM25 命中数。
        hits_vector: 向量召回命中数。
        hits_subfield: `subfields` 标签命中数。
        notes: 供人看的告警（如"向量路未跑：全库向量不存在"）。
    """

    direction: str
    paper_ids: tuple[str, ...]
    hits_fts: int = 0
    hits_vector: int = 0
    hits_subfield: int = 0
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.paper_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# L2 结构层（0 token）
# ══════════════════════════════════════════════════════════════════════


class SpreadState(str, Enum):
    """跨会扩散的三态。**就只有这三个。**

    `str` 混入是为了 JSON 往返（沿用 `dasc/srcfile.py` 的 `SrcKind` 做法）。

    为什么没有第四态（比如 `退潮`）
    ------------------------------

四会的投稿/决策时间在 2025（并入 2024 后仍按会议）内是有序的（ICLR → ICML → NeurIPS），
所以"出现在几个会"能当作一根**粗粒度三点时间轴**。但它是一根"会议数轴"、
**不承担跨年比较**（本项目主口径不做跨年，见文件头修订框）。

    所以：

    * `铺开` = 出现在 ≥ `spread_min` 个会 ⇒ **持续**
    * `新进` = 只出现在末位会 ⇒ **新进**
    * `单会` = 其余 ⇒ **说不出来涨还是跌**

    ⚠️ `单会` 是**不可知**，不是"不热"。把它读成"在退"就是把
    "没测到"读成"测得是 0"——导师库为这个病登记过十几次。
    """

    SPREAD = "铺开"
    NEW = "新进"
    SINGLE = "单会"


@dataclass(frozen=True)
class VenueProfile:
    """一个簇在各会上的分布。**纯计数，不含任何 LLM 判断。**

    这是"热点"在本语料下**唯一的客观依据**（方案 §3.2）。
    """

    counts: dict[str, int] = field(default_factory=dict)
    #: 参与时间轴的会（不含 `venue_off_axis`，如 CoRL）
    on_axis: tuple[str, ...] = ()
    #: 命中的会数（分母是 `on_axis`，不是全部会）
    n_venues: int = 0
    #: 三态标签。**由计数直接算出，不让 LLM 判断。**
    state: SpreadState = SpreadState.SINGLE
    #: 按年份分组的会计数：`{year: {venue: n}}`。**保留分年能力**；
    #: 三态判定仍用总 `counts`（跨年混合口径），此字段供辅助验证/分年展示。
    counts_by_year: dict[int, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "counts": dict(self.counts),
            "on_axis": list(self.on_axis),
            "n_venues": self.n_venues,
            "state": self.state.value,
            "counts_by_year": {str(k): v for k, v in (self.counts_by_year or {}).items()},
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VenueProfile:
        return cls(
            counts=dict(d.get("counts") or {}),
            on_axis=tuple(d.get("on_axis") or ()),
            n_venues=int(d.get("n_venues") or 0),
            state=SpreadState(d.get("state") or SpreadState.SINGLE.value),
            counts_by_year={int(k): dict(v) for k, v
                            in (d.get("counts_by_year") or {}).items()},
        )


@dataclass(frozen=True)
class ClusterMetric:
    """一个簇的结构化指标。L3 的输入，L4 的论据。

    Attributes:
        cluster_id: 簇号。**噪声 `-1` 不会出现在这里**——它被明确排除
            （沿用 `dasc/cluster/clusterers.py` 的纪律：`n_noise` 与
            `n_clusters` 永远并排打印，不合并）。
        size: 簇内论文数。
        venue: 跨会分布。
        top_terms: 簇的判别词（TF-IDF），供人扫读用。
        representative_ids: 代表性论文。⚠️ 这是**质心最近的**，
            不是"最重要的"——它是最典型的，不是最有代表性的。
        neighbor_ids: 质心余弦最近的相邻簇（"哪些方向在融合"）。
    """

    cluster_id: int
    size: int
    venue: VenueProfile
    top_terms: tuple[str, ...] = ()
    representative_ids: tuple[str, ...] = ()
    neighbor_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["venue"] = self.venue.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterMetric:
        return cls(
            cluster_id=int(d["cluster_id"]),
            size=int(d["size"]),
            venue=VenueProfile.from_dict(d.get("venue") or {}),
            top_terms=tuple(d.get("top_terms") or ()),
            representative_ids=tuple(d.get("representative_ids") or ()),
            neighbor_ids=tuple(int(x) for x in (d.get("neighbor_ids") or ())),
        )


# ══════════════════════════════════════════════════════════════════════
# L3 压缩表示层（唯一花钱处）
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ClusterCard:
    """一个簇的压缩表示。**LLM 唯一在这里工作。**

    输入是簇内成员的 **title + abstract**，不是正文——导师库正文
    100% 含未展开 LaTeX 宏、84% 有"宏名粘着后文"吞词（已登记未修），
    本来就不能喂 LLM（方案 §不做清单）。

    ⚠️ `spread_state` / `scale` / `venues` **不是 LLM 产出的**，是从
    :class:`ClusterMetric` 直接抄过来的客观计数。LLM 只填
    `label` / `core_problems` / `dominant_methods` / `open_questions`。
    把它和 LLM 的输出分开存，是为了让"哪些字是模型写的"永远可查。

    ⚠️ `evidence_ids` 必须**都在输入里出现过**。这是防幻觉的第一道闸——
    在 L3 就把"引了不存在的论文"挡掉，而不是等到 L5。
    """

    cluster_id: int
    # ── 以下四项来自 L2，是客观计数，不是 LLM 写的 ──
    scale: int
    venue: VenueProfile
    # ── 以下四项是 LLM 写的 ──
    label: str
    core_problems: tuple[str, ...] = ()
    dominant_methods: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    # ── 溯源 ──
    evidence_ids: tuple[str, ...] = ()
    #: 非空表示这张卡是降级产出的（如 LLM 两次都没给出合法 JSON）。
    #: **降级不是失败**，但要看得见。
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["venue"] = self.venue.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterCard:
        return cls(
            cluster_id=int(d["cluster_id"]),
            scale=int(d.get("scale") or 0),
            venue=VenueProfile.from_dict(d.get("venue") or {}),
            label=str(d.get("label") or ""),
            core_problems=tuple(d.get("core_problems") or ()),
            dominant_methods=tuple(d.get("dominant_methods") or ()),
            open_questions=tuple(d.get("open_questions") or ()),
            evidence_ids=tuple(d.get("evidence_ids") or ()),
            degraded_reason=str(d.get("degraded_reason") or ""),
        )


# ══════════════════════════════════════════════════════════════════════
# L4 检索层
# ══════════════════════════════════════════════════════════════════════


class Route(str, Enum):
    """问题路由。**纯规则判定，不调 LLM**（方案 §2.4 第 8 项）。

    `REFUSE` 是**一等公民**，不是异常分支：超范围拒答是幻觉重灾区，
    也是评测集里专门的一类（方案 §5.1）。
    """

    GLOBAL = "global"      # 热点/趋势类 → 簇卡片 + 跨会矩阵
    LOCAL = "local"        # 事实类 → FTS5 现取摘要
    REFUSE = "refuse"      # 三路召回并集为空 → 明确说"本库无相关内容"


@dataclass(frozen=True)
class Context:
    """装配好的上下文。**顺序是契约的一部分，不是实现细节。**

    DeepSeek 的缓存是**从头开始的前缀匹配**。所以：

    1. 固定前缀（系统人设 + 输出格式 + 引用规范）—— 永不变
    2. 半固定（簇卡片集 / 检索到的摘要）—— 同会话不变
    3. 可变（用户问题）—— 必然未命中

    ⚠️ 时间戳、随机 ID 等"每次都变"的内容**绝不能进 `prefix`**，
    否则缓存永远不命中，成本翻 50 倍（方案 §2.5）。
    """

    route: Route
    prefix: str = ""
    body: str = ""
    question: str = ""
    #: 本次上下文里出现过的全部合法引用 ID（`[P:...]` / `[C:...]` 的值）。
    #: L5 校验**只认这个集合**——不在这里的一律判为编造。
    allowed_ids: frozenset[str] = frozenset()

    def render(self) -> str:
        return f"{self.prefix}\n\n{self.body}\n\n{self.question}".strip()


# ══════════════════════════════════════════════════════════════════════
# L5 推理层
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Claim:
    """答案里的一条事实性陈述，连同它声称的引用。"""

    text: str
    cited_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """对一条 :class:`Claim` 的判定。

    ⚠️ `unsupported` 与 `missing_id` 必须分开：
    * `missing_id` = 引用了一个**根本不存在**的 ID ⇒ 编造
    * `unsupported` = ID 存在，但这条 claim 在原文里**找不到依据** ⇒ 推理失败

    混在一起就看不见"是检索错了还是想错了"——而 KET-RAG 2026 的结论正是
    **73–84% 的错误是推理失败，不是检索失败**（方案 §1.2）。
    """

    claim: Claim
    status: str            # "supported" | "unsupported" | "missing_id" | "no_citation"
    detail: str = ""


@dataclass(frozen=True)
class Answer:
    """最终输出。"""

    route: Route
    text: str = ""
    verdicts: tuple[Verdict, ...] = ()
    refused: bool = False
    refuse_reason: str = ""

    @property
    def n_filtered(self) -> int:
        """被过滤掉的无据陈述数。**要显示给用户看，不能悄悄吞掉。**"""
        return sum(1 for v in self.verdicts if v.status != "supported")
