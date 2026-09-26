"""跨会扩散矩阵（方案 §3.2）—— **"热点"在本语料下唯一的客观依据**。

**0 token，纯计数，不让 LLM 判断。**

四会的投稿/决策时间在 2025 年内是有序的（ICLR → ICML → NeurIPS），所以
「一篇出现在几个会」能当一根**粗粒度三点时间轴**。三态口径（config `spread`）：

| 态 | 判据（对 `on_axis` 会） |
|---|---|
| `铺开` | 命中 ≥ `spread_min` 个会 |
| `新进` | 只命中末位会（`venue_order[-1]`） |
| `单会` | 其余 —— **不可知，不是"不热"** |

⚠️ 红线（方案 §1.3 / schema）：**没有第四态**。`单会` 是"说不出来涨还是跌"，
不是"在退"。任何地方不得引入"退潮/下降/衰退"。
"""

from __future__ import annotations

from ..schema import SpreadState, VenueProfile

__all__ = [
    "spread_state",
    "venue_profile",
    "build_venue_map",
    "ON_AXIS_NOTES",
]


def spread_state(
    counts: dict[str, int],
    *,
    venue_order: tuple[str, ...],
    spread_min: int,
    off_axis: tuple[str, ...] = (),
) -> SpreadState:
    """由计数直接算出三态。调用方保证 `counts` 是**客观计数**，不是 LLM 判断。

    `venue_order` = 参与时间轴的会，按决策时间升序（`[ICLR, ICML, NeurIPS]`）。
    `off_axis`（如 CoRL）不进 `on_axis`、不算 `n_venues`、也不参与三态判定。
    """
    on_axis = [v for v in venue_order if v not in off_axis]
    present = [v for v in on_axis if counts.get(v, 0) > 0]

    if len(present) >= spread_min:
        return SpreadState.SPREAD
    if len(present) == 1 and on_axis and present[0] == on_axis[-1]:
        return SpreadState.NEW
    return SpreadState.SINGLE


def venue_profile(
    counts: dict[str, int],
    *,
    venue_order: tuple[str, ...],
    spread_min: int,
    off_axis: tuple[str, ...] = (),
    counts_by_year: dict[int, dict[str, int]] | None = None,
) -> VenueProfile:
    """把一个簇在各会上的计数变成 :class:`VenueProfile`（含三态）。

    `counts_by_year`（`{year: {venue: n}}`）**保留分年能力**，但不参与三态判定——
    `state` 仍由跨年混合的总 `counts` 算（主口径跨会、不做跨年）。
    """
    on_axis = tuple(v for v in venue_order if v not in off_axis)
    counts = dict(counts)
    state = spread_state(counts, venue_order=venue_order,
                         spread_min=spread_min, off_axis=off_axis)
    return VenueProfile(
        counts=counts,
        on_axis=on_axis,
        n_venues=sum(1 for v in on_axis if counts.get(v, 0) > 0),
        state=state,
        counts_by_year=dict(counts_by_year or {}),
    )


def build_year_map(papers) -> dict[str, int]:
    """`paper_id -> year` 的映射（供跨会全体的分年计数用）。"""

    return {p.paper_id: p.year for p in papers if p is not None and p.year is not None}


def build_venue_map(papers) -> dict[str, str]:
    """`paper_id -> venue` 的映射。取不到 venue 的记 `""`（空，不参与计数）。"""
    return {p.paper_id: (p.venue or "") for p in papers}


#: 供人看的说明串（写进 metrics 产物时带上，免得下个读的人重推一遍）。
ON_AXIS_NOTES = (
    "跨会扩散三态：铺开=命中>=spread_min 个 on_axis 会；"
    "新进=只命中末位会；单会=其余（不可知，不是'不热'）。"
    "⚠️ 无第四态，禁止'退潮/下降/衰退'措辞。"
)
