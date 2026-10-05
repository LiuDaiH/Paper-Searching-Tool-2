"""L4 上下文装配 —— **负责固定前缀顺序以命中 DeepSeek 前缀缓存**（方案 §2.5）。

缓存是**从头开始的前缀匹配**，所以装配顺序是契约的一部分：

1. `prefix`（固定 —— 人设 + 输出格式 + 引用规范）——— **永不变，命中缓存，全局唯一**
2. `body`（半固定 —— 簇卡片集/检索到的摘要，同一会话不变）
3. `question`（可变 —— 必然未命中）

⚠️ 时间戳、随机 ID 等"每次都变"的内容**绝不能进 `prefix`**，否则缓存永远不命中。
"""

from __future__ import annotations

from ..schema import Context, Route

__all__ = ["SYSTEM_PROMPT", "assemble"]

#: 固定前缀。**改它 = 全库缓存失效**，所以标"别动"（除非有意识重来）。
#: 一段话里同时给了「回答什么」「怎么引」「不许说什么」，见方案 §2.5 / §1.3。
SYSTEM_PROMPT = (
    "你是「移动云杯 · 本地论文库研究方向热点挖掘」助手。语料是本机 2025 年 "
    "NeurIPS / ICLR / ICML / CoRL 四会论文的标题与摘要，以及由它们聚类而成的"
    "主题簇卡片。\n"
    "回答规则：\n"
    "1. **每个事实性陈述的结尾必须立刻跟一个引用标记** `[P:<paper_id>]`（论文）"
    "或 `[C:<cluster_id>]`（簇卡片）。标记的值必须是输入材料里真实出现过、且被下方"
    "「允许引用」列出的 ID；**没有跟引用的句子会被判为无据而丢弃**。\n"
    "2. 只依据输入材料作答；材料里没有的依据不得编造。\n"
    "3. 关于主题热度，只能用这三种状态描述：铺开（多个会议都在做）、"
    "新进（只出现在末位会议）、单会（其余）。**绝不使用『退潮/下降/衰退』**——"
    "本语料是单年快照，没有『去年』可比。\n"
    "4. 若与本库无关或材料不足，明确说『本库无相关内容』，不要硬答。\n"
    "5. 输出格式示例：`离线 RL 热点集中在分布外泛化与数据质量 [C:11]；其中方法"
    "受 TD3 思想启发 [P:abcd1234]。`——方括号里的 ID 换成材料中**真实存在**的，"
    "**每一个陈述都紧跟引用标记**。"
)

#: GLOBAL 路径的附加固定后缀（跨会三态如何融入叙述）。
GLOBAL_SUFFIX = (
    "\n\n以下是按主题聚类得到的簇卡片。回答『什么最火』时，优先参考"
    "跨会扩散状态：『铺开』表示持续、『新进』表示新出现的主题；每张卡引用"
    "用 `[C:<cluster_id>]`，涉及具体论文用 `[P:<paper_id>]`。"
)


def assemble(
    route: Route,
    *,
    body: str = "",
    question: str = "",
    allowed_ids: frozenset[str] = frozenset(),
) -> Context:
    """装配上下文。`prefix` 固定；`body` 随路径变化；`question` 是用户原问。"""
    prefix = SYSTEM_PROMPT + (GLOBAL_SUFFIX if route is Route.GLOBAL else "")
    return Context(
        route=route,
        prefix=prefix,
        body=body,
        question=question,
        allowed_ids=frozenset(allowed_ids),
    )
