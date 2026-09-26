"""L4 问题路由 —— **规则兜底 + LLM 增强**（方案 §2.4 第 8 项）。

在线问答环节：单次成本极小，用一圈轻量 LLM 换用户的**使用舒适度**是划算的——
关键词规则挡不住"最近大家研究方向的热点在哪"这类模糊措辞，交给 LLM 判更准。

判定策略：
1. `REFUSE`：方向选择器三路并集为空 ⇒ 明确拒答（**硬规则**，永不交给 LLM）。
2. 命中热点关键词（"最火/热点/趋势/研究方向/前沿"，中英）⇒ 直接 `GLOBAL`（省一次调用）。
3. 其余 ⇒ 有 `llm` 时由 LLM 判 `GLOBAL`/`LOCAL`，失败退 `LOCAL`（**规则兜底**）。

`llm=None` 时退化为纯规则（确定性/离线/评测基线），行为与旧版一致。
"""

from __future__ import annotations

from ..schema import Pool, Route

__all__ = ["route", "GLOBAL_PATTERNS", "REFUSE_REASON", "llm_route"]


#: 触发 GLOBAL 的关键词（快速路径，命中即走 GLOBAL，省一次 LLM 调用）。
GLOBAL_PATTERNS: tuple[str, ...] = (
    "hotspot", "hot spot", "hot topic", "trend", "most popular",
    "what's hot", "what is hot",
    "热点", "最火", "趋势", "研究方向", "前沿",
)

REFUSE_REASON = "本库（2025 NeurIPS / ICLR / ICML / CoRL）没有与您所述方向相关的内容。"


def route(question: str, pool: Pool, *, llm=None) -> Route:
    """路由。`llm` 为可用的 LLM client（可选）；不给则纯规则（确定性/离线）。"""
    if pool.size == 0:
        return Route.REFUSE
    low = question.lower()
    if any(p in low for p in GLOBAL_PATTERNS):
        return Route.GLOBAL
    if llm is not None:
        return llm_route(question, llm)
    return Route.LOCAL


def llm_route(question: str, client, *, system: str | None = None) -> Route:
    """用 LLM 判 GLOBAL / LOCAL。失败（非法 JSON / 调用异常）退 `LOCAL` 兜底。

    `REFUSE` 不由这里判——它由方向选择器的空池硬规则在 `route()` 里提前拦下。
    """
    schema = {
        "type": "object",
        "properties": {"route": {"type": "string"}},
        "required": ["route"],
    }
    sys = system or (
        "你是问题路由。判断用户问题是想问『当前哪些研究方向最火/最受关注』"
        "（回复 global），还是想问某个具体方法/具体细节（回复 local）。"
        "只输出 JSON：{\"route\": \"global\" | \"local\"}。"
    )
    try:
        call = client.ask(question, system=sys, schema=schema, required=("route",))
        r = (call.data or {}).get("route", "").strip().lower()
        return Route.GLOBAL if r == "global" else Route.LOCAL
    except Exception:  # noqa: BLE001 —— 路由失败不崩溃，退兜底
        return Route.LOCAL
