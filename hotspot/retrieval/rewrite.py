"""L4 在线查询改写 —— 把口语问题改成适合检索的查询，提升召回。

规则不做这步（纯关键词召回对口语/长问句损失大）；有 `llm` 时用轻量 LLM 改写一次，
失败退原文（兜底）。**0 依赖、可离线退化为直通。**
"""

from __future__ import annotations

__all__ = ["rewrite"]

_SYSTEM = (
    "把下面的用户问题改写成一段适合论文库检索的英文关键词/短语查询，"
    "保留检索信息、去掉口语与语气词。只输出 JSON：{\"query\": \"...\"}。"
    "若本来就是英文技术词，照原样输出。"
)


def rewrite(question: str, *, client=None, cfg=None) -> str:
    """返回改写后的查询串。`client` 为空时直通原文（不改写）。"""
    if client is None:
        return question
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    try:
        call = client.ask(question, system=_SYSTEM, schema=schema, required=("query",))
        q = (call.data or {}).get("query", "")
        return q.strip() if q and q.strip() else question
    except Exception:  # noqa: BLE001 —— 改写失败退原文
        return question
