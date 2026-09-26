"""L5 推理层 —— 带引用生成答案（方案 §3.5 / §2.4 第 10 项）。

输入是 :class:`Context`（已装配好的固定前缀 + 簇卡/摘要 + 问题）。
输出是**一段自然语言答案**，但格式被 `prefix` 强制：每个事实性陈述后必须跟
`[P:<paper_id>]`（论文）或 `[C:<cluster_id>]`（簇卡片）。引用合法性在
:mod:`.verify` 校验。

⚠️ `model_pro` 是 config 里 L5 答案生成用的模型（比 L3 命名贵一点的档）。
"""

from __future__ import annotations

from ..config import load_config
from ..llm import make_client
from ..schema import Context

__all__ = ["ANSWER_SCHEMA", "generate_answer"]


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def generate_answer(context: Context, *, cfg=None, dry_run: bool = False,
                    model: str | None = None) -> str:
    """生成答案文本。`context.prefix` 里已把引用格式与三态红线写死。"""
    cfg = cfg or load_config()
    pro = (cfg.get("llm") or {}).get("model_pro")
    client = make_client(cfg, model=model or pro or "deepseek-chat", dry_run=dry_run)
    call = client.ask(
        context.question,
        system=context.prefix,
        schema=ANSWER_SCHEMA,
        required=("answer",),
    )
    return call.data["answer"] if call.ok else ""
