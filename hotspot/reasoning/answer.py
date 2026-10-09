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
    """生成答案文本。`context.prefix` 里已把引用格式与三态红线写死。

    用 `ask_free`（纯文本，不 JSON）——JSON object 模式会让部分模型把 schema
    结构回显而非产出带引用的自然语言（实测移动云模型会返回 `{"answer":{...}}`）。
    """
    cfg = cfg or load_config()
    pro = (cfg.get("llm") or {}).get("model_pro")
    client = make_client(cfg, model=model or pro or "deepseek-chat", dry_run=dry_run)
    user = (context.body + "\n\n" if context.body and context.body.strip() else "") \
        + "用户问题：" + context.question
    # 摘要出境闸：dasc 上限 8000 字符（MAX_INPUT_CHARS）。若装配出的材料过长
    # （簇卡片+证据累积），裁剪到安全长度再发，避免整体被拒导致路线图/回答丢失。
    max_chars = int((cfg.get("llm") or {}).get("max_input_chars") or 8000)
    if len(user) > max_chars:
        # 保留问题与「材料过长」提示；body 截断到留足问题长度 + 余量。
        tail = "\n…（材料过长已截断，仅保留开头）\n用户问题：" + context.question
        user = user[:max(0, max_chars - len(tail))] + tail
        user = user[:max_chars]   # 兜底，保证不超闸
    call = client.ask_free(
        user,
        system=context.prefix,
    )
    if call.ok:
        return (call.data.get("text") if call.data else None) or call.raw
    return ""
