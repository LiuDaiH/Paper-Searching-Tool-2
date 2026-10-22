"""L5 推理层 —— 带引用生成答案（方案 §3.5 / §2.4 第 10 项）。

输入是 :class:`Context`（已装配好的固定前缀 + 簇卡/摘要 + 问题）。
输出是**一段自然语言答案**，但格式被 `prefix` 强制：每个事实性陈述后必须跟
`[P:<paper_id>]`（论文）或 `[C:<cluster_id>]`（簇卡片）。引用合法性在
:mod:`.verify` 校验。

⚠️ `model_pro` 是 config 里 L5 答案生成用的模型（比 L3 命名贵一点的档）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import load_config
from ..llm import make_client
from ..schema import Context

__all__ = [
    "ANSWER_SCHEMA",
    "CLAIM_SCHEMA",
    "ClaimAnswer",
    "generate_answer",
    "generate_claim_answer",
    "extract_claim_answer",
]


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "kind": {"type": "string"},
                },
                "required": ["text", "evidence_ids"],
            },
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["claims"],
}

_CITATION_MARK = re.compile(r"\[(?:P|C):[^\]]+\]")


@dataclass(frozen=True)
class ClaimAnswer:
    """结构化答案生成结果。`text` 已由程序统一渲染引用。"""

    ok: bool
    text: str = ""
    claims: tuple[dict, ...] = ()
    limitations: tuple[str, ...] = ()
    reason: str = ""


def _compose_user(context: Context, cfg: dict, *, reserve: int = 0) -> str:
    user = (context.body + "\n\n" if context.body and context.body.strip() else "") \
        + "用户问题：" + context.question
    max_chars = int((cfg.get("llm") or {}).get("max_input_chars") or 8000)
    max_chars = max(1, max_chars - max(0, reserve))
    if len(user) > max_chars:
        tail = "\n…（材料过长已截断，仅保留开头）\n用户问题：" + context.question
        user = user[:max(0, max_chars - len(tail))] + tail
    return user[:max_chars]


def _render_claim(text: str, ids: list[str]) -> str:
    marks = "".join(
        f"[C:{pid}]" if pid.isdigit() else f"[P:{pid}]"
        for pid in ids
    )
    value = text.strip()
    if value and value[-1] in "。.!?！？":
        return f"{value[:-1]}{marks}{value[-1]}"
    return f"{value} {marks}"


def _extract_claim_fallback(context: Context) -> tuple[dict, ...]:
    """结构化模型漏掉 claim 时，从已装配的论文证据中抽取直接陈述。"""
    chunks = re.split(r"\n{2,}", context.body or "")
    claims = []
    seen: set[str] = set()
    for chunk in chunks:
        match = re.match(r"\[([PC]):([^\]]+)\]\s*(.*)", chunk.strip(), re.DOTALL)
        if not match:
            continue
        kind, pid, content = match.group(1), match.group(2), match.group(3).strip()
        if pid in seen or pid not in context.allowed_ids or not content:
            continue
        seen.add(pid)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else pid
        snippet = " ".join(lines[1:]) if len(lines) > 1 else title
        snippet = _CITATION_MARK.sub("", snippet).strip()[:240]
        if not snippet:
            continue
        prefix = "主题卡片" if kind == "C" else f"《{title}》报告"
        claims.append({
            "text": f"{prefix}：{snippet}",
            "evidence_ids": [pid],
            "kind": "extractive",
        })
        if len(claims) >= 3:
            break
    return tuple(claims)


def extract_claim_answer(context: Context) -> ClaimAnswer:
    claims = _extract_claim_fallback(context)
    if not claims:
        return ClaimAnswer(False, reason="材料中没有可抽取的 [P:]/[C:] claim")
    return ClaimAnswer(
        True,
        text="\n".join(
            _render_claim(str(item["text"]), list(item["evidence_ids"]))
            for item in claims
        ),
        claims=claims,
        reason="使用证据抽取式 claim",
    )


def generate_claim_answer(
    context: Context, *, cfg=None, dry_run: bool = False,
    model: str | None = None, client=None,
) -> ClaimAnswer:
    """先生成可校验的 claims，再渲染为当前契约要求的自然语言 text。

    模型只负责提出 claim 和 evidence_ids；所有引用标记均由本地程序添加，
    非法 evidence ID 在进入最终答案前被丢弃。
    """
    cfg = cfg or load_config()
    pro = (cfg.get("llm") or {}).get("model_pro")
    client = client or make_client(
        cfg, model=model or pro or "deepseek-chat", dry_run=dry_run,
    )
    system = (
        context.prefix
        + "\n\n你现在必须输出结构化 JSON，而不是自然语言答案。"
        "只输出 {claims:[{text,evidence_ids,kind}],limitations:[]}。"
        "claims 中每项只能包含一个可独立核查的事实陈述，text 不要写引用标记；"
        "evidence_ids 必须来自允许集合，且每个 claim 至少一个。"
        "没有证据支持的内容不要写入 claims；没有可直接回答的证据时返回空 claims。"
    )
    instruction = "\n\n只能引用材料正文中实际出现的 [P:] / [C:] ID；程序会丢弃其他 ID。"
    user = _compose_user(context, cfg, reserve=len(instruction)) + instruction
    call = client.ask(
        user, system=system, schema=CLAIM_SCHEMA, required=("claims",),
    )
    if not call.ok:
        extracted = extract_claim_answer(context)
        if extracted.ok:
            return extracted
        return ClaimAnswer(False, reason=call.error or "结构化生成失败")
    data = call.data or {}
    claims = []
    limitations = []
    for raw in (data.get("claims") or []):
        if not isinstance(raw, dict):
            continue
        text = _CITATION_MARK.sub("", str(raw.get("text") or "")).strip()
        if not text:
            continue
        ids = []
        for raw_id in (raw.get("evidence_ids") or []):
            pid = str(raw_id).strip()
            if pid.startswith(("P:", "C:")):
                pid = pid[2:]
            if pid in context.allowed_ids and pid not in ids:
                ids.append(pid)
        if not ids:
            continue
        claims.append({
            "text": text,
            "evidence_ids": ids,
            "kind": str(raw.get("kind") or "finding"),
        })
    for raw in (data.get("limitations") or []):
        text = str(raw).strip()
        if text:
            limitations.append(text)
    if not claims:
        extracted = extract_claim_answer(context)
        if extracted.ok:
            return ClaimAnswer(
                True, text=extracted.text, claims=extracted.claims,
                limitations=tuple(limitations), reason=extracted.reason,
            )
        return ClaimAnswer(
            False,
            limitations=tuple(limitations),
            reason="结构化结果没有带合法 evidence ID 的 claim",
        )
    return ClaimAnswer(
        True,
        text="\n".join(
            _render_claim(str(item["text"]), list(item["evidence_ids"]))
            for item in claims
        ),
        claims=tuple(claims),
        limitations=tuple(limitations),
    )


def generate_answer(context: Context, *, cfg=None, dry_run: bool = False,
                    model: str | None = None, client=None) -> str:
    """生成答案文本。`context.prefix` 里已把引用格式与三态红线写死。

    用 `ask_free`（纯文本，不 JSON）——JSON object 模式会让部分模型把 schema
    结构回显而非产出带引用的自然语言（实测移动云模型会返回 `{"answer":{...}}`）。

    `client` 可传入复用连接（服务层传同一个 client，才能把本次 token 用量
    统一记到 `.usage` 上）；不传则自建一个（旧调用方式保持可用）。
    """
    cfg = cfg or load_config()
    pro = (cfg.get("llm") or {}).get("model_pro")
    client = client or make_client(cfg, model=model or pro or "deepseek-chat",
                                   dry_run=dry_run)
    user = _compose_user(context, cfg)
    call = client.ask_free(
        user,
        system=context.prefix,
    )
    if call.ok:
        return (call.data.get("text") if call.data else None) or call.raw
    return ""
