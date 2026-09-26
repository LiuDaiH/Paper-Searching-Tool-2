"""L5 引用校验 —— **防幻觉最后防线**（方案 §3.5 / §2.4 第 11 项）。

对答案里的每条 :class:`Claim` 判四种状态，`missing_id` 与 `unsupported` **必须分开**
（KET-RAG 2026 结论：73–84% 的错误是推理失败，不是检索失败）：

| 状态 | 判据 |
|---|---|
| `no_citation` | claim 不带任何 `[P:]` / `[C:]` |
| `missing_id` | 引用的 ID **不在**上下文允许集（`Context.allowed_ids`）⇒ 编造 |
| `unsupported` | ID 存在，但 claim 在对应证据里**找不到依据** ⇒ 推理失败 |
| `supported` | ID 存在且证据支持该 claim |

不通过的计进 `Answer.n_filtered`，**要显示给用户看，不能悄悄吞掉**。
"""

from __future__ import annotations

import re

from ..llm import make_client
from ..schema import Answer, Claim, Context, Route, Verdict

__all__ = ["parse_claims", "classify", "verify_claims", "build_answer"]

#: `[P:orid:xxx]` / `[C:3]`
REF = re.compile(r"\[([PC]):([A-Za-z0-9_.\-:]+)\]")
_SENT_SPLIT = re.compile(r"(?<=[。!?！？])\s*|\n+")


def parse_claims(answer_text: str) -> list[Claim]:
    """把答案切成一个个事实陈述，附带该陈述里出现的引用 ID。"""
    claims: list[Claim] = []
    for s in _SENT_SPLIT.split(answer_text):
        s = s.strip()
        if not s:
            continue
        ids = tuple(m.group(2) for m in REF.finditer(s))
        claims.append(Claim(text=s, cited_ids=ids))
    return claims


def classify(claim: Claim, *, allowed_ids: frozenset[str]) -> tuple[str, str]:
    """纯规则判定：no_citation / missing_id。id 都存在时返回 unsupported（待 LLM 判）。"""
    if not claim.cited_ids:
        return "no_citation", "该陈述没有引用任何 [P:/[C:] 标记"
    missing = [i for i in claim.cited_ids if i not in allowed_ids]
    if missing:
        return "missing_id", f"引用了上下文允许集之外的 ID：{missing}"
    return "unsupported", ""


def judge_claim(claim: Claim, evidence: str, client, *, cfg=None) -> tuple[str, str]:
    """用 LLM（flash）判 claim 是否被 `evidence` 支持。无 client/evidence ⇒ 保守 unsupported。"""
    if client is None or not evidence:
        return "unsupported", "无证据文本或未接审判模型，保守判为无据"
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"},
                       "reason": {"type": "string"}},
        "required": ["verdict"],
    }
    body = (
        "给定以下材料，判断陈述是否**被材料支持**（supported）还是**材料不支持**"
        "（unsupported）。只判这两者，不判观点好坏。\n\n材料：\n"
        f"{evidence[:3000]}\n\n陈述：{claim.text}\n引用的 ID：{claim.cited_ids}"
    )
    call = client.ask(body, system="引用校验员。输出 JSON {verdict, reason}。",
                      schema=schema, required=("verdict",))
    v = (call.data or {}).get("verdict") if call.ok else ""
    supported = v.strip().lower() == "supported"
    return ("supported", (call.data or {}).get("reason", "")) if supported \
        else ("unsupported", "LLM 判为材料不支持")


def verify_claims(claims: list[Claim], *, allowed_ids: frozenset[str],
                  evidence_lookup: dict[str, str],
                  client=None, cfg=None) -> list[Verdict]:
    """逐条判定。`evidence_lookup`：id -> 该论文/该簇的摘要/卡片文本。"""
    out: list[Verdict] = []
    for c in claims:
        status, detail = classify(c, allowed_ids=allowed_ids)
        if status == "unsupported":
            ev = "\n".join(evidence_lookup.get(i, "") for i in c.cited_ids)
            status, detail = judge_claim(c, ev, client, cfg=cfg)
        out.append(Verdict(claim=c, status=status, detail=detail))
    return out


def build_answer(route: Route, text: str, verdicts: list[Verdict], *,
                 refused: bool = False, refuse_reason: str = "") -> Answer:
    return Answer(route=route, text=text, verdicts=tuple(verdicts),
                  refused=refused, refuse_reason=refuse_reason)
