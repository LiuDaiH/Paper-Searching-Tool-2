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
        "以下是证据材料与一条陈述。判定：若陈述能从材料**合理得出或与材料一致**"
        "（包括由材料可推断的规模、跨会、分年、趋势等数字），判 `supported`；"
        "仅当材料**完全无法支撑、明显无据或与材料矛盾**时才判 `unsupported`。"
        "宁可因材料覆盖不足而偏向 `supported`，也不要把有据推断误判为无据。\n\n"
        f"材料：\n{evidence[:3000]}\n\n陈述：{claim.text}"
    )
    call = client.ask(body, system=(
        "引用校验员：只输出 JSON {\"verdict\": \"supported\" | \"unsupported\", \"reason\": \"\"}。"
        "按上述偏向判定。"),
        schema=schema, required=("verdict",))
    v = (call.data or {}).get("verdict") if call.ok else ""
    supported = v.strip().lower().startswith("supported")
    return ("supported", (call.data or {}).get("reason", "")) if supported \
        else ("unsupported", "LLM 判为材料不支持")


def verify_claims(claims: list[Claim], *, allowed_ids: frozenset[str],
                  evidence_lookup: dict[str, str],
                  client=None, cfg=None, progress=None,
                  batch: bool = True) -> list[Verdict]:
    """判定引用。

    规则判定仍逐条；需要 LLM 的 claim 默认**合并成一次调用**（batch=True），
    失败或返回不完整时自动回退到逐条 `judge_claim`。progress 仍逐条报告。
    """
    out: list[Verdict] = []
    total = len(claims)
    pending: list[tuple[int, Claim, str]] = []
    for i, c in enumerate(claims):
        if progress:
            progress("校验引用", i, total)          # 真实 i/total
        status, detail = classify(c, allowed_ids=allowed_ids)
        if status == "unsupported":
            ev = "\n".join(evidence_lookup.get(i, "") for i in c.cited_ids)
            pending.append((i, c, ev))
            out.append(Verdict(claim=c, status="unsupported", detail=""))
        else:
            out.append(Verdict(claim=c, status=status, detail=detail))

    # 批量判定；任何异常/缺项都按逐条回退，保持“宁可慢，不可漏”。
    if pending and client is not None:
        filled: set[int] = set()
        if batch and len(pending) > 1:
            try:
                filled = _judge_batch(pending, client, out)
            except Exception:  # noqa: BLE001
                filled = set()
        for i, c, ev in pending:
            if i in filled:
                continue
            status, detail = judge_claim(c, ev, client, cfg=cfg)
            out[i] = Verdict(claim=c, status=status, detail=detail)
    elif pending:
        for i, c, ev in pending:
            status, detail = judge_claim(c, ev, client, cfg=cfg)
            out[i] = Verdict(claim=c, status=status, detail=detail)
    if progress:
        progress("校验引用", total, total)
    return out


def _judge_batch(pending: list[tuple[int, Claim, str]], client,
                 out: list[Verdict]) -> set[int]:
    """一次 LLM 判定多条 claim；返回已填充的 claim 下标集合。"""
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "verdict": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "verdict"],
                },
            }
        },
        "required": ["verdicts"],
    }
    payload = [
        {"id": i, "claim": c.text, "evidence": (ev or "")[:1200]}
        for i, c, ev in pending
    ]
    body = (
        "逐条判定下列 claim 是否能由对应 evidence 支持。"
        "只输出 JSON：{\"verdicts\":[{\"id\":整数,\"verdict\":\"supported|unsupported\","
        "\"reason\":\"短理由\"}]}。id 必须原样返回。\n\n"
        + repr(payload)
    )
    call = client.ask(
        body,
        system="引用校验员。逐条判定，不要合并 id；无法判断时判 unsupported。",
        schema=schema,
        required=("verdicts",),
    )
    if not call.ok:
        return set()
    items = (call.data or {}).get("verdicts") or []
    by_id = {int(x["id"]): x for x in items
             if isinstance(x, dict) and isinstance(x.get("id"), (int, float))}
    filled: set[int] = set()
    for i, c, _ev in pending:
        item = by_id.get(i)
        if not item:
            continue
        v = str(item.get("verdict") or "").strip().lower()
        if v not in ("supported", "unsupported"):
            continue
        out[i] = Verdict(claim=c, status=v,
                         detail=str(item.get("reason") or ""))
        filled.add(i)
    return filled


def build_answer(route: Route, text: str, verdicts: list[Verdict], *,
                 refused: bool = False, refuse_reason: str = "",
                 subtopics: tuple[str, ...] = (), meta: dict | None = None) -> Answer:
    return Answer(route=route, text=text, verdicts=tuple(verdicts),
                  refused=refused, refuse_reason=refuse_reason,
                  subtopics=subtopics, meta=meta)
