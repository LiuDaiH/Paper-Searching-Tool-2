"""Mandatory evidence rendering and citation verification for agent output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..api.contract import Citation, Verdict as ContractVerdict
from ..core.models import AgentAnswer, AnswerClaim
from ..llm import make_client
from ..schema import Claim
from .verify import verify_claims

_CITE = re.compile(r"\[[PC]:[^\]]+\]")


@dataclass(frozen=True)
class GroundingResult:
    text: str = ""
    refused: bool = False
    refuse_reason: str = ""
    verdicts: tuple[ContractVerdict, ...] = ()
    citation_ids: tuple[str, ...] = ()


def _render(text: str, evidence_ids: list[str]) -> str:
    clean = _CITE.sub("", text or "").strip()
    marks = "".join(f"[P:{pid}]" for pid in evidence_ids)
    if clean and clean[-1] in "。.!?！？":
        return f"{clean[:-1]}{marks}{clean[-1]}"
    return f"{clean} {marks}".strip()


def verify_and_render(
    answer: AgentAnswer,
    *,
    corpus,
    cfg: dict,
    client=None,
) -> GroundingResult:
    """Enforce fetched IDs, citation verification, and deterministic rendering."""
    if not answer.claims:
        reason = answer.no_answer_reason.strip() or "没有找到足以回答问题的论文证据"
        return GroundingResult(refused=True, refuse_reason=reason)

    evidence_ids = list(dict.fromkeys(
        pid
        for item in answer.claims
        for pid in item.evidence_ids
        if str(pid).strip()
    ))
    details = {
        pid: corpus.get(pid)
        for pid in evidence_ids
    }
    missing = [pid for pid, paper in details.items() if paper is None]
    if missing:
        return GroundingResult(
            refused=True,
            refuse_reason=f"证据 ID 不在本地语料中：{missing}",
        )

    schema_claims = [
        Claim(
            text=item.text,
            cited_ids=tuple(str(pid) for pid in item.evidence_ids),
        )
        for item in answer.claims
    ]
    evidence_lookup = {
        pid: f"{paper.title or ''}\n{paper.abstract or ''}"
        for pid, paper in details.items()
        if paper is not None
    }
    client = client or make_client(cfg)
    verdicts = verify_claims(
        schema_claims,
        allowed_ids=frozenset(evidence_ids),
        evidence_lookup=evidence_lookup,
        client=client,
        cfg=cfg,
        batch=True,
    )

    rendered = []
    contract_verdicts = []
    used_ids = []
    for verdict in verdicts:
        claim = verdict.claim
        text = _render(claim.text, list(claim.cited_ids))
        contract_verdicts.append(ContractVerdict(
            status=verdict.status,
            text=text,
            detail=verdict.detail,
        ))
        if verdict.status == "supported":
            rendered.append(text)
            used_ids.extend(claim.cited_ids)

    if not rendered:
        return GroundingResult(
            refused=True,
            refuse_reason="所有候选 claim 都未通过引用校验",
            verdicts=tuple(contract_verdicts),
            citation_ids=tuple(dict.fromkeys(used_ids)),
        )
    return GroundingResult(
        text="\n".join(rendered),
        verdicts=tuple(contract_verdicts),
        citation_ids=tuple(dict.fromkeys(used_ids)),
    )


def make_citations(corpus, citation_ids: list[str] | tuple[str, ...]) -> list[Citation]:
    out = []
    for pid in dict.fromkeys(str(x) for x in citation_ids):
        paper = corpus.get(pid)
        if paper is None:
            continue
        out.append(Citation(
            id=pid,
            kind="paper",
            title=paper.title or "",
            venue=paper.venue or "",
            year=paper.year,
            abstract=(paper.abstract or "")[:500],
            arxiv_id=getattr(paper, "arxiv_id", "") or "",
            url=getattr(paper, "url", "") or "",
        ))
    return out


def extractive_fallback(
    corpus,
    fetched_ids: list[str] | tuple[str, ...],
    *,
    limit: int = 3,
) -> AgentAnswer:
    """Create a minimal grounded answer when the model cannot emit valid claims."""
    claims = []
    for pid in dict.fromkeys(str(x) for x in fetched_ids):
        paper = corpus.get(pid)
        if paper is None:
            continue
        snippet = re.sub(r"\s+", " ", paper.abstract or paper.title or "").strip()
        if not snippet:
            continue
        if len(snippet) > 260:
            snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
        claims.append(AnswerClaim(
            text=f"《{paper.title or pid}》指出：{snippet}",
            evidence_ids=[pid],
        ))
        if len(claims) >= limit:
            break
    if not claims:
        return AgentAnswer(no_answer_reason="没取到可抽取的论文证据")
    return AgentAnswer(
        claims=claims,
        limitations=["模型未产出合法结构化 claim，已使用抽取式证据降级"],
    )
