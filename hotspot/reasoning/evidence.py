"""Concept/anchor evidence sufficiency for answerability decisions.

This is deliberately deterministic first: it checks concepts and exact anchors
against title+abstract, extracts short evidence snippets, and returns a
decision that the engine can log or enforce. A Jev/NLI classifier can later
replace the ambiguous middle band.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EvidenceResult", "evaluate"]

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _phrase_match(text: str, phrase: str) -> bool:
    """按词元匹配短语，避免 cats 匹配 categories。"""
    words = [w for w in _WORD.findall((phrase or "").lower()) if w]
    return bool(words) and set(words) <= _tokens(text)


def _snippet(text: str, phrases: list[str], limit: int = 360) -> str:
    sentences = [s.strip() for s in _SENT.split(text or "") if s.strip()]
    for s in sentences:
        if any(_phrase_match(s, p) for p in phrases):
            return s[:limit]
    return (text or "")[:limit]


@dataclass(frozen=True)
class EvidenceResult:
    decision: str                 # answer / insufficient_evidence / clarify / out_of_scope
    reason: str
    matched_concepts: tuple[str, ...] = ()
    matched_anchors: tuple[str, ...] = ()
    snippets: tuple[str, ...] = ()
    best_concept_coverage: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "matched_concepts": list(self.matched_concepts),
            "matched_anchors": list(self.matched_anchors),
            "snippets": list(self.snippets),
            "best_concept_coverage": self.best_concept_coverage,
            "score": self.score,
        }


def evaluate(
    corpus,
    hits,
    plan: dict,
    *,
    top_n: int = 30,
) -> EvidenceResult:
    """Evaluate whether the retrieved candidates contain enough evidence.

    `plan` is the output of ``parse_query``. It should contain
    ``core_concepts`` (objects with name/aliases/required) and optional
    ``anchors``. If it does not, the caller should use the old coverage path.
    """
    concepts = [
        c for c in (plan.get("core_concepts") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    anchors = [str(a) for a in (plan.get("anchors") or []) if str(a).strip()]
    intent = str(plan.get("intent") or "fact")
    if not concepts and not anchors:
        return EvidenceResult("answer", "no structured concept gate",
                              score=0.0)

    candidates = []
    for h in list(hits or [])[:top_n]:
        pid = getattr(h, "paper_id", "")
        p = corpus.get(pid) if pid else None
        if p is None:
            continue
        text = f"{p.title or ''} {p.abstract or ''}"
        matched = []
        for c in concepts:
            aliases = [str(c.get("name") or "")] + [
                str(x) for x in (c.get("aliases") or []) if str(x).strip()
            ]
            aliases = [a for a in aliases if a.strip()]
            if any(_phrase_match(text, a) for a in aliases):
                matched.append(str(c["name"]))
        matched_anchors = [a for a in anchors if _phrase_match(text, a)]
        # Prefer candidates that satisfy required concepts and anchors.
        req = [c for c in concepts if bool(c.get("required", False))]
        req_names = {str(c["name"]) for c in req}
        required_ok = req_names <= set(matched)
        anchor_ok = set(anchors) <= set(matched_anchors)
        cov = len(matched) / max(len(concepts), 1)
        score = (0.5 if required_ok else 0.0) + (0.3 if anchor_ok else 0.0) + 0.2 * cov
        candidates.append({
            "paper_id": pid,
            "matched": matched,
            "matched_anchors": matched_anchors,
            "coverage": cov,
            "score": score,
            "snippet": _snippet(text, [
                *[str(x) for c in concepts for x in (c.get("aliases") or [c.get("name")])],
                *anchors,
            ]),
        })

    if not candidates:
        return EvidenceResult("out_of_scope", "没有可检查的候选论文")

    candidates.sort(key=lambda x: (x["score"], x["coverage"]), reverse=True)
    best = candidates[0]
    matched_union = set().union(*(set(c["matched"]) for c in candidates))
    anchors_union = set().union(*(set(c["matched_anchors"]) for c in candidates))
    required = {str(c["name"]) for c in concepts if c.get("required")}
    snippet_ok = bool(best["snippet"].strip())

    missing_anchors = [a for a in anchors if a not in anchors_union]
    if anchors and missing_anchors:
        return EvidenceResult(
            "insufficient_evidence",
            f"缺少 anchor 证据：{missing_anchors}",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
        )
    # An explicit anchor is the hard entity constraint. Without an anchor,
    # required concepts remain mandatory.
    if required and not anchors and not required <= matched_union:
        return EvidenceResult(
            "insufficient_evidence",
            f"缺少核心概念：{sorted(required - matched_union)}",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
        )
    if intent in {"fact", "compare"} and not anchors and best["coverage"] < 0.5:
        return EvidenceResult(
            "insufficient_evidence",
            "候选只覆盖少量核心概念",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
        )
    if intent == "compare" and len(concepts) >= 2 and best["coverage"] < 1.0:
        return EvidenceResult(
            "clarify",
            "对比对象至少在单篇候选证据中未同时出现",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
        )
    if not snippet_ok:
        return EvidenceResult(
            "insufficient_evidence", "候选摘要没有可用证据片段",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (), best["coverage"], best["score"],
        )
    return EvidenceResult(
        "answer", "概念/锚点与摘要证据满足",
        tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
        (best["snippet"],), best["coverage"], best["score"],
    )
