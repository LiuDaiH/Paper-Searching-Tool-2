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
_REQUIRED_COVERAGE = 2 / 3
_RELEVANCE_GENERIC = {
    "a", "an", "and", "approach", "approaches", "area", "areas",
    "algorithm", "algorithms", "application", "applications", "based",
    "challenge", "challenges", "control", "core", "current", "data",
    "development", "direction", "directions", "framework", "hot", "hotspot",
    "hotspots", "how", "idea", "ideas", "improvement", "improvements",
    "in", "innovation", "innovations", "latest", "learning", "main",
    "method", "methods", "model", "models", "network", "networks", "new",
    "of", "on", "optimization", "paper", "papers", "performance", "popular",
    "prediction", "progress", "recent", "recently", "research", "researches",
    "segmentation", "study", "system", "systems", "the", "to", "topic",
    "topics", "trend", "trends", "what", "which", "with", "work", "works",
    "year", "years",
}


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
    evidence_id: str = ""
    requirement_results: tuple[dict[str, Any], ...] = ()
    required_coverage: float = 0.0
    domain_coverage: float = 0.0
    matched_domain_terms: tuple[str, ...] = ()
    domain_support_ids: tuple[str, ...] = ()
    domain_gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "matched_concepts": list(self.matched_concepts),
            "matched_anchors": list(self.matched_anchors),
            "snippets": list(self.snippets),
            "best_concept_coverage": self.best_concept_coverage,
            "score": self.score,
            "evidence_id": self.evidence_id,
            "requirement_results": [dict(r) for r in self.requirement_results],
            "required_coverage": self.required_coverage,
            "domain_coverage": self.domain_coverage,
            "matched_domain_terms": list(self.matched_domain_terms),
            "domain_support_ids": list(self.domain_support_ids),
            "domain_gate_reason": self.domain_gate_reason,
        }


def _requirement_phrases(value: str, concepts: list[dict]) -> list[str]:
    """Expand a requirement with aliases of an equivalent core concept."""
    needle = (value or "").strip().casefold()
    if not needle:
        return []
    phrases = [value]
    for concept in concepts:
        names = [
            str(concept.get("name") or ""),
            *[str(x) for x in (concept.get("aliases") or []) if str(x).strip()],
        ]
        if needle in {x.casefold() for x in names if x}:
            phrases.extend(x for x in names if x)
            break
    return list(dict.fromkeys(x for x in phrases if x))


def _evaluate_requirements(
    candidates: list[dict[str, Any]], plan: dict, concepts: list[dict],
) -> list[dict[str, Any]]:
    """Evaluate semantic requirements independently of the best single paper."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for req in (plan.get("requirements") or [])[:12]:
        if not isinstance(req, dict):
            continue
        kind = str(req.get("kind") or "other").strip().lower()
        value = str(req.get("value") or "").strip()
        if not value:
            continue
        key = f"{kind}:{value.casefold()}"
        if key in seen:
            continue
        seen.add(key)
        phrases = _requirement_phrases(value, concepts)
        matched_ids: list[str] = []
        for candidate in candidates:
            paper = candidate.get("paper")
            matched = False
            if kind == "time":
                years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", value)]
                matched = bool(years) and getattr(paper, "year", None) in years
            elif kind == "venue":
                venue = str(getattr(paper, "venue", "") or "")
                matched = bool(venue) and _phrase_match(venue, value)
            else:
                matched = any(
                    _phrase_match(candidate.get("text") or "", p) for p in phrases
                )
            if matched:
                matched_ids.append(str(candidate.get("paper_id") or ""))
        out.append({
            "kind": kind,
            "value": value,
            "required": bool(req.get("required", True)),
            "satisfied": bool(matched_ids),
            "paper_ids": list(dict.fromkeys(pid for pid in matched_ids if pid)),
        })
    return out


def _anchor_belongs(anchor: str, concept: dict) -> bool:
    """Return whether an anchor names this concept or one of its versions."""
    needle = str(anchor or "").strip().casefold()
    if not needle:
        return False
    names = [
        str(concept.get("name") or ""),
        *[str(x) for x in (concept.get("aliases") or []) if str(x).strip()],
    ]
    for name in names:
        key = name.casefold()
        if key == needle:
            return True
        # Dreamer -> DreamerV2/DreamerV3 are alternatives of one series.
        if key.startswith(needle):
            suffix = key[len(needle):].lstrip(" _-")
            if suffix.startswith("v") and suffix[1:].isdigit():
                return True
    return False


def _anchor_groups(anchors: list[str], concepts: list[dict]) -> list[list[str]]:
    """Group version-like anchors under their shared core concept."""
    groups: list[list[str]] = []
    grouped: set[str] = set()
    by_version_base: dict[str, list[str]] = {}
    for anchor in anchors:
        base = re.sub(r"[ _-]?v?\d+$", "", anchor.casefold()).strip()
        if base:
            by_version_base.setdefault(base, []).append(anchor)
    for members in by_version_base.values():
        if len(members) > 1:
            groups.append(members)
            grouped.update(members)
    for concept in concepts:
        members = [
            a for a in anchors
            if a not in grouped and _anchor_belongs(a, concept)
        ]
        if members:
            groups.append(members)
            grouped.update(members)
    groups.extend([a] for a in anchors if a not in grouped)
    return groups


def _domain_groups(plan: dict, concepts: list[dict]) -> tuple[list[dict], str]:
    """Build hard-gate groups from domain-bearing query terms.

    Generic research words alone are not evidence that a question belongs to
    this corpus. Aliases remain alternatives within one group, so paraphrases
    do not have to match verbatim.
    """
    raw_terms = [
        str(x).strip() for x in (plan.get("core_terms") or [])
        if str(x).strip()
    ]
    if not raw_terms:
        raw_terms = [
            str(c.get("name") or "").strip() for c in concepts
            if str(c.get("name") or "").strip()
        ]
    if not raw_terms:
        raw_terms = [
            str(x).strip() for x in (plan.get("keywords") or [])
            if str(x).strip()
        ]

    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    had_ascii = False
    for term in raw_terms:
        aliases = [term]
        needle = term.casefold()
        for concept in concepts:
            names = [
                str(concept.get("name") or ""),
                *[str(x) for x in (concept.get("aliases") or []) if str(x).strip()],
            ]
            if needle in {name.casefold() for name in names if name}:
                aliases.extend(name for name in names if name)
                break
        aliases = list(dict.fromkeys(aliases))
        ascii_words = {
            w for alias in aliases for w in _WORD.findall(alias.lower())
        }
        if ascii_words:
            had_ascii = True
        significant = {
            w for w in ascii_words
            if len(w) > 1 and not w.isdigit() and w not in _RELEVANCE_GENERIC
        }
        if not significant:
            continue
        key = "|".join(sorted(significant))
        if key in seen:
            continue
        seen.add(key)
        groups.append({
            "name": term,
            "aliases": aliases,
            "significant": significant,
        })
    if groups:
        return groups, "ok"
    if raw_terms and had_ascii:
        return [], "generic_only"
    return [], "no_ascii_scope_terms"


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
    answer_mode = str(plan.get("answer_mode") or "")

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
            "paper": p,
            "text": text,
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
    requirement_results = _evaluate_requirements(candidates, plan, concepts)
    requirement_hits = {
        str(candidate["paper_id"]): sum(
            str(candidate["paper_id"]) in r["paper_ids"]
            for r in requirement_results if r["required"]
        )
        for candidate in candidates
    }
    candidates.sort(
        key=lambda x: (
            requirement_hits.get(str(x["paper_id"]), 0),
            x["score"],
            x["coverage"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best_id = str(best["paper_id"])
    matched_union = set().union(*(set(c["matched"]) for c in candidates))
    anchors_union = set().union(*(set(c["matched_anchors"]) for c in candidates))
    required = {str(c["name"]) for c in concepts if c.get("required")}
    required_coverage = len(required & matched_union) / max(len(required), 1)
    domain_groups, domain_gate_reason = _domain_groups(plan, concepts)
    matched_domain_terms: list[str] = []
    domain_support_ids = []
    for candidate in candidates:
        matches = [
            str(group["name"]) for group in domain_groups
            if any(_phrase_match(candidate["text"], alias)
                   for alias in group["aliases"])
        ]
        candidate["domain_matches"] = matches
        if matches:
            domain_support_ids.append(str(candidate["paper_id"]))
            matched_domain_terms.extend(matches)
    matched_domain_terms = list(dict.fromkeys(matched_domain_terms))
    domain_support_ids = list(dict.fromkeys(domain_support_ids))
    domain_coverage = (
        len(matched_domain_terms) / len(domain_groups)
        if domain_groups else 0.0
    )
    snippet_ok = bool(best["snippet"].strip())
    result_kw = {
        "requirement_results": tuple(requirement_results),
        "required_coverage": required_coverage,
        "domain_coverage": domain_coverage,
        "matched_domain_terms": tuple(matched_domain_terms),
        "domain_support_ids": tuple(domain_support_ids),
        "domain_gate_reason": domain_gate_reason,
    }

    # 语义需求常用于组织答案和提示缺失项；只有对比对象是硬约束。
    # 方法/指标名称常被论文改写，逐字硬门控会造成大面积误拒。
    hard_requirement_kinds = {"comparison_side"}
    enforce_requirements = (
        answer_mode in {"paper_fact", "compare"}
        or intent in {"fact", "compare"}
    )
    missing_requirements = [
        r for r in requirement_results
        if (enforce_requirements and r["required"]
            and r["kind"] in hard_requirement_kinds and not r["satisfied"])
    ]
    if missing_requirements:
        missing_text = "、".join(
            f"{r['kind']}:{r['value']}" for r in missing_requirements
        )
        has_comparison = any(r["kind"] == "comparison_side" for r in requirement_results)
        return EvidenceResult(
            "clarify" if has_comparison else "insufficient_evidence",
            f"缺少需求证据：{missing_text}",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )

    anchor_groups = _anchor_groups(anchors, concepts)
    missing_anchor_groups = [
        group for group in anchor_groups
        if not any(anchor in anchors_union for anchor in group)
    ]
    if anchors and missing_anchor_groups:
        return EvidenceResult(
            "insufficient_evidence",
            f"缺少 anchor 证据：{missing_anchor_groups}",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    anchors_satisfied = bool(anchors) and not missing_anchor_groups
    if domain_gate_reason == "generic_only" and not anchors_satisfied:
        return EvidenceResult(
            "clarify", "问题只有泛化研究词，缺少可验证的核心领域词",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    if domain_groups and not anchors_satisfied:
        required_domain = max(1, (2 * len(domain_groups) + 2) // 3)
        min_domain_papers = (
            min(2, len(candidates))
            if (answer_mode == "global_trend"
                or intent in {"trend", "worth", "newfield"})
            else 1
        )
        if (len(matched_domain_terms) < required_domain
                or len(domain_support_ids) < min_domain_papers):
            return EvidenceResult(
                "out_of_scope",
                "候选证据未覆盖问题的核心领域"
                f"（{len(matched_domain_terms)}/{len(domain_groups)}，"
                f"支持论文 {len(domain_support_ids)} 篇）",
                tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
                (best["snippet"],) if snippet_ok else (),
                best["coverage"], best["score"],
                evidence_id=best_id, **result_kw,
            )
    # An explicit anchor is the hard entity constraint. Without an anchor,
    # required concepts remain mandatory.
    required_hits = required & matched_union
    if (required and not anchors and not required <= matched_union
            and not (required_coverage >= _REQUIRED_COVERAGE
                     and len(required_hits) >= 2)):
        return EvidenceResult(
            "insufficient_evidence",
            f"缺少核心概念：{sorted(required - matched_union)}"
            f"（覆盖率 {required_coverage:.2f}）",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    if (intent in {"fact", "compare"} and required and not anchors
            and best["coverage"] < 0.5 and required_coverage < 0.5):
        return EvidenceResult(
            "insufficient_evidence",
            "候选只覆盖少量核心概念",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    if (intent == "compare" and not requirement_results
            and len(concepts) >= 2 and best["coverage"] < 1.0):
        return EvidenceResult(
            "clarify",
            "对比对象至少在单篇候选证据中未同时出现",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (best["snippet"],) if snippet_ok else (),
            best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    if not snippet_ok:
        return EvidenceResult(
            "insufficient_evidence", "候选摘要没有可用证据片段",
            tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
            (), best["coverage"], best["score"],
            evidence_id=best_id, **result_kw,
        )
    return EvidenceResult(
        "answer", "概念/锚点与需求矩阵证据满足",
        tuple(sorted(matched_union)), tuple(sorted(anchors_union)),
        (best["snippet"],), best["coverage"], best["score"],
        evidence_id=best_id, **result_kw,
    )
