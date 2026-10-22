"""Private adapters for maturing the minimal tool chain.

These functions normalize existing implementations, but are not registered and
must not be imported by frontend code.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from .contract import ToolObservation


def paper_search(
    corpus,
    query: str,
    *,
    cfg: dict,
    index,
    bm25,
    top_k: int = 50,
    keywords: list[str] | None = None,
    core_terms: list[str] | None = None,
    scope: dict | None = None,
    min_coverage: float | None = None,
) -> ToolObservation:
    """Run hybrid lexical+dense retrieval and normalize the observation."""
    from ..retrieval.hybrid import retrieve  # noqa: PLC0415

    started = time.perf_counter()
    try:
        result = retrieve(
            corpus,
            query,
            cfg=cfg,
            bm25=bm25,
            index=index,
            top_k=top_k,
            keywords=keywords,
            core_terms=core_terms,
            scope=scope,
            min_coverage=min_coverage,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolObservation.failure(
            "paper_search_failed",
            f"{type(exc).__name__}: {exc}",
            retryable=True,
            tool_version="0.1",
        )
    hits = list(result.hits)
    stats = dict(result.stats or {})
    return ToolObservation(
        status="ok" if hits else "empty",
        ids=tuple(hit.paper_id for hit in hits),
        ranks=tuple(range(1, len(hits) + 1)),
        scores=tuple(float(hit.score) for hit in hits),
        counts={
            "returned": len(hits),
            "bm25_candidates": int(stats.get("bm25_candidates") or 0),
            "vector_candidates": int(stats.get("vector_candidates") or 0),
            "fused": int(stats.get("fused") or 0),
        },
        flags={
            "answerable": bool(stats.get("answerable", False)),
            "rare_uncovered": bool(stats.get("rare_uncovered", False)),
            "deterministic": not bool(stats.get("reranked", False)),
        },
        data={
            "query_coverage": stats.get("query_coverage"),
            "min_coverage": stats.get("min_coverage"),
            "rare_terms": list(stats.get("rare_terms") or []),
            "notes": list(result.notes or []),
            "stats": stats,
            "evidence_ids": [hit.paper_id for hit in (result.evidence_hits or hits)],
        },
        elapsed_s=round(time.perf_counter() - started, 3),
        tool_version="0.1",
    )


def fetch_papers(corpus, paper_ids: list[str] | tuple[str, ...]) -> ToolObservation:
    """Fetch known paper records without changing ranking or scope."""
    unique = list(dict.fromkeys(str(pid) for pid in paper_ids if str(pid).strip()))
    papers: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for pid in unique:
        paper = corpus.get(pid)
        if paper is None:
            missing.append(pid)
            continue
        papers[pid] = {
            "title": getattr(paper, "title", "") or "",
            "abstract": getattr(paper, "abstract", "") or "",
            "year": getattr(paper, "year", None),
            "venue": getattr(paper, "venue", "") or "",
        }
    return ToolObservation(
        status="ok" if papers else "empty",
        ids=tuple(papers),
        counts={"requested": len(unique), "returned": len(papers),
                "missing": len(missing)},
        flags={"partial": bool(missing), "deterministic": True},
        data={"papers": papers, "missing_ids": missing},
        tool_version="0.1",
    )


def candidate_filter(
    corpus,
    paper_ids: list[str] | tuple[str, ...],
    *,
    years: list[int] | None = None,
) -> ToolObservation:
    """Remove unknown IDs and apply the corpus's explicit year filter."""
    unique = list(dict.fromkeys(str(pid) for pid in paper_ids if str(pid).strip()))
    known = [pid for pid in unique if corpus.get(pid) is not None]
    years_set = {int(y) for y in (years or [])}
    filtered = known
    if years_set and years_set != {2024, 2025}:
        filtered = [
            pid for pid in known
            if getattr(corpus.get(pid), "year", None) in years_set
        ]
    return ToolObservation(
        status="ok" if filtered else "empty",
        ids=tuple(filtered),
        counts={
            "input": len(unique),
            "unknown": len(unique) - len(known),
            "year_filtered": len(known) - len(filtered),
            "returned": len(filtered),
        },
        flags={
            "years_applied": bool(years_set and years_set != {2024, 2025}),
            "deterministic": True,
        },
        tool_version="0.1",
    )


def coarse_cluster_select(
    paper_ids: list[str] | tuple[str, ...],
    *,
    cfg: dict,
    index,
) -> ToolObservation:
    """Select stable coarse clusters before online local clustering."""
    from ..retrieval.coverage import (  # noqa: PLC0415
        load_global_coverage,
        select_candidates,
    )

    started = time.perf_counter()
    try:
        coverage = load_global_coverage(cfg, index=index)
        if coverage is None:
            return ToolObservation.failure(
                "coverage_index_unavailable",
                "global coverage index unavailable",
                tool_version="0.1",
            )
        kept, selection = select_candidates(coverage, list(paper_ids))
    except Exception as exc:  # noqa: BLE001
        return ToolObservation.failure(
            "coarse_cluster_select_failed",
            f"{type(exc).__name__}: {exc}",
            retryable=True,
            tool_version="0.1",
        )
    return ToolObservation(
        status="ok" if kept else "empty",
        ids=tuple(kept),
        counts={
            "input": len(set(paper_ids)),
            "returned": len(kept),
            "dropped": len(set(paper_ids)) - len(kept),
        },
        flags={
            "fallback_reason": str(selection.get("fallback_reason") or ""),
            "deterministic": True,
        },
        data={"selection": selection},
        elapsed_s=round(time.perf_counter() - started, 3),
        tool_version="0.1",
    )


def local_cluster(
    vectors,
    paper_ids: list[str] | tuple[str, ...],
    *,
    intent: str,
) -> ToolObservation:
    """Cluster an already filtered candidate subset without venue/year labels."""
    from dasc.cluster import clusterers  # noqa: PLC0415

    ids = [str(pid) for pid in paper_ids]
    if len(ids) != len(vectors):
        return ToolObservation.failure(
            "local_cluster_shape_mismatch",
            f"ids={len(ids)} vectors={len(vectors)}",
            tool_version="0.1",
        )
    n_components = max(
        2,
        min(32, int(vectors.shape[0]) - 1, int(vectors.shape[1])),
    )
    mcs = 8
    if len(ids) < 100:
        mcs = 3
    elif len(ids) < 500:
        mcs = 5
    if intent in {"worth", "newfield"}:
        mcs = max(2, mcs - 3)
    cluster_cfg = {
        "min_cluster_size": mcs,
        "min_samples": max(2, min(3, mcs - 1)),
    }
    started = time.perf_counter()
    try:
        result = clusterers.run(
            vectors,
            ids,
            min_cluster_size=cluster_cfg["min_cluster_size"],
            min_samples=cluster_cfg["min_samples"],
            prep="whiten",
            n_components=n_components,
            metric="euclidean",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolObservation.failure(
            "local_cluster_failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            tool_version="0.1",
        )
    clusters = [
        {
            "cluster_id": cluster.label,
            "size": cluster.size,
            "members": list(cluster.members),
            "representative": cluster.centroid_nearest,
        }
        for cluster in result.clusters
    ]
    return ToolObservation(
        status="ok" if clusters else "empty",
        ids=tuple(str(cluster["representative"]) for cluster in clusters),
        counts={
            "input": len(ids),
            "clusters": len(clusters),
            "clustered": sum(int(cluster["size"]) for cluster in clusters),
        },
        flags={"deterministic": True},
        data={"cluster_cfg": cluster_cfg, "clusters": clusters},
        elapsed_s=round(time.perf_counter() - started, 3),
        tool_version="0.1",
    )


def evidence_gate(
    corpus,
    candidate_ids: list[str] | tuple[str, ...],
    plan: dict,
) -> ToolObservation:
    """Evaluate a query plan against already retrieved candidate papers."""
    from ..reasoning.evidence import evaluate  # noqa: PLC0415

    started = time.perf_counter()
    hits = [SimpleNamespace(paper_id=str(pid)) for pid in candidate_ids]
    try:
        evidence = evaluate(corpus, hits, plan)
    except Exception as exc:  # noqa: BLE001
        return ToolObservation.failure(
            "evidence_gate_failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            tool_version="0.1",
        )
    evidence_id = str(evidence.evidence_id or "")
    return ToolObservation(
        status="ok",
        ids=(evidence_id,) if evidence_id else (),
        counts={
            "candidates": len(hits),
            "matched_concepts": len(evidence.matched_concepts),
            "matched_anchors": len(evidence.matched_anchors),
            "domain_support": len(evidence.domain_support_ids),
        },
        flags={
            "decision": evidence.decision,
            "reason": evidence.reason,
            "domain_gate_reason": evidence.domain_gate_reason,
            "deterministic": True,
        },
        data={"evidence": evidence.to_dict()},
        elapsed_s=round(time.perf_counter() - started, 3),
        tool_version="0.1",
    )
