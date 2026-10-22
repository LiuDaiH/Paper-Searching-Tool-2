"""Global coarse-topic coverage for retrieved candidates.

The on-disk cluster members are the stable coarse assignment. Papers outside
that membership set are assigned to the nearest cluster centroid on demand.
This keeps the first version small: no full-corpus assignment matrix is built.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config, paths_from

__all__ = [
    "GlobalCoverageIndex",
    "coverage_policy",
    "load_global_coverage",
    "select_candidates",
    "summarize",
]

_CACHE: dict[tuple, "GlobalCoverageIndex"] = {}
DEFAULT_MIN_MARGIN = 0.02


@dataclass(frozen=True)
class GlobalCoverageIndex:
    assignment: dict[str, int]
    centroids: np.ndarray
    centroid_ids: tuple[int, ...]
    cluster_sizes: dict[int, int]
    spread_state: dict[int, str]
    vector_ids: list[str]
    vector_rows: dict[str, int]
    index_vectors: np.ndarray
    min_margin: float = DEFAULT_MIN_MARGIN

    def assign(self, paper_id: str) -> tuple[int | None, str]:
        cid = self.assignment.get(str(paper_id))
        if cid is not None:
            return cid, "cluster_membership"
        row = self.vector_rows.get(str(paper_id))
        if row is None or not len(self.centroids):
            return None, "unassigned"
        scores = self.centroids @ self.index_vectors[row]
        if len(scores) > 1:
            top2 = np.partition(scores, -2)[-2:]
            if float(top2[1] - top2[0]) < self.min_margin:
                return None, "low_confidence"
        pos = int(np.argmax(scores))
        return int(self.centroid_ids[pos]), "nearest_centroid"


def coverage_policy(parsed: dict) -> str:
    """Map the answer mode to its coarse-to-fine policy."""
    mode = str(parsed.get("answer_mode") or "").strip().lower()
    if mode == "global_trend":
        return "coarse_then_fine"
    if mode == "compare":
        return "per_side"
    if mode == "relation":
        return "dual_concept"
    return "paper_first"


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def load_global_coverage(cfg=None, *, index=None) -> GlobalCoverageIndex | None:
    """Load cluster members/metrics and build centroid vectors if available."""
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    members_path = ws / "metrics" / "cluster_members.json"
    metrics_path = ws / "metrics" / "metrics.json"
    if index is None:
        from .vector import load_index  # noqa: PLC0415

        index = load_index(cfg)
    X, ids = index
    key = (
        str(members_path), str(metrics_path), len(ids), X.shape,
        members_path.stat().st_mtime_ns if members_path.exists() else 0,
        metrics_path.stat().st_mtime_ns if metrics_path.exists() else 0,
    )
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    members_raw = _load_json(members_path, {})
    metrics_raw = _load_json(metrics_path, {})
    row = {str(pid): i for i, pid in enumerate(ids)}
    assignment: dict[str, int] = {}
    member_rows: dict[int, list[int]] = {}
    cluster_sizes: dict[int, int] = {}
    for raw_cid, raw_members in members_raw.items():
        try:
            cid = int(raw_cid)
        except (TypeError, ValueError):
            continue
        members = [str(pid) for pid in (raw_members or [])]
        cluster_sizes[cid] = len(members)
        rows = []
        for pid in members:
            assignment[pid] = cid
            if pid in row:
                rows.append(row[pid])
        if rows:
            member_rows[cid] = rows

    centroid_ids = []
    centroid_vectors = []
    for cid, rows in sorted(member_rows.items()):
        vec = np.mean(X[rows], axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        centroid_ids.append(cid)
        centroid_vectors.append(vec / norm)
    centroids = (
        np.stack(centroid_vectors).astype(np.float32, copy=False)
        if centroid_vectors else np.empty((0, X.shape[1]), dtype=np.float32)
    )
    spread_state = {
        int(item.get("cluster_id")): str((item.get("venue") or {}).get("state") or "")
        for item in (metrics_raw.get("metrics") or [])
        if item.get("cluster_id") is not None
    }
    out = GlobalCoverageIndex(
        assignment=assignment,
        centroids=centroids,
        centroid_ids=tuple(centroid_ids),
        cluster_sizes=cluster_sizes,
        spread_state=spread_state,
        vector_ids=list(ids),
        vector_rows=row,
        index_vectors=X,
        min_margin=DEFAULT_MIN_MARGIN,
    )
    _CACHE[key] = out
    return out


def summarize(cfg=None, *, pids: list[str], index=None, top_n: int = 5) -> dict[str, Any]:
    """Return normalized coarse-topic coverage for a candidate ID list."""
    unique = list(dict.fromkeys(str(pid) for pid in (pids or []) if pid))
    try:
        coverage = load_global_coverage(cfg, index=index)
    except Exception as exc:  # noqa: BLE001 - observability must not break retrieval
        coverage = None
        unavailable_reason = f"coverage_index_error:{type(exc).__name__}"
    else:
        unavailable_reason = "coverage_index_unavailable"
    if coverage is None:
        return {
            "assigned_coverage": 0.0,
            "unassigned": len(unique),
            "low_confidence": 0,
            "dominant_cluster_id": None,
            "dominant_share": 0.0,
            "n_cluster_members": 0,
            "spread_state": "",
            "topic_assignment_source": "unavailable",
            "coverage_gap_reason": unavailable_reason,
            "top_clusters": [],
        }
    assignments = []
    sources = []
    reasons = []
    for pid in unique:
        cid, reason = coverage.assign(pid)
        reasons.append(reason)
        if cid is not None:
            assignments.append(cid)
            sources.append(reason)
    counts = Counter(assignments)
    top = counts.most_common(max(1, top_n))
    dominant = top[0][0] if top else None
    assigned = len(assignments)
    dominant_share = (top[0][1] / assigned) if top and assigned else 0.0
    source = (
        "mixed" if len(set(sources)) > 1
        else (sources[0] if sources else "unassigned")
    )
    gap = ""
    if not unique:
        gap = "no_candidates"
    elif not assigned:
        gap = "no_cluster_assignment"
    elif "low_confidence" in reasons and len(assignments) < len(unique):
        gap = "low_confidence_cluster"
    elif len(assignments) < len(unique):
        gap = "partial_cluster_assignment"
    return {
        "assigned_coverage": round(assigned / max(len(unique), 1), 4),
        "unassigned": len(unique) - assigned,
        "low_confidence": sum(reason == "low_confidence" for reason in reasons),
        "dominant_cluster_id": dominant,
        "dominant_share": round(dominant_share, 4),
        "n_cluster_members": int(coverage.cluster_sizes.get(dominant, 0))
        if dominant is not None else 0,
        "spread_state": coverage.spread_state.get(dominant, "")
        if dominant is not None else "",
        "topic_assignment_source": source,
        "coverage_gap_reason": gap,
        "top_clusters": [
            {
                "cluster_id": cid,
                "count": count,
                "share": round(count / assigned, 4) if assigned else 0.0,
                "cluster_size": int(coverage.cluster_sizes.get(cid, 0)),
                "spread_state": coverage.spread_state.get(cid, ""),
            }
            for cid, count in top
        ],
    }


def select_candidates(
    coverage: GlobalCoverageIndex,
    pids: list[str],
    *,
    max_clusters: int = 3,
    min_dominant_share: float = 0.35,
    target_share: float = 0.80,
) -> tuple[list[str], dict[str, Any]]:
    """Select stable coarse clusters before online fine clustering.

    This is intentionally conservative: weak or sparse assignment falls back to
    the complete candidate set. A selected cluster must have at least two local
    candidates, so one accidental nearest-centroid assignment cannot steer a
    global trend answer.
    """
    unique = list(dict.fromkeys(str(pid) for pid in (pids or []) if pid))
    assignments: dict[str, int] = {}
    for pid in unique:
        cid, _ = coverage.assign(pid)
        if cid is not None:
            assignments[pid] = cid
    assigned = len(assignments)
    counts = Counter(assignments.values())
    info: dict[str, Any] = {
        "policy": "coarse_then_fine",
        "selected_cluster_ids": [],
        "selected_share": 0.0,
        "retained_candidates": len(unique),
        "dropped_candidates": 0,
        "fallback_reason": "",
    }
    if not unique:
        info["fallback_reason"] = "no_candidates"
        return [], info
    if assigned < max(3, int(np.ceil(len(unique) * 0.5))):
        info["fallback_reason"] = "insufficient_cluster_assignment"
        return unique, info

    ordered = counts.most_common()
    dominant_share = ordered[0][1] / assigned
    if dominant_share < min_dominant_share:
        info["fallback_reason"] = "weak_dominant_topic"
        return unique, info

    selected: list[int] = []
    accumulated = 0
    for cid, count in ordered:
        if len(selected) >= max(1, max_clusters):
            break
        if count < 2:
            break
        selected.append(cid)
        accumulated += count
        if accumulated / assigned >= target_share:
            break
    selected_set = set(selected)
    keep = [
        pid for pid in unique
        if assignments.get(pid) in selected_set or pid not in assignments
    ]
    minimum_keep = max(1, min(4, len(unique) // 4))
    if len(keep) < minimum_keep:
        info["fallback_reason"] = "coarse_filter_too_small"
        return unique, info

    info.update({
        "selected_cluster_ids": selected,
        "selected_share": round(accumulated / assigned, 4),
        "retained_candidates": len(keep),
        "dropped_candidates": len(unique) - len(keep),
    })
    return keep, info
