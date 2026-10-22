"""Audit corpus-wide coarse-topic coverage without loading an LLM.

    python scripts/audit_coverage.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.config import load_config, paths_from  # noqa: E402
from hotspot.retrieval.coverage import load_global_coverage  # noqa: E402
from hotspot.retrieval.vector import load_index  # noqa: E402


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "p10": round(float(p10), 4),
        "p50": round(float(p50), 4),
        "p90": round(float(p90), 4),
    }


def audit(cfg: dict) -> dict:
    index = load_index(cfg)
    X, ids = index
    coverage = load_global_coverage(cfg, index=index)
    if coverage is None:
        raise RuntimeError("global coverage index unavailable")

    assigned: dict[str, int] = {}
    source = Counter()
    pending_rows: list[int] = []
    pending_ids: list[str] = []
    for row, pid in enumerate(ids):
        cid, why = coverage.assign(pid)
        source[why] += 1
        if why in {"nearest_centroid", "low_confidence"}:
            pending_rows.append(row)
            pending_ids.append(pid)
        if cid is not None:
            assigned[pid] = cid

    margins = np.empty(0, dtype=np.float32)
    top_scores = np.empty(0, dtype=np.float32)
    if pending_rows and len(coverage.centroids):
        chunks = []
        chunk_margins = []
        chunk_scores = []
        for start in range(0, len(pending_rows), 2048):
            rows = pending_rows[start:start + 2048]
            scores = coverage.index_vectors[rows] @ coverage.centroids.T
            if scores.shape[1] > 1:
                part = np.partition(scores, -2, axis=1)[:, -2:]
                part.sort(axis=1)
                chunk_margins.extend((part[:, 1] - part[:, 0]).tolist())
                chunk_scores.extend(part[:, 1].tolist())
            else:
                chunk_margins.extend([1.0] * len(rows))
                chunk_scores.extend(scores[:, 0].tolist())
            chunks.append(scores.argmax(axis=1))
        nearest_pos = np.concatenate(chunks)
        for pid, pos in zip(pending_ids, nearest_pos):
            expected = assigned.get(pid)
            actual = int(coverage.centroid_ids[int(pos)])
            if expected is not None and expected != actual:
                raise RuntimeError(f"nearest-centroid audit mismatch: {pid}")
        margins = np.asarray(chunk_margins, dtype=np.float32)
        top_scores = np.asarray(chunk_scores, dtype=np.float32)

    counts = Counter(assigned.values())
    total = len(ids)
    dominant_id, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    stable = source.get("cluster_membership", 0)
    nearest = source.get("nearest_centroid", 0)
    low_confidence = source.get("low_confidence", 0)
    unassigned = total - len(assigned)
    return {
        "total_papers": total,
        "assigned": len(assigned),
        "assigned_coverage": round(len(assigned) / max(total, 1), 4),
        "stable_cluster_members": stable,
        "stable_coverage": round(stable / max(total, 1), 4),
        "nearest_centroid": nearest,
        "nearest_coverage": round(nearest / max(total, 1), 4),
        "unassigned": unassigned,
        "low_confidence": low_confidence,
        "ambiguous_margin_lt_002": int((margins < 0.02).sum()),
        "margin_quantiles": _quantiles(margins),
        "top_similarity_quantiles": _quantiles(top_scores),
        "n_assigned_clusters": len(counts),
        "n_metric_clusters": len(coverage.spread_state),
        "dominant_cluster_id": dominant_id,
        "dominant_share": round(dominant_count / max(len(assigned), 1), 4),
        "cluster_size_quantiles": _quantiles(
            np.asarray(list(counts.values()), dtype=np.float32)
        ),
        "top_clusters": [
            {
                "cluster_id": cid,
                "count": count,
                "share": round(count / max(len(assigned), 1), 4),
                "stable_members": coverage.cluster_sizes.get(cid, 0),
                "spread_state": coverage.spread_state.get(cid, ""),
            }
            for cid, count in counts.most_common(12)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="全库粗主题覆盖审计")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config()
    result = audit(cfg)
    out = Path(args.out) if args.out else (
        paths_from(cfg)["workspace"] / "ledger" / "coverage_audit.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(
        "coverage audit | "
        f"total={result['total_papers']} stable={result['stable_coverage']:.4f} "
        f"nearest={result['nearest_coverage']:.4f} "
        f"unassigned={result['unassigned']} "
        f"low_confidence={result['low_confidence']} "
        f"ambiguous={result['ambiguous_margin_lt_002']} "
        f"clusters={result['n_assigned_clusters']} out={out}"
    )
    hard_unassigned = (
        result["unassigned"] - result["low_confidence"]
    )
    return 0 if hard_unassigned == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
