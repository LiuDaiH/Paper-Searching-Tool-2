"""Read-only topic snapshots built from offline cluster artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..config import load_config, paths_from
from ..core.models import TopicHit, TopicOverview
from ..corpus import open_corpus

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "with", "from", "into", "using", "based",
    "what", "which", "how", "are", "is", "of", "to", "in", "on",
    "a", "an", "latest", "recent", "research", "method", "methods",
    "approach", "approaches", "paper", "papers", "topic", "topics",
}


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD.findall((text or "").lower())
        if len(word) > 2 and word not in _STOP
    }


class TopicStore:
    """Compatibility view over the existing precomputed cluster snapshot."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        ws = paths_from(self.cfg)["workspace"]
        self.metrics_path = ws / "metrics" / "metrics.json"
        self.members_path = ws / "metrics" / "cluster_members.json"
        self._corpus = None
        self._rows: list[dict] | None = None

    @property
    def available(self) -> bool:
        return self.metrics_path.exists() and self.members_path.exists()

    def _load(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        members = json.loads(self.members_path.read_text(encoding="utf-8"))
        self._corpus = open_corpus(self.cfg)
        rows = []
        for metric in metrics.get("metrics") or []:
            cid = int(metric["cluster_id"])
            member_ids = [str(x) for x in members.get(str(cid), [])]
            titles = []
            for pid in member_ids[:8]:
                paper = self._corpus.get(pid)
                if paper is not None and paper.title:
                    titles.append(paper.title)
            representative_ids = [
                str(x) for x in (metric.get("representative_ids") or [])
            ]
            representative_title = ""
            if representative_ids:
                rep = self._corpus.get(representative_ids[0])
                representative_title = rep.title if rep is not None else ""
            venue = metric.get("venue") or {}
            rows.append({
                "cluster_id": cid,
                "size": int(metric.get("size") or 0),
                "spread_state": str(venue.get("state") or ""),
                "venue_counts": dict(venue.get("counts") or {}),
                "year_counts": {
                    int(year): int(count)
                    for year, values in (venue.get("counts_by_year") or {}).items()
                    for count in [sum(int(x) for x in values.values())]
                },
                "representative_id": representative_ids[0] if representative_ids else "",
                "representative_title": representative_title,
                "terms": _terms(" ".join([representative_title, *titles])),
            })
        self._rows = rows
        return rows

    def search(self, query: str, *, limit: int = 5) -> TopicOverview:
        q_terms = _terms(query)
        if not q_terms:
            return TopicOverview(query=query, topics=[])
        scored = []
        for row in self._load():
            overlap = len(q_terms & row["terms"])
            if overlap:
                scored.append((overlap / len(q_terms), row))
        scored.sort(
            key=lambda item: (item[0], item[1]["size"]),
            reverse=True,
        )
        topics = [
            TopicHit(
                cluster_id=row["cluster_id"],
                label=row["representative_title"] or f"cluster {row['cluster_id']}",
                size=row["size"],
                spread_state=row["spread_state"],
                venue_counts=row["venue_counts"],
                year_counts=row["year_counts"],
                representative_id=row["representative_id"],
                representative_title=row["representative_title"],
                score=round(float(score), 4),
            )
            for score, row in scored[:max(1, min(limit, 10))]
        ]
        return TopicOverview(query=query, topics=topics)
