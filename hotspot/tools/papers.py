"""Fetch full paper records for evidence IDs returned by search."""

from __future__ import annotations

from ..core.models import PaperDetail
from ..index.lance_store import LancePaperStore


def fetch_papers(
    store: LancePaperStore,
    paper_ids: list[str],
    *,
    limit: int = 8,
) -> list[PaperDetail]:
    """Fetch title/abstract metadata for up to ``limit`` known paper IDs."""
    ids = [str(pid) for pid in paper_ids if str(pid).strip()][:max(1, min(limit, 20))]
    return store.get_papers(ids)
