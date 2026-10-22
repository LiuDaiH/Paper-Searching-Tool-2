"""Offline topic snapshot tool."""

from __future__ import annotations

from ..core.models import TopicOverview
from ..index.topic_store import TopicStore


def get_topic_overview(store: TopicStore, query: str, *, limit: int = 5) -> TopicOverview:
    """Return precomputed topic clusters and their venue/year spread."""
    return store.search(query, limit=limit)
