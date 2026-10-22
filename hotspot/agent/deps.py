"""Request-scoped dependencies for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import load_config
from ..corpus import open_corpus
from ..index.lance_store import LancePaperStore
from ..index.topic_store import TopicStore


@dataclass
class AgentDeps:
    cfg: dict = field(default_factory=load_config)
    source: str = "meeting"
    years: list[int] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)
    store: LancePaperStore | None = None
    topics: TopicStore | None = None
    corpus: object | None = None
    seen_ids: set[str] = field(default_factory=set)
    fetched: set[str] = field(default_factory=set)
    fetched_order: list[str] = field(default_factory=list)
    search_calls: int = 0
    fetch_calls: int = 0
    topic_calls: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = LancePaperStore(self.cfg, source=self.source)
        if self.topics is None:
            self.topics = TopicStore(self.cfg)
        if self.corpus is None:
            self.corpus = open_corpus(self.cfg)
