"""Offline checks for the LanceDB search and fetch tools.

    python scripts/check_tools_v2.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.config import load_config
from hotspot.corpus import open_corpus
from hotspot.core.models import SearchFilters
from hotspot.index.lance_store import LancePaperStore
from hotspot.index.topic_store import TopicStore
from hotspot.tools.papers import fetch_papers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config()
    store = LancePaperStore(cfg)
    corpus = open_corpus(cfg)
    if not store.available:
        print("✗ LanceDB index missing")
        return 1

    questions = json.loads(
        (ROOT / "eval" / "retrieval_questions.json").read_text(encoding="utf-8")
    )
    probes = [
        pid
        for item in questions
        for pid in item.get("relevant_ids", [])
        if corpus.get(pid) is not None
    ][:max(1, args.limit)]

    hits = 0
    failures = []
    for pid in probes:
        paper = corpus.get(pid)
        result = store.search(paper.title, limit=10)
        returned = [item.paper_id for item in result.results]
        if pid in returned:
            hits += 1
        else:
            failures.append(pid)

    fetched = fetch_papers(store, probes[:3])
    filtered = store.search(
        "reinforcement learning",
        filters=SearchFilters(years=[2025]),
        limit=10,
    )
    years_ok = all(hit.year == 2025 for hit in filtered.results)
    topic_store = TopicStore(cfg)
    topics = topic_store.search("reinforcement learning", limit=3)
    print(
        f"{'✓' if not failures and len(fetched) else '✗'} "
        f"tool_v2 probes={len(probes)} title_hits={hits} "
        f"fetched={len(fetched)} years_ok={years_ok} "
        f"topics={len(topics.topics)}"
    )
    if failures or not years_ok or not topics.topics:
        print("misses:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
