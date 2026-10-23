"""Build a deterministic, score-blinded qrels candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402
from hotspot.index.lance_store import LancePaperStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "retrieval_questions.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "eval" / "qrels_v2"))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-rank", type=int, default=60)
    parser.add_argument("--hard-negatives", type=int, default=15)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--random-negatives", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    store = LancePaperStore(cfg)
    if not store.available:
        print("✗ LanceDB index missing")
        return 1
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    from hotspot.corpus import open_corpus

    all_ids = [p.paper_id for p in open_corpus(cfg).list_papers(limit=None)]
    if args.query_limit:
        questions = questions[:args.query_limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rows = []
    pool_rows = []
    for item in questions:
        qid = str(item["id"])
        query = str(item["query"])
        keywords = [str(x) for x in (item.get("keywords") or [])]
        result = store.search(
            query,
            keywords=keywords,
            limit=max(1, min(max(args.top_k, args.max_rank), 100)),
        )
        candidates: dict[str, dict] = {}
        for rank, hit in enumerate(result.results[:args.top_k], start=1):
            candidates[hit.paper_id] = {
                "paper_id": hit.paper_id,
                "retrieval_rank": rank,
                "retrieval_source": "lance_hybrid",
            }
        rng = random.Random(
            int(hashlib.sha256(f"{qid}:hard".encode("utf-8")).hexdigest()[:16], 16)
        )
        hard = result.results[args.top_k:max(args.top_k, args.max_rank)]
        for hit in rng.sample(hard, min(args.hard_negatives, len(hard))):
            candidates.setdefault(hit.paper_id, {
                "paper_id": hit.paper_id,
                "retrieval_rank": result.results.index(hit) + 1,
                "retrieval_source": "hard_negative",
            })
        for pid in item.get("relevant_ids") or []:
            candidates.setdefault(pid, {
                "paper_id": str(pid),
                "retrieval_rank": None,
                "retrieval_source": "seed_relevant",
            })
        if args.random_negatives > 0:
            seed = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16], 16)
            remaining = [pid for pid in all_ids if pid not in candidates]
            for pid in random.Random(seed).sample(
                remaining,
                min(args.random_negatives, len(remaining)),
            ):
                candidates[pid] = {
                    "paper_id": pid,
                    "retrieval_rank": None,
                    "retrieval_source": "random_negative",
                }
        details = {p.paper_id: p for p in store.get_papers(list(candidates))}
        query_rows.append({
            "id": qid,
            "type": item.get("type", "retrieval"),
            "source": item.get("source", "meeting"),
            "query": query,
            "search_query": item.get("search_query", query),
            "keywords": keywords,
            "filters": {
                "years": item.get("years") or [],
                "venues": item.get("venues") or [],
            },
            "expected_refusal": bool(item.get("expect_refuse", False)),
        })
        for pid, meta in candidates.items():
            paper = details.get(pid)
            if paper is None:
                continue
            blind = hashlib.sha256(f"{qid}::{pid}".encode("utf-8")).hexdigest()
            pool_rows.append({
                "query_id": qid,
                "paper_id": pid,
                **meta,
                "blind_order": blind,
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "venue": paper.venue,
            })
    pool_rows.sort(key=lambda row: (row["query_id"], row["blind_order"]))
    (out_dir / "queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in query_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "pool.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pool_rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"qrels pool | queries={len(query_rows)} candidates={len(pool_rows)} "
        f"out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
