"""Retrieval-only evaluation for P1-1.

默认读 `eval/retrieval_questions.json`，在样例/测试配置上计算
Recall@k、nDCG@10、Precision@10、MRR 和检索耗时；`--grid` 会扫描
hybrid 候选数/RRF 参数，按 nDCG@10 排序后给出建议配置。

用法::

    $env:HOTSPOT_CONFIG='测试数据集/config.yaml'
    python eval/retrieval_eval.py
    python eval/retrieval_eval.py --grid
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hotspot  # noqa: E402,F401
from hotspot.api.engine import Engine  # noqa: E402
from hotspot.config import load_config, paths_from  # noqa: E402
from hotspot.corpus import open_corpus  # noqa: E402
from hotspot.retrieval.hybrid import (  # noqa: E402
    _bm25_scores,
    _dense_scores,
    retrieve,
)


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _metrics(ids: list[str], relevant: set[str], k: int) -> dict:
    top = ids[:k]
    hits = [1.0 if pid in relevant else 0.0 for pid in top]
    n_rel = len(relevant)
    recall = sum(hits) / n_rel if n_rel else None
    precision = sum(hits) / k if k else None
    idcg = _dcg([1.0] * min(n_rel, k)) if n_rel else 0.0
    ndcg = (_dcg(hits) / idcg) if idcg else None
    rr = 0.0
    for i, pid in enumerate(top, start=1):
        if pid in relevant:
            rr = 1.0 / i
            break
    return {"recall": recall, "precision": precision, "ndcg": ndcg, "mrr": rr}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _load(cfg):
    corpus = open_corpus(cfg)
    eng = Engine(cfg=cfg)
    return (
        corpus,
        eng._vector_index("meeting", cfg),
        eng._bm25_index("meeting", corpus, cfg),
    )


def evaluate(items, cfg, *, mode: str = "hybrid", top_k: int = 50) -> dict:
    corpus, index, bm25 = _load(cfg)
    bm, bm_ids = bm25
    rows = []
    for item in items:
        q = item["query"]
        t0 = time.perf_counter()
        if mode == "bm25":
            rows_bm = _bm25_scores(bm, bm_ids, [q], top_k)
            ids = [pid for pid, _ in rows_bm]
            notes = []
        elif mode == "vector":
            rows_vec = _dense_scores(
                q, cfg=cfg, index=index,
                model=str((cfg.get("embed") or {}).get("model") or "BAAI/bge-m3"),
            )
            ids = [pid for pid, _ in rows_vec[:top_k]]
            notes = []
        else:
            hr = retrieve(
                corpus, q, cfg=cfg, bm25=bm25, index=index, top_k=top_k
            )
            ids = [h.paper_id for h in hr.hits]
            notes = hr.notes
        elapsed = round(time.perf_counter() - t0, 3)
        rel = set(item.get("relevant_ids") or [])
        m = _metrics(ids, rel, 10) if rel else {
            "recall": None, "precision": None, "ndcg": None, "mrr": None
        }
        m20 = _metrics(ids, rel, 20) if rel else m
        m50 = _metrics(ids, rel, 50) if rel else m
        rows.append({
            "id": item["id"], "query": q, "mode": mode,
            "returned": len(ids), "empty": not ids,
            "expect_refuse": bool(item.get("expect_refuse", False)),
            "top_ids": ids[:10],
            "metrics@10": m, "metrics@20": m20, "metrics@50": m50,
            "elapsed_s": elapsed, "notes": notes,
        })
    return {
        "mode": mode,
        "n": len(rows),
        "recall@10": _mean([r["metrics@10"]["recall"] for r in rows]),
        "recall@20": _mean([r["metrics@20"]["recall"] for r in rows]),
        "recall@50": _mean([r["metrics@50"]["recall"] for r in rows]),
        "precision@10": _mean([r["metrics@10"]["precision"] for r in rows]),
        "ndcg@10": _mean([r["metrics@10"]["ndcg"] for r in rows]),
        "mrr": _mean([r["metrics@10"]["mrr"] for r in rows]),
        "elapsed_s": _mean([r["elapsed_s"] for r in rows]),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "retrieval_questions.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("hybrid", "bm25", "vector"), default="hybrid")
    ap.add_argument("--grid", action="store_true", help="扫描 hybrid 参数")
    ap.add_argument("--top-k", type=int, default=50)
    a = ap.parse_args()

    cfg = load_config()
    items = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    out = Path(a.out) if a.out else paths_from(cfg)["workspace"] / "ledger" / "retrieval_eval.json"

    if a.grid:
        grid = []
        for top in (50, 100, 200):
            for rrf_k in (20, 60):
                for weight in (0.5, 1.0, 1.5):
                    c = copy.deepcopy(cfg)
                    r = c.setdefault("retrieval", {})
                    r.update({"hybrid_top_k": top, "rrf_k": rrf_k,
                              "vector_weight": weight})
                    res = evaluate(items, c, mode="hybrid", top_k=a.top_k)
                    grid.append({
                        "hybrid_top_k": top, "rrf_k": rrf_k, "vector_weight": weight,
                        "ndcg@10": res["ndcg@10"], "recall@20": res["recall@20"],
                        "elapsed_s": res["elapsed_s"],
                    })
                    print(top, rrf_k, weight, "ndcg@10=", res["ndcg@10"],
                          "recall@20=", res["recall@20"], "sec=", res["elapsed_s"])
        grid.sort(key=lambda x: (
            -(x["ndcg@10"] if x["ndcg@10"] is not None else -1),
            x["elapsed_s"] if x["elapsed_s"] is not None else 1e9,
        ))
        payload = {"grid": grid}
    else:
        payload = evaluate(items, cfg, mode=a.mode, top_k=a.top_k)
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"},
                         ensure_ascii=False, indent=2))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("落盘:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
