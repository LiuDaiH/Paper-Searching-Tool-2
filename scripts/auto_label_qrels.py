"""Generate silver retrieval qrels with the configured Mobile Cloud LLM.

    python scripts/auto_label_qrels.py --query-limit 1
    python scripts/auto_label_qrels.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402

PROMPT_VERSION = "qrels-v1"
INSTRUCTIONS = """你是论文检索相关性标注员。只根据给定问题和候选论文标题/摘要判断相关性。

等级：
2 = 论文直接讨论问题核心，能够提供回答所需的核心证据。
1 = 主题相关，但不能直接回答核心问题，或只提供背景/相邻方法。
0 = 不相关、主题不同，或违反给定的时间/会议等硬约束。

不能使用外部知识，不能因为方法常见就给高分。每条候选必须返回一次，item_id 原样返回。
"""


class QrelLabel(BaseModel):
    item_id: int
    grade: Literal[0, 1, 2]
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = ""
    reason: str = ""


class QrelBatch(BaseModel):
    labels: list[QrelLabel]


def _key(cfg: dict) -> str:
    llm = cfg.get("llm") or {}
    value = str(llm.get("api_key") or "").strip()
    env_name = str(llm.get("api_key_env") or "").strip()
    if not value and env_name:
        value = str(os.environ.get(env_name, "")).strip()
    if not value:
        raise RuntimeError("未配置 Mobile Cloud API key")
    return value


def _model(cfg: dict):
    llm = cfg.get("llm") or {}
    provider = OpenAIProvider(
        base_url=str(llm.get("base_url") or "").rstrip("/"),
        api_key=_key(cfg),
    )
    return OpenAIChatModel(
        str(llm.get("model_flash") or llm.get("model_pro") or "deepseek-chat"),
        provider=provider,
    )


def _cache_key(query_id: str, paper_id: str, model: str) -> str:
    return hashlib.sha256(
        f"{PROMPT_VERSION}|{model}|{query_id}|{paper_id}".encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["cache_key"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(ROOT / "eval" / "qrels_v2"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    queries = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (data_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    pool = [
        json.loads(line)
        for line in (data_dir / "pool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_ids = list(dict.fromkeys(row["query_id"] for row in pool))
    if args.query_limit:
        query_ids = query_ids[:args.query_limit]
    grouped = defaultdict(list)
    for row in pool:
        if row["query_id"] in query_ids:
            grouped[row["query_id"]].append(row)

    cfg = load_config()
    model_name = str((cfg.get("llm") or {}).get("model_flash") or "")
    agent = Agent(
        _model(cfg),
        output_type=QrelBatch,
        instructions=INSTRUCTIONS,
        retries=1,
    )
    cache_path = data_dir / "silver.cache.jsonl"
    cache = {} if args.force else _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for qid in query_ids:
        query = queries[qid]
        rows = grouped[qid]
        pending = [
            row for row in rows
            if _cache_key(qid, row["paper_id"], model_name) not in cache
        ]
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start:start + max(1, args.batch_size)]
            payload = {
                "question": query["query"],
                "search_query": query["search_query"],
                "hard_filters": query["filters"],
                "expected_refusal": query["expected_refusal"],
                "candidates": [
                    {
                        "item_id": i,
                        "title": row.get("title") or "",
                        "abstract": (row.get("abstract") or "")[:1400],
                        "year": row.get("year"),
                        "venue": row.get("venue"),
                    }
                    for i, row in enumerate(batch)
                ],
            }
            result = agent.run_sync(
                json.dumps(payload, ensure_ascii=False),
                timeout=args.timeout,
                usage_limits=UsageLimits(request_limit=2),
            )
            for label in result.output.labels:
                if not (0 <= label.item_id < len(batch)):
                    continue
                row = batch[label.item_id]
                record = {
                    "cache_key": _cache_key(qid, row["paper_id"], model_name),
                    "query_id": qid,
                    "paper_id": row["paper_id"],
                    "grade": int(label.grade),
                    "confidence": float(label.confidence),
                    "reason_code": label.reason_code,
                    "reason": label.reason,
                    "label_source": "llm",
                    "prompt_version": PROMPT_VERSION,
                    "model": model_name,
                }
                cache[record["cache_key"]] = record
                with cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{qid} | batch={start // max(1, args.batch_size) + 1} "
                f"labeled={len(batch)}"
            )

    final_rows = []
    for row in pool:
        if row["query_id"] not in query_ids:
            continue
        record = cache.get(_cache_key(row["query_id"], row["paper_id"], model_name))
        if record:
            final_rows.append(record)
        else:
            final_rows.append({
                "query_id": row["query_id"],
                "paper_id": row["paper_id"],
                "grade": None,
                "confidence": 0.0,
                "reason_code": "missing_llm_label",
                "reason": "",
                "label_source": "missing",
                "prompt_version": PROMPT_VERSION,
                "model": model_name,
            })
    out = data_dir / "silver.jsonl"
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in final_rows) + "\n",
        encoding="utf-8",
    )
    missing = sum(row["grade"] is None for row in final_rows)
    grades = Counter(
        int(row["grade"]) for row in final_rows if row["grade"] is not None
    )
    manifest = {
        "kind": "silver",
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
        "queries": len(query_ids),
        "candidates": len(final_rows),
        "missing": missing,
        "grade_counts": {str(k): grades[k] for k in sorted(grades)},
        "low_confidence": sum(
            float(row["confidence"]) < 0.75
            for row in final_rows
            if row["grade"] is not None
        ),
        "warning": "LLM-generated silver labels are for development, not gold reporting.",
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"silver | labels={len(final_rows)} missing={missing} out={out}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
