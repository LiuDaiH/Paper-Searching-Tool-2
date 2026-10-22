"""Request-scoped agent trace without secrets or full answer bodies."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import paths_from


def trace_path(cfg: dict) -> Path:
    return paths_from(cfg)["workspace"] / "ledger" / "retrieval_trace.jsonl"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def append_trace(cfg: dict, record: dict[str, Any]) -> Path:
    path = trace_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "trace_version": "2",
        "at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
    return path


def sanitize_answer_text(text: str) -> dict[str, Any]:
    value = str(text or "")
    return {"chars": len(value), "sha256": _hash(value)}
