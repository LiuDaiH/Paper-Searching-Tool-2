"""成本记账 + 报表（harness 卖点：省 30–120 倍的实测证据）。

每次问答记一行到 `workspace/ledger/cost.jsonl`；`report` 汇总累计问答数、
LLM 调用数、估算 token 与估算 ¥。**精确计价需 DeepSeek 官方 usage/定价**，
这里用估算（每调用 input/output token 常量和 config 单价）作为量级证据。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import load_config, paths_from

__all__ = ["record_qa", "report"]


def record_qa(cfg, *, question: str, n_calls: int,
              est_in: int = 0, est_out: int = 0, note: str = "") -> None:
    """记录一次问答的估算成本。追加 JSONL，幂等可累积。"""
    ws = paths_from(cfg)["workspace"]
    d = ws / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": (question or "")[:80],
        "n_calls": n_calls,
        "est_tokens_in": est_in, "est_tokens_out": est_out, "note": note,
    }
    with open(d / "cost.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(cfg=None) -> dict:
    """汇总 `workspace/ledger/cost.jsonl`。"""
    cfg = cfg or load_config()
    p = paths_from(cfg)["workspace"] / "ledger" / "cost.jsonl"
    if not p.exists():
        return {"n_questions": 0}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_q = len(rows)
    calls = sum(r.get("n_calls", 0) or 0 for r in rows)
    tin = sum(r.get("est_tokens_in", 0) or 0 for r in rows)
    tout = sum(r.get("est_tokens_out", 0) or 0 for r in rows)
    llm = cfg.get("llm") or {}
    pin = float(llm.get("cost_per_1m_in") or 0)
    pout = float(llm.get("cost_per_1m_out") or 0)
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    return {"n_questions": n_q, "n_llm_calls": calls, "est_tokens_in": tin,
            "est_tokens_out": tout, "est_cost_yuan": round(cost, 3)}
