"""Summarize retrieval Trace without loading question or answer text.

    python scripts/summarize_trace.py workspace/ledger/retrieval_trace.jsonl
    python scripts/summarize_trace.py TRACE.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _retrieve(trace: dict) -> tuple[dict, dict]:
    for action in trace.get("actions") or []:
        if action.get("name") == "retrieve":
            observation = action.get("observation") or {}
            return (
                observation.get("retrieval") or {},
                observation.get("evidence") or {},
            )
    return {}, {}


def _coverage(retrieval: dict) -> dict:
    direct = retrieval.get("global_coverage") or {}
    if direct:
        return direct
    for side in retrieval.get("sides") or []:
        if isinstance(side, dict) and side.get("global_coverage"):
            return side["global_coverage"]
    return {}


def _record(trace: dict) -> dict:
    outcome = trace.get("outcome") or {}
    retrieval, observation_evidence = _retrieve(trace)
    evidence = retrieval.get("evidence") or observation_evidence or {}
    coverage = _coverage(retrieval)
    actions = trace.get("actions") or []
    return {
        "id": str(trace.get("question_id") or ""),
        "type": str(trace.get("question_type") or "unknown"),
        "route": str(outcome.get("route") or ""),
        "refused": bool(outcome.get("refused")),
        "evidence": str(evidence.get("decision") or ""),
        "domain_gate": str(evidence.get("domain_gate_reason") or ""),
        "policy": str(retrieval.get("coverage_policy") or ""),
        "assigned": coverage.get("assigned_coverage"),
        "dominant": coverage.get("dominant_cluster_id"),
        "share": coverage.get("dominant_share"),
        "error": bool(outcome.get("error")),
        "latency": {
            str(action.get("name")): action.get("elapsed")
            for action in actions
            if action.get("elapsed") is not None
        },
    }


def _avg(values) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 3) if vals else None


def _group(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "routes": dict(Counter(row["route"] or "-" for row in rows)),
        "refused": sum(row["refused"] for row in rows),
        "errors": sum(row["error"] for row in rows),
        "evidence": dict(Counter(row["evidence"] or "-" for row in rows)),
        "domain_gate": dict(Counter(row["domain_gate"] or "-" for row in rows)),
        "coverage_policy": dict(Counter(row["policy"] or "-" for row in rows)),
        "assigned_avg": _avg(row["assigned"] for row in rows),
        "dominant_share_avg": _avg(row["share"] for row in rows),
        "latency_avg": {
            stage: _avg(
                row["latency"].get(stage) for row in rows
            )
            for stage in ("parse", "retrieve", "name", "answer", "verify")
        },
    }


def summarize(path: Path) -> dict:
    rows = []
    invalid = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(_record(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid += 1
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    return {
        "overall": {**_group(rows), "invalid_lines": invalid},
        "by_type": {
            key: _group(group) for key, group in sorted(by_type.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Trace JSONL")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.path.exists():
        print(f"trace missing: {args.path}")
        return 1
    result = summarize(args.path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    overall = result["overall"]
    print(
        f"trace | n={overall['n']} invalid={overall['invalid_lines']} "
        f"routes={overall['routes']} refused={overall['refused']} "
        f"errors={overall['errors']}"
    )
    print(
        f"evidence | decisions={overall['evidence']} "
        f"domain={overall['domain_gate']}"
    )
    print(
        f"coverage | policies={overall['coverage_policy']} "
        f"assigned_avg={overall['assigned_avg']} "
        f"dominant_share_avg={overall['dominant_share_avg']}"
    )
    print(f"latency | avg_s={overall['latency_avg']}")
    for question_type, group in result["by_type"].items():
        print(
            f"type={question_type} | n={group['n']} routes={group['routes']} "
            f"refused={group['refused']} evidence={group['evidence']} "
            f"coverage={group['coverage_policy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
