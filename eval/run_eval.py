"""评测框架（方案 §5 + 2026-09-30-测试计划-大创接入评测.md）：跑 G0–G3 对比 + 4 项指标汇总。

用法：:

    # dry-run（默认，不真调）：只跑流程收指标骨架
    python eval/run_eval.py --questions eval/questions.json --group g0
    # 真调：走 `config.yaml` 的 `llm_eval`（kimi-k3）+ 该组开关
    python eval/run_eval.py --questions eval/questions.json --group g1 --live
    # 冷缓存评测：隔离 LLM 缓存，避免复用历史答案
    python eval/run_eval.py --group g0 --live \
        --llm-cache-dir workspace/ledger/llm_cache_g0

组（`--group`）：g0=无大创 / g1=只 rlRAG / g2=只 Streit-Garg / g3=全接入。
同 4 项指标（§5.3）：Cost per Query / Citation Support / Hallucination / 拒答正确率。
汇总输出该组均值 + 相对 g0 的 Δ（成对对比由外层多组结果合并比较）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=SyntaxWarning)
import hotspot  # noqa: E402 —— 装闸① + 接线

from hotspot.config import load_config, paths_from
from hotspot.api import AskRequest, Engine  # noqa: E402
from hotspot.api.contract import ApiError, E_CORPUS_UNAVAILABLE  # noqa: E402
from hotspot.retrieval.assemble import SYSTEM_PROMPT  # noqa: E402
from hotspot.subcluster import PARSE_SCHEMA  # noqa: E402

GROUPS = {
    "g0": dict(rlrag=False, sg=False),
    "g1": dict(rlrag=True, sg=False),
    "g2": dict(rlrag=False, sg=True),
    "g3": dict(rlrag=True, sg=True),
}

TRACE_VERSION = "1"
_TRACE_STAGES = ("parse", "retrieve", "name", "answer", "verify")

#: 「拒答」语义标记——模型回「本库无相关内容」这类，即便未走 refused 通道也算正确拒答
_REFUSAL_TEXT = ("本库无相关", "本库没有", "本库无法", "本库不包含",
                 "没有相关内容", "无相关内容", "无法回答", "没有找到相关",
                 "拒绝回答", "语料中无")


def _is_refusal(text: str) -> bool:
    return any(k in (text or "") for k in _REFUSAL_TEXT)


def _eval_cfg(group: str) -> tuple[dict, dict]:
    """评测 cfg：`llm_eval` 合入 `llm` + 设该组大创开关。返回 (cfg, 组标记)。"""
    cfg = load_config()
    llm_eval = cfg.get("llm_eval") or {}
    llm = dict(cfg.get("llm") or {})
    if llm_eval.get("base_url"):
        llm["base_url"] = llm_eval["base_url"]
    if llm_eval.get("model"):
        llm["model_flash"] = llm_eval["model"]
        llm["model_pro"] = llm_eval["model"]
    if llm_eval.get("api_key_env"):
        llm["api_key_env"] = llm_eval["api_key_env"]
    if llm_eval.get("timeout"):
        llm["timeout"] = llm_eval["timeout"]
    ec = copy.deepcopy(cfg)
    ec["llm"] = llm
    rg = ec.setdefault("rlrag", {})
    rg["enabled"] = GROUPS[group]["rlrag"]
    sg = ec.setdefault("streit_garg", {})
    sg["enabled"] = GROUPS[group]["sg"]
    return ec, GROUPS[group]


_CITE = re.compile(r"\[[PC]:[^\]]+\]")


def _metrics_for(resp) -> dict:
    """把 `AskResponse` 的契约字段翻译成评测指标（不依赖旧 Answer 对象）。"""
    claims = list(resp.verdicts)
    n = len(claims)
    citings = [v for v in claims if _CITE.search(v.text or "")]
    supported = sum(1 for v in claims if v.status == "supported")
    unsupported = sum(1 for v in claims if v.status == "unsupported")
    missing_id = sum(1 for v in claims if v.status == "missing_id")
    no_citation = sum(1 for v in claims if v.status == "no_citation")
    return {
        "n_claims": n,
        "n_filtered": resp.n_filtered,
        "supported": supported,
        "unsupported": unsupported,
        "missing_id": missing_id,
        "no_citation": no_citation,
        "citation_support": (supported / len(citings)) if citings else None,
        # 幻觉 = 编造(unsupported) + 越权引用(missing_id)，**不含 no_citation**
        # （拒答"本库无相关内容"这类无引用陈述不算幻觉，见测试计划 §2 口径）。
        "hallucination": ((unsupported + missing_id) / n) if n else None,
        "cost_yuan": float(getattr(resp.usage, "cost_yuan", 0.0) or 0.0),
    }


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v)) if v else None


def _sha256(value) -> str:
    raw = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _trace_sanitize(value, *, key: str = ""):
    """Trace 不落完整摘要/答案正文；文本只保留长度与指纹。"""
    if key in {"snippets", "abstract", "body", "answer_text"}:
        if isinstance(value, list):
            return [
                {"chars": len(str(x)), "sha256": _sha256(str(x))}
                for x in value
            ]
        text = str(value or "")
        return {"chars": len(text), "sha256": _sha256(text)}
    if isinstance(value, dict):
        return {
            str(k): _trace_sanitize(v, key=str(k))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_trace_sanitize(v, key=key) for v in value]
    return value


def _trace_prompt_versions() -> dict:
    return {
        "answer_system_sha256": _sha256(SYSTEM_PROMPT),
        "parse_schema_sha256": _sha256(PARSE_SCHEMA),
    }


def _trace_config(cfg: dict) -> dict:
    safe = copy.deepcopy(cfg)
    llm = safe.get("llm") or {}
    llm.pop("api_key", None)
    safe["llm"] = llm
    return {
        "sha256": _sha256(safe),
        "model": str(llm.get("model_pro") or llm.get("model_flash") or ""),
        "base_url": str(llm.get("base_url") or ""),
        "rlrag": bool((cfg.get("rlrag") or {}).get("enabled", False)),
        "streit_garg": bool((cfg.get("streit_garg") or {}).get("enabled", False)),
    }


def _trace_actions(meta: dict, metrics: dict, resp=None) -> list[dict]:
    stages = (meta.get("usage") or {}).get("stages") or {}
    plan = meta.get("plan") or {}
    retrieval = _trace_sanitize(meta.get("retrieval") or {})
    evidence = _trace_sanitize(meta.get("evidence") or {})
    observations = {
        "parse": {"plan": _trace_sanitize(plan)},
        "retrieve": {"retrieval": retrieval, "evidence": evidence},
        "name": {
            "topics": [
                {"cluster_id": t.cluster_id, "label": t.label, "size": t.size}
                for t in (getattr(resp, "topics", []) or [])
            ]
            if resp is not None else [],
        },
        "answer": {
            "route": getattr(resp, "route", None),
            "refused": getattr(resp, "refused", None),
            "answer_chars": len(getattr(resp, "text", "") or ""),
            "citation_ids": [c.id for c in (getattr(resp, "citations", []) or [])],
        } if resp is not None else {},
        "verify": {"metrics": metrics},
    }
    out = []
    cursor = 0.0
    for name in _TRACE_STAGES:
        start = stages.get(f"{name}_start")
        end = stages.get(name)
        if start is None and end is None:
            continue
        elapsed = None
        if end is not None:
            # Most stages emit once, so start == end. In that case stage cost is
            # the interval since the previous completed stage.
            if start is not None and float(end) > float(start) + 1e-9:
                elapsed = round(float(end) - float(start), 3)
            else:
                elapsed = round(float(end) - cursor, 3)
            cursor = float(end)
        out.append({
            "name": name,
            "params": {},
            "observation": observations.get(name, {}),
            "elapsed": elapsed,
        })
    return out


def _trace_record(*, item: dict, row: dict, resp, cfg: dict, group: str) -> dict:
    meta = dict(getattr(resp, "meta", {}) or {})
    usage = getattr(resp, "usage", None)
    if usage is not None:
        meta["usage"] = usage.to_dict()
    metrics = row.get("metrics") or {}
    return {
        "trace_version": TRACE_VERSION,
        "at": datetime.now(timezone.utc).isoformat(),
        "group": group,
        "request_id": getattr(resp, "request_id", None),
        "question": str(item.get("question") or ""),
        "question_id": item.get("id"),
        "question_type": item.get("type"),
        "source": getattr(resp, "source", None),
        "mode": getattr(resp, "mode", None),
        "plan": _trace_sanitize(meta.get("plan") or {}),
        "actions": _trace_actions(meta, metrics, resp),
        "prompt_versions": _trace_prompt_versions(),
        "model_config": _trace_config(cfg),
        "outcome": {
            "route": getattr(resp, "route", None),
            "refused": getattr(resp, "refused", None),
            "refuse_reason": getattr(resp, "refuse_reason", ""),
            "latency_s": getattr(usage, "elapsed_s", None) if usage else None,
            "cost_yuan": getattr(usage, "cost_yuan", None) if usage else None,
            "warnings": list(getattr(resp, "warnings", []) or []),
            "error": getattr(resp, "error", None),
            "metrics": metrics,
        },
    }


def _trace_error_record(*, item: dict, row: dict, cfg: dict, group: str) -> dict:
    return {
        "trace_version": TRACE_VERSION,
        "at": datetime.now(timezone.utc).isoformat(),
        "group": group,
        "request_id": None,
        "question": str(item.get("question") or ""),
        "question_id": item.get("id"),
        "question_type": item.get("type"),
        "source": "meeting",
        "mode": "new",
        "plan": {},
        "actions": [],
        "prompt_versions": _trace_prompt_versions(),
        "model_config": _trace_config(cfg),
        "outcome": {
            "route": None,
            "refused": False,
            "refuse_reason": "",
            "latency_s": None,
            "cost_yuan": None,
            "warnings": [],
            "error": row.get("error"),
            "metrics": {},
        },
    }


def _eval_line(row: dict) -> str:
    """One terminal line per question; full payload stays in the JSON file."""
    meta = row.get("meta") or {}
    retrieval = meta.get("retrieval") or {}
    evidence = (
        retrieval.get("evidence")
        or meta.get("evidence")
        or {}
    )
    coverage = retrieval.get("global_coverage") or {}
    if not coverage:
        sides = retrieval.get("sides") or []
        for side in sides:
            if isinstance(side, dict) and side.get("global_coverage"):
                coverage = side["global_coverage"]
                break
    policy = retrieval.get("coverage_policy") or "-"
    assigned = coverage.get("assigned_coverage")
    dominant = coverage.get("dominant_cluster_id")
    share = coverage.get("dominant_share")
    coverage_text = (
        f"{policy},assigned={assigned:.3f},dominant={dominant}@{share:.3f}"
        if isinstance(assigned, (int, float)) and isinstance(share, (int, float))
        else policy
    )
    parts = [
        str(row.get("id") or "-"),
        str(row.get("route") or "-"),
        f"refused={bool(row.get('refused'))}",
        f"evidence={evidence.get('decision') or '-'}",
        f"coverage={coverage_text}",
    ]
    if row.get("error"):
        parts.append(f"error={str(row['error'])[:100]}")
    return " | ".join(parts)


def _append_trace(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group", choices=sorted(GROUPS), default="g0")
    ap.add_argument("--live", action="store_true", default=False,
                    help="真调 LLM（默认 dry-run 只跑流程）")
    ap.add_argument("--resume", action="store_true",
                    help="读取 --out 中已有的 partial rows，跳过已完成题")
    ap.add_argument("--resume-from", default=None,
                    help="配合 --resume；从该题 id 开始重跑，丢弃它及之后的旧结果")
    ap.add_argument("--llm-cache-dir", default=None,
                    help="覆盖 dasc LLM 缓存目录，用于冷缓存/分组隔离评测")
    ap.add_argument("--llm-egress-log", default=None,
                    help="覆盖 LLM 出境审计日志路径，便于分组留存")
    ap.add_argument("--trace-out", default=None,
                    help="append-only JSONL Trace；默认 workspace/ledger/retrieval_trace.jsonl")
    ap.add_argument("--llm-timeout", type=float, default=None,
                    help="覆盖本次评测的 LLM 单请求超时秒数")
    a = ap.parse_args()

    if a.llm_cache_dir or a.llm_egress_log:
        import dasc.llm.client as llm_client  # noqa: PLC0415

        if a.llm_cache_dir:
            llm_client.CACHE_DIR = Path(a.llm_cache_dir)
        if a.llm_egress_log:
            llm_client.EGRESS_LOG = Path(a.llm_egress_log)

    ec, gflags = _eval_cfg(a.group)
    if a.llm_timeout is not None:
        ec.setdefault("llm", {})["timeout"] = max(1.0, float(a.llm_timeout))
    engine = Engine(cfg=ec)
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    ws = paths_from(ec)["workspace"]
    out = Path(a.out) if a.out else ws / "ledger" / f"eval_{a.group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_out = (
        Path(a.trace_out) if a.trace_out
        else ws / "ledger" / "retrieval_trace.jsonl"
    )

    rows: list[dict] = []
    if a.resume and out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            rows = list(old.get("rows") or [])
            if a.resume_from:
                ids = [str(r.get("id") or "") for r in rows]
                if a.resume_from in ids:
                    rows = rows[:ids.index(a.resume_from)]
            print(f"[resume] 已有 {len(rows)} 题，从 {a.resume_from or '未完成部分'} 继续")
        except Exception as e:  # noqa: BLE001
            print(f"[resume] 读取失败，重新开始：{type(e).__name__}: {e}")
            rows = []
    completed = {str(r.get("id") or "") for r in rows}

    def write_progress() -> None:
        """每题写一次，长评测被中断也不丢已完成结果。"""
        out.write_text(json.dumps(
            {"group": a.group, "partial": True, "n_done": len(rows), "rows": rows},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")

    for item in qs:
        if str(item.get("id") or "") in completed:
            continue
        resp = None
        try:
            resp = engine.ask(AskRequest.from_dict({
                "question": item["question"],
                "source": "meeting",
                "dry_run": not a.live,
            }))
            row = {"id": item["id"], "type": item["type"],
                   "route": resp.route, "refused": resp.refused,
                   "answer": resp.text,   # 记答案原文，可复盘/引用核对
                   "verdicts": [(v.status, (v.text or "")[:300],
                                 (v.detail or "")[:200]) for v in resp.verdicts],
                   "metrics": _metrics_for(resp),
                   "meta": {
                       **dict(resp.meta or {}),
                       "topics": len(resp.topics),
                       "graph": resp.graph is not None,
                       "usage": resp.usage.to_dict(),
                   }}
            # 空答案（端点失败被静默吞掉）→ 视为失败，不复用，避免污染指标
            if not resp.refused and not (resp.text or "").strip():
                row["error"] = "空答案（LLM 调用未返回内容，疑似端点失败）"
                row["metrics"] = {}
        except ApiError as e:
            if e.code == E_CORPUS_UNAVAILABLE:
                raise RuntimeError(f"评测中止：{e.message}") from e
            row = {
                "id": item["id"], "type": item["type"],
                "error": f"{type(e).__name__}: {e}",
                "route": None, "refused": False, "answer": "",
                "verdicts": [], "metrics": {},
            }
        except Exception as e:  # noqa: BLE001 —— 单题失败不中断评测，记 error 继续
            row = {"id": item["id"], "type": item["type"], "error": f"{type(e).__name__}: {e}",
                   "route": None, "refused": False, "answer": "", "verdicts": [],
                   "metrics": {}}
        rows.append(row)
        write_progress()
        trace = (
            _trace_record(
                item=item, row=row, resp=resp, cfg=ec, group=a.group,
            )
            if resp is not None
            else _trace_error_record(
                item=item, row=row, cfg=ec, group=a.group,
            )
        )
        _append_trace(trace_out, trace)
        print(_eval_line(row))

    refused_ok = sum(1 for r in rows if r["type"] == "refuse"
                     and (r["refused"] or _is_refusal(r.get("answer") or "")))
    n_refuse = sum(1 for r in rows if r["type"] == "refuse")
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["metrics"])
    summary = {"n": len(rows), "rlrag": gflags["rlrag"], "sg": gflags["sg"],
               "refuse_recall": (refused_ok / n_refuse) if n_refuse else None}
    # 大创专项指标：rlrag_ranked 是否在意图路径产出、key_path 是否可行（体现大创真正发挥的值）
    rl_meta = [r for r in rows if (r.get("meta") or {}).get("rlrag_ranked")]
    kp_meta = [r for r in rows if (r.get("meta") or {}).get("graph")]
    summary["rlrag_produced"] = len(rl_meta)
    summary["kl_key_produced"] = len(kp_meta)
    # rlrag_ranked 前 k 命中 query_hits 的比例（粗糙的「排序有用性」信号）
    hit_rates = []
    for r in rl_meta:
        m = r["meta"]
        qh = set(m.get("query_hits") or [])
        top = [it.get("paper") for it in m.get("rlrag_ranked", [])[:10]]
        if qh:
            hit_rates.append(len(qh & set(top)) / min(len(top), len(qh)))
    summary["rlrag_top10_hit_rate"] = _mean(hit_rates) if hit_rates else None
    kp_feas = kp_meta
    summary["key_path_feasible"] = len(kp_feas)
    for t, ms in by_type.items():
        # 兼容 error 行（metrics 为空 dict）：用 .get 且跳过空 metrics，避免 KeyError
        ms = [m for m in ms if m]
        summary[t] = {
            "citation_support": _mean([m.get("citation_support") for m in ms]),
            "hallucination": _mean([m.get("hallucination") for m in ms]),
            "n_filtered_avg": _mean([m.get("n_filtered") for m in ms]),
        }
    payload = {"group": a.group, **summary, "rows": rows}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"summary | group={a.group} | n={summary['n']} | "
        f"refuse_recall={summary['refuse_recall']} | out={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

