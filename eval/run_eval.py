"""评测框架（方案 §5 + 测试计划-大创接入评测.md）：跑 G0–G3 对比 + 4 项指标汇总。

用法：:

    # dry-run（默认，不真调）：只跑流程收指标骨架
    python eval/run_eval.py --questions eval/questions.json --group g0
    # 真调：走 `config.yaml` 的 `llm_eval`（kimi-k3）+ 该组开关
    python eval/run_eval.py --questions eval/questions.json --group g1 --live

组（`--group`）：g0=无大创 / g1=只 rlRAG / g2=只 Streit-Garg / g3=全接入。
同 4 项指标（§5.3）：Cost per Query / Citation Support / Hallucination / 拒答正确率。
汇总输出该组均值 + 相对 g0 的 Δ（成对对比由外层多组结果合并比较）。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hotspot  # noqa: E402 —— 装闸① + 接线

from hotspot.config import load_config, paths_from
from hotspot.api import AskRequest, Engine  # noqa: E402

GROUPS = {
    "g0": dict(rlrag=False, sg=False),
    "g1": dict(rlrag=True, sg=False),
    "g2": dict(rlrag=False, sg=True),
    "g3": dict(rlrag=True, sg=True),
}

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group", choices=sorted(GROUPS), default="g0")
    ap.add_argument("--live", action="store_true", default=False,
                    help="真调 LLM（默认 dry-run 只跑流程）")
    a = ap.parse_args()

    ec, gflags = _eval_cfg(a.group)
    engine = Engine(cfg=ec)
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    ws = paths_from(ec)["workspace"]
    out = Path(a.out) if a.out else ws / "ledger" / f"eval_{a.group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in qs:
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
        except Exception as e:  # noqa: BLE001 —— 单题失败不中断评测，记 error 继续
            row = {"id": item["id"], "type": item["type"], "error": f"{type(e).__name__}: {e}",
                   "route": None, "refused": False, "answer": "", "verdicts": [],
                   "metrics": {}}
        rows.append(row)
        if row.get("error"):
            print(f"[{a.group}/{item['id']}/{item['type']}] ERROR {row['error'][:120]}")
        else:
            print(f"[{a.group}/{item['id']}/{item['type']}] route={row['route']} "
                  f"refused={row['refused']} m={row['metrics']}")

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
    print(f"\n[{a.group}] 落盘：{out}")
    print(json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

