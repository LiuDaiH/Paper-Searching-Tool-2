"""评测框架（方案 §5）：对每题跑一次问答管线，收集 4 项指标到 `workspace/ledger/`。

用法：:

    python eval/run_eval.py --questions eval/questions.json --out workspace/ledger/eval.json

指标（§5.3）：
  · Cost per Query  —— 走 `workspace/ledger/` 统计
  · Citation Support —— verdict.supported / 总带引用 claim
  · Hallucination Rate —— unsupported + missing_id / 总 claim
  · 拒答正确率（refuse 类）：refused=True 的比例

跨会留出验证（Precision@10 + Spearman）是专门实验，另写脚本（§5.4）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hotspot  # noqa: E402 —— 装闸① + 接线

from hotspot.config import load_config, paths_from
from hotspot.ui.app import run_qa  # noqa: E402 —— 复用端到端编排


def _metrics_for(ans) -> dict:
    claims = list(ans.verdicts)
    n = len(claims)
    citings = [v for v in claims if v.claim.cited_ids]
    return {
        "n_claims": n,
        "n_filtered": ans.n_filtered,
        "supported": sum(1 for v in claims if v.status == "supported"),
        "unsupported": sum(1 for v in claims if v.status == "unsupported"),
        "missing_id": sum(1 for v in claims if v.status == "missing_id"),
        "no_citation": sum(1 for v in claims if v.status == "no_citation"),
        "citation_support": (sum(1 for v in citings if v.status == "supported")
                             / len(citings)) if citings else None,
        "hallucination": (ans.n_filtered / n) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    a = ap.parse_args()

    cfg = load_config()
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    ws = paths_from(cfg)["workspace"]
    out = Path(a.out) if a.out else ws / "ledger" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in qs:
        ans = run_qa(item["question"], cfg=cfg, dry_run=a.dry_run)
        row = {"id": item["id"], "type": item["type"],
               "route": ans.route.value, "refused": ans.refused,
               "metrics": _metrics_for(ans)}
        rows.append(row)
        print(f"[{item['id']}/{item['type']}] route={ans.route.value} "
              f"refused={ans.refused} metrics={row['metrics']}")

    refused_ok = sum(r["refused"] for r in rows if r["type"] == "refuse")
    n_refuse = sum(1 for r in rows if r["type"] == "refuse")
    out.write_text(json.dumps({"rows": rows,
                               "refuse_recall": (refused_ok / n_refuse)
                               if n_refuse else None,
                               "summary": {}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
