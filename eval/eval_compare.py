"""合并 `eval_g*.json` → G0–G3 成对对比表（测试计划-大创接入评测 §6）。

用法：先分别跑 `run_eval.py --group g{0..3} --live`，再：
    python eval/eval_compare.py [--dir workspace/ledger]

输出：各指标在 g0–g3 的值与相对 g0 的 Δ。跨会留出（Precision@10）是另一独立实验脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="workspace/ledger")
    a = ap.parse_args()
    d = Path(a.dir)
    datas = {}
    for g in ("g0", "g1", "g2", "g3"):
        p = d / f"eval_{g}.json"
        if p.exists():
            datas[g] = json.loads(p.read_text(encoding="utf-8"))
    if "g0" not in datas:
        print("缺 g0 —— 先跑 run_eval.py --group g0")
        return 1

    sums = {g: data.get("rows", data) for g, data in datas.items()}
    base = datas["g0"]
    print(f"问题数 g0={base.get('n')}；已跑组：{sorted(datas)}")
    print("\n指标对比（Δ = 该组 − g0）:\n")

    def cell(v, delta):
        if v is None:
            return "  -   "
        ds = "" if delta is None else f"({delta:+.3f})"
        return f"{v:.3f}{ds}"

    def aggregate(data, metric):
        rows = data.get("rows") or []
        vals = [r["metrics"][metric] for r in rows if r["metrics"].get(metric) is not None]
        return (sum(vals) / len(vals)) if vals else None

    for metric in ("citation_support", "hallucination"):
        b = aggregate(base, metric)
        print(f"{metric:16s} " + "  ".join(
            cell(aggregate(data, metric),
                 None if g == "g0" or aggregate(data, metric) is None or b is None
                 else aggregate(data, metric) - b)
            for g, data in [(g, datas[g]) for g in ("g0", "g1", "g2", "g3") if g in datas]))
    for t in ("refuse_recall",):
        b = base.get(t)
        print(f"{t:16s} " + "  ".join(
            cell(data.get(t), None if g == "g0" or data.get(t) is None or b is None
                 else data.get(t) - b)
            for g, data in [(g, datas[g]) for g in ("g0", "g1", "g2", "g3") if g in datas]))
    print(f"\n列序: {[g for g in ('g0','g1','g2','g3') if g in datas]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
