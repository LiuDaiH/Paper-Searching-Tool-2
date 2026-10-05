"""成本报表 CLI：`python scripts/cost_report.py` —— 汇总 workspace/ledger/cost.jsonl。

若想清空累计：删 `workspace/ledger/cost.jsonl` 即可（产物可重建）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402
from hotspot.config import load_config  # noqa: E402
from hotspot.cost import report  # noqa: E402


def main() -> int:
    r = report(load_config())
    if not r["n_questions"]:
        print("尚无成本记录（workspace/ledger/cost.jsonl 为空或不存在）。")
        return 0
    print(f"累计问答数      : {r['n_questions']}")
    print(f"累计 LLM 调用   : {r['n_llm_calls']}")
    print(f"估算 input token: {r['est_tokens_in']}")
    print(f"估算 output token: {r['est_tokens_out']}")
    print(f"估算成本(¥)     : {r['est_cost_yuan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
