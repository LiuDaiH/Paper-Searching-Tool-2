"""评测全量 4 组（G0–G3）串行跑完 —— 一次进程内依序 g0→g1→g2→g3。

复用 `eval/run_eval.py`；每 subprocess 首次加载 embedding 模型（进程内缓存，4 组共 4 次加载）。
在**独立终端**运行最稳（不受会话生命周期影响）；也支持 `automation` 定时跑。

用法：`python scripts/run_eval_all.py [--questions eval/questions.json] [--live]`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

GROUPS = ["g0", "g1", "g2", "g3"]


def run_all(questions: str, live: bool) -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    for g in GROUPS:
        t0 = time.time()
        print(f"\n===== 开始 {g}（{time.strftime('%H:%M:%S')}）=====", flush=True)
        cmd = [sys.executable, "eval/run_eval.py",
               "--questions", questions, "--group", g]
        if live:
            cmd.append("--live")
        r = subprocess.run(cmd, cwd=root)
        if r.returncode != 0:
            failures += 1
        print(f"===== {g} 结束，用时 {(time.time() - t0) / 60:.1f} 分钟"
              f"（{time.strftime('%H:%M:%S')}）=====", flush=True)
    print(f"\n全部完成：{'有失败' if failures else 'OK'}（{len(GROUPS)} 组）")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="评测 G0–G3 四组串行跑完")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    return run_all(a.questions, a.live)


if __name__ == "__main__":
    raise SystemExit(main())
