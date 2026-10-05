"""依赖图续跑抽取 CLI：`python scripts/run_deps.py [--limit N] [--start I] [roots...]`

每篇 `.src` 抽取被引 arxiv id，逐篇落盘 `--out/refs/*.json`；已存在跳过（可续跑）。
默认根：G 盘 24 的 cache。全库跑：`python scripts/run_deps.py G:\\rl_data24\\cache G:\\rl_data\\cache`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402
from hotspot.deps_extract import extract_resumable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/depgraph")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("roots", nargs="*", default=["G:\\rl_data24\\cache"])
    a = ap.parse_args()
    rep = extract_resumable(a.roots, out_dir=a.out, limit=a.limit, start=a.start)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
