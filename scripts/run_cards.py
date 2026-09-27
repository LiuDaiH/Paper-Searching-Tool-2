"""L3 簇卡片生成 CLI：`python scripts/run_cards.py [--limit N] [--dry-run]`

先 `--limit N` 少量探路（确认真调通、schema/validator 合规、落盘正常），再全量 71 簇。
断点续跑：已生成 `workspace/cards/card_*.json` 的簇会跳过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hotspot.config import load_config  # noqa: E402
from hotspot.corpus import open_corpus  # noqa: E402
from hotspot.compress.cluster_card import make_cards  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 生成簇卡片")
    ap.add_argument("--limit", type=int, default=None, help="只生成前 N 簇（探路）")
    ap.add_argument("--dry-run", action="store_true", help="不真调 LLM")
    a = ap.parse_args()

    cfg = load_config()
    corpus = open_corpus(cfg)
    rep = make_cards(corpus, cfg=cfg, dry_run=a.dry_run, max_clusters=a.limit)
    print("\n卡片生成报告:")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
