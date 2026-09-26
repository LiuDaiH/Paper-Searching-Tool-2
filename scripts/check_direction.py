"""D2 验证：方向选择器三路召回（探针 7）。

跑法（在 `移动云杯/` 下）::

    python scripts/check_direction.py

对齐方案 §3.1 验收与探针 7：
「给定 'RLHF 的对齐方法'，输出子语料规模 + 三路命中数」，且三路**都有非零命中
或明确报说哪一路没跑**——`subfields` 盲区必须看得见，不能合并成一个数。

⚠️ `with c.lib` 用闸④（`Corpus.__exit__` 会吞掉审计差异），见 README §三。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402 —— 装闸① + 接线
from hotspot.corpus import open_corpus  # noqa: E402
from hotspot import direction  # noqa: E402
from hotspot import config as hcfg  # noqa: E402

CASES = [
    "RLHF 的对齐方法",
    "model-based reinforcement learning",
    "offline reinforcement learning",
    "漫游器身外接设备方向",   # 应该是三路全空 → 拒答口径
]


def main() -> int:
    cfg = hcfg.load_config()
    print("=" * 72)
    print("D2 方向选择器 · 三路召回验证")
    print("=" * 72)

    c = open_corpus(cfg)
    try:
        with c.lib():                   # no-op；读 jsonl 无写面
            for case in CASES:
                p = direction.select(c, case, cfg=cfg)
                print()
                print(f"方向：「{case}」")
                print(f"  子语料规模      = {p.size}")
                print(f"  并集             = {len(p.paper_ids)} 篇")
                print(f"  FTS5 命中        = {p.hits_fts}")
                print(f"  subfield 命中    = {p.hits_subfield}")
                print(f"  vector 命中      = {p.hits_vector}")
                for n in p.notes:
                    print(f"  ⚠️  {n}")
    finally:
        pass

    print()
    print("=" * 72)
    print("结论：三路分开报 → 盲区看得见。向量路待 D3/D4 产物接入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
