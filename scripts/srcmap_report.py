"""汇总 `srcmap/*.json` → 命中率 + `digest → paper_id` 映射统计。

用法：`python scripts/srcmap_report.py [--out workspace/depgraph]`
输出：
- 命中率 / 未命中数 / 错误数 / kind 分布
- 命中映射条数（唯一 digest → paper_id）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gen_hits(sm_dir: Path):
    """迭代命中记录 `{src_path, paper_id}`。"""
    for rf in sm_dir.glob("*.json"):
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("paper_id"):
            yield d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/depgraph")
    a = ap.parse_args()
    sm = Path(a.out) / "srcmap"
    if not sm.is_dir():
        print(f"srcmap 目录不存在: {sm}")
        return 1

    total = matched = err = 0
    kinds: dict[str, int] = {}
    errtypes: dict[str, int] = {}
    unmatchedex = 0
    first = []
    seen_digest: set[str] = set()
    digest_to_paper: dict[str, str] = {}
    for rf in sm.glob("*.json"):
        total += 1
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            err += 1
            continue
        kinds[d.get("kind") or "?"] = kinds.get(d.get("kind") or "?", 0) + 1
        if d.get("error"):
            err += 1
            et = str(d["error"]).split(":")[0]
            errtypes[et] = errtypes.get(et, 0) + 1
        if d.get("paper_id"):
            matched += 1
            if len(first) < 5:
                first.append((Path(d["src"]).name[:12], d["paper_id"], d.get("title", "")[:40]))
            digest = Path(d["src"]).stem
            seen_digest.add(digest)
            digest_to_paper[digest] = d["paper_id"]

    print(f"srcmap 总记录: {total}")
    print(f"  命中 paper_id: {matched}  ({(matched / max(total, 1)) * 100:.1f}%)")
    print(f"  错误记录: {err}  {errtypes}")
    print(f"  kind 分布: {kinds}")
    print(f"  唯一 digest→paper 映射: {len(digest_to_paper)}")
    for name, pid, t in first:
        print(f"    {name} -> {pid}  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
