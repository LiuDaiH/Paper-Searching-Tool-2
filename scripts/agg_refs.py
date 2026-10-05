"""把依赖图 `refs` 与 `srcmap` 聚合 → 论文级引用关系，供 relation 引用层用。

产物 `out_dir/paper_refs.json`:
{
  "paper_refs": {paper_id: [被引 arxiv_ids]},
  "ref_to_papers": {arxiv_id: [引用了它的 paper_ids]},
  "stats": {srcs, mapped_papers, ref_edges, unmatched_srcs}
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/depgraph")
    a = ap.parse_args()
    out = Path(a.out)
    refs_dir = out / "refs"
    sm_dir = out / "srcmap"

    digest_to_paper: dict[str, str] = {}
    src_to_paper: dict[str, str] = {}
    for rf in sm_dir.glob("*.json"):
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("paper_id") and d.get("src"):
            digest_to_paper[Path(d["src"]).stem] = d["paper_id"]
            src_to_paper[str(d["src"])] = d["paper_id"]

    paper_refs: dict[str, list[str]] = {}
    unmatched_srcs = 0
    srcs_total = 0
    for rf in refs_dir.glob("*.json"):
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("src"):
            continue
        srcs_total += 1
        pid = src_to_paper.get(str(d["src"]))
        if not pid:
            unmatched_srcs += 1
            continue
        arx = d.get("arxiv_ids") or []
        if arx:
            paper_refs.setdefault(pid, [])
            paper_refs[pid].extend(a for a in arx if a not in paper_refs[pid])

    ref_to_papers: dict[str, list[str]] = {}
    for pid, arx in paper_refs.items():
        for a in arx:
            ref_to_papers.setdefault(a, [])
            ref_to_papers[a].append(pid)

    stats = {
        "srcs": srcs_total,
        "mapped_papers": len(paper_refs),
        "ref_edges": sum(len(v) for v in paper_refs.values()),
        "refs_arxiv": len(ref_to_papers),
        "unmatched_srcs": unmatched_srcs,
    }
    payload = {"paper_refs": paper_refs, "ref_to_papers": ref_to_papers, "stats": stats}
    (out / "paper_refs.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
