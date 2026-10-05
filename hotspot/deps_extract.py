"""依赖图**续跑抽取驱动**（2026-09-28）。

dasc 只有单篇抽取器 `latexcite.extract_path`，没有全库驱动的落盘。本模块补上：
**逐 `.src` 抽取引用 → 每篇单独落盘 `refs/*.json` → 已存在跳过** ⇒ 断了重跑续跑、不白跑。

每篇记录：`{"src": 路径, "arxiv_ids": [被引的 arxiv id 集合], "error": 可选}`。
这些 `arxiv_ids` 就是依赖图的**边**（该篇引用了谁）；`nodes`=论文，aggregate 可幂等重算。
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["extract_resumable"]


def _arxiv_ids(extraction) -> list[str]:
    """取 CiteExtraction 的被引 arxiv id 集合（方法或属性皆兼容）。"""
    m = getattr(extraction, "arxiv_ids", None)
    if callable(m):
        return sorted(str(x) for x in m())
    vals = extraction.arxiv_ids if m is not None else []
    return sorted(str(x) for x in vals)


def extract_resumable(src_roots, *, out_dir, limit: int | None = None,
                      start: int = 0) -> dict:
    """逐 `.src` 抽取并落盘，`refs/{i:05d}.json` 已存在即跳过（续跑）。

    - `src_roots`: 一个或多个含 `.src` 的目录（会递归 `rglob`）。
    - `limit` / `start`: 只处理编号区间（`start`..`start+limit`），方便分批/续跑定位。
    - 返回 `{n_src, done, skipped, error, out}`。
    """
    from dasc.deps import latexcite  # noqa: PLC0415

    out_dir = Path(out_dir)
    refs = out_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)

    srcs: list[Path] = []
    for r in (src_roots if isinstance(src_roots, (list, tuple)) else [src_roots]):
        rp = Path(r)
        if rp.exists():
            srcs.extend(sorted(rp.rglob("*.src")))

    if start:
        srcs = srcs[start:]
    if limit is not None:
        srcs = srcs[:limit]

    done = skipped = err = 0
    for i, s in enumerate(srcs, start=start):
        out = refs / f"{i:05d}.json"
        if out.exists():
            skipped += 1
            continue
        rec: dict = {"src": str(s)}
        try:
            ex = latexcite.extract_path(s)
            rec["arxiv_ids"] = _arxiv_ids(ex)
        except Exception as e:  # noqa: BLE001 —— 单篇失败也落盘（错误标记），不整批崩
            rec["error"] = f"{type(e).__name__}: {e}"
            err += 1
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1

    return {"n_src": len(srcs), "done": done, "skipped": skipped,
            "error": err, "out": str(refs)}
