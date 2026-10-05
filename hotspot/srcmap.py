"""建 `paper ↔ .src` 映射（解包反推版，2026-09-29）。

目标：把依赖图 `refs/*.json` 里的每条 `.src`（内容是摘要哈希 digest 的无意义文件名）
关联回语料里的论文。方法：解包 `.src` → 提 `\\title{}` → 归一化匹配 corpus 标题。

产物：`out_dir/srcmap/{i:05d}.json`，每条 `{"src": 完整路径, "paper_id": …, "title": …,
"via": "title"|"unmatched"}`。**已存在即跳过**，断了重跑续跑、不白跑。
只覆盖 `refs/*.json` 里出现过的 src，不重扫无关文件。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HOTSPOT = Path(__file__).resolve().parent
_ROOT = _HOTSPOT.parent
_OB = Path(r"D:\大学编程学习\obsidian")
sys.path.insert(0, str(_HOTSPOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_OB))

from hotspot.corpus import open_corpus  # noqa: E402
from dasc.srcfile import open_src, SrcKind  # noqa: E402

__all__ = ["build_srcmap", "norm", "extract_titles", "clean_title"]

#: 剥 LaTeX 命令，但保留其花括号里的"真文本"（`\\textit{Foo}` → `Foo`）。
_CMD_KEEP = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\s*\{")
_CMD_DROP = re.compile(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?")
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def clean_title(t: str) -> str:
    """把提出来、可能含 LaTeX 表述的标题洗干净，够匹配即可。"""
    t = re.sub(r"(?<!\\)%.*$", "", t, flags=re.M)  # 行注释
    t = t.replace("\\\\", " ")  # `\\` 换行
    out, i, n = "", 0, len(t)
    while i < n:
        m = _CMD_KEEP.match(t, i)
        if m:
            out += t[m.end():m.end()]  # 命令名丢弃，从 `{` 开始递归
            i += m.end() - 1
            # 从 `{` 后继续，等到配平的 `}`
            depth = 0
            while i < n:
                if t[i] == "{":
                    depth += 1
                elif t[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            continue
        m = _CMD_DROP.match(t, i)
        if m:
            i = m.end()
            continue
        out += t[i]
        i += 1
    out = out.replace("~", " ")
    out = out.replace("{", " ").replace("}", " ")
    out = re.sub(r"\\([%&$#_])", r"\1", out)  # 转义的表义字符保留
    return _WS.sub(" ", out).strip()


def extract_titles(a) -> list[str]:
    """从一个已打开的 `.src` 里收集候选标题（可以是多个 `\\title` 出现）。"""
    if a.kind in (SrcKind.PDF, SrcKind.UNREADABLE):
        return []
    texts: list[str] = []
    if a.kind is SrcKind.SINGLE:
        texts.append(a.all_text)
    else:
        for n in a.tex_files:
            try:
                texts.append(a.read(n).decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001
                continue
    out: list[str] = []
    for text in texts:
        for m in re.finditer(r"\\title\s*(?:\[[^\]]*\])?\s*\{", text):
            i = m.end()
            depth = 1
            start = i
            while i < len(text) and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            out.append(text[start:i - 1])
    return out


def build_srcmap(src_paths, *, out_dir: Path, start: int = 0, limit: int | None = None,
                 retry_failed: bool = False) -> dict:
    """对提供的 src 路径逐一建映射并落盘（续跑）。

    `retry_failed=True` 时，重跑已落盘但**带 error 的记录**（如当时文件暂时读不到），
    普通未命中不重跑（标题匹配不上，重跑也救不回）。
    """
    corpus = open_corpus()
    title_map: dict[str, str] = {}
    for p in corpus.list_papers():
        if p.title:
            title_map.setdefault(norm(p.title), p.paper_id)

    srcs = sorted({str(s) for s in src_paths})
    if start:
        srcs = srcs[start:]
    if limit is not None:
        srcs = srcs[:limit]

    out_dir = Path(out_dir)
    sm = out_dir / "srcmap"
    sm.mkdir(parents=True, exist_ok=True)

    done = skipped = err = 0
    matched = unmatched = 0
    for i, sp in enumerate(srcs, start=start):
        out = sm / f"{i:05d}.json"
        if out.exists():
            if not retry_failed:
                skipped += 1
                continue
            try:
                old = json.loads(out.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                old = {}
            if "error" not in old or old.get("paper_id"):
                skipped += 1
                continue
        rec: dict = {"src": sp, "via": "unmatched"}
        try:
            a = open_src(sp)
            rec["kind"] = a.kind.value
            if a.kind not in (SrcKind.PDF, SrcKind.UNREADABLE):
                cands = [clean_title(t) for t in extract_titles(a)]
                cands = [c for c in cands if c]
                for c in cands:
                    pid = title_map.get(norm(c))
                    if pid:
                        rec["paper_id"] = pid
                        rec["title"] = c[:200]
                        rec["via"] = "title"
                        matched += 1
                        break
                else:
                    unmatched += 1
                    if cands:
                        rec["title"] = cands[0][:200]
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            err += 1
        out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        done += 1

    return {"n_src": len(srcs), "done": done, "skipped": skipped,
            "matched": matched, "unmatched": unmatched, "error": err,
            "out": str(sm)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/depgraph")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--retry", action="store_true", help="只重跑带 error 的记录")
    a = ap.parse_args()

    out = Path(a.out)
    refs = out / "refs"
    paths = []
    for rf in refs.glob("*.json"):
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("src"):
            paths.append(d["src"])
    rep = build_srcmap(paths, out_dir=out, start=a.start, limit=a.limit,
                       retry_failed=a.retry)
    for k, v in rep.items():
        print(f"  {k}: {v}")
