#!/usr/bin/env python3
"""索引树自检：树还在不在（`移动云杯` 版）。离线、秒级、确定性——不碰网络。

查三件事：
  1. 悬空双链      [[X]] 指向不存在的文件
  2. 孤儿 md       从根 README 出发走不到（"从根就能读全"的承诺，
                   这是它唯一能被证伪的地方）
  3. README 覆盖   每个含 md 的一级文件夹是否都有自己的 README（或直辖豁免）

本项目的两张「不在树上」的表：
  · 数据文件  workspace/           —— 向量/聚类/metrics/cards/ledger，产物、可重建，不读
  · skill 源  .zhanlu/ 、科研设施/research-log/ —— 是 skill 与模板，不是树的文档
  · 代码      hotspot/scripts/eval —— 是代码 *.py，不在树的 md 扫描范围里

用法（在 `移动云杯/` 下）：
    python scripts/check_tree.py --vault .
退出码：0 = 全绿，1 = 有问题。

⚠️ 先跑，再按报告修配置，别按想象修配置。
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from collections import deque
from pathlib import Path

CONFIG = {
    # 树的根：可达性从这个文件算起
    "root_md": "README.md",

    # 子 README 的合法文件名（见手册 §2 命名一节）
    "readme_names": {"README.md", "_index.md"},

    # **数据/配置**不是**文档**：不参与"从根可达"的检查，也**不被读取**。
    # 规则按**连续路径段**匹配：`workspace` 命中任意层级下的 `workspace/`，
    # `research-log` 命中 `科研设施/research-log/`（skill 源）。
    "data_rules": [
        ".git", ".zhanlu", "__pycache__", ".venv", "node_modules",
        "workspace",       # 数据产物：向量/聚类/metrics/cards/ledger，可重建，不读
        "research-log",    # skill 源（照搬自 /日志 skill），是模板不是树的文档
    ],

    # 有意不进树的 md：相对路径 → 为什么它合法（每条都要写原因）。
    # 已消失的条目会在报告里点出来。
    "exempt": {},

    # **按设计就没有子 README** 的一级文件夹（"直辖"文件夹，见手册 §2）。
    # 不登记它，检查会把设计报成缺陷。
    "no_readme_ok": {
        "日志": "根直辖，索引在根 README §2（folder-readme 手册 §2）",
    },
}

#: 必须先去代码块与行内反引号，否则文档里**作为例子**写的 `[[README]]`
#: 会被当成真链接。
#: `{3,}` 支持 **3 个及以上**反引号的围栏——skill 全文内嵌时通常用 4 个
#: 反引号（````…````）以免与内部 ``` 冲突，这类围栏也要整体剥掉。
FENCE = re.compile(r"^[ \t]*`{3,}.*?^[ \t]*`{3,}[ \t]*$", re.S | re.M)
INLINE = re.compile(r"`[^`\n]*`")
#: 兼容 markdown 表格里的转义竖线：`[[A/README\|打开]]`
WIKILINK = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]*)?\]\]")

#: 子 README 的合法文件名（登记脚本时也认这个）
SCRIPT_README_NAMES = ("_index.md", "README.md")

#: 「脚本速查」节的标题。check_tree 靠它找登记了哪些 `.py`。
SCRIPT_SECTION = "脚本速查"


def _readme_scripts(raw: str) -> set[str]:
    """提取 README 「脚本速查」节里登记的文件名（反引号内的 `xxx.py`）。

    只认「脚本速查」这个二级标题之后、下一个二级标题之前的部分，避免把
    别处提到的 `.py` 也当成登记。
    """
    m = re.search(rf"^##\s*{SCRIPT_SECTION}.*?(?=^## |\Z)", raw, re.S | re.M)
    if not m:
        return set()
    found = set(re.findall(r"`([A-Za-z0-9_.\-]+\.py)`", m.group(0)))
    # `__init__.py` 是包脚手架，扫描侧不要求登记；README 里提到它只作说明，不视为登记
    return {f for f in found if f != "__init__.py"}


def _script_issues(vault: Path, rules) -> list[str]:
    """脚本速查一致性：每个含直接 `.py` 的目录，其 README 是否登记齐全且无冗余。

    返回问题列表（空 = 一致）。
    """
    by_dir: dict[str, list[str]] = {}
    for py in vault.rglob("*.py"):
        rel = py.relative_to(vault).as_posix()
        if is_data(rel, rules) or rel.endswith("__init__.py"):
            # data（产物/第三方）与 `__init__.py`（成包的脚手架，无独立功能）不要求登记
            continue
        by_dir.setdefault(posixpath.dirname(rel), []).append(posixpath.basename(rel))

    issues: list[str] = []
    for d, files in sorted(by_dir.items()):
        files = sorted(files)
        readme = None
        for name in SCRIPT_README_NAMES:
            cand = (vault / d / name) if d else (vault / name)
            if cand.exists():
                readme = cand
                break
        if readme is None:
            issues.append(f"{(d or '.').rstrip('/')}/ 有 {len(files)} 个 .py：{files}"
                          f"，但没有 README/_index 来登记脚本速查")
            continue
        listed = _readme_scripts(readme.read_text(encoding="utf-8", errors="replace"))
        missing = [f for f in files if f not in listed]
        stale = sorted(listed - set(files))
        if missing:
            issues.append(f"{(d or '.').rstrip('/')}/ 未在 README 脚本速查登记：{missing}")
        if stale:
            issues.append(f"{(d or '.').rstrip('/')}/ README 登记了不存在的脚本：{stale}")
    return issues


def strip_code(text: str) -> str:
    return INLINE.sub("", FENCE.sub("", text))


def is_data(rel: str, rules) -> bool:
    parts = rel.split("/")
    for r in rules:
        rp = r.split("/")
        n = len(rp)
        if any(parts[i:i + n] == rp for i in range(len(parts) - n + 1)):
            return True
    return False


def resolve(target: str, vault: Path, md_by_path: dict, by_stem: dict, cur_dir: str):
    t = target.strip().rstrip("\\").strip()
    t = t.split("#")[0].strip()
    if not t:
        return None, "空链接"

    variants = []
    for base in (t, posixpath.normpath(posixpath.join(cur_dir, t))):
        variants += [base, base + ".md"]
    for v in variants:
        if v in md_by_path:
            return v, None
        if (vault / v).is_file():
            return v, None

    stem = posixpath.basename(t)
    if stem.endswith(".md"):
        stem = stem[:-3]
    hits = by_stem.get(stem, [])
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, f"歧义（{len(hits)} 个同名：{', '.join(sorted(hits)[:3])}…）"
    return None, "悬空"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=CONFIG["root_md"], help="树根（vault 相对）")
    ap.add_argument("--vault", default=".", help="库根目录")
    ap.add_argument("--data", action="append", default=[],
                    help="额外的数据目录（可重复），如 --data workspace")
    args = ap.parse_args()

    rules = CONFIG["data_rules"] + args.data
    vault = Path(args.vault).resolve()
    md_by_path = {p.relative_to(vault).as_posix(): p for p in vault.rglob("*.md")}

    by_stem: dict[str, list[str]] = {}
    for rel in md_by_path:
        stem = posixpath.basename(rel)[:-3] if rel.endswith(".md") else rel
        by_stem.setdefault(stem, []).append(rel)

    graph: dict[str, set[str]] = {rel: set() for rel in md_by_path}
    dangling, ambiguous = [], []
    readme_raw: list[tuple[str, str]] = []

    for rel, path in md_by_path.items():
        if is_data(rel, rules):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if posixpath.basename(rel) in CONFIG["readme_names"]:
            readme_raw.append((posixpath.dirname(rel), raw))
        for m in WIKILINK.finditer(strip_code(raw)):
            tgt, why = resolve(m.group(1), vault, md_by_path, by_stem,
                               posixpath.dirname(rel))
            if tgt:
                graph[rel].add(tgt)
            else:
                (dangling if why == "悬空" else ambiguous).append(
                    f"{rel} → [[{m.group(1)}]]（{why}）"
                )

    if args.root not in md_by_path:
        print(f"!! 根 {args.root} 不在库里，无法算可达性", file=sys.stderr)
        return 1

    reachable, dq = {args.root}, deque([args.root])
    while dq:
        for nxt in graph.get(dq.popleft(), ()):
            if nxt not in reachable:
                reachable.add(nxt)
                dq.append(nxt)

    doc_nodes = {
        r for r in md_by_path
        if not is_data(r, rules) and r not in CONFIG["exempt"]
    }
    for r in doc_nodes:
        if r in reachable:
            continue
        base = posixpath.basename(r)
        for d, raw in readme_raw:
            if (posixpath.relpath(r, d) if d else r) in raw or base in raw:
                reachable.add(r)
                break

    orphans = sorted(doc_nodes - reachable)

    missing_readme, tops = [], {}
    for r in doc_nodes:
        if "/" in r:
            tops.setdefault(r.split("/")[0], []).append(r)
    for top, files in sorted(tops.items()):
        if top in CONFIG["no_readme_ok"]:
            continue
        names = {posixpath.basename(f) for f in files}
        if not (names & CONFIG["readme_names"]):
            missing_readme.append(f"{top}/（含 {len(files)} 份 md，但没有 README）")

    dead_exempt = [k for k in CONFIG["exempt"] if k not in md_by_path]
    dead_no_readme = [k for k in CONFIG["no_readme_ok"]
                      if not (vault / k).is_dir()]

    n_data = len(md_by_path) - len(doc_nodes)
    print(f"md 总数：{len(md_by_path)}（文档 {len(doc_nodes)} / 数据与豁免 {n_data}）")
    print(f"从 {args.root} 可达：{len(reachable & doc_nodes)} / {len(doc_nodes)}")

    ok = True
    for title, items in (
        ("悬空双链", dangling),
        ("歧义双链（同名撞车）", ambiguous),
        ("孤儿 md —— 从根走不到", orphans),
        ("缺 README 的文件夹", missing_readme),
    ):
        if items:
            ok = False
            print(f"\n[{title}] {len(items)} 处")
            for x in items[:30]:
                print("   -", x)
            if len(items) > 30:
                print(f"   …另有 {len(items) - 30} 处")

    for label, dead in (("豁免表", dead_exempt), ("直辖登记表", dead_no_readme)):
        if dead:
            print(f"\n[{label}已失效] {len(dead)} 条指向不存在的路径，可以删了")
            for x in dead:
                print("   -", x)

    # ---- 脚本速查一致性 ----
    script_issues = _script_issues(vault, rules)
    if script_issues:
        ok = False
        print(f"\n[脚本速查不一致] {len(script_issues)} 处")
        for x in script_issues[:40]:
            print("   -", x)
        if len(script_issues) > 40:
            print(f"   …另有 {len(script_issues) - 40} 处")

    print("\n树是完整的" if ok else "\n树有缺口——上面每一处都是某个新会话会走进的死胡同")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
