"""D1 接入验证：四道闸在**本项目**这条路上真的生效。

跑法（在 `移动云杯/` 下）::

    python scripts/check_access.py

它证明四件事，**每件都要能报红**：

1. 闸①  `sys.dont_write_bytecode` 已装
2. 闸③  `rllib` 没被 import 进来
3. 闸②  开库前没有非空 `-wal`
4. 闸④  本次会话读过导师库、**一行没写**（靠 `LibraryView.end_audit()` 的差异列表）

⚠️ 为什么不用 `with Corpus() as c:`
------------------------------------

`Corpus.__exit__`（`dasc/corpora/conference.py:337-338`）是这么写的::

    def __exit__(self, *exc_info: object) -> None:
        self.lib.end_audit()          # ← 返回值被丢掉了

而 `LibraryView.__exit__`（`dasc/readonly.py:298-303`）是::

    diffs = self.end_audit()
    if diffs:
        raise LibraryTouched(...)

**同一个 diff，`LibraryView` 会 raise，`Corpus` 会咽下去。** 所以凡是拿
`Corpus` 当上下文管理器的代码，闸④ 在关闭那一刻是**静默失效**的——
而"静默失效的闸"比"没有闸"更坏，因为它长得和"刚核对过"一模一样。

本脚本因此用 `c.lib` 做审计上下文，不用 `Corpus` 自己的 `__enter__/__exit__`。
（这是 dasc 侧的缺陷，已向用户报告；未擅自改 dasc。）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 先把本项目根放上 sys.path，好让 `import hotspot` 找得到
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402  —— 装闸① + 接线 sys.path（必须在任何 dasc import 之前）
from dasc.corpora import Corpus  # noqa: E402
from dasc.readonly import LibraryView, StaleIndex  # noqa: E402

_fails: list[str] = []


def say(ok: bool, what: str, detail: str = "") -> bool:
    """打印一条结论。``ok=False`` 记进 `_fails`，最后决定退出码。"""
    mark = "OK  " if ok else "红  "
    line = f"[{mark}] {what}"
    if detail:
        line += f"  —— {detail}"
    print(line)
    if not ok:
        _fails.append(what)
    return ok


def main() -> int:
    print("=" * 72)
    print("D1 接入验证 · 移动云杯 热点挖掘")
    print("=" * 72)
    print()

    # ── 闸① ──────────────────────────────────────────────────────────
    say(sys.dont_write_bytecode is True,
        "闸① sys.dont_write_bytecode 已装",
        f"dont_write_bytecode={sys.dont_write_bytecode!r}")

    # ── 闸③ ──────────────────────────────────────────────────────────
    say("rllib" not in sys.modules,
        "闸③ rllib 未被 import",
        "sys.modules 里没有 rllib" if "rllib" not in sys.modules else "rllib 已被导入！")

    # ── 负向对照：先证明这条核对**不是白给的** ──────────────────────
    # 不拍起始账就 end_audit，必须返回非空（哨兵值），而不是空列表。
    # 否则"干净"就是默认答案，闸④ 报了绿也说明不了任何事。
    bare = LibraryView()
    sentinel = bare.end_audit()
    say(bool(sentinel),
        "负向对照：没拍起始账时 end_audit 会报「核对等于没做」",
        f"{sentinel[0] if sentinel else '（空！核对会假绿）'}")

    # ── 开库读 ───────────────────────────────────────────────────────
    c = Corpus()
    print()
    print(f"导师库：{c.lib.root}")
    print()

    try:
        with c.lib:                    # ← 用 LibraryView 的闸④，不用 Corpus 的
            # 闸②：开库前没有非空 -wal（index_conn / verify 都会查，这里显式再查一次）
            try:
                c.lib.assert_no_hot_wal()
                say(True, "闸② 开库前无未 checkpoint 的 -wal")
            except StaleIndex as e:
                say(False, "闸② 发现非空 -wal，已拒绝读", str(e)[:120])

            # 读全库（Corpus._load 读 papers.jsonl）
            st = c.stats()
            total = st.get("papers") or st.get("total") or st.get("n_papers")
            say(bool(total), "读到 papers.jsonl", f"总篇数 = {total}")

            # 取第一篇，核对 C2 契约字段真的带出来了
            papers = c.list_papers(limit=1)
            if papers:
                p = papers[0]
                say(bool(p.title and p.abstract and p.venue),
                    "Paper 带出 title/abstract/venue",
                    f"{p.paper_id}  venue={p.venue}  year={p.year}  "
                    f"abstract={len(p.abstract)} 字符")
                # ⚠️ 只印元信息，不印摘要正文（导师库铁律 4：聚合计数，不打印论文内容）
            else:
                say(False, "list_papers 返回空")

            # FTS5 检索这条路（L4 的 LOCAL 路径靠它）
            hits = c.search("reinforcement learning", k=5)
            say(len(hits) > 0, "FTS5 检索可用",
                f"'reinforcement learning' 命中 {len(hits)} 条")

        print()
        # `with c.lib` 正常退出 = end_audit 返回空 = 一行没写。
        # 有差异的话上面那行会直接 raise LibraryTouched，跑不到这里。
        say(True, "闸④ 本次会话读过导师库、一行没写", "end_audit() 差异列表为空")

    except Exception as e:                              # noqa: BLE001
        print()
        say(False, "闸④ 读取过程中出事", f"{type(e).__name__}: {e}")

    # ── 收尾 ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    if _fails:
        print(f"结论：{len(_fails)} 项红 —— " + "；".join(_fails))
        return 1
    print("结论：全绿。四道闸在本项目这条路上都生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
