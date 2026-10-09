"""从 LaTeX 源码里抽引用。**两条路必须都走，且分别记数。**

抄自何处
--------

逻辑移植自 `科研阅读路径/probe/t3_citation_probe.py`（2026-09-15 写的探针，
2026-09-20 移植）。函数级对应关系：

| 这里 | 那边 |
|---|---|
| :func:`clean_latex` / :func:`norm_title` / :func:`split_bibitems` / :func:`guess_title` | `_clean_latex` / `_norm_title` / `_split_bibitems` / `_guess_title` |
| :func:`parse_bib` | `parse_bib` |
| :func:`bibtex_title` | **新写的**——原探针没有，见下 |
| :func:`extract` | `probe_one` 的抽取部分 |

⚠️ **有一处刻意不抄**：探针用 `importlib.util.spec_from_file_location` 直接加载
导师库里的 `rllib/extract/latex.py`（`t3_citation_probe.py:44-47`）。
**那会执行导师库里的代码、并把 `.pyc` 写进导师库**——正是泄漏点②，
也正是 `dasc.readonly` 闸①②③ 拦的那件事。探针跑在 2026-09-15，那时闸还没建；
`dasc/` 不能这么干，所以解包器是自己写的（:mod:`dasc.srcfile`）。
**抄的是算法，不是它的依赖路子。**

为什么「两条路都要走」不是一句口号
------------------------------------

综述 `2509.02547` 的源码包里**只有 `.bib`，没有 `.bbl`**。而
`_guess_title()` 是**为 `\\bibitem` 正文写的**启发式（靠 `\\newblock` 分段）；
把它喂给 BibTeX 条目原文（`@article{key, title={…}}`）猜出来的是垃圾，
归一化后匹配不上 ⇒ **标题匹配数 = 0**。补一个 BibTeX 感知的 `title={…}` 抽取器后，
从 0 变成 **96**。（`T4-REPORT.md:109-113`、`语料库验收要求.md:102` 的 A4）

机制在这里：导师库的 `rllib/extract/latex.py:150` 的 `_TEXT_EXTS = (".tex", ".ltx", ".bbl")`
**把 `.bib` 过滤掉了**。所以"只写一路"不是"少一点点"——
**是整整一类论文一篇都抽不到，而且不报错。**

本地 160 个真实 `.src` 的分布（2026-09-20 实测）：

| 有 `.bib` | 有 `.bbl` | 个数 |
|---|---|---|
| ✅ | ❌ | **106** |
| ✅ | ✅ | 18 |
| ❌ | ❌ | 17 |
| ❌ | ✅ | 15 |

**106/156 = 68% 只有 `.bib`。** 只走 `.bbl` 那一路，本地这批就静默零命中。

⚠️ 所以本模块的返回值**不允许**把"这条路不适用"和"这条路跑了但没抽到"
混成一个 0。每一条路都有 :attr:`Route.applicable`，跑没跑、抽到几条，分开报。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..srcfile import SrcArchive, SrcKind, open_src

__all__ = [
    "Entry",
    "Route",
    "CiteExtraction",
    "extract",
    "extract_path",
    "clean_latex",
    "norm_title",
    "split_bibitems",
    "guess_title",
    "bibtex_title",
    "parse_bib",
    "rule_fingerprint",
]

# ── 解析规则（抄自 t3_citation_probe.py:54-60） ────────────────────────────

#: LaTeX 注释：`%` 到行尾。**必须能穿过它**，见下。
_CMT = r"(?:\s*%[^\n]*\n)*"

#: `\cite` / `\citep` / `\citet` / `\citealp` / `\citeauthor` … 含可选参数与多键
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")

#: `\bibitem[短标签]{键}`。**三处 `\s*` / `_CMT` 都是实测逼出来的**，
#: 不是防御性编程——本地 160 个 `.src` 里各有一个包踩中：
#:
#: | 真实写法 | 不加会怎样 |
#: |---|---|
#: | `\bibitem [\protect…]{key}`（`[` 前有空格） | 整个包 38 条一条都抽不到 |
#: | `\bibitem[Abdelaal…]%`⏎`{key}`（`]` 与 `{` 之间是注释） | 整个包 33 条一条都抽不到 |
#: | `\providecommand\bibitemStop`（宏定义，不是引用） | `(?![a-zA-Z])` 挡住；没有它，子串判据会认成"有原料" |
#:
#: ⚠️ `[^\]]*` 假定可选标签里没有 `]`。真遇到再改——**但现在它是量过的**，
#: 不是猜的（160 个包里没有一个可选标签含 `]`）。
_BIBITEM_RE = re.compile(
    r"\\bibitem(?![a-zA-Z])" + _CMT + r"\s*(?:\[[^\]]*\]" + _CMT + r")?\s*\{([^}]*)\}"
)
#: 新式 arXiv 编号（2015-04 之后）。旧式（`hep-th/9901001`）另有一路，见 `_ARXIV_OLD_RE`。
_ARXIV_NEW_RE = re.compile(r"arXiv[:\s/]*(\d{4}\.\d{4,5})", re.I)
_ARXIV_OLD_RE = re.compile(r"arXiv[:\s/]*([a-z-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
#: BibTeX 条目的开头：`@article{key,`
_BIB_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]*)", re.I)

#: **不是参考文献**的 BibTeX 条目类型。`@string{icalp = "…"}` 长得和
#: `@article{key,` 一模一样，不排掉就会被当成一条引用——本地实测它
#: 制造了重复键，**让"并集"反而小于单条路线**（那个不等式是数学上恒成立的，
#: 它一破就说明解析进了脏东西）。
_NON_REF_TYPES = frozenset({"string", "preamble", "comment"})

#: 三条路线的名字。**列在这里是为了让 :func:`rule_fingerprint` 覆盖到
#: "有哪几条路"这件事**——加一条路而指纹不变，快照就会把两次不可比的结果
#: 当成可比的。
ROUTES = ("bbl", "inline", "bib")

#: 标题启发式里要剥掉的 LaTeX 片段
_BRACE_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\s*")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------- #
# 纯函数：文本处理（可单测，没有 IO）
# ---------------------------------------------------------------------- #

def clean_latex(s: str) -> str:
    """bibitem 文本 → 可读纯文本（够用来做标题匹配即可）。"""
    s = s.replace("\\newblock", " ")
    s = _BRACE_CMD_RE.sub(" ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("~", " ")
    s = re.sub(r"\\([%&$#_])", r"\1", s)
    return _WS_RE.sub(" ", s).strip()


def norm_title(t: str) -> str:
    """标题归一化：只留字母数字，用于模糊匹配。"""
    return re.sub(r"[^a-z0-9]", "", t.lower())


def split_bibitems(bbl: str) -> list[tuple[str, str]]:
    """`.bbl` → `[(键, 条目文本)]`。**同时兼容 `thebibliography` 直接写在 `.tex` 里的情况。**"""
    hits = list(_BIBITEM_RE.finditer(bbl))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(bbl)
        body = bbl[m.end():end]
        cut = body.find("\\end{thebibliography}")
        if cut != -1:
            body = body[:cut]
        out.append((m.group(1).strip(), body))
    return out


def guess_title(body: str) -> str:
    """从 **`\\bibitem` 正文**里猜标题。

    ⚠️ **不要拿它猜 BibTeX 条目**——它是为 `.bbl` 写的，见模块 docstring。
    BibTeX 要走 :func:`bibtex_title`。

    两档启发式，够用才继续，不够就返回空（**宁可标失败也不瞎猜**）：
    1. `\\newblock` 分段：arXiv/ACL 风格是 [作者][标题][出处]，取第 2 段。
    2. 没有 `\\newblock`：按句号切，取**含年份之前最长的那一段**——
       作者段通常短（"A. Author, B. Author"），标题段最长。
    """
    if "\\newblock" in body:
        segs = [clean_latex(s) for s in body.split("\\newblock")]
        segs = [s for s in segs if len(s) > 12]
        if len(segs) >= 2:
            return segs[1]
        if segs:
            return segs[0]

    cleaned = clean_latex(body)
    ym = list(_YEAR_RE.finditer(cleaned))
    head = cleaned[: ym[-1].start()] if ym else cleaned
    parts = [p.strip() for p in re.split(r"\.\s+", head) if len(p.strip()) > 12]
    if not parts:
        return ""
    return max(parts, key=len)


def _braced_field(body: str, name: str) -> str:
    """从 BibTeX 条目里取 `name = {...}` 的值，**按花括号配平**。

    不能写成 `re.search(r"title\\s*=\\s*\\{(.*?)\\}")`——BibTeX 标题里
    **允许嵌套花括号**保大写（`title={{The {N}ew York Times}}`），
    非贪婪匹配会在第一个 `}` 就停，得到 `{The {N` 这种半截。
    """
    m = re.search(rf"\b{name}\s*=\s*", body, re.I)
    if not m:
        return ""
    rest = body[m.end():].lstrip()
    if not rest:
        return ""
    if rest[0] == "{":
        depth, i = 0, 0
        while i < len(rest):
            if rest[i] == "{":
                depth += 1
            elif rest[i] == "}":
                depth -= 1
                if depth == 0:
                    return rest[1:i]
            i += 1
        return rest[1:]  # 没配平：返回剩下的，**不假装成功**
    if rest[0] == '"':
        end = rest.find('"', 1)
        return rest[1:end] if end != -1 else rest[1:]
    return re.split(r"[,\n]", rest, 1)[0].strip()


def bibtex_title(body: str) -> str:
    """BibTeX 条目的标题。**这是 T4 补上的那一条腿。**

    原探针只有 `_guess_title()`（为 `\\bibitem` 写的），喂 BibTeX 条目时
    猜出来的是垃圾 ⇒ 匹配数虚低为 0。这里直接读 `title={...}`。
    """
    raw = _braced_field(body, "title")
    if not raw:
        return ""
    return _WS_RE.sub(" ", clean_latex(raw)).strip()


def arxiv_ids(text: str) -> set[str]:
    """文本里的 arXiv ID。新式 + 旧式两种编号都收。"""
    return {m.group(1) for m in _ARXIV_NEW_RE.finditer(text)} | {
        m.group(1) for m in _ARXIV_OLD_RE.finditer(text)
    }


def parse_bib(content: str) -> list[tuple[str, str, str]]:
    """BibTeX 内容 → `[(键, 标题, 条目原文)]`。

    不做完整 BibTeX 解析（那是另一个坑），按 `@type{` 切段后逐段取。
    `@string` / `@preamble` / `@comment` 被排掉——**它们不是参考文献**
    （见 :data:`_NON_REF_TYPES`）。

    ⚠️ 已知限制：不认 `@comment{...@article{假条目...}}` 这种**嵌套在注释里**的
    条目——按 `@type{` 切段看不见花括号层级，会把注释里的东西切出来。
    本地 160 个包里没有这种写法，所以没花代价处理；**遇到再改，
    但那时它会在计数上现形**（段数虚高）。
    """
    hits = list(_BIB_ENTRY_RE.finditer(content))
    out: list[tuple[str, str, str]] = []
    for i, m in enumerate(hits):
        if m.group(1).lower() in _NON_REF_TYPES:
            continue
        end = hits[i + 1].start() if i + 1 < len(hits) else len(content)
        body = content[m.start():end]
        out.append((m.group(2).strip(), bibtex_title(body), body))
    return out


# ---------------------------------------------------------------------- #
# 结果类型
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class Entry:
    """一条参考文献。`title` 可能为空——**那是"没猜出来"，不是"没有标题"**。"""

    key: str
    title: str
    raw: str = field(repr=False, default="")

    @property
    def has_title(self) -> bool:
        return bool(self.title.strip())


@dataclass(frozen=True)
class Route:
    """一条抽取路线的结果。**`applicable` 与 `entries` 是两件事。**

    | applicable | entries | 含义 |
    |---|---|---|
    | False | 0 | **这条路根本没跑**——包里没有它要的原料 |
    | True | 0 | 跑了，**确实一条都没抽到** |
    | True | >0 | 抽到了 |

    把前两行混成一个 0，就是本库的签名病。本地那 106 个 `.bib`-only 的包，
    在"只走 `.bbl`"的实现下会全落在第 1 行，而报告里显示成第 2 行。
    """

    name: str                 # "bbl" / "inline" / "bib"
    applicable: bool          # 原料在不在
    entries: tuple[Entry, ...] = ()
    sources: tuple[str, ...] = ()   # 用了哪些成员
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.entries)

    @property
    def n_titled(self) -> int:
        return sum(1 for e in self.entries if e.has_title)

    @property
    def unique_keys(self) -> int:
        """**同一路线内部**去重后的条数。`n` 与它的差 = 包内重复键。

        为什么单列一个：:attr:`CiteExtraction.entries` 是跨路线按 key 去重的，
        所以 `len(entries) ≥ max(单路的 unique_keys)` 是**数学上恒成立的**。
        写成 `≥ max(r.n)` 就不恒成立——本地 `12811beca7f4` 那个包
        `.bib` 里 n=851、unique=850，并集 850 < 851，看着像并集算错了，
        其实是那条路线自己有重复键。

        **判据要挑恒成立的那个写**；恒不成立的那个留着当"脏东西报警器"。
        """
        return len({e.key for e in self.entries})


@dataclass(frozen=True)
class CiteExtraction:
    """一篇论文的引用抽取结果。**按路线分开报，另有并集。**"""

    src: Path
    kind: SrcKind
    routes: tuple[Route, ...]
    cite_keys: tuple[str, ...] = ()       # `\cite{}` 里出现的键
    resolved_keys: tuple[str, ...] = ()   # 能对上 bibitem 的键
    note: str = ""

    def route(self, name: str) -> Route | None:
        return next((r for r in self.routes if r.name == name), None)

    @property
    def ran(self) -> tuple[str, ...]:
        """**真的跑了**的路线名（`applicable` 为真）。"""
        return tuple(r.name for r in self.routes if r.applicable)

    @property
    def entries(self) -> tuple[Entry, ...]:
        """并集。**按 key 去重**——`.bib` 与 `.bbl` 会命中同一批文献。"""
        seen: dict[str, Entry] = {}
        for r in self.routes:
            for e in r.entries:
                # 有标题的那条优先留下：`.bbl` 猜出来的常常是空
                if e.key not in seen or (e.has_title and not seen[e.key].has_title):
                    seen[e.key] = e
        return tuple(seen.values())

    @property
    def arxiv_ids(self) -> set[str]:
        out: set[str] = set()
        for r in self.routes:
            for e in r.entries:
                out |= arxiv_ids(e.raw)
        return out

    def summary(self) -> str:
        parts = []
        for r in self.routes:
            if not r.applicable:
                parts.append(f"{r.name}: 不适用")
            else:
                parts.append(f"{r.name}: {r.n} 条（{r.n_titled} 条有标题）")
        return " | ".join(parts) or "（没有可走的路线）"


# ---------------------------------------------------------------------- #
# 主入口
# ---------------------------------------------------------------------- #

def extract(archive: SrcArchive) -> CiteExtraction:
    """从一个已打开的源码包里抽引用。**两条路都跑，分别记数。**"""
    routes: list[Route] = []
    note = ""

    if archive.kind is SrcKind.PDF:
        return CiteExtraction(
            src=archive.path,
            kind=archive.kind,
            routes=(),
            note=(
                "这是 PDF 形态的 `.src` —— **两条 LaTeX 路都不适用**，"
                "不是「这篇没有参考文献」。要抽得先有 PDF 正文解析，"
                "而 `data/text/` 里的正文已经不带参考文献结构了。"
            ),
        )
    if archive.kind is SrcKind.UNREADABLE:
        return CiteExtraction(
            src=archive.path, kind=archive.kind, routes=(),
            note="源码包解不开 ⇒ **什么都判断不了**，不是「没有引用」。",
        )

    if archive.kind is SrcKind.SINGLE:
        texs = {"<single>": archive.all_text}
        bibs: dict[str, str] = {}
        bbls: dict[str, str] = {}
    else:
        texs = {n: _safe_read(archive, n) for n in archive.tex_files}
        bibs = {n: _safe_read(archive, n) for n in archive.bib_files}
        bbls = {n: _safe_read(archive, n) for n in archive.bbl_files}

    # ① `.bbl` 路线
    bbl_entries: list[Entry] = []
    for name, content in bbls.items():
        for key, body in split_bibitems(content):
            bbl_entries.append(Entry(key=key, title=guess_title(body), raw=body))
    routes.append(Route(
        name="bbl", applicable=bool(bbls),
        entries=tuple(bbl_entries), sources=tuple(bbls),
        note="" if bbls else "包里没有 `.bbl`",
    ))

    # ② `.tex` 内联 `thebibliography` 路线（没有 `.bbl` 时的兜底）
    #
    # ⚠️ 判据是 **`_BIBITEM_RE` 真的匹配**，不是 `"\bibitem" in v`。
    #    子串判据会被宏定义骗：本地 `1abfa1446e1d` 那个包里有
    #    `\providecommand \bibitemStop [0]{}`，子串命中 235 次 ⇒ 报成
    #    "这条路的原料在"，于是 applicable=True / n=0——**看着像跑过了没抽到，
    #    其实压根没有参考文献**。这正是本库那个签名病的一个新变种。
    inline_entries: list[Entry] = []
    inline_src = [n for n, v in texs.items() if _BIBITEM_RE.search(v)]
    for name in inline_src:
        for key, body in split_bibitems(texs[name]):
            inline_entries.append(Entry(key=key, title=guess_title(body), raw=body))
    routes.append(Route(
        name="inline", applicable=bool(inline_src),
        entries=tuple(inline_entries), sources=tuple(inline_src),
        note="" if inline_src else "`.tex` 里没有内联 `\\bibitem`",
    ))

    # ③ `.bib` 路线 —— **导师库的 unpack 把这一路丢掉了**，见模块 docstring
    bib_entries: list[Entry] = []
    for name, content in bibs.items():
        for key, title, body in parse_bib(content):
            bib_entries.append(Entry(key=key, title=title, raw=body))
    routes.append(Route(
        name="bib", applicable=bool(bibs),
        entries=tuple(bib_entries), sources=tuple(bibs),
        note="" if bibs else "包里没有 `.bib`",
    ))

    # ④ `\cite` 键：看有多少引用能对上（对应 K2 判据）
    cite_keys: set[str] = set()
    for v in texs.values():
        for m in _CITE_RE.finditer(v):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cite_keys.add(k)
    bibitem_keys = {e.key for e in bbl_entries} | {e.key for e in inline_entries}
    resolved = cite_keys & bibitem_keys

    if not any(r.applicable for r in routes):
        note = (
            "⚠️ **三条路都不适用**：包里既没有 `.bbl`/内联 `\\bibitem`，也没有 `.bib`。"
            "这不等于「这篇没有参考文献」——只等于**这份源码包给不出**。"
            "本地 160 个里有 17 个是这种。"
        )
    return CiteExtraction(
        src=archive.path,
        kind=archive.kind,
        routes=tuple(routes),
        cite_keys=tuple(sorted(cite_keys)),
        resolved_keys=tuple(sorted(resolved)),
        note=note,
    )


def _safe_read(archive: SrcArchive, name: str) -> str:
    try:
        return archive.read(name).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 单个成员读不出不该让整篇失败
        return ""


def extract_path(path: Path | str) -> CiteExtraction:
    """便利函数：从路径直接抽。`open_src` 的 `SrcError` 会照常抛出。"""
    return extract(open_src(path))


def rule_fingerprint() -> str:
    """抽取规则的指纹。**从规则本身算出来，不是手写的版本号。**

    手写的版本号会忘着改——改了正则忘了 bump，于是两份规则不同的抽取结果
    顶着同一个版本号进快照，然后被当成"可以比较"。这里直接把所有正则的
    `pattern` 串起来取 sha1：**规则一变，指纹就变，没有"忘了改"这回事**。

    ⚠️ 它只覆盖正则常量，**不覆盖 :func:`extract` 里的流程**（比如"三条路都跑"
    这件事本身）。这是一种取舍：流程变了指纹不变，可能会漏报一次不兼容。
    但流程比正则难改得多，且改了流程的 diff 一定很大、一定有人看见；
    而正则改一个字符是很容易漏掉的。
    """
    rules = [
        _BIBITEM_RE.pattern,
        _CITE_RE.pattern,
        _ARXIV_NEW_RE.pattern,
        _ARXIV_OLD_RE.pattern,
        _BIB_ENTRY_RE.pattern,
        ",".join(sorted(_NON_REF_TYPES)),
        ",".join(sorted(ROUTES)),
    ]
    return hashlib.sha1("\x00".join(rules).encode("utf-8")).hexdigest()[:12]
