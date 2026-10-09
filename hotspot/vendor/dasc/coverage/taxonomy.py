"""覆盖单元：从综述 `2509.02547` 的源码包里，抽出**现成的**分类学。

为什么用别人的分类学，不自己设计
--------------------------------

`大创计划-科研阅读路径规划.md` §2.2 原来把覆盖单元定义成「(子主题, 贡献类型)」——
而那个定义**是从自己的聚类结果里长出来的**，于是"覆盖得够不够"就成了自说自话：
聚类分出来的格子，当然会被自己的聚类填满。风险表里那条
「覆盖度定义自说自话」说的就是这个。

T4 探针给出了出路：综述 `2509.02547` 的源码包里有 `sections/` 结构，
**本身就是一张分类学**。用它当外部锚，覆盖单元就不是我们发明的了。
（`T4-REPORT.md:79-91`）

⚠️ 两个踩过的坑，都写进代码了
------------------------------

**坑一：必须按 `\\label` 解析，不能按文件名。** 实测对不上的有四对：

| 文件名 | `\\label` | 显示标题 |
|---|---|---|
| `capability/long-horizon.tex` | `subsec:long` | **Others** |
| `position/envstate.tex` | `subsec:env` | Environment State |
| `position/transition.tex` | `subsec:trans` | Transition Dynamics |
| `task/vision.tex` | `subsec:vision-understand` | Vision Agent |

按文件名解析会得到 `long-horizon` 这么一个**看起来很合理、但综述里根本没有**的轴。
这是「注释不是判据」的同一族：文件名读起来最顺，顺到不会有人去验它。

**坑二：`\\input` 有嵌套，且 `.tex` 后缀时有时无。** `main.tex` 载入六个 hub 文件，
hub 文件再 `\\input` 自己的子文件——而且同一个文件里两种写法混着用
（`\\input{sections/capability/plan.tex}` 与 `\\input{sections/capability/memory}` 相邻）。
所以必须**递归**，且两种写法都要认。

单点依赖：2026-09-20 已解开——而**解开它的关键是换掉判据**
----------------------------------------------------------

到 2026-09-20 上午为止，本模块的结论是「概念分类学只有 `2509.02547` 一篇有」，
理由写的是「`2510.16724` 没有 `sections/`」。**那个判据是错的。**

错在把**偶然特征**当成了**必要条件**：`sections/` 只是作者碰巧用了这个目录名。
当天实测的 `2503.21460`（《LLM Agent: A Survey on Methodology, Applications
and Challenges》）**根本没有 `sections/`**，却用**扁平编号文件名**编码了同一件事：

```
5_method.tex
5.1_construction.tex  5.2_collaboration.tex  5.3_evolution.tex
```

⇒ 它解出来 4 根带真 label 的轴（`sec:method` / `sec:dataset` / `sec:realworld`
/ `sec:app`），`Agent Construction` 下还有 Profile Definition / Memory Mechanism
/ Planning Capability / Action Execution 四个子类——**与 `2509.02547` 的
`position` 轴结构同源**。

判据从「有没有 `sections/` 目录」换成了**层级分解**（`flat_numbered` 那一类因此
浮出来）。但**层级分解也只能判"这包有没有章节"，判不了"哪几根枝是轴"**。

⚠️⚠️ 2026-09-20 当天更晚，拿负向对照把这件事量了：从导师库 `cache/` 的
160 个**普通论文** `.src` 里抽**按名字排序的前 60 个**跑层级分解，
**32 个**被判成"可作分类学来源"——**假阳性 32/60 = 53%**。
"用词表认叙事标题"的启发式因此**整个废掉**。它此前看着能用，只是因为
`survey_cache/` 里全是**按标题含 survey 筛出来的综述**：样本选的，不是判据判的。
**这两个数现在是现量的**，复现在 `dasc/checks/check_axes.py`——不再只活在注释里。

⚠️ **53% 是那条退役判据的，不是接替它的 :attr:`Shape.chaptered` 的。**
`chaptered` 去掉了词表筛选 ⇒ 条件**恒**比退役判据松 ⇒ 同一个样本上它命中
**54/60 = 90%**，离正向组（人判有轴的综述，8/8）只差 10 个点。
**把 53% 挂到 `chaptered` 头上，就是让一个更松的谓词去继承一个更严的谓词的读数**
——这跟"注释不是判据"是同一族。`check_axes.py` 的 ② 号判据
（`chaptered 命中数 ≥ retired 命中数`，逻辑上恒真）就钉这件事，两个数都从那里来。

⇒ 现在的分工是：**机器报结构事实，人判轴并落盘**
（`axes_by_hand.json`，署名与日期在里面），代码只读人判。读不到就说
「未经人判」，**不许退回机器猜**。

⚠️ 一个仍未证的事：两篇的轴**词表几乎不重合**（见 :func:`cross_check_sources`）。
所以"第二来源"现在的用途是**互补**，不是"互相校验"——把它接进覆盖单元之前，
先看那份对照，别把"找到了"读成"验过了"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..srcfile import SrcKind, SrcArchive, open_src

__all__ = [
    "DEFAULT_VAULT",
    "DEFAULT_SURVEY",
    "SURVEY_CACHE",
    "SECOND_SURVEY",
    "CONFIRMED_SECOND",
    "AXES",
    "AXES_BY_HAND",
    "CrossCheck",
    "Node",
    "Shape",
    "SourceCrossCheck",
    "Taxonomy",
    "TaxonomyError",
    "cross_check_sources",
    "describe_structure",
    "human_axes",
    "human_axes_note",
    "judged_sources",
    "load_taxonomy",
    "clean_title",
    "second_source_hint",
    "survey_sources",
]

#: 库根（`obsidian\`）。`dasc/coverage/taxonomy.py` → parents[2]。
DEFAULT_VAULT: Path = Path(__file__).resolve().parents[2]

#: 分类学的**唯一**来源。在 `科研阅读路径/probe/survey_cache/`——**不是导师库**，
#: 所以读它不涉及只读网关（那道闸只管 `rl_acdemical_paper/`）。
DEFAULT_SURVEY: Path = (
    DEFAULT_VAULT / "科研阅读路径" / "probe" / "survey_cache" / "2509.02547.src"
)

#: 放综述 e-print 的地方。**在导师库外面**，加候选就是往这里丢文件。
#: 读它不涉及只读网关（那道闸只管 `rl_acdemical_paper/`）。
SURVEY_CACHE: Path = DEFAULT_VAULT / "科研阅读路径" / "probe" / "survey_cache"

#: 2026-09-20 之前被点名的"第二来源候选"。**它确实没有 `sections/`**——
#: 但当时据此下的结论（"所以没有第二来源"）是错的，见模块 docstring。
#: 留着是因为 `check_coverage.py` 拿它当**负向对照**（一份不是分类学的源码包
#: 必须被判成"不成"），那条对照现在仍然有效、仍然要跑。
SECOND_SURVEY: Path = SURVEY_CACHE / "2510.16724.src"

#: 2026-09-20 实测可用作第二来源的那一篇。**它是靠层级分解判出来的，
#: 不是靠目录名**——它没有 `sections/`。
CONFIRMED_SECOND: Path = SURVEY_CACHE / "2503.21460.src"

#: 四根轴：`\\label` → (轴名, 中文说明)。
#:
#: 前三根是计划里那条「(MDP 部件 × 能力 × 任务)」的三个维度；`resources` 是第四根，
#: 由 T4 一并列出（`T4-REPORT.md:81-89`）。`future` **不是轴**——它是挑战与展望，
#: 不参与笛卡尔积，但仍在树里，所以照样解析、照样报数。
AXES: dict[str, tuple[str, str]] = {
    "sec:position": ("position", "MDP 部件轴——RL 在 LLM 里的定位"),
    "sec:capability": ("capability", "智能体能力轴"),
    "sec:task": ("task", "任务域轴"),
    "sec:resources": ("resources", "资源轴——环境与框架"),
}
#: 在树里但不是轴的那些顶层枝。
NON_AXIS_BRANCHES: dict[str, str] = {
    "sec:future_directions": "挑战与展望（不是轴，不进笛卡尔积）",
    "sec:introduction": "引言（不是轴）",
    "sec:conclusion": "结论（不是轴）",
}

_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}

#: `\section*{...}` 也要认（星号版不编号，但照样是结构）。
_SEC_RE = re.compile(
    r"\\(section|subsection|subsubsection)(\*)?\s*\{(?P<title>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
)
#: ⚠️ **只认结构型 label**，且前缀有**两种写法**。
#:
#: 写成裸 ``\\label{...}`` 会抓到标题后面第一个 `\\label`——而那里常常是
#: `\\label{fig:...}`（图注的）。那样解析出来的轴名看着正常，其实指向一张图。
#:
#: `subsub:` 这一种是**交叉核对抓出来的**：2026-09-20 首跑时，`trees.tex` 引用了
#: `subsub:search_open` / `subsub:search_close`，而解析只认 `subsubsec:`，
#: 于是那两个子节以「没有 label」的样子混了过去。**光看解析结果看不出来**——
#: 它们显示的标题是对的，只有拿总图当外部判据才暴露。
_SECTION_LABEL_RE = re.compile(
    r"\\label\s*\{(?P<label>(?:sec|subsec|subsubsec|subsub|app):[^{}]+)\}"
)
_INPUT_RE = re.compile(r"\\input\s*\{(?P<path>[^{}]+)\}")
_CREF_RE = re.compile(r"\\cref\s*\{(?P<label>[^{}]+)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


class TaxonomyError(RuntimeError):
    """解析漏了，**不是**这包没有分类学。两件事的读数一模一样，必须分开报。"""


@dataclass(frozen=True)
class Node:
    """分类学里的一个节点。**`label` 是它的身份，`source`（文件名）不是。**"""

    level: int
    title: str
    label: str
    source: str

    @property
    def kind(self) -> str:
        return {1: "section", 2: "subsection", 3: "subsubsection"}.get(
            self.level, f"level{self.level}"
        )


@dataclass(frozen=True)
class CrossCheck:
    """拿总图核对解析结果的**差异**。两边都空 = 全部对上。"""

    #: `trees.tex` 引用了、解析却没抓到 —— **这条最可疑，通常意味着解析漏了**。
    only_tree: tuple[str, ...] = ()
    #: 解析到了、总图没引用 —— 可能是作者没画全，也可能是解析多抓了。
    only_parsed: tuple[str, ...] = ()
    #: 作者没给 label 的子节数。**不是差异，是背景**。
    unlabeled: int = 0

    @property
    def clean(self) -> bool:
        """两种差异都空才算干净。`unlabeled` 不参与——理由见 :meth:`Taxonomy.cross_check`。"""
        return not self.only_tree and not self.only_parsed

    def describe(self) -> list[str]:
        out: list[str] = []
        if self.only_tree:
            out.append(f"⚠️ 总图引用了、我们没解析到（{len(self.only_tree)}）：{list(self.only_tree)}")
        if self.only_parsed:
            out.append(f"· 我们解析到、总图没引用（{len(self.only_parsed)}）：{list(self.only_parsed)}")
        if self.unlabeled:
            out.append(f"· 作者没给 label 的子节：{self.unlabeled} 个（不计入差异）")
        return out or ["  —— 全部对上 ——"]


@dataclass(frozen=True)
class Taxonomy:
    """抽好的分类学。`nodes` 按**文档顺序**（读论文的顺序），不是字母序。"""

    source_path: Path
    nodes: tuple[Node, ...]
    #: `trees.tex`（那张总图）里 `\\cref` 引用到的 label 集合。
    #: 拿它当**外部判据**核对解析结果——见 :meth:`cross_check`。
    tree_labels: frozenset[str]
    #: 归档里全部成员名，供排查用。
    members: tuple[str, ...]

    # ---------------------------------------------------------------- 取

    def branches(self) -> tuple[Node, ...]:
        """顶层枝（`\\section`）。"""
        return tuple(n for n in self.nodes if n.level == 1)

    def by_label(self) -> dict[str, Node]:
        return {n.label: n for n in self.nodes}

    def children_of(self, label: str) -> tuple[Node, ...]:
        """某个节点的**直接**下级。靠文档顺序切段，不靠 label 的前缀——
        `subsec:mdp` 的前缀里并没有 `sec:position`，前缀法在这里是错的。"""
        out: list[Node] = []
        idx = {n.label: i for i, n in enumerate(self.nodes)}
        if label not in idx:
            return ()
        start = idx[label]
        base = self.nodes[start].level
        for n in self.nodes[start + 1:]:
            if n.level <= base:
                break
            if n.level == base + 1:
                out.append(n)
        return tuple(out)

    def axis(self, label: str) -> tuple[Node, ...]:
        """某根轴下的**全部**后代（含子子节），拍平。"""
        out: list[Node] = []
        idx = {n.label: i for i, n in enumerate(self.nodes)}
        if label not in idx:
            return ()
        start = idx[label]
        base = self.nodes[start].level
        for n in self.nodes[start + 1:]:
            if n.level <= base:
                break
            out.append(n)
        return tuple(out)

    def axis_counts(self, *, deep: bool = False) -> dict[str, int]:
        """四根轴各有多少个单元（轴自身不算，它只是枝头）。

        ⚠️ **两种口径差得很远，别混着报。** `deep=False`（默认）只数**直接下级**，
        与 :meth:`units` 同粒度——`task` 是 **8**。`deep=True` 连子子节一起数，
        `task` 变成 **22**。两个数都对，但它们回答的不是同一个问题；
        报数时不写清用的是哪个，读的人会以为其中一个错了。
        """
        pick = self.axis if deep else self.children_of
        return {AXES[lab][0]: len(pick(lab)) for lab in AXES}

    def units(self) -> tuple[tuple[str, str, str], ...]:
        """覆盖单元 = **前三根轴的笛卡尔积**。

        ⚠️ **绝大多数格子是空的，这是对的、不是缺陷。** 按 T4 的原话，
        覆盖单元是「(MDP 部件 × 能力 × 任务) 的格子」，「跨轴重叠天然存在」——
        一篇论文同时属于某个 MDP 部件与某个能力，正是区分度的来源。
        覆盖度问的是「这些格子里填了几个」，**不是**"填满才算合格"。

        `resources` 不参与：它是资源轴（环境/框架），与前三轴正交，
        拿来当第四个维度会把格子数再乘 3，而它回答的是另一个问题。

        ⚠️ **取各轴的「直接下级」，不把子子节也摊进来。** 理由：T4 说的那个三轴，
        粒度是**同一层**的（`position/` × `capability/` × `task/` 各自的子节点）。
        混着层级算，`task` 会从 8 个单元涨成 22 个（它下面还有一层 `subsubsec`），
        于是格子数从 392 虚涨到 1078——而多出来的那些格子**不是新的覆盖单元，
        是同一个单元的更细划分**。子子节照常解析、照常报数，只是不参与笛卡尔积。
        """
        picks = [
            tuple(d.label for d in self.children_of(lab))
            for lab in ("sec:position", "sec:capability", "sec:task")
        ]
        return tuple((a, b, c) for a in picks[0] for b in picks[1] for c in picks[2])

    # ---------------------------------------------------------------- 核

    def cross_check(self) -> "CrossCheck":
        """拿 `trees.tex` 当外部判据核对。

        为什么这算"外部"：`trees.tex` 是作者画的那张总图，它引用哪些 label
        是**作者的决定**，不是我解析逻辑的产物。两边对不上时**不自动判决谁对**，
        只把差异摆出来——首跑就是靠它抓出 `subsub:` 那个漏掉的前缀的。

        ⚠️ **`<无 label>` 的节点不算进差异**。它们和"解析到但总图没引用"是两件事：
        前者是**作者压根没给这个子节起 label**（综述里常见），后者才可能是解析问题。
        把两者混在一起报，首跑 16 条差异里有 13 条是噪声，真问题会被淹掉。
        """
        got = {lbl for lbl in self.by_label() if not lbl.startswith("<无 label")}
        unlabeled = sum(1 for n in self.nodes if n.label.startswith("<无 label"))
        return CrossCheck(
            only_tree=tuple(sorted(self.tree_labels - got)),
            only_parsed=tuple(sorted(got - self.tree_labels)),
            unlabeled=unlabeled,
        )


# ---------------------------------------------------------------------- #
# 解析
# ---------------------------------------------------------------------- #

#: 已知的"一层包裹"宏，取内层文字。**这是白名单，不是黑名单。**
#:
#: `xhyu` 是这一篇的作者宏（修订标记用的高亮）。它**只在这一篇里存在**——
#: 换一篇综述就没有。加白名单而不是"见到 `\\foo{...}` 就剥"，
#: 是因为后者会把 `\\cite{...}`、`\\ref{...}` 里的关键信息也一起吃掉。
#:
#: ⚠️ 遇到**不在名单里**的包裹宏，标题会带着 `\\foo` 原样出来。
#: **这是有意的**：带着宏的标题一眼能看见，被猜错的标题看不出来。
_KNOWN_WRAPPERS = (
    "textbf", "emph", "textit", "textsc", "texttt", "mathrm", "text",
    "xhyu",
)


def clean_title(raw: str) -> str:
    """把 LaTeX 标题洗成人话。**只做无歧义的替换**，不做解释。

    `\\textbf{...}` 取内层、`~` 变空格、`\\\\` 变空格、`\\%` 还原成 `%`。
    洗不干净时宁可留着 —— 一个留着 `\\text` 的标题能被看见，
    一个被猜错的标题看不出来。
    """
    s = raw
    s = re.sub(
        r"\\(?:" + "|".join(_KNOWN_WRAPPERS) + r")\s*\{([^{}]*)\}", r"\1", s
    )
    s = s.replace("~", " ")
    s = re.sub(r"\\\\", " ", s)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_comments(tex: str) -> str:
    """去 `%` 注释。**`\\%` 不算注释**（负向后顾挡掉），否则 "95\\% 的" 会被砍掉半行。"""
    return _COMMENT_RE.sub("", tex)


def _read_all(archive: SrcArchive) -> dict[str, str]:
    """把包里的文本成员一次性读进来。**只读文本成员**，图/样式不碰。"""
    out: dict[str, str] = {}
    if archive.kind is SrcKind.SINGLE:
        out["<single>"] = archive.single_text.decode("utf-8", "replace")
        return {k: _strip_comments(v) for k, v in out.items()}
    for n in archive.find(".tex", ".ltx", ".bib", ".bbl"):
        try:
            out[n] = _strip_comments(archive.read(n).decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - 单个成员读不出不该让整篇失败
            continue
    return out


def _resolve(name: str, files: dict[str, str]) -> str | None:
    """`\\input{sections/capability/memory}` 补 `.tex` 后再找。**两种写法都认。**"""
    for cand in (name, f"{name}.tex", f"{name}.ltx"):
        if cand in files:
            return cand
    return None


def load_taxonomy(src_path: Path | str = DEFAULT_SURVEY) -> Taxonomy:
    """解开综述源码包，抽出分类学。**全在内存里，不落盘。**

    :raises FileNotFoundError: 源码包不在。**不静默返回空分类学**——
        一个空分类学会让后面每一处判断都"通过"，那比报错糟得多。
    """
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(
            f"综述源码包不在：{p}\n"
            f"（它是分类学的**唯一**来源，见本模块 docstring。"
            f"文件不在就该停，不要退化成「没有轴」。）"
        )
    archive = open_src(p)
    files = _read_all(archive)

    # 总图里的 \cref —— 外部判据
    tree_labels: frozenset[str] = frozenset()
    for name in ("sections/intro/trees.tex", "sections/intro/trees"):
        if name in files:
            tree_labels = frozenset(m.group("label") for m in _CREF_RE.finditer(files[name]))
            break

    nodes: list[Node] = []
    seen_files: set[str] = set()

    def walk(name: str) -> None:
        if name in seen_files:  # 防环：正常不会有，但递归里不能靠"正常"
            return
        seen_files.add(name)
        tex = files[name]
        matches = list(_SEC_RE.finditer(tex))
        # `\section` 与 `\input` **按出现位置交错**处理，两种包形态共用这一条路径。
        # ⚠️ 2026-09-20：此前只把 `main.tex` 当 `\input` 清单用、**从不 walk 它自己**，
        # 于是"章节直接写在 main.tex 里、不走 `\input`"的包解析出 0 节点
        # （`2504.21277`：`_SEC_RE` 命中 49 次，节点 0）。顺序与正文一致也靠这个交错。
        events: list[tuple[int, str, int, re.Match[str]]] = [
            (m.start(), "sec", i, m) for i, m in enumerate(matches)
        ]
        events += [(m.start(), "input", -1, m) for m in _INPUT_RE.finditer(tex)]
        events.sort(key=lambda e: e[0])
        for _, kind, i, m in events:
            if kind == "input":
                target = _resolve(m.group("path"), files)
                if target is not None:
                    walk(target)
                continue
            # label 通常紧跟其后（下一行），也可能同段。**只在到下一条结构命令
            # 之间找**——否则一条没有 label 的 `\subsection` 会捞走后面图注的 label。
            stop = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
            lab_m = _SECTION_LABEL_RE.search(tex, m.end(), stop)
            label = lab_m.group("label") if lab_m else f"<无 label@{name}#{m.start()}>"
            nodes.append(
                Node(
                    level=_LEVELS[m.group(1)],
                    title=clean_title(m.group("title")),
                    label=label,
                    source=name,
                )
            )

    if "main.tex" in files:
        # ⚠️ `main.tex` **自己也要走一遍**，不能只拿它当 `\input` 清单。
        walk("main.tex")
    else:
        # 没有 main.tex 就退回全量扫（`.src` 形态不止一种）
        for name in sorted(files):
            walk(name)

    if not nodes:
        # 全包里扫得到结构命令、却解析出 0 节点 ⇒ **是解析漏了**。
        # 而"全包一条都扫不到"才是诚实的空（那包确实没有分类学）。
        # 这两个读数长得**一模一样**，所以必须在这里分开——本模块 docstring 承诺的
        # "不静默返回空分类学"，实现上就落在这一条上。
        stray = sum(len(list(_SEC_RE.finditer(t))) for t in files.values())
        if stray:
            raise TaxonomyError(
                f"包里扫到 {stray} 条 \\section/\\subsection，却解析出 0 个节点：{p.name}\n"
                f"（这几乎总是解析漏了，不是「这包没有分类学」——别把它读成后者。）"
            )

    return Taxonomy(
        source_path=p,
        nodes=tuple(nodes),
        tree_labels=tree_labels,
        members=archive.names,
    )


# ---------------------------------------------------------------------- #
# 判据：一个源码包算不算「分类学来源」
# ---------------------------------------------------------------------- #

#: 叙事节标题里的**词**。顶层枝的标题**整个就是**这些词（可加 `and`/`,` 串起来）时，
#: 才当它叙事节。**是整词匹配，不是包含匹配。**
#:
#: ⚠️ 2026-09-20 实测七个包，包含匹配暴露了自相矛盾：`\bmethod\b` 排掉了
#: `2503.21460` 的 "Agent Methodology"（下级 3 的轴），却放过了 `2509.06733` 的
#: "RL Methods for Agentic Research"（同样是轴，只是复数字形没有 `\b` 边界）。
#: **同一个词根的两种字形落到两边** ⇒ 包含匹配与"叙事 vs 分类"没有稳定关系。
#:
#: ⚠️⚠️ **但这张表整个是废的——别再用它下任何结论。** 2026-09-20 拿导师库
#: `cache/` 的 160 个**普通论文** `.src` 做负向对照（抽**按名字排序的前 60 个**
#: 跑 `描述结构`）：**32 个**被判成"可作分类学来源"，假阳性 **32/60 = 53%**
#: （现量的复现在 `checks/check_axes.py`，样本口径与当年一致）。
#: 原因很直白：普通论文的顶层枝标题是 ``Key Design and Optimization of RL
#: Algorithms in LLMs/MLLMs`` 这种长句，**不等于**表里任何一个裸词，于是全被
#: 当成了轴。这个判据实际等价于"这是个有章节的 tar 包"。
#: 它此前看起来能用，只是因为 `survey_cache/` 里**全是按标题含 survey 筛出来的
#: 综述**——样本选的，不是判据判的。
#:
#: 所以"哪几根是轴"**改由人判、落盘、署名**，见 :func:`human_axes` 与
#: `axes_by_hand.json`。
#:
#: ⚠️ 这张表现在**没有任何现役代码在用它**（:meth:`Shape.chaptered` 也不用它——
#: 它只数顶层枝，不筛）。唯一的使用者是 `checks/check_axes.py` 里的
#: `retired_verdict()`，那份代码的作用是**把当年的 32/60 原样复现出来**，
#: 好让这个数可重跑。**别在新代码里引它。**
_LINEAR_ATOM = (
    r"introduction|related works?|background|preliminar\w*|"
    r"methods?|methodolog(?:y|ies)|experiments?|experimental (?:setup|results)|results?|"
    r"evaluations?|conclusions?|discussions?|limitations?|appendix|acknowledge?ments?|"
    r"abstract|future works?|future (?:trends|directions)|challenges?|trends|directions|"
    r"reproducibility(?: checklist)?|references?|implementation details|"
    r"ablations?(?: studies)?|setup|analysis"
)
_LINEAR_WORDS = re.compile(
    rf"(?:open\s+)?(?:the\s+)?(?:{_LINEAR_ATOM})"
    rf"(?:\s*(?:and|&|,|/)\s*(?:the\s+)?(?:{_LINEAR_ATOM}))*",
    re.I,
)

#: 人判的轴名单。**机器判不了这个**（见 :data:`_LINEAR_WORDS` 上的实测）。
AXES_BY_HAND: Path = SURVEY_CACHE / "axes_by_hand.json"

#: 顶层枝至少这么多根，才够开一次"它有没有轴"的讨论。
#: ⚠️ 这是**门槛**，不是判据。现量的读数（`checks/check_axes.py`）：
#: 过了这道门槛的**普通论文**，退役判据下来 32/60 = **53%**，
#: `chaptered` 下来 54/60 = **90%**。两个数合起来才是"门槛不是判据"的证据。
_MIN_AXES = 3


def _encoding(names: tuple[str, ...]) -> str:
    """这个包的层级是用**什么方式**编码的。

    ⚠️ 返回的是**方式**，不是"有没有 `sections/`"。2026-09-20 之前的判据把
    `sections_dir` 当成了唯一合格值，于是漏掉了 `flat_numbered` 那一类。
    """
    tex = [n for n in names if n.lower().endswith((".tex", ".ltx"))]
    if any(n.startswith("sections/") for n in names):
        return "sections_dir"
    if len(tex) == 1:
        return "single_file"
    if sum(1 for n in tex if re.match(r"^\d+(\.\d+)*[_-]", Path(n).name)) >= 3:
        return "flat_numbered"
    if len(tex) > 1:
        return "multi_file"
    return "none"


def _axes_blob() -> dict[str, object]:
    """人判名单里的 `surveys` 段。读不出/没有就当空——**但不假装判过**。"""
    if not AXES_BY_HAND.is_file():
        return {}
    try:
        blob = json.loads(AXES_BY_HAND.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    surveys = blob.get("surveys")
    return surveys if isinstance(surveys, dict) else {}


def human_axes(src_path: Path | str) -> tuple[str, ...] | None:
    """这个包的轴，**取自已落盘的人判名单**。没有记录就返回 `None`。

    ⚠️ `None` 的意思是**「这篇还没人判过」**，不是「它没有轴」。
    调用方**不许**把 `None` 退回词表去猜——`:data:`_LINEAR_WORDS` 在负向对照上
    的命中率是 **53%**（退役版）/ **90%**（现役 :attr:`Shape.chaptered`
    那个更松的版本），猜出来的东西两个数都不支持。要报就报「未经人判」。
    """
    rec = _axes_blob().get(Path(src_path).name)
    if not isinstance(rec, dict):
        return None
    axes = rec.get("axes")
    if not isinstance(axes, list):
        return None
    return tuple(str(a) for a in axes)


def human_axes_note(src_path: Path | str) -> str:
    """人判名单上给这篇写的备注（含判它的依据与保留意见）。没有就空串。"""
    rec = _axes_blob().get(Path(src_path).name)
    return rec.get("note", "") if isinstance(rec, dict) else ""


def judged_sources() -> frozenset[str]:
    """人判名单里**已判过**的包名。用途是**查漏**。

    漏一个，那个包就会以「未经人判」的样子混过去——而"没判过"和"没有轴"
    的读数一模一样，正是本库的头号病。
    """
    return frozenset(str(k) for k in _axes_blob())


@dataclass(frozen=True)
class Shape:
    """一个源码包的**结构形态**。⚠️ **这不是"能不能当分类学来源"的判据**——
    那件事机器判不了（见 :data:`_LINEAR_WORDS`）。这里只报**结构事实**。"""

    path: Path
    kind: str
    encoding: str
    #: 顶层枝标题，按文档顺序。
    tops: tuple[str, ...]
    #: `(顶层枝标题, 直接下级数)`。计数一起带着，是为了不用回头再解一遍包。
    branching: tuple[tuple[str, int], ...]
    n_level2: int
    n_level3: int
    #: 顶层枝里 label 命中 :data:`AXES` 的那些，存 `(轴名, label, 标题)`。
    #: 只有 `2509.02547` 会有——它的轴在 :data:`AXES` 里手工确认过，
    #: **轴的身份是 label**（`sec:position` 的标题写着 "Preliminary: …"）。
    declared: tuple[tuple[str, str, str], ...] = ()

    @property
    def axes(self) -> tuple[str, ...] | None:
        """人判的轴。`None` = **这篇还没人判过**，不是"它没有轴"。

        ⚠️ 返回 `None` 时**不许**退回词表猜——实测下来词表那条路在普通论文上
        命中 53%（退役版）/ 90%（现役 `chaptered`，见下）。这两个数由
        `checks/check_axes.py` **现量**，不是注释里的旧账。
        """
        return human_axes(self.path)

    @property
    def chaptered(self) -> bool:
        """至少三根顶层枝、其中两根下面还有一层。**弱事实，不是判据。**

        ⚠️ 2026-09-20 实测（`checks/check_axes.py` 现量，负向样本 =
        导师库 160 个普通论文里**按名字排序的前 60 个**）：
        **54 个满足 —— 命中率 54/60 = 90%**，而正向组（人判有轴的综述）是 8/8 = 100%
        ⇒ 两组只差 10 个点。它只说明"这包有章节层级"，**完全不说明它有分类轴**。

        ⚠️ **别把 53% 挪到这里。** 那个数是**退役判据**
        （多一道词表筛选，`checks/check_axes.py` 的 `retired_verdict()`）的。
        两个谓词不是同一个：词表只**删**候选 ⇒ `chaptered` 恒更松 ⇒ 命中数恒 ≥。
        拿 53% 给 `chaptered` 背书 = 让更松的谓词继承更严的谓词的读数。
        判来源请用 :attr:`axes`（人判）。
        """
        return len(self.tops) >= _MIN_AXES and sum(
            1 for _, n in self.branching if n > 0
        ) >= 2

    def describe(self) -> list[str]:
        ax = self.axes
        verdict = (
            f"人判 {len(ax)} 根轴" if ax is not None
            else "**未经人判**（机器判不了「哪几根是轴」——见 _LINEAR_WORDS 上的实测）"
        )
        head = (f"{self.path.name}  [{self.kind}/{self.encoding}]  "
                f"顶层 {len(self.tops)}，其中有下级的 "
                f"{sum(1 for _, n in self.branching if n > 0)}"
                f"  ->  {verdict}")
        marked = set(ax or ())
        detail = [
            f"      {t}  (下级 {n}){'  <- 轴' if t in marked else ''}"
            for t, n in self.branching
        ]
        if self.declared:
            detail.append(
                "      · taxonomy.AXES 里有确认 label 的："
                + ", ".join(f"{t} → {name}" for name, _, t in self.declared)
            )
        return [head] + detail


def describe_structure(src_path: Path | str) -> Shape:
    """解一个 `.src`，报它的**结构事实**（顶层枝、下级数、编码方式、人判的轴）。

    ⚠️ **它不判"这包算不算分类学来源"。** 那件事机器判不了——负向对照现量的读数：
    普通论文里 53%（退役判据）/ 90%（现役 :attr:`Shape.chaptered`）都会命中。
    轴由 :func:`human_axes` 给，机器只负责把结构摆出来让人能判。
    """
    p = Path(src_path)
    t = load_taxonomy(p)
    a = open_src(p)
    return Shape(
        path=p,
        kind=a.kind.value,
        encoding=_encoding(a.names),
        tops=tuple(b.title for b in t.branches()),
        branching=tuple((b.title, len(t.children_of(b.label))) for b in t.branches()),
        n_level2=sum(1 for n in t.nodes if n.level == 2),
        n_level3=sum(1 for n in t.nodes if n.level == 3),
        declared=tuple(
            (AXES[b.label][0], b.label, b.title)
            for b in t.branches() if b.label in AXES
        ),
    )


def survey_sources() -> tuple[Path, ...]:
    """`survey_cache/` 下**全部**候选源码包。

    ⚠️ 这里刻意返回全部而不是"那一个被点名的"。旧版把候选硬编码成
    `2510.16724`，于是 2026-09-20 新取的 6 篇**根本没有被它的报告看见**——
    一个只认识一个文件的检查，在第二个候选出现时会说"没有"。
    """
    if not SURVEY_CACHE.is_dir():
        return ()
    return tuple(sorted(SURVEY_CACHE.glob("*.src")))


def second_source_hint() -> str:
    """第二来源的**如实**报告。**不要把这个函数改写成"找到了"。**

    ⚠️ 2026-09-20 换过一次判据，见模块 docstring。旧版只看 `sections/` 目录，
    于是把 `2503.21460` 判成"没有第二来源"。现在的结论来自结构，
    **且每条都带着判据的原始材料（顶层枝与下级数）一起报**。
    """
    cands = survey_sources()
    if not cands:
        return f"survey_cache 里一个 .src 都没有：{SURVEY_CACHE}"
    lines: list[str] = []
    judged: list[str] = []
    unjudged: list[str] = []
    for p in cands:
        try:
            s = describe_structure(p)
        except Exception as exc:  # noqa: BLE001 - 单个包读不出不该让整份报告失败
            lines.append(f"  {p.name}  读不出（{type(exc).__name__}: {exc}）")
            continue
        lines.extend(s.describe())
        (judged if s.axes is not None else unjudged).append(p.name)
    n = len(cands)
    head = (f"分类学来源：{n} 个候选，其中 {len(judged)} 个人判过轴"
            + (f"（{', '.join(judged)}）" if judged else "")
            + (f"，{len(unjudged)} 个**还没人判**：{', '.join(unjudged)}" if unjudged else ""))
    if unjudged:
        head += ("\n  ⚠️ 「还没人判」不是「它没有轴」——去 "
                 f"{AXES_BY_HAND.name} 里补一条，**不要**退回词表猜。")
    return head + "\n" + "\n".join(lines)


@dataclass(frozen=True)
class SourceCrossCheck:
    """两套分类学在**轴这一层**的对照。

    ⚠️ **它证明的比读起来少。** 两份综述各用各的词，轴名不重合是**正常的**，
    不是失败。这一比能回答的只有一件事：**两套轴是不是在同一粒度上分解同一件事**
    （`n_axes` 与 `deep` 的对照）。词表重不重合要另说，见 :meth:`describe` 结尾。
    """

    a: Shape
    b: Shape
    shared: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    #: 两篇**全部后代节点标题**（归一化后）的重合数。近 0 是预期，不是缺陷。
    shared_descendants: int

    def describe(self) -> list[str]:
        ax_a, ax_b = self.a.axes, self.b.axes
        if ax_a is None or ax_b is None:
            who = [s.path.name for s in (self.a, self.b) if s.axes is None]
            return ["", f"  比不了：{', '.join(who)} **还没人判过轴**"
                        f"（去 {AXES_BY_HAND.name} 里补，不要退回词表猜）"]
        out = ["", f"  A = {self.a.path.name}  |  B = {self.b.path.name}",
               f"  A {len(ax_a)} 根轴   B {len(ax_b)} 根轴   "
               f"（两边都是**人判**的，同一口径）"]
        out.append(f"  轴名完全重合的：{list(self.shared) or '无'}")
        out.append(f"  只在 A 的：{list(self.only_a)}")
        out.append(f"  只在 B 的：{list(self.only_b)}")
        out.append(f"  全部后代标题的重合数：{self.shared_descendants}"
                   f"（**近 0 是预期**——两篇各用各的词表）")
        out.append("  结论只到「粒度」这一层：两套轴是**互补**的，不是互相校验的。"
                   "把 B 接进覆盖单元之前先接受这一点。")
        return out


def cross_check_sources(
    a_path: Path | str = DEFAULT_SURVEY,
    b_path: Path | str = CONFIRMED_SECOND,
) -> SourceCrossCheck:
    """把两套分类学的**轴**摆在一起比。**不做语义判断。**"""
    sa = describe_structure(a_path)
    sb = describe_structure(b_path)
    ta = load_taxonomy(a_path)
    tb = load_taxonomy(b_path)

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # 两边都取**人判**的轴（`axes_by_hand.json`），同一口径才比得起来。
    # ⚠️ 有一边没判过就直接报"比不了"，**不退回词表**——见 `_LINEAR_WORDS` 的实测。
    aa = {norm(t): t for t in (sa.axes or ())}
    ab = {norm(t): t for t in (sb.axes or ())}
    shared = tuple(sorted(set(aa) & set(ab)))
    only_a = tuple(aa[k] for k in sorted(set(aa) - set(ab)))
    only_b = tuple(ab[k] for k in sorted(set(ab) - set(aa)))

    da = {norm(n.title) for n in ta.nodes}
    db = {norm(n.title) for n in tb.nodes}
    return SourceCrossCheck(
        a=sa, b=sb, shared=shared, only_a=only_a, only_b=only_b,
        shared_descendants=len(da & db),
    )



def main() -> int:
    tax = load_taxonomy()
    print("=" * 72)
    print(f"覆盖单元 · 分类学  ←  {tax.source_path}")
    print("=" * 72)
    print(f"归档成员 {len(tax.members)} 个 | 解析出节点 {len(tax.nodes)} 个")
    print()
    for b in tax.branches():
        tag = AXES[b.label][0] if b.label in AXES else NON_AXIS_BRANCHES.get(b.label, "?")
        kids = tax.axis(b.label)
        print(f"[{tag:10s}] {b.title}   ({b.label})")
        for k in kids:
            print(f"    {k.level - 1}· {k.title:<34s} {k.label}   ← {k.source}")
    print()
    print("四轴计数（直接下级 / 含子子节）:",
          tax.axis_counts(), "/", tax.axis_counts(deep=True))
    print("覆盖单元（前三轴直接下级笛卡尔积）:", len(tax.units()), "个格子")
    print()
    print("交叉核对（拿 trees.tex 当外部判据）:")
    for d in tax.cross_check().describe():
        print("  " + d)
    print()
    print("第二来源:")
    print("  " + second_source_hint().replace("\n", "\n  "))
    print()
    print("两套分类学的交叉核对（只比到「轴」这一层）:")
    try:
        for line in cross_check_sources().describe():
            print(line)
    except FileNotFoundError as exc:
        print(f"  没跑成：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
