"""导师库的窄接口。

`大创执行步骤` 第 1 步的四件事里，这一件是「接导师库，实现窄接口
`list_papers` / `get_fulltext` / `search` / `get_citations`」。

为什么要"窄"
-------------

`语料库验收要求.md` §0.3 里那条理由：窄接口是**演示版与推广版之间唯一的通道**。
导师库有 25 个字段，而 M1 只用到其中一小部分；把 25 个字段原样透出去，
将来换语料（2024/25 arXiv 层）时就要么全改调用方、要么造一个假装兼容的适配层。
所以这里只暴露 :class:`Paper` 上那几项——**C2 契约里写死的那几项**。

⚠️ 本模块最要紧的不是它能查到什么，是它**查不到时怎么说**
--------------------------------------------------------

本库反复栽的那个病是「分不清『没有』与『没读到』」（导师库 `README.md` 里
至少有五次记录）。窄接口是它最容易复发的地方，因为**返回值都是"空"**：

* 论文没有全文 ⟷ 全文在没插的移动硬盘上 ⟷ 全文文件坏了
* arXiv 没有源码包 ⟷ 源码包在本机 cache 里没有（在 G: 上）

所以这一层**不返回裸的 `str` / `[]` / `None`**，一律返回带 :class:`Availability`
的结果对象。那个枚举是**接口的一部分**，不是日志。

**2026-09-20 实测的本地存量**（决定了这些状态各自会有多少）：

| 项 | 数 |
|---|---|
| `papers.jsonl` | 13,119 篇（其中有 `arxiv_id` 的 9,798） |
| `data/text/*.txt.gz` | **9,827 个**，与 `paper_id` 归一化后**逐一对上，0 个无主文件** |
| `cache/*.src` | **160 个**，且**全部属于周报库**（1069 篇），**主库一个都没有** |
| `pdfs/` | 1,318 个 |

正文那一行是**对出来的，不是估的**：把 9,827 个文件名与 13,119 个 `paper_id`
各按 :func:`safe_stem` 归一化后取交集——**有主 9,827 / 无主 0 / 缺文件 3,292**，
两边一个不多一个不少。（所以 `has_fulltext` 直接看文件在不在就是可靠的。）
→ 这一步顺带查出**正文有第二条来路**：缺正文的 3,292 篇里，**78 篇的 PDF
就在本机**只是没抽过，另有 32 篇**没有 arxiv_id 却有正文**（ICML/PMLR，从 PDF 抽的）。
那 32 篇证伪了"没有 arXiv ID ⇒ 不会有正文"，也把 :meth:`Corpus.get_fulltext`
的 `ABSENT` 判据收紧了——详见那个方法的注释。

⚠️ **别把这张表当成常量。** 它是 2026-09-20 的快照，而**存量会变**（正文天天在增）。
要数现数：`python -m dasc.checks.check_corpus`。

⇒ **主库的源码包 100% 不在本机**（都在没插的移动硬盘上）。所以 :meth:`Corpus.get_src`
对主库论文必然返回 ``OFF_DISK``——那不是"arXiv 没有源码包"，是"我们这块盘上没有"。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shlex
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..readonly import LibraryView

# ---------------------------------------------------------------------- #
# 可用性：本模块存在的理由
# ---------------------------------------------------------------------- #

class Availability(str, Enum):
    """一份东西"有没有"。**这四个值不能压成两个。**

    `ABSENT` 与 `OFF_DISK` 的区别是本库签名病的正面答复：前者是**关于论文/语料
    的事实**（这东西**不可能**存在），后者是**关于我们自己的缺口**
    （它可能存在，只是没在本机）。

    ⚠️ **判据（2026-09-20 收紧）**：用 `ABSENT` **必须能说出"它不可能存在"的理由**；
    说不出就用 `OFF_DISK`。默认值是 `OFF_DISK`——因为把"我没取到"说成
    "世界上没有"是本库反复栽的那个方向，反过来只是保守。
    """

    OK = "ok"                # 拿到了
    ABSENT = "absent"        # 语料里确实没有 —— 关于论文的事实
    OFF_DISK = "off_disk"    # 原料不在这块盘上 —— 关于我们的缺口
    UNREADABLE = "unreadable"  # 文件在，但解不开（gzip 坏、编码坏）

    @property
    def is_ours(self) -> bool:
        """这个"没有"是不是**我们的问题**。"""
        return self in (Availability.OFF_DISK, Availability.UNREADABLE)


# ---------------------------------------------------------------------- #
# 窄接口的返回类型（`语料库验收要求.md` §C2）
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paper:
    """C2 契约要求 `record` 至少含这些。

    ⚠️ `first_seen` **不是可选项**——B1 判据（周更）与预印本的时效排序都靠它，
    而它一旦缺了，"这篇是什么时候进来的"就只能靠猜。
    """

    paper_id: str
    arxiv_id: str | None
    title: str                 # 原文，不归一化（B4：标题匹配是主力，实测比 arXiv ID 多 6 倍命中）
    abstract: str
    year: int | None
    venue: str | None
    subfields: tuple[str, ...]  # 多标签（B3）
    is_rl: bool
    first_seen: str
    citations: int | None
    url: str | None
    has_fulltext: bool          # 本机是否有正文；**只回答"在不在"，不含"能不能读"**
    has_local_pdf: bool = False  # 本机是否有 PDF（第二条正文来路，见下）

    @classmethod
    def from_jsonl(cls, o: dict[str, Any], *, has_fulltext: bool) -> Paper:
        rl = o.get("rl") or {}
        return cls(
            paper_id=o["paper_id"],
            arxiv_id=o.get("arxiv_id"),
            title=o.get("title") or "",
            abstract=o.get("abstract") or "",
            year=o.get("year"),
            venue=o.get("venue"),
            subfields=tuple(s for s in (rl.get("subfield") or []) if s),
            is_rl=bool(rl.get("is_rl")),
            first_seen=o.get("first_seen") or "",
            citations=o.get("citations"),
            url=o.get("url"),
            has_fulltext=has_fulltext,
            has_local_pdf=bool(o.get("local_pdf")),
        )


@dataclass(frozen=True)
class FullText:
    paper_id: str
    status: Availability
    text: str = ""
    chars: int = 0
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Availability.OK


@dataclass(frozen=True)
class Src:
    """arXiv 源码包（tar.gz / 单文件 gzip / 也可能是 PDF）。

    ⚠️ 内容形态**不只一种**：多文件投稿外面有 tar 层，单 `.tex` 投稿**没有**——
    那种的 gzip 流能整条解出、CRC 也对，但解出来是 LaTeX 源码本身而不是 tar。
    判据写成"头部是 gzip 就算过"会静默放过它、到抽取时才炸（导师库实测踩过）。
    所以这里只负责**把它取出来**，形态判定交给解析方。
    """

    paper_id: str
    status: Availability
    path: Path | None = None
    data: bytes | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Availability.OK


@dataclass(frozen=True)
class Citation:
    """一条引用边。``target`` 可能是 arXiv ID、DOI 或纯标题键，看来源格式。"""

    source: str
    target: str
    kind: str = ""     # "arxiv_id" | "doi" | "title_key" | "raw"


@dataclass(frozen=True)
class Citations:
    """引用关系。**第 1 步只定形状，抽取器属第 2 步。**

    `edges` 与 `count` 是两回事，别混：
    * `count` = 这篇**被引**多少次（`papers.jsonl` 的 `citations` 字段，走 OpenAlex）。
    * `edges` = 这篇**引了谁**（依赖图的边），只能从 `.src` 的 `.bbl`/`.bib` 抽。

    ⚠️ **`count` 的三种"零"必须分开**，否则 `citations: null` 会被读成"没人引用"。
    2026-09-20 实测 13,119 篇的分布：

    ============  =======  ====================================
    `citations`   篇数     含义
    ============  =======  ====================================
    `None`        13,069   **我们从没查过**（99.6%）
    `0`           44       查过，确实是 0
    `> 0`         6        查过，有被引
    ============  =======  ====================================

    ⇒ 全库**只有 50 篇**有被引数据，所以 `citations` 现在**撑不起任何排序**，
    只能当"这 50 篇是热的"用。

    `count_known = (citations is not None)` 这条判据**是量过的，不是推的**：
    那 50 篇非 `None` 的**全部都有 `citations_updated`**（另查 `citations_attempted`
    有 46 篇是日期、4 篇是 `None`——所以 `citations_updated` 才是可靠的那个信号）。
    于是 `0` 与 `None` 的分界确实落在"查过 / 没查过"上。
    **代价**：这是 50 行的经验相关，不是文档保证。哪天真出现 `citations: 0` 而
    无 `citations_updated` 的样本，这条就得改成读 `citations_updated`。
    """

    paper_id: str
    edges: tuple[Citation, ...] = ()
    edges_status: Availability = Availability.OFF_DISK
    count: int | None = None
    count_known: bool = False
    note: str = ""


# ---------------------------------------------------------------------- #
# 抄来的小工具（注明出处，避免"抄一份就漂且不报警"）
# ---------------------------------------------------------------------- #

#: 抄自 `rl_acdemical_paper/rllib/paths.py`（2026-09-20）。
#: paper_id 形如 `orid:004uTlSufe`，`:` 在 Windows 上是非法文件名字符。
#: **必须与导师库用同一条规则**，否则同一篇论文在磁盘上会有两个"身份"。
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(paper_id: str) -> str:
    return _UNSAFE.sub("-", paper_id).strip("-")


#: 抄自 `rl_acdemical_paper/rllib/extract/text.py:75`（2026-09-20），
#: 缓存键 `sha1(该 URL)` + `.src`（`rllib/http.py` 的 `_cache_path`）。
#: **2026-09-20 实测核对过**：本地 160 个 `.src` 与周报库 1069 篇的
#: `sha1(/src/<id>)` **160/160 全中**，0 个对不上。
def arxiv_src_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/src/{arxiv_id}"


def src_cache_name(arxiv_id: str) -> str:
    return hashlib.sha1(arxiv_src_url(arxiv_id).encode()).hexdigest() + ".src"


#: 抄自 `rllib/search/query.py`（2026-09-20）。
#: bm25 列权重：标题命中远比正文命中更能说明这篇讲的就是这个主题。
BM25_WEIGHTS = (10.0, 3.0, 1.0)


def fts_query(text: str, *, any_term: bool = False) -> str:
    """把用户输入变成安全的 FTS5 表达式（抄自 `rllib/search/query.py:_fts_query`）。

    不能直接塞进 `MATCH`：FTS5 语法里 `"` `*` `-` `:` `(` 都有语义，标题里的
    标点会让查询**报错**（不是返回空，是抛 `sqlite3.OperationalError`）。
    引号内转义靠双写 `"`；词尾的 `*` 留在引号**外**——`"optim"*` 是合法前缀
    语法，而 `"optim*"` 会把星号当字面量。
    """
    try:
        chunks = shlex.split(text, posix=False)
    except ValueError:
        chunks = text.split()

    terms: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip().strip('"').strip("'").strip()
        if not chunk:
            continue
        prefix = chunk.endswith("*")
        stem = chunk[:-1] if prefix else chunk
        if not stem:
            continue
        quoted = '"' + stem.replace('"', '""') + '"'
        terms.append(quoted + "*" if prefix else quoted)

    if not terms:
        return ""
    return (" OR " if any_term else " AND ").join(terms)


# ---------------------------------------------------------------------- #
# 语料
# ---------------------------------------------------------------------- #

class Corpus:
    """导师库的窄接口。**只读**——所有读取都走 :class:`~dasc.readonly.LibraryView`。

    用法::

        with Corpus() as c:
            rlhf = c.list_papers(pool="rlhf")
            print(c.get_fulltext(rlhf[0].paper_id).status)
            print(c.search("verifiable rewards", k=5))
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.lib = LibraryView(root) if root is not None else LibraryView()
        self._papers: dict[str, Paper] | None = None
        self._order: list[str] = []

    # ---------------------------------------------------------------- 加载

    def _load(self) -> None:
        """把 `papers.jsonl` 读进来。幂等，只读一次。

        **一次读全库而不是流式**：13,119 篇在内存里约几十 MB，而 M1 的
        聚类/贪心都要反复扫全部候选；每次查询重读 31.7 MB 的 jsonl 是浪费。
        """
        if self._papers is not None:
            return
        text_dir = self.lib.text_dir
        papers: dict[str, Paper] = {}
        order: list[str] = []
        with self.papers_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                pid = o["paper_id"]
                has_text = (text_dir / (safe_stem(pid) + ".txt.gz")).exists()
                papers[pid] = Paper.from_jsonl(o, has_fulltext=has_text)
                order.append(pid)
        self._papers = papers
        self._order = order

    @property
    def papers_jsonl(self) -> Path:
        return self.lib.papers_jsonl

    # ---------------------------------------------------------------- 上下文

    def __enter__(self) -> Corpus:
        self.lib.begin_audit()
        self.lib.verify()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.lib.end_audit()

    # ---------------------------------------------------------------- 窄接口 ①

    def list_papers(
        self,
        pool: str | None = None,
        *,
        limit: int | None = None,
        rl_only: bool = False,
    ) -> list[Paper]:
        """列论文。``pool`` 是子方向池名（如 `"rlhf"`），``None`` = 全库。

        ⚠️ **`pool=None` 不是"只 RL"**。导师库的 `is_rl` 是**筛选面不是闸门**
        （检索那条设计红线），13,119 篇里只有 1,585 篇 `is_rl`。要只要 RL 就
        显式传 ``rl_only=True``。默认给全量是因为主题聚类与覆盖度都要在
        **未经过那道闸门**的集合上做——否则筛选器的召回漏洞会被当成"世界上没有"。
        """
        self._load()
        assert self._papers is not None
        out: list[Paper] = []
        for pid in self._order:
            p = self._papers[pid]
            if pool is not None and pool not in p.subfields:
                continue
            if rl_only and not p.is_rl:
                continue
            out.append(p)
            if limit is not None and len(out) >= limit:
                break
        return out

    def get(self, paper_id: str) -> Paper | None:
        self._load()
        assert self._papers is not None
        return self._papers.get(paper_id)

    # ---------------------------------------------------------------- 窄接口 ②

    def get_fulltext(self, paper_id: str) -> FullText:
        """取正文（LaTeX 扁平化后的纯文本）。

        三个"空"要分开——见 :class:`Availability`。
        """
        self._load()
        assert self._papers is not None
        p = self._papers.get(paper_id)
        if p is None:
            return FullText(paper_id, Availability.ABSENT, note="库里没有这个 paper_id")

        path = self.lib.text_dir / (safe_stem(paper_id) + ".txt.gz")
        if not path.exists():
            # ⚠️ 这里**一律 OFF_DISK，从不 ABSENT**——是量过之后改的，见下。
            #
            # 原先写的是"没有 arxiv_id ⇒ 本来就不会有正文 ⇒ ABSENT"。2026-09-20 对完账
            # 发现那是错的：**库里有第二条正文来路**——32 篇 ICML/PMLR 论文
            # （`proceedings.mlr.press`）没有 arxiv_id，却有正文，是从 PDF 抽的。
            # 那 32 篇全都 `local_pdf` 有值、PDF 就在 `pdfs/`。
            #
            # 于是把缺正文的 3,292 篇拆开看：
            #   · 无 arxiv_id、无 PDF  3,211    ← NeurIPS 1442 / ICLR 916 / ICML 830 / CoRL 23
            #   · 无 arxiv_id、**有 PDF** 78    ← 原料在手，只是没抽过
            #   · 有 arxiv_id、无 PDF      3
            # 三家会议**都有可取的 PDF**（ICML 那 32 篇就是证据），所以这 3,292 篇里
            # 一篇都不是"这篇没有全文"，全都是"我们还没去取"。
            # 把它们叫 ABSENT 就是本库签名病的原样复现。
            #
            # ⇒ 规则收紧成：**`ABSENT` 必须能说出"这东西不可能存在"的理由**（比如
            # 非 arXiv 论文要 arXiv 源码包）；说不出就是 `OFF_DISK`。
            if p.has_local_pdf:
                why = "PDF 就在本机，只是还没抽成正文"
            elif p.arxiv_id:
                why = "有 arxiv_id，但源码包不在本机（主库的 .src 全在移动硬盘上）"
            else:
                why = ("既无 arxiv_id 也无本地 PDF——但**不等于这篇没全文**："
                       "NeurIPS/ICLR/ICML 三家的 PDF 本来就得现取，属我们的缺口")
            return FullText(paper_id, Availability.OFF_DISK, note=why)

        try:
            raw = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        except (OSError, EOFError) as exc:
            return FullText(paper_id, Availability.UNREADABLE, note=f"{type(exc).__name__}: {exc}")
        return FullText(paper_id, Availability.OK, text=raw, chars=len(raw))

    # ---------------------------------------------------------------- 窄接口 ③

    def search(self, query: str, k: int = 20, **filters: Any) -> list[str]:
        """全文检索，返回 paper_id 列表（C2 契约签名）。

        **复用导师库那张 FTS5 索引，不自己写 BM25。** 索引是产物、派生自
        `papers.jsonl` + `data/text/*.gz`，可随时重建；正因为它已经在那儿，
        它也是**现成的对照基线**（`大创执行步骤` 第 1 步原话）。

        ⚠️ 三条会静默出错的：
        * 索引是**无内容 FTS5 表** ⇒ 正文 SELECT 不回来，命中段落要回读
          `.txt.gz` 现切（这里不切，只要 id）；
        * `tokenize='porter unicode61'` 是**英文词干** ⇒ 中文查询查不到；
        * bm25 权重**必须内联**进 SQL，不能走占位符——SELECT 在 WHERE 之前，
          占位符按 SQL 文本顺序绑定，三个权重参数会把 `MATCH` 的表达式挤到
          后面去，报 `fts5: syntax error near "."`（一个看不出真正原因的错）。
        """
        expr = fts_query(query)
        if not expr:
            return []
        where = ["docs MATCH ?"]
        params: list[Any] = [expr]
        if (venue := filters.get("venue")):
            where.append("p.venue = ?")
            params.append(venue)
        if (year := filters.get("year")):
            where.append("p.year = ?")
            params.append(year)
        if (subfield := filters.get("subfield")):
            # 逗号分隔列里的精确成员匹配：前后补逗号再 LIKE，避免 "rl" 命中 "rlhf"
            where.append("(',' || p.subfields || ',') LIKE ?")
            params.append(f"%,{subfield},%")
        if filters.get("rl_only"):
            where.append("p.is_rl = 1")

        weights = ", ".join(repr(float(w)) for w in BM25_WEIGHTS)
        sql = f"""
            SELECT p.paper_id
            FROM docs JOIN papers p ON p.doc_rowid = docs.rowid
            WHERE {' AND '.join(where)}
            ORDER BY bm25(docs, {weights})
            LIMIT ?
        """
        params.append(k)
        with self.lib.index_conn() as conn:
            return [r["paper_id"] for r in conn.execute(sql, params)]

    # ---------------------------------------------------------------- 窄接口 ④

    def get_src(self, paper_id: str) -> Src:
        """取 arXiv 源码包。**依赖图的唯一原始输入**。

        第 1 步只把它**取出来**（路径 + 字节），解析 `.bbl`/`.bib` 属第 2 步。
        """
        self._load()
        assert self._papers is not None
        p = self._papers.get(paper_id)
        if p is None:
            return Src(paper_id, Availability.ABSENT, note="库里没有这个 paper_id")
        if not p.arxiv_id:
            # 这条 ABSENT 站得住（按上面那条判据）：arXiv 的源码包**只**按 arxiv_id 取，
            # 没有 ID 就没有入口，不是"我们还没去拿"。
            # 但**理由要说准**：不能说"arXiv 上没有它的源码包"——PMLR/ICLR 的论文
            # 很多在 arXiv 上也有，只是库里没把 ID 对上。那是**我们的知识缺口**，
            # 所以两句都写出来，别让读的人以为已经查过 arXiv 了。
            return Src(paper_id, Availability.ABSENT,
                       note="库里没有它的 arxiv_id ⇒ **我们无从定位**源码包"
                            "（不等于 arXiv 上没有——只是 ID 没对上）")

        path = self.lib.cache_dir / src_cache_name(p.arxiv_id)
        if not path.exists():
            return Src(
                paper_id, Availability.OFF_DISK, path=path,
                note=f"本机 cache 里没有 {path.name}——源码包在移动硬盘上（主库的 .src 全不在本机）",
            )
        return Src(paper_id, Availability.OK, path=path, data=path.read_bytes())

    def get_citations(self, paper_id: str) -> Citations:
        """引用关系。

        **第 1 步只定形状，抽取器属第 2 步。** 这一版：
        * `count` 走 `papers.jsonl` 的 `citations` 字段（OpenAlex，库里绝大多数是
          `None`）——用 `count_known` 区分"确实是 0"与"我们没去查过"，
          否则 `citations: null` 会被读成"没人引用"。
        * `edges` 恒为空，`edges_status` 报出**为什么**空（源码包不在本机／
          抽取器还没写），而不是给一个看起来正常的 `[]`。
        """
        self._load()
        assert self._papers is not None
        p = self._papers.get(paper_id)
        if p is None:
            return Citations(paper_id, edges_status=Availability.ABSENT, note="库里没有这个 paper_id")

        src = self.get_src(paper_id)
        return Citations(
            paper_id=paper_id,
            edges=(),
            edges_status=src.status,
            count=p.citations,
            count_known=p.citations is not None,
            note=(
                "边（引了谁）只能从 .src 的 .bbl/.bib 抽，抽取器属第 2 步；"
                "`count` 是被引次数，两者不是一回事"
                + (f"；源码包：{src.note}" if src.note else "")
            ),
        )

    # ---------------------------------------------------------------- 盘点

    def stats(self) -> dict[str, Any]:
        """本地存量盘点。**聚合计数，不打印任何论文内容**（导师库铁律 4）。"""
        self._load()
        assert self._papers is not None
        ps = list(self._papers.values())
        with_text = sum(1 for p in ps if p.has_fulltext)
        rl = sum(1 for p in ps if p.is_rl)
        return {
            "papers": len(ps),
            "with_arxiv_id": sum(1 for p in ps if p.arxiv_id),
            "with_fulltext": with_text,
            "is_rl": rl,
            "pools": len({s for p in ps for s in p.subfields}),
            "src_local": sum(1 for f in self.lib.cache_dir.glob("*.src")),
        }
