"""源码包（`.src`）的统一读取口。**全程在内存里解，一个字节都不落盘。**

为什么要有这一层
----------------

`.src` 不是一种格式，是**四种**，而且它们**长得一模一样**（都是那个 sha1 名字的文件）。
2026-09-20 在本地 160 个真实 `.src` 上量的形态分布：

| 形态 | 个数 | 是什么 |
|---|---|---|
| `TAR` | 156 | gzip 解出来是 tar —— 多文件投稿 |
| `PDF` | **4** | **根本不是 gzip**，头四字节是 `%PDF` |
| `SINGLE` | 0（本地没撞上） | gzip 解出来直接是 LaTeX 源，**没有 tar 层** |
| `UNREADABLE` | 0 | 解不开 |

⚠️ **`SINGLE` 这一种本地 160 个里一个都没有，但它是真实存在的**——
交接包 `00-先读我.md` §4 记过：单 `.tex` 投稿**没有 tar 层**，
gzip 流能整条解出、CRC 也对，但解出来是 LaTeX 源码不是 tar。
**判据写成"头部是 gzip 就算过"会静默放过它**，一直到抽取时才炸。
所以这里必须**先试 tar，失败了再看解出来的是不是文本**，而不是只看头四字节。

为什么返回值是枚举而不是 `None`
--------------------------------

本库的签名病是「分不清『没有』与『没读到』」。这一层是它最容易复发的地方，
因为四种形态里有三种都表现为"读不出来"。所以 :class:`SrcKind` 是**必填**的，
调用方必须显式写 `if a.kind is SrcKind.PDF:` 才能分支——**没有默认分支可走**。

为什么不落盘
------------

两个理由，第二个更要紧：

1. 抽取只需要**文本**，不需要文件树。落盘只是把临时目录管理问题引进来。
2. 一旦落盘，就有了"往哪儿落"的问题——而 `dasc/` 的红线是**不往导师库写任何东西**
   （见 `dasc/README.md` §6）。**不落盘 = 这个问题不存在**，比"记得落对地方"可靠。
"""

from __future__ import annotations

import gzip
import io
import tarfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ["SrcKind", "SrcArchive", "open_src", "PDF_MAGIC"]

#: PDF 的魔数。本地 4 个 `.src` 就是它——**这不是错误数据，是真实的形态**：
#: 有些投稿的 "source" 就是 PDF 本身。
PDF_MAGIC = b"%PDF"

#: LaTeX 源码里一眼能认出的记号，用来判 `SINGLE`（gzip 解出来是源而不是 tar）。
#: 刻意放宽：只要出现任何一个就算，因为目标是**把它和 tar 分开**，不是判它是不是好源码。
_TEX_MARKERS = (b"\\documentclass", b"\\begin{document}", b"\\section", b"\\input")


class SrcKind(str, Enum):
    """一个 `.src` 解出来到底是什么。**四种，没有"其他"。**

    `str` 混入是为了能直接进 JSON，不必再写一层转换。
    """

    #: gzip → tar：多文件投稿。`.bib` / `.bbl` / `.tex` 都从这里取。
    TAR = "tar"

    #: gzip → 单个 LaTeX 源（**没有 tar 层**）。交接包 §4 记过的那种。
    SINGLE = "single"

    #: 不是 gzip，是 PDF。**不是坏数据**，它本来就是这个。
    PDF = "pdf"

    #: 解不开。到这一步才是真的"读不出来"——**它和上面三种不是一回事**。
    UNREADABLE = "unreadable"


class SrcError(ValueError):
    """`.src` 打不开。**只在真的解不开时抛**——`PDF` 不抛，它是合法形态。"""


@dataclass(frozen=True)
class SrcArchive:
    """一个打开了的 `.src`。**已经全部在内存里**，不持有文件句柄。

    :attr:`kind` 是 :class:`SrcKind`，**不是可选的**——调用方必须自己分支。
    取成员之前先看 kind，这是本模块存在的全部意义。

    ⚠️ `TAR` 之外的三种，:meth:`names` 一律是空元组。
    **空元组不代表"这个包里没有 `.bib`"**，它代表"这个包根本没有成员这个概念"。
    要区分这两件事，就看 :attr:`kind`——这正是本库那条「`ABSENT` 必须能说出
    不可能存在的理由」在文件层的版本。
    """

    path: Path
    kind: SrcKind
    #: 只有 `TAR` 有；其余为空。
    names: tuple[str, ...] = ()
    #: `SINGLE` 时是解出来的源码文本；其余为空。
    single_text: bytes = b""
    #: 原始字节。tar 成员要现取，所以留着。
    raw: bytes = b""
    #: 只读的解包句柄（仅 `TAR`）。**不暴露**，靠 :meth:`read` 取。
    _tar: tarfile.TarFile | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ 取

    def read(self, name: str) -> bytes:
        """按成员名取字节。**只有 `TAR` 能取**——其余抛 :class:`SrcError`。

        抛而不是返回 `b""`：返回空字节会让调用方把"取不到"当成"这篇没引文"，
        而这两件事一个错一个对。
        """
        if self.kind is not SrcKind.TAR or self._tar is None:
            raise SrcError(
                f"{self.path.name} 是 {self.kind.value}，没有成员可读。"
                f"先看 kind 再取，别把「没有成员」读成「这个成员是空的」。"
            )
        try:
            f = self._tar.extractfile(name)
        except KeyError as exc:
            raise SrcError(f"{name} 不在 {self.path.name} 里") from exc
        if f is None:  # 目录条目
            raise SrcError(f"{name} 是目录条目，不是文件")
        return f.read()

    def find(self, *suffixes: str) -> tuple[str, ...]:
        """按后缀挑成员名。**`TAR` 之外一律空元组**，理由见类 docstring。"""
        if self.kind is not SrcKind.TAR:
            return ()
        low = tuple(s.lower() for s in suffixes)
        return tuple(n for n in self.names if n.lower().endswith(low))

    @property
    def tex_files(self) -> tuple[str, ...]:
        return self.find(".tex", ".ltx")

    @property
    def bib_files(self) -> tuple[str, ...]:
        """`.bib` 路线（BibTeX 式）的原料。"""
        return self.find(".bib")

    @property
    def bbl_files(self) -> tuple[str, ...]:
        """`\\bibitem` 路线（`.bbl` 式）的原料。"""
        return self.find(".bbl")

    @property
    def all_text(self) -> str:
        """**所有**能当文本看的成员拼起来（`.tex`/`.ltx`/`.bib`/`.bbl`）。

        给"这个包里到底有没有引用"这类**总览**用。取引用**不要**用它——
        两条路线必须分开走、分开记数，合起来看就分不出是哪条命中的
        （那正是 T4 那个「0 → 96」踩的坑）。
        """
        if self.kind is SrcKind.SINGLE:
            return self.single_text.decode("utf-8", "replace")
        if self.kind is not SrcKind.TAR:
            return ""
        parts: list[str] = []
        for n in self.find(".tex", ".ltx", ".bib", ".bbl"):
            try:
                parts.append(self.read(n).decode("utf-8", "replace"))
            except SrcError:
                continue
        return "\n".join(parts)


def _looks_like_tex(blob: bytes) -> bool:
    head = blob[:65536]
    return any(m in head for m in _TEX_MARKERS)


def open_src(path: Path | str) -> SrcArchive:
    """打开一个 `.src`，判出它的 :class:`SrcKind`，全在内存里。

    **判序是刻意的**：先看 PDF 魔数，再试 gzip，再试 tar，最后才回头猜 SINGLE。
    理由——`tarfile` 对非 tar 数据抛的是 `tarfile.ReadError`，
    而它对**被截断的 tar** 也抛同一个异常；拿它当唯一判据，
    一个坏掉的 tar 会被误判成 SINGLE，然后以"这包里没有 `.bib`"的样子混过去。
    """
    p = Path(path)
    raw = p.read_bytes()

    # ① 不是 gzip 的一类。PDF 是**真实形态**，不是错误。
    if raw[:4] == PDF_MAGIC:
        return SrcArchive(path=p, kind=SrcKind.PDF, raw=raw)

    # ② gzip？解不开就是真的读不出来。
    try:
        plain = gzip.decompress(raw)
    except (OSError, EOFError) as exc:
        raise SrcError(
            f"{p.name} 既不是 PDF 也不是 gzip（头四字节 {raw[:4]!r}）：{exc}"
        ) from exc

    # ③ gzip 解出来是 tar 吗？
    try:
        t = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    except tarfile.ReadError:
        # ④ 不是 tar。**到这里才判 SINGLE**，而且要有 LaTeX 的记号才认——
        #    否则一个解开了但内容不认识的包会被当成"没有引文的投稿"。
        if _looks_like_tex(plain):
            return SrcArchive(path=p, kind=SrcKind.SINGLE, single_text=plain, raw=raw)
        raise SrcError(
            f"{p.name} 解出来既不是 tar 也不像 LaTeX 源（{len(plain)} 字节）。"
            f"⚠️ 这一条**没有**被算成「这包没引文」——它是未知形态，要人看一眼。"
        )

    names = tuple(m.name for m in t.getmembers() if m.isfile())
    # 名义上的防线：本模块从不 extractall，所以现在无害；
    # 写在这里是为了让**以后**想加落盘的人先撞上这行。
    if any(n.startswith("/") or ".." in Path(n).parts for n in names):
        raise SrcError(f"{p.name} 里有绝对路径或 .. 成员，拒绝打开：{names[:3]}")
    return SrcArchive(path=p, kind=SrcKind.TAR, names=names, raw=raw, _tar=t)
