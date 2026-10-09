"""只读网关：四道闸，防误写导师库。

四道闸分别拦哪一条泄漏向量
---------------------------

| # | 泄漏向量 | 闸 |
|---|---|---|
| ① | `import rllib` 写 `.pyc` 进库内 | `sys.dont_write_bytecode = True`（**在 import 之前**） |
| ② | 直接连库时 SQLite 建 `-wal` / `-shm`；`rllib.search.index.connect()` 执行 `PRAGMA journal_mode=WAL` 与 DDL | 只读 URI，且**必须带 `immutable=1`**；**永不调用 `connect()`** |
| ③ | 闸①装晚了（有人先 `import rllib`），或以后有人图省事 import 它 | 运行期断言 `'rllib' not in sys.modules` |
| ④ | 路径拼错、上层拿到裸 `Path` 自己写、缓存/临时目录被指到库内 | 只暴露 :class:`LibraryView`，**它没有任何写方法**；并在关闭时核对库未被改动 |

⚠️ **闸② 是本文件最要紧的一条，因为文档里那句话是欠定的。**
`CLAUDE.md` §二、`语料库验收要求.md` §C、导师库 `README.md` §2 三处都只写
「只读 URI 连接」。2026-09-20 在本机**实测**（`data/index.sqlite`，250,720,256 B，
连接前后比对文件清单与 mtime）：

| 连接串 | 查到 | 新建的文件 |
|---|---|---|
| `mode=ro` | 13,119 篇 | **`-wal`(0 B) + `-shm`(32,768 B)** ← **这是写** |
| `mode=ro&immutable=1` | 13,119 篇 | 无 |

也就是说「只读 URI」若只写 `mode=ro`，**闸是漏的**：它挡得住 `PRAGMA journal_mode=WAL`，
挡不住 SQLite 为 WAL 库建共享内存文件这一步。`immutable=1` 才是真正一行不写。

**代价必须一起记住**：`immutable=1` 让 SQLite **完全无视 `-wal`**。
若 WAL 里还有没 checkpoint 的帧，我们会静默读到旧数据——那正是本库的签名病
（分不清「没有」与「没读到」）。所以配一条判据：**开库前若发现非空的 `-wal`，
报错而不是读**。见 :func:`LibraryView.connect_index`。

⚠️ **闸③ 与闸④ 的口径来源（2026-09-20 已结）**
本模块落地时，本库文档一致记录的是**两道闸**（只读 URI + `dont_write_bytecode`），
而 `论文趋势挖掘/计划/大创执行步骤.md` 第 1 步写的是「**四道闸**」，另两道**任何
文档里都没有定义**——当时这里按「四条泄漏向量各配一闸」补的，标为待确认口径。

**用户 2026-09-20 拍：正式写成四道闸。** 已回写进 `CLAUDE.md` §二与
`科研阅读路径/规范/语料库验收要求.md` §C1，**这三处现在同口径**——
改这里就要改那两处，否则又回到"文档与实现对不上"。
"""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ── 闸① 再装一次 ──────────────────────────────────────────────────────
# 它在 dasc/__init__.py 已装；这里重复是为了本模块被单独导入时也成立。
# 幂等，无副作用。
sys.dont_write_bytecode = True

#: 导师库默认位置：与 `dasc/` 同级。
DEFAULT_LIBRARY: Path = Path(__file__).resolve().parents[1] / "rl_acdemical_paper"

#: 库内**绝不应该被任何人 import 的包名**。见闸③。
FORBIDDEN_MODULES: tuple[str, ...] = ("rllib",)

#: 审计快照的观测面。
#:
#: 分两类是因为它们坏掉的方式不同：**单文件**要盯住字节数和 mtime（真的被写），
#: **目录**要盯住（文件数, 总字节, 最新 mtime）三元组——新增/删除一个文件时
#: 总字节可能恰好不变，但计数会变。
_WATCH_FILES: tuple[str, ...] = (
    "data/papers.jsonl",
    "data/index.sqlite",
    "data/index.sqlite-wal",
    "data/index.sqlite-shm",
)
_WATCH_DIRS: tuple[str, ...] = ("data/text", "vault", "pdfs", "cache")


class LibraryTouched(RuntimeError):
    """导师库在本次会话期间被改动了。闸④ 的判决。"""


class StaleIndex(RuntimeError):
    """`-wal` 非空 ⇒ `immutable=1` 会读到旧数据。**不读，报错。**"""


@dataclass(frozen=True)
class Gate:
    """一道闸的自检结果。``ok=False`` 时 :attr:`detail` 要能指出**哪一条**泄漏向量。"""

    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - 仅用于人眼
        return f"[{'OK ' if self.ok else 'RED'}] {self.name} — {self.detail}"


# ---------------------------------------------------------------------- #
# 审计快照
# ---------------------------------------------------------------------- #

def _dir_triple(d: Path) -> tuple[int, int, int]:
    """目录的 (文件数, 总字节, 最新 mtime_ns)。

    空目录与不存在都返回 ``(0, 0, 0)``——**这两者在这里不加区分是安全的**，
    因为审计只做前后比对，而"不存在 → 不存在"与"空 → 空"都不会被误判成改动。
    """
    n = size = newest = 0
    if not d.exists():
        return (0, 0, 0)
    for root, _dirs, files in os.walk(d):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))
            except OSError:
                continue
            n += 1
            size += st.st_size
            if st.st_mtime_ns > newest:
                newest = st.st_mtime_ns
    return (n, size, newest)


def _file_pair(f: Path) -> tuple[int, int] | None:
    if not f.exists():
        return None
    st = f.stat()
    return (st.st_size, st.st_mtime_ns)


def audit(root: Path) -> dict[str, object]:
    """给导师库拍一张账。**只 stat，不读内容**（9828 个正文文件也只要几十毫秒）。"""
    snap: dict[str, object] = {}
    for rel in _WATCH_FILES:
        snap[rel] = _file_pair(root / rel)
    for rel in _WATCH_DIRS:
        snap[rel] = _dir_triple(root / rel)
    return snap


# ---------------------------------------------------------------------- #
# 只读视图
# ---------------------------------------------------------------------- #

class LibraryView:
    """导师库的只读句柄。**没有任何写方法**——这是闸④ 的本体。

    用法::

        with LibraryView() as lib:
            lib.verify()                    # 跑四道闸，红了会抛
            with lib.index_conn() as conn:  # 只读连接
                ...

    刻意**不**提供 `__getattr__` 转发到 `Path`、也不暴露可写的目录路径成裸 `Path`：
    一旦上层拿到能 `.write_bytes()` 的东西，"没有写方法"就只是措辞。
    :meth:`text_dir` 等方法返回的是 `Path`，但那是**读**的入口，且网关在
    :meth:`verify` 里已核对过它没被改动。
    """

    def __init__(self, root: Path | str = DEFAULT_LIBRARY) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"导师库不在：{self.root}")
        self._before: dict[str, object] | None = None

    # ---------------------------------------------------------------- 路径

    @property
    def papers_jsonl(self) -> Path:
        return self.root / "data" / "papers.jsonl"

    @property
    def index_sqlite(self) -> Path:
        return self.root / "data" / "index.sqlite"

    @property
    def text_dir(self) -> Path:
        return self.root / "data" / "text"

    @property
    def vault_papers_dir(self) -> Path:
        return self.root / "vault" / "papers"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    # ---------------------------------------------------------------- 闸③

    def gate_no_rllib(self) -> Gate:
        """闸③：确认导师库的**代码**从没被 import 进来。

        为什么这条不能省：闸① 靠的是"我先设了 `dont_write_bytecode`"，
        而那只在**它先于 import 执行**时成立。这里从结果侧反查一次——
        如果 `sys.modules` 里已经有 `rllib`，说明那一下已经发生过了，
        `.pyc` 可能已经落进库内，而闸① 全程不会报错。
        """
        leaked = [m for m in FORBIDDEN_MODULES if m in sys.modules]
        if leaked:
            return Gate(
                "③ 不 import 导师库代码",
                False,
                f"sys.modules 里已有 {leaked} ——闸① 装晚了。"
                f"它可能已把 .pyc 写进库内，而闸① 不会报错。检查 "
                f"{self.root}/rllib/__pycache__/ 的 mtime。",
            )
        return Gate("③ 不 import 导师库代码", True, "sys.modules 干净")

    # ---------------------------------------------------------------- 闸④

    def begin_audit(self) -> dict[str, object]:
        self._before = audit(self.root)
        return self._before

    def end_audit(self) -> list[str]:
        """比对本次会话前后的库账。返回**差异列表**（空 = 一行没动）。"""
        if self._before is None:
            return ["（没拍过起始账——begin_audit 没被调用，这条核对等于没做）"]
        after = audit(self.root)
        diffs: list[str] = []
        for key in sorted(set(self._before) | set(after)):
            b, a = self._before.get(key), after.get(key)
            if b != a:
                diffs.append(f"{key}: {b} -> {a}")
        self._before = None
        return diffs

    # ---------------------------------------------------------------- 闸②

    def assert_no_hot_wal(self) -> None:
        """开只读连接**之前**的前置判据。

        `immutable=1` 会让 SQLite 无视 `-wal`。若 WAL 里还有没 checkpoint 的帧，
        我们会拿到**旧数据且毫无察觉**——本库签名病的原样复现。
        宁可不读，也不要读出一个看起来正常的旧答案。
        """
        wal = Path(str(self.index_sqlite) + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            raise StaleIndex(
                f"{wal.name} 有 {wal.stat().st_size} 字节未 checkpoint。"
                f"immutable=1 会无视它、静默读到旧数据。"
                f"先确认导师库没有任务在写（`python scripts/resume.py --status`），"
                f"或让它正常 checkpoint 之后再来。"
            )

    @contextmanager
    def index_conn(self) -> Iterator[sqlite3.Connection]:
        """打开 `data/index.sqlite` 的**真正一行不写**的只读连接。

        `mode=ro` 单独用是不够的（实测会建 `-wal`/`-shm`），必须带 `immutable=1`。
        """
        self.assert_no_hot_wal()
        uri = f"file:{self.index_sqlite.as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- 汇总

    def verify(self, *, strict: bool = True) -> list[Gate]:
        """跑一遍**开库前**能跑的闸（①③ + 前置判据），返回逐条结果。

        `strict=True` 时任何一条红都抛——**红了要停，不是记一笔继续**。
        闸④ 是**收尾**那条（它要比对关闭时的账），在 :meth:`begin_audit` /
        :meth:`end_audit` 里，不在这里。
        """
        gates: list[Gate] = [
            Gate(
                "① 禁用字节码写入",
                sys.dont_write_bytecode,
                "sys.dont_write_bytecode = True"
                if sys.dont_write_bytecode
                else "**没有装上** —— import 导师库会写 .pyc",
            ),
            self.gate_no_rllib(),
        ]
        try:
            self.assert_no_hot_wal()
            gates.append(Gate("② 前置：无未 checkpoint 的 WAL", True, "-wal 不存在或为空"))
        except StaleIndex as exc:
            gates.append(Gate("② 前置：无未 checkpoint 的 WAL", False, str(exc)))

        if strict:
            red = [g for g in gates if not g.ok]
            if red:
                raise LibraryTouched("只读网关自检未通过：\n" + "\n".join(str(g) for g in red))
        return gates

    # ---------------------------------------------------------------- 上下文

    def __enter__(self) -> LibraryView:
        self.begin_audit()
        self.verify()
        return self

    def __exit__(self, *exc_info: object) -> None:
        diffs = self.end_audit()
        if diffs:
            raise LibraryTouched(
                "本次会话改动了导师库（闸④ 拦下）：\n  " + "\n  ".join(diffs)
            )
