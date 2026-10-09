"""两个具体的问法：**给论文打覆盖标签**、**给 broad 锚判类型**。

⚠️ 标签**从分类学里取，不是我编的**
------------------------------------

贡献标签的取值必须来自 2a 抽出的四轴分类学（:mod:`dasc.coverage.taxonomy`），
**不是我在这里硬写一张 enum**。理由很直接：硬写的话，标签集和覆盖单元
（三轴直接下级的笛卡尔积，392 格）就会各说各话，而覆盖度是拿后者当分母的——
**两套口径一起用，谁都不知道漏在哪**。所以 :func:`contribution_schema`
从 `taxonomy.children_of()` 现取。

⚠️ 但本库记过一条：**「哪些变了」≠「变成了什么」**（`changed-set-vs-changed-content`）。
分类学一改，标签集就变，旧结果**不可比**。所以每次问都记下分类学来源文件的
sha256（:func:`taxonomy_fingerprint`），随结果落盘。

9 个 broad 锚是**算出来的，不是我挑的**
----------------------------------------

`probe/t3_pool_anchors.json` 里 19 个池各有一个 `top30`。
把出现在 **≥10 个池**里的挑出来，**恰好 9 个**（2026-09-20 实测）：

| arXiv | 出现在几个池 | 被引 |
|---|---|---|
| 1707.06347 | 17/19 | 55 |
| 2303.08774 | 14/19 | 36 |
| 2407.21783 | 12/19 | 34 |
| 2312.11805 | 12/19 | 17 |
| 1312.5602 | 12/19 | 6 |
| 2204.05862 | 11/19 | 50 |
| 2307.09288 | 10/19 | 33 |
| 1909.08593 | 10/19 | 29 |
| 2110.14168 | 10/19 | 24 |

⚠️ 阈值放低到 9 就变成 **11 个**（多出 `2402.03300`、`2005.01643`）。
**这个"9"是阈值切出来的，不是数据里天然有 9 个。**改阈值要一起改这句话。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..coverage.taxonomy import DEFAULT_VAULT, Taxonomy, load_taxonomy

__all__ = [
    "ANCHORS_JSON", "ANCHOR_THRESHOLD", "ANCHOR_KINDS",
    "broad_anchors", "taxonomy_fingerprint",
    "contribution_schema", "contribution_prompt",
    "anchor_schema", "anchor_prompt",
]

ANCHORS_JSON: Path = (DEFAULT_VAULT / "科研阅读路径" / "probe" / "t3_pool_anchors.json")

#: 出现在这么多个池里的锚，才算「broad」。见模块 docstring：这个数**是切出来的**。
ANCHOR_THRESHOLD = 10

#: 锚的类型。**五选一**，对应计划 C 那句「概念前置 vs 工具/数据集/基线/通用组件」。
ANCHOR_KINDS = ("概念前置", "工具", "数据集", "基线", "通用组件", "说不清")

SYSTEM_ZH = (
    "你是一个严谨的科研论文分类助手。你只根据给出的标题与摘要判断，"
    "不许引入外部知识猜测论文内容。判断不了就选「说不清」，"
    "**不要为了给出答案而勉强归类**。所有输出必须是合法 JSON，不要任何解释性前后缀。"
)


# ---------------------------------------------------------------------- #
# 分类学指纹
# ---------------------------------------------------------------------- #

def taxonomy_fingerprint(tax: Taxonomy | None = None) -> str:
    """分类学来源文件的 sha256 前 12 位。**标签集一变，它就变。**"""
    p = (tax.source_path if tax is not None else DEFAULT_VAULT / "科研阅读路径" / "probe"
         / "survey_cache" / "2509.02547.src")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------- #
# broad 锚
# ---------------------------------------------------------------------- #

def broad_anchors(threshold: int = ANCHOR_THRESHOLD,
                  path: Path | None = None) -> list[dict]:
    """跨池出现 ≥ `threshold` 次的锚，按出现次数降序。

    ⚠️ 这是**冻结产物**（2026-09-15 的 T3 探针跑出来的），不是现算的。
    它记的是当时那张 `.src` 快照上的引用图——所以它也必须进快照，
    否则报告里的锚点会随抓取漂（`B5` 那条）。
    """
    d = json.loads((path or ANCHORS_JSON).read_text(encoding="utf-8"))
    counts: dict[str, dict] = {}
    for pool, v in d["pools"].items():
        for a in v.get("top30", []):
            e = counts.setdefault(
                a["arxiv_id"], {"arxiv_id": a["arxiv_id"], "pools": 0,
                                "max_cited_by": 0, "in": []})
            e["pools"] += 1
            e["in"].append(pool)
            e["max_cited_by"] = max(e["max_cited_by"], a.get("cited_by", 0))
    out = [e for e in counts.values() if e["pools"] >= threshold]
    out.sort(key=lambda e: (-e["pools"], -e["max_cited_by"]))
    return out


def anchor_schema() -> dict:
    return {
        "kind": f"五选一：{'/'.join(ANCHOR_KINDS)}",
        "confidence": "0 到 1 之间的小数，表示你对这个判断的把握",
        "rationale": "一句中文，说明**摘要里的哪句话**支持这个判断",
    }


def anchor_prompt(title: str, abstract: str, *, n_pools: int, cited_by: int) -> str:
    """锚类型判定。**输入的只有标题与摘要**（出境检查会挡住正文）。"""
    return (
        f"下面这篇论文被 {n_pools} 个子方向的论文共同引用（池内被引 {cited_by} 次），"
        f"属于跨子方向的「broad 锚」。\n\n"
        f"请判断它在这些论文里扮演的角色，五选一：\n\n"
        f"- **概念前置**：它提出的是这些工作共同依赖的概念/框架/问题设定"
        f"（如 PPO 之于 RLHF、Transformer 之于一切）\n"
        f"- **工具**：它提供的是被反复使用的**方法或实现**（如某个优化器、某个采样技巧）\n"
        f"- **数据集**：它主要贡献是数据/评测集\n"
        f"- **基线**：它主要作为被比较的对象存在\n"
        f"- **通用组件**：它是个通用部件，子方向并不真正依赖它的思想"
        f"（如某个通用语言模型、某个库）\n"
        f"- **说不清**：摘要信息不足以判断\n\n"
        f"标题：{title}\n\n摘要：{abstract}"
    )


# ---------------------------------------------------------------------- #
# 覆盖标签
# ---------------------------------------------------------------------- #

def _axis_options(tax: Taxonomy, axis: str) -> dict[str, str]:
    """一根轴的可选值：`{label: 标题}`。**从分类学现取。**"""
    return {n.label: n.title for n in tax.children_of(axis)}


def taxonomy_axes(tax: Taxonomy | None = None) -> dict[str, dict[str, str]]:
    tax = tax or load_taxonomy()
    return {
        "position": _axis_options(tax, "sec:position"),
        "capability": _axis_options(tax, "sec:capability"),
        "task": _axis_options(tax, "sec:task"),
    }


def contribution_schema(tax: Taxonomy | None = None) -> dict:
    axes = taxonomy_axes(tax)
    return {
        "position": {f"列表，取值必须来自：{sorted(v)}": "可多选；都不沾就空列表 []"}
        for k, v in axes.items()
    } | {
        "coverage": "字符串：'命中'（至少一个轴非空）/ '未覆盖'（三个轴全空）/ '说不清'",
        "confidence": "0 到 1 之间的小数",
        "rationale": "一句中文，说明**摘要里的哪句话**支持这个判断",
    }


def contribution_prompt(title: str, abstract: str, tax: Taxonomy | None = None) -> str:
    axes = taxonomy_axes(tax)
    lines = [
        "下面这篇论文要被打上**覆盖标签**。标签取值来自一篇综述的分类学，逐轴给：\n",
    ]
    for axis, opts in axes.items():
        lines.append(f"**{axis}**（可多选，不沾就 []）：")
        for lab, t in opts.items():
            lines.append(f"  - `{lab}` = {t}")
        lines.append("")
    lines += [
        "规则：",
        "1. **只根据标题与摘要判断**，摘要里没提到的轴就留空列表"
        "——空不是「没归类」，是「这篇没做这个轴」。",
        "2. 三个轴**互相独立**（跨轴重叠是允许的），不要为了凑而选。",
        "3. `coverage` 只有在**三个轴全空**时才填 `未覆盖`。",
        "",
        f"标题：{title}",
        "",
        f"摘要：{abstract}",
    ]
    return "\n".join(lines)


def contribution_validator(tax: Taxonomy | None = None):
    """校验标签**真的来自那根轴**。**不合规就重试，不静默接受。**"""
    axes = taxonomy_axes(tax)

    def check(data: dict, raw: str) -> None:
        for axis, opts in axes.items():
            got = data.get(axis, [])
            if not isinstance(got, list):
                raise ValueError(f"{axis} 不是列表，是 {type(got).__name__}")
            bad = [x for x in got if x not in opts]
            if bad:
                raise ValueError(
                    f"{axis} 里有不在分类学里的标签：{bad}（合法值 {sorted(opts)}）")
        cov = data.get("coverage")
        if cov not in ("命中", "未覆盖", "说不清"):
            raise ValueError(f"coverage 取值非法：{cov!r}")
        empty = all(not data.get(a) for a in axes)
        if cov == "命中" and empty:
            raise ValueError("coverage 说「命中」但三根轴全是空的——自相矛盾")
        if cov == "未覆盖" and not empty:
            raise ValueError(f"coverage 说「未覆盖」但有轴非空："
                             f"{ {a: data.get(a) for a in axes} }")
        if len(str(data.get("rationale", ""))) < 4:
            raise ValueError("rationale 太短，没有说明依据")

    return check
