"""覆盖单元：拿一篇现成综述的分类学当外部锚。

**这一层不设计分类学，只搬运它。** 理由见 :mod:`dasc.coverage.taxonomy` 的
模块 docstring——自己设计的分类学会让"覆盖得够不够"变成自说自话。

对外暴露两类：

- **抽结构**：:class:`Taxonomy` / :func:`load_taxonomy`（唯一的入口）、
  :data:`AXES`（`2509.02547` 那四根轴的定义）。
- **判来源**：:func:`human_axes`（读**人判**的轴名单）、:func:`judged_sources`
  （查漏用）、:data:`AXES_BY_HAND`、:func:`describe_structure` /
  :func:`cross_check_sources`。

⚠️ **"哪几根枝是轴"机器判不了**（负向对照现量：普通论文里 53% / 90% 会命中，
见 `checks/check_axes.py`），所以它由人判、落盘、署名——见 :func:`human_axes`
的 docstring。
"""

from __future__ import annotations

from .taxonomy import (
    AXES,
    AXES_BY_HAND,
    DEFAULT_SURVEY,
    DEFAULT_VAULT,
    SECOND_SURVEY,
    Node,
    Shape,
    Taxonomy,
    cross_check_sources,
    describe_structure,
    human_axes,
    judged_sources,
    load_taxonomy,
    second_source_hint,
    survey_sources,
)

__all__ = [
    "AXES",
    "AXES_BY_HAND",
    "DEFAULT_SURVEY",
    "DEFAULT_VAULT",
    "SECOND_SURVEY",
    "Node",
    "Shape",
    "Taxonomy",
    "cross_check_sources",
    "describe_structure",
    "human_axes",
    "judged_sources",
    "load_taxonomy",
    "second_source_hint",
    "survey_sources",
]
