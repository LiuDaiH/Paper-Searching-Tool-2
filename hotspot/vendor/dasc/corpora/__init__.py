"""语料接入层。

窄接口的**唯一实现**是 :mod:`dasc.corpora.conference`（接导师库）。
将来 2024/25 arXiv 层接进来时是**新增一个实现**，不是改这个——那正是
「接口留好、实现欠着」的意思（`大创执行步骤` §0.3）。

```python
from dasc.corpora import Corpus
with Corpus() as c:
    c.list_papers(pool="rlhf")[:3]
```
"""

from __future__ import annotations

from .conference import Availability, Citations, Corpus, FullText, Paper, Src

__all__ = ["Availability", "Citations", "Corpus", "FullText", "Paper", "Src"]
