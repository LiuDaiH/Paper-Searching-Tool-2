---
tags: [索引]
---

# hotspot/structure · `_index`

## 1. 这个文件夹是什么

离线论文表示和会议统计。旧聚类实现已删除，embedding 与三态会议矩阵保留。

## 2. 当前工作状态

- `embed.py`：1024 维论文/查询 embedding，作为新检索栈的数据基础。
- `venue_matrix.py`：跨会扩散三态，热点口径的唯一客观统计。
- 新主题模型在 v2 中重新选型，不恢复旧 `cluster.py`。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无（产物在 `workspace/vectors|metrics/`）。

## 脚本速查

- `embed.py` —— BGE-M3 分块向量化，可续跑。
- `venue_matrix.py` —— 铺开 / 新进 / 单会三态计数。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `venue_matrix.py`
2. `embed.py`

## 5. 相关

- 上位指针：`../_index.md`
