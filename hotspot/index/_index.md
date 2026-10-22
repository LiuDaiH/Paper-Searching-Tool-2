---
tags: [索引]
---

# hotspot/index · `_index`

## 1. 这个文件夹是什么

离线论文索引和在线候选召回。当前使用 LanceDB 的 vector + FTS 混合检索。

## 2. 当前工作状态

- `lance_store.py` 是唯一检索后端边界。
- `topic_store.py` 只读取离线 topic snapshot，不在线聚类。
- 索引从已有 1024 维 BGE-M3 分片构建，不重新编码全库。
- 工具层不直接访问 LanceDB。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无（LanceDB 在 `workspace/lance/`）。

## 脚本速查

- `lance_store.py` —— LanceDB 构建、论文读取、vector+FTS 混合召回。
- `topic_store.py` —— 读取离线 cluster metrics，提供 topic snapshot 查询。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `lance_store.py`

## 5. 相关

- 上位指针：`../_index.md`
