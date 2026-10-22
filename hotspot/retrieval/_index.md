---
tags: [索引]
---

# hotspot/retrieval · `_index`

## 1. 这个文件夹是什么

检索层。旧 BM25、向量召回、RRF/MMR、路由、改写和 coverage 已于 2026-10-22
从 v2 分支删除，等待新检索工具接入。

## 2. 当前工作状态

- 当前只保留上下文装配 `assemble.py`。
- 新检索实现应通过独立工具接入，不在本目录恢复旧的 `subcluster_adaptive` 单函数编排。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `assemble.py` —— 将候选证据装配为带引用约束的模型上下文。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `assemble.py`

## 5. 相关

- 上位指针：`../_index.md`
- 重建前实现：Git commit `b517988`
