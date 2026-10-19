---
tags: [索引]
---

# hotspot/structure · `_index`（L2 结构层）

> **给 AI**：先读 `venue_matrix.py`（三态判定）→ `embed.py`；`cluster.py` 只在调参/换聚类时读。

## 1. 这个文件夹是什么

L2 结构层：把 13,119 篇摘要压成「哪些题多大、铺得多开、彼此多近」——**0 token**。

## 2. 当前工作状态

- 代码 ✅ 已写（分块向量化 / 聚类 / 跨会三态，导入验证通过）
- 跑数 ⏸ 待导师库恢复（探针 1–3、探针 6）

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无（产物在 `workspace/vectors|clusters|metrics/`）。

## 脚本速查

- `embed.py` —— L2 分块向量化：bge-m3 全库向量化，**可续跑**（R1 唯一对策）。
- `cluster.py` —— L2 聚类：PCA 白化 + HDBSCAN，产出带跨会三态的 `ClusterMetric` 与 `cluster_id→members`。
- `venue_matrix.py` —— L2 跨会扩散矩阵：纯计数算「铺开/新进/单会」三态。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `venue_matrix.py`（护城河核心的三态判定）
2. `embed.py` → `cluster.py`（数据流）

## 5. 相关

- 上位指针：`hotspot/_index.md`
- 口径：`../../规划/热点挖掘系统-方案设计.md` §3.2
