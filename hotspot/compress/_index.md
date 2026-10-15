---
tags: [索引]
---

# hotspot/compress · `_index`（L3 压缩层）

## 1. 这个文件夹是什么

L3 压缩表示层：每簇一张 `ClusterCard`。**★ 唯一花钱处**（LLM 只命名、抄客观字段、强制溯源）。

## 2. 当前工作状态

- 代码 ✅ 已写（导入 + validator 单测通过）
- 生成卡片 ⏸ 待 `workspace/cards` 数据 + `DEEPSEEK_API_KEY`

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无（卡片产物在 `workspace/cards/`）。

## 脚本速查

- `cluster_card.py` —— L3 生成每簇 ClusterCard：`evidence_ids` 必须取自输入、断点续跑、失败降级。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `cluster_card.py`

## 5. 相关

- 上位指针：`hotspot/_index.md`
- 口径：`../../规划/热点挖掘系统-方案设计.md` §3.3
