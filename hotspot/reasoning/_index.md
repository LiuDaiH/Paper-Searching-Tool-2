---
tags: [索引]
---

# hotspot/reasoning · `_index`（L5 推理层）

> **给 AI**：先读 `answer.py` → `verify.py`；四态定义 grep `schema.py` 的 `class Verdict`，别整篇读 schema。

## 1. 这个文件夹是什么

L5 推理层：带引用生成答案 + 引用校验（防幻觉最后防线）。

## 2. 当前工作状态

- 代码 ✅ 已写（parse/classify 单测通过，裸 id 约定一致）
- 端到端 ⏸ 待 LLM key + 数据

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无。

## 脚本速查

- `answer.py` —— L5 带引用生成答案（强制 `[P:id]` / `[C:id]`）。
- `verify.py` —— L5 引用校验：missing_id / unsupported 分开判，防幻觉。
- `evidence.py` —— L5.5 可答性证据：core_concepts/anchors 匹配、摘要片段、answer/insufficient/clarify/out_of_scope 决策。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `verify.py`（防幻觉关键）

## 5. 相关

- 上位指针：`hotspot/_index.md`
- 口径：`../../规划/2026-09-21-热点挖掘系统-方案设计.md` §3.5
