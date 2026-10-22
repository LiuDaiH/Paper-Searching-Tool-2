---
tags: [索引]
---

# hotspot/observability · `_index`

## 1. 这个文件夹是什么

Agent 请求级 append-only Trace。记录问题、工具调用、验证状态和结果摘要，
不记录 API key、摘要正文或完整答案。

## 2. 当前工作状态

- Trace v2 写入 `workspace/ledger/retrieval_trace.jsonl`。
- 工具事件由 `AgentDeps.events` 提供。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无（Trace 属于可重建运行产物）。

## 脚本速查

- `trace.py` —— append-only Trace 写入与答案文本脱敏。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `trace.py`

## 5. 相关

- 上位指针：`../_index.md`
