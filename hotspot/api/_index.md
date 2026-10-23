---
tags: [索引]
---

# hotspot/api · `_index`

## 1. 这个文件夹是什么

前端与算法之间唯一的稳定接口面：`meta` + `ask`，JSON / SSE。
2026-10-22 起算法 v2 已接入 PydanticAI agent、LanceDB 工具和强制引用校验。

## 2. 当前工作状态

- 契约 v1 保留；前端仍可依赖请求/响应形状。
- `engine.py` 将 `AskRequest` 转为 request-scoped agent 运行，再执行强制 grounding。
- dry-run 暂不生成答案；真实问答需要 LLM key 与 `workspace/lance/`。
- scope / agent / verify 使用独立超时；错误码区分 timeout、connection 和 output-invalid。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `contract.py` —— 请求/响应 DTO、参数白名单、错误码。
- `engine.py` —— API ↔ PydanticAI agent 适配与强制证据/引用校验。
- `http.py` —— 标准库 HTTP、SSE、静态托管。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `contract.py`
2. `engine.py`
3. `http.py`

## 5. 相关

- 上位指针：`../_index.md`
- 接口契约：`../../规划/2026-10-18-接口契约-前端与算法.md`
- 重建前实现：Git commit `b517988`
