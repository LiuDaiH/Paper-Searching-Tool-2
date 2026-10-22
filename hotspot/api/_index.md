---
tags: [索引]
---

# hotspot/api · `_index`

## 1. 这个文件夹是什么

前端与算法之间唯一的稳定接口面：`meta` + `ask`，JSON / SSE。
2026-10-22 起算法 v2 重建，旧编排已移除，保留契约与 HTTP 外壳。

## 2. 当前工作状态

- 契约 v1 保留；前端仍可依赖请求/响应形状。
- `engine.py` 当前是运行壳：`meta` 可用，`ask` 明确返回重建中错误。
- 新 Controller/工具链接入时只替换 `Engine.ask` 内部。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `contract.py` —— 请求/响应 DTO、参数白名单、错误码。
- `engine.py` —— 稳定 API 壳，等待 v2 agent 与工具链。
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
