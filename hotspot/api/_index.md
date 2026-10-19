---
tags: [索引]
---

# hotspot/api · `_index`（L6.5 服务层 / 接口面）

> **给 AI**：改接口只读 `contract.py`（形状）+ `engine.py` 的 `Engine.ask`；
> `http.py` 只在改传输/端口时读。字段表在 `../../规划/接口契约-前端与算法.md`。

## 1. 这个文件夹是什么

**前端与算法之间唯一的接口面**。把 L1–L5 的编排收在 `engine.py`，把对外形状收在
`contract.py`，把传输收在 `http.py`。前端（HTML/JS 或 Streamlit）只依赖这里。

对外只有两个业务方法 / 两个 HTTP 路由：`meta` + `ask`。字段表见
`规划/接口契约-前端与算法.md`（**改接口先改那份文档**）。

## 2. 当前工作状态

- 契约 v1 ✅ 落地（`AskRequest` / `AskResponse` / `meta` / 13 个白名单参数）
- 引擎 ✅ 编排已从 `hotspot/ui/app.py` 搬入（意图路径 / 兜底路径 / 追问 / 图谱 / 用量）
- HTTP ✅ 标准库实现（含 SSE 进度、静态托管、CORS），**零第三方依赖**
- 自检 ✅ `python scripts/check_api.py` 全绿（契约往返 / 参数校验 / HTTP / SSE）
- 样例数据回归 ✅ 2026-10-19：`测试数据集/config.yaml` + 真实 `POST /api/ask` 返回 200，
  `topics.cluster_id` 与答案 `[C:x]` 对齐；追问 `citation_ids` 已实测
- 待接：账号系统（前端）、全量 G 盘语料回归

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无。

## 脚本速查

- `contract.py` —— 接口契约 v1：请求/响应 DTO、参数白名单 `PARAM_SPEC`、错误码、进度事件；**只增不改**。
- `engine.py` —— 服务层引擎：`Engine.meta()` / `Engine.ask()`，把 L1–L5 串成一次问答（无会话状态）。
- `http.py` —— HTTP 适配：标准库 `http.server`，`GET /api/meta`、`POST /api/ask`（JSON / SSE）、`/api/health`、静态托管。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

新会话接手接口：**只读三份**，不用读算法内部。

1. `../../规划/接口契约-前端与算法.md`（字段表 + 两侧接手指南）
2. `contract.py`（真实形状与校验）
3. `engine.py` 的 `ask()` / `meta()`（编排入口，内部细节可跳过）

## 5. 相关

- 上位指针：`../_index.md`
- 契约文档：`../../规划/接口契约-前端与算法.md`
- 前端适配：`../ui/app.py`（Streamlit）、`scripts/run_server.py` + `web/`（HTTP）
- 自检：`scripts/check_api.py`
- 内部数据契约：`../schema.py`（L1–L5 的口径，**不是**对前端的那一份）
