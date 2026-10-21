---
tags: [索引]
---

# hotspot/ui · `_index`（L6 交互层 · Streamlit 适配）

> **给 AI**：只需读 `app.py` 的 `main` 与 `_render_assistant`；接口细节不在这里，去 `../api/`。

## 1. 这个文件夹是什么

**Streamlit 渲染适配**：只负责画页面、存会话、存本机 API 配置。
问答/检索/聚类/校验/图谱/token 用量全部来自 `hotspot/api`（服务层），
这里**不 import 任何算法模块**。

新前端（HTML/CSS/JS）走 `hotspot/api/http.py`，与本目录无关——两边共用同一个引擎。

## 2. 当前工作状态

- 代码 ✅ `app.py` 已瘦身为纯渲染（原 844 行里约 500 行编排搬进 `hotspot/api/engine.py`）
- 代码 ✅ 会话/配置存储拆到 `store.py`（等账号系统接管，替换时不用动服务层）
- 验证 ⏸ 需真数据（G 盘语料或 `测试数据集/`）跑一次完整问答
- 说明：Streamlit 仍是**过渡期**前端；正式前端在 `web/`（参考实现）+ 同学的账号系统

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无（运行时写 `workspace/exports/sessions.json`、`workspace/settings.json`，gitignore）。

## 脚本速查

- `app.py` —— Streamlit 单页问答（**只渲染**）：会话列表、数据源切换、API 配置、进度面板、主题簇/引用/校验/路径图展示。
- `store.py` —— 前端本地存储：`settings.json`（API 配置）与 `exports/sessions.json`（多会话历史），迁移到 MySQL 时只换这一层。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `app.py`（看它怎么用 `Engine`：一次 `ask` + 一组 stage 事件，没有算法细节）
2. `store.py`（会话/配置落盘格式）
3. 需要改接口时去 `../api/_index.md`，不要在这里加编排

## 5. 相关

- 上位指针：`../_index.md`
- 服务层：`../api/_index.md`
- 契约：`../../规划/2026-10-18-接口契约-前端与算法.md`
- 启动：`scripts/run_ui.py`（Streamlit）/ `scripts/run_server.py`（HTTP）
