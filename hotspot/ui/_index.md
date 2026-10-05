---
tags: [索引]
---

# hotspot/ui · `_index`（L6 交互层）

## 1. 这个文件夹是什么

L6 交互层：Streamlit 单页问答（端到端编排）。

## 2. 当前工作状态

- 代码 ✅ 多会话前端已完成并验证（`streamlit run scripts/run_ui.py --server.port 8502`）
- 演示 ✅ dry-run 问 RLHF 拿到带 `[C:]/[P:]` 答案；会话历史持久化到 `workspace/exports/sessions.json`

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无。

## 脚本速查

- `app.py` —— L6 Streamlit 多会话问答：**数据源切换（会议 24+25 / arXiv 周报）**、新对话/追问（基于前文不重检索）/会话历史持久化（`sessions.json`）/运行态 API 配置（不落盘）/dry-run 开关→路由→装配→答案→校验→展示。
- 启动须用 `scripts/run_ui.py`（streamlit 唯一正确入口，sys.path 接线），**勿直接 `streamlit run app.py`**（相对导入报错）。

## 4. 建议阅读顺序

1. `app.py`

## 5. 相关

- 上位指针：`hotspot/_index.md`
- 口径：`热点挖掘系统-方案设计.md` §3.6
