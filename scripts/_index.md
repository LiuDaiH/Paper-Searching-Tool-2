---
tags: [索引]
---

# scripts · `_index`

## 1. 这个文件夹是什么

运行入口、检查脚本和数据维护脚本。旧检索/大创算法专用脚本已删除。

## 2. 当前工作状态

- 服务入口、数据 embedding/周报更新和接口检查保留。
- 检索评测、G0–G3 评测和单一检索链脚本将在 v2 工具完成后再建。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `check_access.py` —— 验证导师库只读接入。
- `check_tree.py` —— 索引树与脚本速查自检。
- `check_api.py` —— 契约与 HTTP/SSE 接口自检。
- `check_tools_v2.py` —— LanceDB 标题探测与论文读取工具离线自检。
- `build_lance_index.py` —— 从现有 BGE-M3 向量分片构建 LanceDB 论文索引。
- `run_server.py` —— HTTP 服务入口。
- `run_ui.py` —— Streamlit 入口。
- `cost_report.py` —— 成本报表。
- `export_arxiv_weekly.py` —— arXiv 周报向量与发布物生成。
- `update_arxiv_weekly.py` —— arXiv 周报增量更新。
- `summarize_trace.py` —— Trace JSONL 汇总，不输出问题/答案原文。
- `build_dist.py` —— 服务端源码打包。
- `build_release.py` —— 发布目录组装。

## 4. 建议阅读顺序

1. `check_api.py`
2. `run_server.py`
3. `check_tree.py`

## 5. 相关

- 上位指针：`../README.md`
- 重建前实现：Git commit `b517988`
