# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-22

## 当前目标

算法 v2 重置。旧检索、聚类、路由和 custom harness 已删除；下一步按目标重新组装
成熟工具、Controller 与评测。

## 基线

- Git 快照：`b517988`（旧实现可完整恢复）。
- 当前分支：`codex/agent-v2-reset`。
- `main` 保留删除前可运行版本。

## 已保留

- 论文数据、metadata、1024 维 embedding 与只读语料层。
- `hotspot.api` 契约、HTTP/SSE、前端适配。
- `reasoning/evidence.py`、`answer.py`、`verify.py`。
- `structure/embed.py`、`venue_matrix.py`。
- 问题集与 evidence/verify fixtures。
- Trace 汇总与发布/服务入口。

## 已删除

- 旧 `subcluster_adaptive`、direction、BM25/dense/RRF/MMR/router/rewrite/vector。
- 旧 global coverage、nearest-centroid coarse-to-fine。
- rlRAG/RF/MPS/共引/依赖抽取/旧聚类卡片。
- custom harness、旧 G0–G3 执行器与检索 benchmark。

## 当前状态

- `Engine.meta` 可用；`Engine.ask` 明确返回“算法 v2 重建中”。
- API 代码可导入，旧算法依赖已从当前分支移除。
- 新检索后端、工具 schema、Controller、评测器尚未建立。

## 下一步

1. 明确 v2 目标产品和验收指标。
2. 选择检索后端并只在固定查询集上验证。
3. 引入成熟 agent/tool 框架，不再自研 registry。
4. 建立最小纵链：search → fetch → evidence → answer → verify。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
```

旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
