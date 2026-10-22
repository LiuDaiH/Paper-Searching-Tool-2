# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-22

## 当前目标

算法 v2 paper-first 纵链已搭起：LanceDB search/fetch → PydanticAI agent →
mandatory verify。下一步补工具级基准与 global topic 工具。

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

- LanceDB 全量索引已建立：22,747 篇、1024 维、vector + FTS。
- `Engine.ask` 已接 PydanticAI agent；真实 DPO 问题跑通并返回带引用答案。
- Agent 可选 `search_papers` / `fetch_papers`；evidence 输出必须有 fetched ID。
- 输出后强制 `verify_and_render`，未通过引用校验的 claim 不进入答案。
- 5 条精确标题探测 5/5 命中；更完整工具基准待补。

## 下一步

1. 扩大离线工具基准，校准 LanceDB 召回质量。
2. 增加 scope/QueryPlan 的硬约束与 OOD 评测。
3. 增加离线 topic snapshot 与 `get_topic_overview`。
4. 将工具事件写入可回放 Trace。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
```

旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
