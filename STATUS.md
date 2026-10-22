# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-22

## 当前目标

算法 v2 paper-first 与离线 topic snapshot 已接通；当前重点是扩大语义 qrels，
再决定是否接入 Jev reranker。

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
- `Engine.ask` 已接 PydanticAI agent；范围内问题可返回带引用答案。
- Agent 可选 `search_papers` / `fetch_papers` / `get_topic_overview`；
  evidence 输出必须有 fetched ID。
- 输出后强制 `verify_and_render`，未通过引用校验的 claim 不进入答案。
- 32 条精确标题探测 32/32；v2 工具集 Recall@10=0.291、Recall@50=0.558。
- 5 条 OOD smoke 5/5 拒答；全局趋势问题会调用离线 topic snapshot。
- Trace v2 已写入 `workspace/ledger/retrieval_trace.jsonl`。

## 下一步

1. 扩充语义检索 qrels，校准 LanceDB 召回质量。
2. 运行覆盖更多 local/compare/relation/OOD 的真实 agent 回归。
3. Jev reranker 实验阻塞：本机模型权重和 API 凭据当前都不可用。
4. 评估 topic snapshot 的可读性，再决定是否需要新主题模型。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
```

旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
