# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-22

## 当前目标

算法 v2 paper-first、离线 topic snapshot、契约 v2 和完整 HTTP 演示已跑通。
当前不继续改检索算法，重点转向稳定性、超时归因和前端联调。

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
- Silver qrels：7 查询、367 候选、0 缺失；grade 0/1/2 = 83/20/264；
  其中 hard negatives + random negatives 已覆盖。
- Silver 工具基线：Recall@10=0.2115、Recall@50=0.8316。
- 契约 v2：旧参数已删除，仅保留 5 个 Agent/回答参数。
- `meta` 已记录 PydanticAI、LanceDB、topic snapshot、trace 能力和 arXiv 索引缺失提示。
- usage 已映射模型请求数、输入/输出 token 与耗时；成本仍为 0（待价格表）。
- 16 题验收：9 条正常带引用回答、3 条 OOD 拒答、1 条范围内拒答、3 条超时。
- 完整 HTTP 演示：DPO 返回 3 supported claims / 3 citations；
  158.036s、4 次模型请求、11032 输入 token、1497 输出 token。

## 下一步

1. 排查 3 条超时和 RLOO 范围内拒答；不改检索召回策略。
2. 完善超时/重试与错误分层，降低 P95 延迟。
3. 前端按 contract v2 联调，验证来源禁用与 usage 展示。
4. 提交当前契约和验收改动；Jev 仍因模型/凭据缺失阻塞。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
```

旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
