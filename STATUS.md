# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、已完成、未完成和固定验收命令。

更新：2026-10-22

## 当前目标

按 10.21 计划实施修正：先完成工具成熟度与边界验收，再冻结 `ToolSpec`/Controller；
当前固定流水线只作 provisional baseline。

## 已完成

- `hotspot/retrieval/coverage.py`：稳定粗簇 + 最近质心归属、趋势题粗到细选择。
- `evidence.py`：核心领域证据硬门；泛词不能单独放行；anchor 仍为实体硬约束。
- `subcluster.py`：趋势题消费粗簇；事实/比较/关系保留各自路径。
- `engine.py`：响应 `meta.retrieval` 记录覆盖策略与观测。
- 开发减负：`check_api.py --quiet`、评测每题单行、`summarize_trace.py`、
  `STATUS.md` 与低成本会话规则。
- 10.21 计划已补充“harness 不是起点”：工具成熟度、离线工具基准和
  `paper_search → evidence_gate → fetch_papers` 最小纵链优先。
- 已清理失败产物与 `__pycache__`，移除旧 `.zhanlu`，并把 42 个历史 smoke
  产物归档到 `workspace/ledger/archive/2026-10-22-smoke-pre-harness/`。
- P0 工具基线：内部 contract、golden fixtures、覆盖审计、paper-search 基线已落；
  矩阵与结果见 `规划/2026-10-22-工具成熟度与离线基准.md`。
- P1 边界拆分：`paper_search / candidate_filter / coarse_cluster_select /
  local_cluster / evidence_gate / fetch_papers` 已成为私有 adapter 并接回主路径。
- `规划/` 已压缩为 7 份活跃文档；已完成汇报、旧测试计划、旧改造路线、
  Linux 迁移备忘和 arXiv 图增强专项设计已删除，必要结论已并入保留文档。
- 离线 `check_api` 全绿；真实回归 `g02` 通过，覆盖策略为 `coarse_then_fine`。

## 未完成

- `paper_search` 未成熟：Recall@50=0.469，平均 11.73s。
- 全库覆盖未成熟：稳定归属 10.53%，46.59% 的 nearest 候选 margin<0.02；
  `coarse_then_fine` 在 offline RL 冒烟中回退 `weak_dominant_topic`。
- 修复上述两项后才允许最小纵链 `version: 1.0` 和 registry。
- 小型真实回归只作验收样本，不再驱动逐题规则修改；不跑 G1。

## 固定验收

```powershell
python -m py_compile hotspot/retrieval/coverage.py hotspot/reasoning/evidence.py hotspot/subcluster.py hotspot/api/engine.py scripts/check_api.py eval/run_eval.py scripts/summarize_trace.py
python scripts/check_api.py --quiet
python scripts/check_tools.py
python scripts/check_tree.py --vault .
python scripts/audit_coverage.py
python scripts/summarize_trace.py workspace/ledger/coverage_gate_v2_trace.jsonl
```

真实模型调用必须走允许联网的执行方式；普通沙箱会把端点拒绝成空回答。
旧 Trace 的单次 emit 阶段延迟会显示为 0；新 Trace 已按相邻阶段结束时间修正。
