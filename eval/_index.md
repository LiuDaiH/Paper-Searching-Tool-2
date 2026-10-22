---
tags: [索引]
---

# eval · `_index`

> **给 AI**：只需读 `run_eval.py` 的 CLI 段 + `questions*.json` 的前几条；其余是评测数据。

## 1. 这个文件夹是什么

评测集与评测脚本（方案 §5）：2 基线 + 3 消融 + 跨会留出验证。

## 2. 当前工作状态

- 评测框架 ✅ 已升级为 **G0–G3 大创开关 + `llm_eval`（kimi-k3）+ 成对对比**（`run_eval.py` + `eval_compare.py`，2026-09-30）。
- **问题集 ✅ 已建 70 条**（`questions.json`：global 20 / local 20 / refuse 20 / relation 10）。
- **跨会留出 ✅ baseline 已跑**（`holdout.py`：venue 切分 + 随机/高频对照 P@10/Spearman，0 token）；
  **subcluster 主题预测**（需向量/内存/LLM）待接入。
- **2026-10-20 P0 迁移**：`run_eval.py` 已从旧 `ui.app.run_qa` 改为 `hotspot.api.Engine.ask`；
  评测走统一契约，`usage.stages` 保留每阶段 start/end。
- **2026-10-20 P1-1**：`retrieval_eval.py` + `retrieval_questions.json`；样例
  BM25/vector/hybrid 对比，hybrid 调参后 nDCG@10=0.788、Recall@20=0.833。

## 3. 文件清单

**计划文件**：

- `questions.json` —— **正式问题集**（70 条：global/local/refuse 各 20 + relation 10），`run_eval.py` 默认读取。
- `questions.example.json` —— 问题集模板（global / local / refuse 三类示例）
- `retrieval_questions.json` —— **P1-1 检索评测集**：查询 + 相关论文 id（样例数据，供 Recall/nDCG 调参）
- `tool_fixtures/` —— 工具级 golden fixtures：evidence gate 类别样本、verify 四态样本；不调用 LLM。
  ⚠️ 正例来自 1,200 篇样例；直接拿到 22,747 篇全量会低估 recall，全量需重新标注。

**数据文件**：无（评测产物落 `workspace/ledger/`，`eval_g{0..3}.json`）。

## 脚本速查

- `run_eval.py` —— 评测框架：`--group g0/g1/g2/g3` 设大创开关，`--live` 走 `llm_eval` 真调；终端每题只印 `id/route/refused/evidence/coverage`，完整结果落 JSON。
- `retrieval_eval.py` —— P1-1 检索评测：BM25 / vector / hybrid 的 Recall@k、nDCG@10、Precision@10、MRR；`--grid` 扫 hybrid 参数。
- `eval_compare.py` —— 合并 `eval_g*.json` 出 G0–G3 成对对比表（指标 Δ 相对 g0）。
- `holdout.py` —— 跨会留出：venue 切分(train=ICLR+ICML25, holdout=NeurIPS25) + 随机/高频 baseline 的 Precision@10/Spearman（0 token，流式轻内存）；subcluster 主题预测占位待接入。

## 4. 建议阅读顺序

1. `questions.example.json`（了解问题集形态）

## 5. 相关

- 上位指针：`README.md`（项目根总目录）
- 评测口径见 `../规划/2026-09-21-热点挖掘系统-方案设计.md` §5
