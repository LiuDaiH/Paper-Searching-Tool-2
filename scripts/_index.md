---
tags: [索引]
---

# scripts · `_index`

> **给 AI**：按下面「脚本速查」找**单条**，不要通读；常用入口 `run_server.py` / `run_ui.py` / `check_api.py`。

## 1. 这个文件夹是什么

一次性/验证脚本：接入验证、方向选择验证、索引树自检。

## 2. 当前工作状态

- D1 接入验证：✅ 通过
- D2 方向选择验证：✅ 通过
- 索引树自检（悬空/孤儿/README 覆盖/**脚本速查一致性**）：✅ 全绿
- 接口自检（`check_api.py`：契约/参数/HTTP/SSE）：✅ 全绿

## 3. 文件清单

**计划文件**：无（全部是代码，见下「脚本速查」）。
**数据文件**：无。

## 脚本速查

- `check_access.py` —— D1 接入验证：四道闸在本项目这条路上真的生效。
- `check_direction.py` —— D2 方向选择器三路召回验证（探针 7）。
- `check_tree.py` —— 索引树自检：悬空/孤儿/README 覆盖/脚本速查一致性。
- `check_mps_ml.py` —— **Streit-Garg 关键路径接入自检**：`hotspot/mps.py` 四算法
  可行 + `subcluster` relation 导入无副作用（数据缺则跳过，非假装绿）。
- `run_cards.py` —— L3 生成簇卡片（真调 DeepSeek，断点续跑）。
- `run_ui.py` —— Streamlit UI 入口（`streamlit run scripts/run_ui.py`）。
- `run_server.py` —— **HTTP 服务入口**（新前端用）：`python scripts/run_server.py --port 8000 --web web`，只暴露 `/api/meta`、`/api/ask`。
- `check_api.py` —— **接口自检**（离线秒级）：契约往返 + 参数白名单 + HTTP/SSE 冒烟，用 StubEngine 不碰语料。
- `check_tools.py` —— 内部工具 golden tests：contract、evidence gate、verify 四态、fetch/私有 adapter。
- `audit_coverage.py` —— 全库粗主题覆盖审计：稳定归属、最近质心、margin、簇规模；不调用 LLM。
- `summarize_trace.py` —— 只读 Trace JSONL，汇总 route、拒答、证据门、覆盖策略和阶段延迟；不输出问题/答案原文。
- `run_query.py` —— 意图驱动聚类端到端实测（parse_query → 检索聚类 → 命名 → 回答）。
- `run_deps.py` —— 依赖图续跑抽取 CLI（逐篇落盘被引边，可续跑）。
- `agg_refs.py` —— 聚合依赖图：`refs/*.json` × `srcmap/*.json` → `workspace/depgraph/paper_refs.json`（paper→被引 arXiv 双向，供 `co_cite`）。
- `srcmap_report.py` —— 汇总 `srcmap/*.json` 命中率 + `digest→paper_id` 映射统计。
- `cost_report.py` —— 成本报表 CLI（汇总 `workspace/ledger/cost.jsonl` 的估算 token/¥）。
- `export_arxiv_weekly.py` —— P3 发布物生成：arXiv 周报分块向量化（`workspace/arxiv_vectors/`，分片续跑=增量）→ 打包 `weekly_arxiv/`（vectors.npy/ids/meta/topics/manifest），`--publish-dir` 拷贝到穿透/同步目录。
- `fetch_arxiv_refs.py` —— R1 周报引用图拉取：Semantic Scholar 单条 GET，增量断点续跑（可配 key），出 `depgraph/cite_graph_arxiv.pkl`。
- `prep_arxiv_rl.py` —— R2 周报 rl_prep：从 `arxiv_vectors` 导 `mobile_corpus_vectors_arxiv.npz` + `global_mean.npy`（`workspace/arxiv_rl/`）。
- `export_future_rf_arxiv.py` —— R3 周报未来被引 RF 重训：时间留出（early/late）+ future 标签 → `rl_prep/future_rf_arxiv.pkl`。
- `update_arxiv_weekly.py` —— **每周新增增量管线**（用户核心诉求）：水位检测新增 → 只对新增做 embedding（`vec_delta_*`）+ 引用（`only=新增`）→ 更新发布物 → 水位前移。`--init` 首次把当前全库设为基线、跳过历史。全幂等、断点可重入。
- `build_dist.py` —— **服务端源码打包**：核心源码（hotspot/、scripts/）+ config + requirements + 运行说明 → `发布/服务端源码-<日期>.zip`（排除 workspace/日志/缓存）。
- `run_eval_all.py` —— 评测 G0–G3 四组串行跑完（复用 run_eval，建议独立终端/automation，避免会话中断）。
- `export_sample.py` —— B 档样例数据集：从全量 24+25 会议+arXiv 周报切可运行子集（语料/向量/BM25），供独立应用快照演示。
- `build_release.py` —— B 档发布：组装「服务端源码.zip + 样例数据集.zip(智引热点独立快照) + 根 README」到 `G:\work!`。

## 4. 建议阅读顺序

无固定顺序（脚本按需 `python scripts/xxx.py` 跑）。

## 5. 相关

- 上位指针：`README.md`（项目根总目录）
- `hotspot/`（脚本所验证的实现层）
