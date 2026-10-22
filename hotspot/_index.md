---
tags: [索引]
---

# hotspot · `_index`

> **给 AI**：先读根 `../AGENTS.md` 的「按任务查表」，别从本目录开始翻。
> 本文件只有 §3 与 §4 值得读；`vendor/`（179k 字符）默认不读。

## 1. 这个文件夹是什么

热挖掘系统的**核心代码包**（L1–L6）。分层见 `../规划/2026-09-21-热点挖掘系统-方案设计.md` §本项目分层。

## 2. 当前工作状态

- L1–L6 代码骨架 ✅ 全部完成（纯逻辑已单测，数据/API 相关待跑数验证）
- L6.5 服务层 ✅ 契约 v1（`hotspot/api/`）：`meta` + `ask` 两个接口，Streamlit 与 HTTP 共用；
  自检 `scripts/check_api.py` 全绿；编排已从 `ui/app.py` 搬入 `engine.py`
- P0（2026-10-20）✅ 查询/向量/BM25/RF 缓存、`direction` 向量召回与阈值、
  `usage.stages` start/end、批量引用校验；样例真实请求约 34s
- P1-1（2026-10-20）✅ `retrieval/hybrid.py`：BM25 + dense → RRF → 可选
  cross-encoder → MMR；已接入 subcluster 与 LOCAL 路径
- 待导师库恢复：`embed → cluster → venue_matrix → cluster_card → eval` 完整跑数

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无（产物在 `workspace/`）。

**子包**（各包自己的 `_index.md` 登记其直接脚本）：

| 子包 | 层 | 它的 _index |
|---|---|---|
| `hotspot/structure/` | L2 | [[hotspot/structure/_index\|打开]] |
| `hotspot/compress/` | L3 | [[hotspot/compress/_index\|打开]] |
| `hotspot/retrieval/` | L4 | [[hotspot/retrieval/_index\|打开]] |
| `hotspot/reasoning/` | L5 | [[hotspot/reasoning/_index\|打开]] |
| `hotspot/harness/` | 内部工具成熟度（非公开） | [[hotspot/harness/_index\|打开]] |
| `hotspot/ui/` | L6（Streamlit 适配） | [[hotspot/ui/_index\|打开]] |
| `hotspot/api/` | L6.5 服务层（**前端唯一接口面**） | [[hotspot/api/_index\|打开]] |

## 脚本速查

- `corpus.py` —— 语料源：`MergedCorpus` 读 G 盘两份 jsonl 并集 22,747 篇保留 `year` 分组；`arXivCorpus`（独立周报源）读 `config.corpus.arxiv_source`，带 `topics`(分类)/`first_seen`(入库日) 与「本周新增」`new_this_week`。
- `subcluster.py` —— **意图驱动聚类（大创两线主路径接入）**：检索 → 意图定聚类规则 → 子聚类 + venue/year 标签单独分析。`config.yaml` 的 `rlrag.enabled`（RF 未来被引重排候选 + 簇代表取 RF 最高）与 `streit_garg.enabled`（非 relation 也算 MPS 关键路径）为**评测 G0–G3 开关**，失败写 `*_error` 不阻塞。**arXiv 源**（`arXivCorpus`）时 rlRAG 走 `rlr.arxiv_ranker`，Streit-Garg 因缺 coverage 降级报错。
- `deps_extract.py` —— **依赖图续跑抽取驱动**：逐 `.src` 抽取被引 arxiv id 边、逐篇落盘、断点可续（支撑 `relation`/共同来源）。
- `srcmap.py` —— **解包反推 `paper↔src` 映射**：解包每个 `.src` 提 `\title{}` 匹配 corpus 标题，落盘 `srcmap/*.json`（全库 17,014 / 命中 11,278 / 66%）；`--retry` 补跑之前的读盘失败。
- `co_cite.py` —— **引用层「共同来源」**：读 `workspace/depgraph/paper_refs.json`，`intent=relation` 时算两主题论文集**共同被引 arXiv** 反推共同来源（`co_cited` 并入 `subcluster`）；语料内 arxiv 反推会议/标题。
- `mps.py` —— **Streit-Garg 关键路径**（`compute_key_path`）：对某方向用 MPS 算
  「必备先序/关键路径」可行词 + 覆盖簇（复用 `dasc/mps/`，无 LLM/无导师库写入，
  数据缺则明确报未运行）。`intent=relation` 时恒由 `subcluster` 调用；`streit_garg.enabled`
  时非 relation 也接入（SG 主路径，供上下文构建）。
- `rlr.py` —— **rlRAG 学过的检索策略（用测试性能最好的 RF(S+A)，不用 RL）**。
  `RFRanker` 懒加载 `dasc/_out/rlr/future_rf.pkl`，`rank(candidates)` 按
  「未来被引概率」排该学/该引顺序。**mmap 按需读全库向量**（不整载 93MB，
  规避 8GB 机 OOM）。实测 RF 是全测试集最优（
  `future_report.json` rf auc=0.7471 > lr > heat；`future_rl_report` rawrf > rl），
  RL(GRPO) 追不上 RF 先验故不采用（详见模块 docstring）。纯 sklearn，只读，
  无写入；模型导出需 G 盘数据一次（`dasc/rlr/export_future_rf.py`）。
  **`arxiv_ranker(cfg)`** 指向周报 RF（R1–R3），供 arXiv 源 rlRAG。
- `cost.py` —— 成本记账 + 报表：每次问答记 `workspace/ledger/cost.jsonl`，汇总估算 token/¥（harness 卖点）。
- `schema.py` —— **内部**全链路数据契约（L1–L5 唯一口径，先读它）；对前端的那份在 `api/contract.py`。
- `direction.py` —— L1 方向选择器：自由文本 → 子语料（三路召回，0 token）。
- `llm.py` —— L3 薄包装 dasc LLM 客户端（从 config 构造 Client/Budget）。
- `config.py` —— 配置加载：从 `config.yaml` 读参数。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `schema.py`（数据契约）
2. 各子包 `_index.md` 按层（L2 → L3 → L4 → L5 → L6）

## 5. 相关

- 上位指针：`README.md`（项目根总目录）
- 分层与排期：`../规划/2026-09-21-热点挖掘系统-方案设计.md`
