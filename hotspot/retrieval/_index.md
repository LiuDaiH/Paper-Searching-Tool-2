---
tags: [索引]
---

# hotspot/retrieval · `_index`（L4 检索层）

> **给 AI**：先读 `router.py` → `assemble.py`；`bm25.py` / `vector.py` 只在改召回算法时读。

## 1. 这个文件夹是什么

L4 检索层：**规则兜底 + LLM 增强**路由 + 在线查询改写 + 上下文装配（命中缓存）+ numpy 暴力向量检索。
在线问答环节单次成本极小，用轻量 LLM 换**使用舒适度**；`llm=None` / `dry-run` 时退纯规则（确定性/离线）。

## 2. 当前工作状态

- `router.py` ✅ 规则兜底单测全绿；`llm_route` 失败退 `LOCAL`
- `rewrite.py` / `assemble.py` / `vector.py` ✅ 导入通过
- `hybrid.py` ✅ P1-1：BM25 × dense → RRF → 可选 cross-encoder → MMR；默认无模型可跑
- 完整在线检索 ⏸ 待向量/卡片数据 + `DEEPSEEK_API_KEY`

## 3. 文件清单

**计划文件**：无（全部是代码）。
**数据文件**：无（索引来自 `workspace/vectors/`）。

## 脚本速查

- `router.py` —— L4 路由：**规则兜底 + LLM 增强**（REFUSE 硬规则；关键词快路径；LLM 判 GLOBAL/LOCAL，失败退 LOCAL）。
- `bm25.py` —— BM25 词法检索（hybrid 词法侧）：加分排序、部分命中召回、离线缓存；与向量构成词×义混合。
- `rewrite.py` —— L4 在线查询改写：口语问题→检索查询，提升召回，失败退原文。
- `assemble.py` —— L4 装配上下文：固定前缀命中 DeepSeek 缓存，引用规范写死。
- `vector.py` —— L4 numpy 暴力余弦检索（不引向量库）。
- `hybrid.py` —— L4 混合检索：BM25 + dense 的 RRF 融合、可选 cross-encoder rerank、MMR 多样性；失败回退 RRF。
- `coverage.py` —— 全局粗主题覆盖：稳定簇成员 + 最近质心兜底、趋势题粗到细候选选择、覆盖观测。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `router.py` → `rewrite.py` → `assemble.py` → `hybrid.py` → `coverage.py` → `vector.py`

## 5. 相关

- 上位指针：`hotspot/_index.md`
- 口径：`../../规划/2026-09-21-热点挖掘系统-方案设计.md` §3.4 / §2.5
