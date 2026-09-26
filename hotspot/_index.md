---
tags: [索引]
---

# hotspot · `_index`

## 1. 这个文件夹是什么

热挖掘系统的**核心代码包**（L1–L6）。分层见 `热点挖掘系统-方案设计.md` §本项目分层。

## 2. 当前工作状态

- L1–L6 代码骨架 ✅ 全部完成（纯逻辑已单测，数据/API 相关待跑数验证）
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
| `hotspot/ui/` | L6 | [[hotspot/ui/_index\|打开]] |

## 脚本速查

- `corpus.py` —— 24+25 合并语料源（`MergedCorpus`）：读 G 盘两份 jsonl 并集 22,747 篇，保留 `year` 分组。
- `schema.py` —— 全链路数据契约（唯一口径，先读它）。
- `direction.py` —— L1 方向选择器：自由文本 → 子语料（三路召回，0 token）。
- `llm.py` —— L3 薄包装 dasc LLM 客户端（从 config 构造 Client/Budget）。
- `config.py` —— 配置加载：从 `config.yaml` 读参数。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `schema.py`（数据契约）
2. 各子包 `_index.md` 按层（L2 → L3 → L4 → L5 → L6）

## 5. 相关

- 上位指针：`README.md`（项目根总目录）
- 分层与排期：`热点挖掘系统-方案设计.md`
