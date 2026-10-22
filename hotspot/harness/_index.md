---
tags: [索引]
---

# hotspot/harness · `_index`

## 1. 这个文件夹是什么

内部工具契约层，用于工具成熟度验收。这里还不是公开 harness，也不接入
Controller；对外接口仍只有 `hotspot.api`。

## 2. 当前工作状态

- `contract.py`：内部 `ToolSpec`、归一化 `ToolObservation`、结构化失败。
- `tools.py`：最小纵链私有 adapter：`paper_search → evidence_gate → fetch_papers`。
- 尚未建立 registry；未通过离线基准的工具不得注册或暴露给 agent。

## 3. 文件清单

**计划文件**：无。
**数据文件**：无（基准结果落 `workspace/ledger/`）。

## 脚本速查

- `contract.py` —— 内部工具输入/输出/失败 observation 与 `ToolSpec` 校验。
- `tools.py` —— 现有检索、取论文、证据门的私有归一化 adapter，尚未注册。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `contract.py`

## 5. 相关

- 上位指针：`../_index.md`
- 长期路线：`../../规划/2026-10-21-检索智能体地基与演进方向.md`
