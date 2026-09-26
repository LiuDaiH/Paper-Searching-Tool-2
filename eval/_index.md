---
tags: [索引]
---

# eval · `_index`

## 1. 这个文件夹是什么

评测集与评测脚本（方案 §5）：2 基线 + 3 消融 + 跨会留出验证。

## 2. 当前工作状态

- 评测框架 ✅ 已写（跑一次问答管线收集 4 项指标）。
- 60 条问题集 / 基线 / 跨会留出验证：⏸ 待导师库恢复 + `DEEPSEEK_API_KEY` 后跑。

## 3. 文件清单

**计划文件**：

- `questions.example.json` —— 问题集模板（global / local / refuse 三类示例）

**数据文件**：无（评测产物落 `workspace/ledger/`）。

## 脚本速查

- `run_eval.py` —— 评测框架：对每题跑一次问答管线，收集 4 项指标到 `workspace/ledger/`。

## 4. 建议阅读顺序

1. `questions.example.json`（了解问题集形态）

## 5. 相关

- 上位指针：`README.md`（项目根总目录）
- 评测口径见 `热点挖掘系统-方案设计.md` §5
