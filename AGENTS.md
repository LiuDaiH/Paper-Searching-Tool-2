# 移动云杯 · 开发入口（AGENTS.md）

**这是每次会话必读的入口。** 树的细节见根 `README.md`（总目录）；设施规则见 `.zhanlu/skill/folder-readme`。

## 1. 项目有哪些块

| 块 | 是什么 | 可写 / 只读 |
|---|---|---|
| 根散件（`README.md`、`AGENTS.md`、`config.yaml`） | 入口文件 | 可写 |
| `规划/` | **长期参考**（方案 / 测试计划 / 分工 / 方向决策 / 迁移备忘） | 可写（先读方案） |
| `日志/` | 研究日志（流水账，一天一篇） | 可写 |
| `hotspot/` `scripts/` `eval/` | 代码（`*.py`） | 可写 |
| `hotspot/api/` | **前端唯一接口面**（契约 `contract.py` / 引擎 `engine.py` / HTTP `http.py`） | 可写，改前先读 `规划/接口契约-前端与算法.md` |
| `web/` | 新前端静态页（参考实现，原生 JS） | 可写 |
| `workspace/` | **数据产物**（可重建） | 可重建，不读 |
| `G:\rl_data24` + `G:\rl_data` | 24+25 语料（`hotspot/corpus.MergedCorpus` 合并，22,747 篇） | **只读** |

## 2. 硬要求

动手前先读那个文件夹的 `_index.md` / `README.md`（例如要改 `hotspot/` 先读 `hotspot/_index.md`）。

## 3. 回写要求

改动之后回写，并**向上传播到根** `README.md`（`folder-readme` skill §三的传播表）。新建计划文件/日志后同步更新对应索引。

## 4. 计划 / 数据只读

**浏览项目只读计划文件**：`README`、`日志/`、`规划/`、经验与方案 md 进上下文；
**数据文件不读**：`workspace/` 产物（向量/聚类/metrics/cards）、skill 源、`__pycache__`——要么可重建、要么该由脚本读。

## 接口红线（2026-10-18 起）

1. **前端只准 import `hotspot.api`**（或走 HTTP）。不准 import `subcluster` / `retrieval.*` / `reasoning.*` / `schema`。
2. **只增不改**：给 `AskResponse` 加字段必须带默认值；删/改语义必须换 `CONTRACT_VERSION` 并写迁移说明。
3. 加可调参数 → 只改 `hotspot/api/contract.py` 的 `PARAM_SPEC` + `config.yaml` 默认值，前端设置页自动出现。
4. 改完接口跑 `python scripts/check_api.py`，全绿再提交。

## 红线（本项目的核心约束）

热点只能表述为三态：**铺开 / 新进 / 单会**；**永不输出「退潮 / 下降 / 衰退」**（语料 24+25，但主口径选择对比跨会、不做跨年，2026-09-22 拍板）。
