"""LLM 打标签：**带预算、缓存、出境留痕的 DeepSeek 客户端**。

两块，刻意分开：

- :mod:`dasc.llm.client` —— 怎么调（缓存 / 预算 / schema 校验 / 出境记录）
- :mod:`dasc.llm.tasks` —— 问什么（覆盖标签、broad 锚类型），**取值从分类学现取**

分开的理由：换模型、换端点、改预算不该碰提示词；改提示词不该碰网络层。

⚠️ **这是本库唯一的数据出境路径**（到 `api.deepseek.com`）。只发标题与摘要，
**不发正文全文**——`client.MAX_INPUT_CHARS` 是那道闸。
⚠️ **key 只在环境变量里**（`DEEPSEEK_API_KEY`），不读文件、不写日志。
"""

from __future__ import annotations

from .client import (
    BASE_URL,
    CACHE_DIR,
    EGRESS_LOG,
    MAX_INPUT_CHARS,
    MODEL,
    Budget,
    BudgetExceeded,
    Client,
    LLMError,
    SchemaViolation,
    Usage,
    check_payload,
    redact,
)
from .tasks import (
    ANCHOR_KINDS,
    ANCHOR_THRESHOLD,
    ANCHORS_JSON,
    anchor_prompt,
    anchor_schema,
    broad_anchors,
    contribution_prompt,
    contribution_schema,
    contribution_validator,
    taxonomy_axes,
    taxonomy_fingerprint,
)

__all__ = [
    "BASE_URL", "MODEL", "MAX_INPUT_CHARS", "CACHE_DIR", "EGRESS_LOG",
    "Client", "Budget", "Usage", "LLMError", "BudgetExceeded", "SchemaViolation",
    "redact", "check_payload",
    "ANCHORS_JSON", "ANCHOR_THRESHOLD", "ANCHOR_KINDS",
    "broad_anchors", "taxonomy_fingerprint", "taxonomy_axes",
    "contribution_schema", "contribution_prompt", "contribution_validator",
    "anchor_schema", "anchor_prompt",
]
