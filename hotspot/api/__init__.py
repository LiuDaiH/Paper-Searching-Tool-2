"""L6.5 服务层 —— 前端与算法之间**唯一**的接口面。

一句话：**前端只准 import 本包，不准 import `hotspot` 里其它任何东西。**

为什么要有这一层
----------------
改造前 `hotspot/ui/app.py` 直接 import 9 个模块、19 个符号，还自己实现了一套编排
（意图路径、兜底路径、追问、卡片装配、关键路径画图）。后果：

* 前端（HTML/JS 重构）要照抄一遍这套编排，或者再写一套 —— 两套必然漂移；
* 算法改一个函数签名，前端就崩；
* 每个 agent 上手都要读 844 行 app.py + 8 个算法文件才能动手。

现在把编排搬进 :class:`hotspot.api.engine.Engine`，对外只留 **两个方法 / 两个 HTTP 路由**：

| 方法 | HTTP | 干什么 |
|---|---|---|
| :meth:`Engine.meta` | `GET /api/meta` | 语料概况 + 可调参数表 + 能力开关（前端据此**自动生成**设置页） |
| :meth:`Engine.ask` | `POST /api/ask` | 问一次（新问题/追问、参数、API 配置、引用、图谱、token 用量，全在一条请求里） |

`GET /api/health` 是运维探活，不算业务接口。

用法（两行）
------------
>>> from hotspot.api import Engine
>>> eng = Engine()
>>> resp = eng.ask({"question": "2025 RLHF 方向最火的是什么？", "source": "meeting"})
>>> resp["answer"]["text"]
"""

from __future__ import annotations

from .contract import (  # noqa: F401
    CONTRACT_VERSION,
    PARAM_SPEC,
    ApiError,
    AskRequest,
    AskResponse,
    MetaResponse,
    STAGES,
)
from .engine import Engine, ask, meta  # noqa: F401

__all__ = [
    "CONTRACT_VERSION", "PARAM_SPEC", "STAGES",
    "ApiError", "AskRequest", "AskResponse", "MetaResponse",
    "Engine", "ask", "meta",
]
