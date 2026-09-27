"""L3 薄包装 `dasc.llm.client` —— 从 `config.yaml` 构造 `Client` 与 `Budget`。

不重写 LLM 客户端（复用 `dasc/llm/client.py`：JSON mode + 缓存 + 出境审计 + 预算熔断）。
这里只做一件事：把配置里的模型名、预算、key 来源对接上。

**key 来源**：以 `config.yaml` 的 `llm.api_key_env` 为准（用户环境变量名，如 `API_KEY`）。
`dasc` 内部读的是 `DEEPSEEK_API_KEY`，这里在进程内做适配：把用户指定的变量名值同步到
`DEEPSEEK_API_KEY`（仅当后者未设）。**key 只来自环境变量，永不落盘、永不打印。**
"""

from __future__ import annotations

import os

from dasc.llm.client import Budget, Client  # noqa: F401

from .config import load_config

__all__ = ["make_client", "make_budget", "ensure_api_key"]


def ensure_api_key(llm: dict) -> str | None:
    """把 `config.llm.api_key_env`（用户 key 变量名）适配到 dasc 期望的 `DEEPSEEK_API_KEY`。

    返回实际生效的 key 变量名（不返回 key 本身）。key 仍在环境变量里，不落盘。
    """
    key_env = (llm or {}).get("api_key_env") or "DEEPSEEK_API_KEY"
    if key_env != "DEEPSEEK_API_KEY" and key_env in os.environ \
            and "DEEPSEEK_API_KEY" not in os.environ:
        os.environ["DEEPSEEK_API_KEY"] = os.environ[key_env]
    return key_env


def make_budget(cfg=None) -> Budget:
    cfg = cfg or load_config()
    b = (cfg.get("llm") or {}).get("budget") or {}
    return Budget(
        max_calls=int(b.get("max_calls") or 2000),
        max_tokens=int(b.get("max_tokens") or 5_000_000),
        max_usd=(float(b["max_usd"]) if b.get("max_usd") else None),
    )


def make_client(cfg=None, *, model: str | None = None, dry_run: bool = False) -> Client:
    """构造 Client。key 从 `config.llm.api_key_env` 指定的环境变量读（未设则抛/走 dry_run）。"""
    cfg = cfg or load_config()
    llm = cfg.get("llm") or {}
    ensure_api_key(llm)
    return Client(
        model=model or llm.get("model_flash") or "deepseek-chat",
        base_url=llm.get("base_url") or "https://api.deepseek.com/v1",
        budget=make_budget(cfg),
        dry_run=dry_run,
    )
