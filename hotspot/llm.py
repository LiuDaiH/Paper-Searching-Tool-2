"""L3 薄包装 `dasc.llm.client` —— 从 `config.yaml` 构造 `Client` 与 `Budget`。

不重写 LLM 客户端（复用 `dasc/llm/client.py`：JSON mode + 缓存 + 出境审计 + 预算熔断）。
这里只做一件事：把配置里的模型名、预算、key 来源对接上。
"""

from __future__ import annotations

from dasc.llm.client import Budget, Client  # noqa: F401

from .config import load_config

__all__ = ["make_client", "make_budget"]


def make_budget(cfg=None) -> Budget:
    cfg = cfg or load_config()
    b = (cfg.get("llm") or {}).get("budget") or {}
    return Budget(
        max_calls=int(b.get("max_calls") or 2000),
        max_tokens=int(b.get("max_tokens") or 5_000_000),
        max_usd=(float(b["max_usd"]) if b.get("max_usd") else None),
    )


def make_client(cfg=None, *, model: str | None = None, dry_run: bool = False) -> Client:
    """构造 Client。key 从环境变量 `DEEPSEEK_API_KEY` 读（未设则抛/走 dry_run）。"""
    cfg = cfg or load_config()
    llm = cfg.get("llm") or {}
    return Client(
        model=model or llm.get("model_flash") or "deepseek-chat",
        base_url=llm.get("base_url") or "https://api.deepseek.com/v1",
        budget=make_budget(cfg),
        dry_run=dry_run,
    )
