"""L3 薄包装 `dasc.llm.client` —— 从 `config.yaml` 构造 `Client` 与 `Budget`。

不重写 LLM 客户端（复用 `dasc/llm/client.py`：JSON mode + 缓存 + 出境审计 + 预算熔断）。
这里只做两件事：把配置里的模型名、预算对接上，并以「前端填入/本地 settings.json」
的明文 key 为唯一来源——**绝不读任何环境变量用于 API key**。
"""

from __future__ import annotations

import os

from dasc.llm.client import Budget, Client, LLMError  # noqa: F401

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


def make_client(cfg=None, *, model: str | None = None, dry_run: bool = False,
                timeout: float | None = None, api_key: str | None = None) -> Client:
    """构造 Client。

    key 优先级：显式 ``api_key`` > ``cfg.llm.api_key`` > ``cfg.llm.api_key_env``
    指向的环境变量。环境变量只在配置里**显式写出变量名**时读取，不扫描任意变量。
    ``timeout`` 取参 > config.llm.timeout > dasc 默认 120s。
    """
    cfg = cfg or load_config()
    llm = cfg.get("llm") or {}
    key = api_key or llm.get("api_key")
    if not key and llm.get("api_key_env"):
        key = os.environ.get(str(llm["api_key_env"]), "")
    if not key and not dry_run:
        raise LLMError(
            "未配置 API key：请在前端侧栏「API 配置」填写 API key 后提问。"
            "（也可在配置中显式设置 llm.api_key_env 指向环境变量。）"
        )
    to = timeout if timeout is not None else llm.get("timeout")
    return Client(
        model=model or llm.get("model_flash") or "deepseek-chat",
        base_url=llm.get("base_url") or "https://api.deepseek.com/v1",
        api_key=key,
        budget=make_budget(cfg),
        dry_run=dry_run,
        timeout=float(to) if to not in (None, "") else 120.0,
    )
