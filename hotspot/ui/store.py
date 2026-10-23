"""Streamlit 前端的本地存储（**前端资产，不是算法资产**）。

只干三件事：运行态 API 配置、检索参数、多会话历史。都落 `workspace/`（gitignore）。

⚠️ 这一层正在被账号系统替代（组长分工：前端做 MySQL 账号库）。替换时
**不需要动服务层**——`Engine.ask` 只从请求里拿 `llm` / `params` / `history`，
不知道这些数据存在哪。迁移时把本文件的函数换成 SQL 即可。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import paths_from

__all__ = [
    "settings_path", "load_api_settings", "save_api_settings",
    "load_params", "save_params",
    "sessions_path", "load_sessions", "save_sessions", "new_session_id",
]


def _workspace(cfg) -> Path:
    return paths_from(cfg)["workspace"]


def settings_path(cfg) -> Path:
    """`workspace/settings.json`：`{base_url, model, api_key, params}`（单机自用）。"""
    return _workspace(cfg) / "settings.json"


def sessions_path(cfg) -> Path:
    return _workspace(cfg) / "exports" / "sessions.json"


def _read_settings(cfg) -> dict:
    try:
        p = settings_path(cfg)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001 —— 配置坏了不该阻止启动
        pass
    return {}


def _write_settings(cfg, data: dict) -> None:
    p = settings_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_api_settings(cfg) -> dict:
    """读 `{base_url, model, api_key}`。读不到就当没配过。"""
    d = _read_settings(cfg)
    return {k: d[k] for k in ("base_url", "model", "api_key") if d.get(k)}


def save_api_settings(cfg, api: dict) -> None:
    """**合并**写入（不覆盖 params）。⚠️ key 明文——账号系统上线后应改为服务端加密。"""
    d = _read_settings(cfg)
    d.update({k: api.get(k) for k in ("base_url", "model", "api_key")})
    _write_settings(cfg, d)


def load_params(cfg) -> dict:
    """读用户上一次的 Agent 参数（形如 `{"search_limit": 12}`）。"""
    return dict(_read_settings(cfg).get("params") or {})


def save_params(cfg, params: dict) -> None:
    d = _read_settings(cfg)
    d["params"] = {k: v for k, v in (params or {}).items() if v is not None}
    _write_settings(cfg, d)


def load_sessions(cfg) -> dict:
    try:
        p = sessions_path(cfg)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def save_sessions(cfg, sessions: dict) -> None:
    p = sessions_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")


def new_session_id() -> str:
    return time.strftime("%H%M%S")
