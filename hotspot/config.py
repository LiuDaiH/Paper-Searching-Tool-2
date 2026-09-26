"""配置加载。**唯一入口出自 `config.yaml`**（方案：不做接口抽象，只有一套环境）。

所有模块都从这里取配置，不各自散落魔数。改参数只动 `config.yaml`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import hotspot

__all__ = ["load_config", "paths_from"]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """读 `config.yaml`。返回原始 dict（不套对象——够用，且不引抽象代价）。"""
    cfg_path = Path(path) if path else hotspot.HOTSPOT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def paths_from(cfg: dict[str, Any]) -> dict[str, Path]:
    """把 `paths` 段解析成绝对 `Path`。相对路径以本项目根为基准。"""
    raw = cfg.get("paths") or {}
    out: dict[str, Path] = {}
    for key, val in raw.items():
        if not val:
            continue
        p = Path(val)
        if not p.is_absolute():
            p = hotspot.HOTSPOT_ROOT / p
        out[key] = p
    return out
