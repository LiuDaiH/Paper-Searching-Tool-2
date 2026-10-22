"""Public API shell for the agent-v2 rebuild.

The legacy retrieval/orchestration implementation was removed. This module
keeps the stable ``meta``/``ask`` boundary importable while tools and the new
controller are assembled.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Optional

from ..config import load_config
from ..corpus import open_arxiv_corpus, open_corpus
from .contract import (
    E_CORPUS_UNAVAILABLE,
    E_INTERNAL,
    PARAM_SPEC,
    SOURCES,
    ApiError,
    AskRequest,
    AskResponse,
    MetaResponse,
    SourceInfo,
    new_request_id,
)

ProgressHook = Optional[Any]
_ENGINE: Optional["Engine"] = None


class Engine:
    """Stable service shell; algorithm v2 will replace only the ask pipeline."""

    def __init__(self, cfg: dict | None = None):
        self._cfg = copy.deepcopy(cfg) if cfg is not None else load_config()
        self._corpora: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _corpus(self, source: str):
        if source not in SOURCES:
            raise ApiError("bad_request", f"未知数据源：{source}")
        with self._lock:
            if source in self._corpora:
                return self._corpora[source]
            try:
                corpus = (
                    open_arxiv_corpus(self._cfg)
                    if source == "arxiv"
                    else open_corpus(self._cfg)
                )
            except Exception as exc:  # noqa: BLE001
                raise ApiError(
                    E_CORPUS_UNAVAILABLE,
                    f"数据源 {source!r} 打不开：{type(exc).__name__}: {exc}",
                    status=503,
                ) from exc
            self._corpora[source] = corpus
            return corpus

    def meta(self) -> MetaResponse:
        out = MetaResponse(params=[p.to_dict(self._cfg) for p in PARAM_SPEC])
        for key, label in SOURCES.items():
            info = SourceInfo(key=key, label=label)
            try:
                stats = self._corpus(key).stats()
                info.papers = int(stats.get("papers") or 0)
            except ApiError as exc:
                info.note = exc.message
                out.warnings.append(exc.message)
            out.sources.append(info)
        out.capabilities = {
            "algorithm_v2_rebuilding": True,
            "stream": True,
            "dry_run": True,
            "sources": list(SOURCES),
        }
        return out

    def ask(
        self,
        request: AskRequest | dict[str, Any],
        on_event: ProgressHook = None,
    ) -> AskResponse:
        req = request if isinstance(request, AskRequest) else AskRequest.from_dict(request)
        message = "算法 v2 正在重建：旧检索与编排已移除，尚未接入新工具链。"
        return AskResponse(
            request_id=new_request_id(),
            source=req.source,
            mode=req.mode,
            route="error",
            text="",
            error={"code": E_INTERNAL, "message": message},
            warnings=[message],
            meta={"rebuild": True},
        )


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


def ask(request: AskRequest | dict[str, Any], on_event: ProgressHook = None) -> dict[str, Any]:
    return get_engine().ask(request, on_event=on_event).to_dict()


def meta() -> dict[str, Any]:
    return get_engine().meta().to_dict()
