"""Public API shell for the agent-v2 rebuild.

The legacy retrieval/orchestration implementation was removed. This module
keeps the stable ``meta``/``ask`` boundary importable while tools and the new
controller are assembled.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Optional

from ..agent.runtime import run_research_agent
from ..config import load_config
from ..corpus import open_arxiv_corpus, open_corpus
from ..index.lance_store import LancePaperStore
from ..reasoning.grounding import make_citations, verify_and_render
from .contract import (
    E_CORPUS_UNAVAILABLE,
    E_INTERNAL,
    E_NO_API_KEY,
    PARAM_SPEC,
    SOURCES,
    ApiError,
    AskRequest,
    AskResponse,
    MetaResponse,
    SourceInfo,
    Usage,
    new_request_id,
    stage_event,
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
        started = time.perf_counter()
        cfg = copy.deepcopy(self._cfg)
        req.llm.apply_to(cfg)
        corpus = self._corpus(req.source)
        store = LancePaperStore(cfg)
        if not store.available:
            raise ApiError(
                E_INTERNAL,
                "LanceDB 索引不存在；先运行 scripts/build_lance_index.py",
                status=503,
            )
        if req.dry_run:
            message = "算法 v2 需要 LLM 与 LanceDB，dry-run 暂不生成回答。"
            return AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_NO_API_KEY, "message": message},
                warnings=[message],
                meta={"agent": {"dry_run": True}},
            )

        if on_event:
            on_event(stage_event("retrieve", elapsed=0.0))
        try:
            run = run_research_agent(
                req.question,
                cfg=cfg,
                history=[m.to_dict() for m in req.history],
            )
        except RuntimeError as exc:
            if "API key" in str(exc):
                raise ApiError(E_NO_API_KEY, str(exc)) from exc
            return AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": str(exc)},
                warnings=[str(exc)],
            )
        except Exception as exc:  # noqa: BLE001
            message = f"算法 v2 agent 失败：{type(exc).__name__}: {exc}"
            return AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": message},
                warnings=[message],
            )
        try:
            if on_event:
                on_event(stage_event("answer", elapsed=time.perf_counter() - started))
            grounded = verify_and_render(
                run.answer,
                corpus=corpus,
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"算法 v2 处理失败：{type(exc).__name__}: {exc}"
            return AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": message},
                warnings=[message],
                meta={"agent": {"tool_events": run.deps.events}},
            )
        if on_event:
            on_event(stage_event("verify", elapsed=time.perf_counter() - started))

        elapsed = round(time.perf_counter() - started, 3)
        return AskResponse(
            request_id=new_request_id(),
            source=req.source,
            mode=req.mode,
            route="refuse" if grounded.refused else "local",
            refused=grounded.refused,
            refuse_reason=grounded.refuse_reason,
            text=grounded.text,
            n_filtered=sum(
                1 for verdict in grounded.verdicts
                if verdict.status != "supported"
            ),
            verdicts=list(grounded.verdicts),
            citations=make_citations(corpus, grounded.citation_ids),
            warnings=list(run.answer.limitations),
            usage=Usage(
                elapsed_s=elapsed,
                stages={
                    "retrieve": 0.0,
                    "answer": elapsed,
                    "verify": elapsed,
                },
            ),
            meta={
                "agent": {
                    "search_calls": run.deps.search_calls,
                    "fetch_calls": run.deps.fetch_calls,
                    "seen_ids": len(run.deps.seen_ids),
                    "fetched_ids": len(run.deps.fetched),
                    "tool_events": run.deps.events,
                },
                "answer_claims": [item.model_dump() for item in run.answer.claims],
            },
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
