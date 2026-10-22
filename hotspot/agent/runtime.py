"""PydanticAI controller for the paper-first research vertical slice."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from ..core.models import AgentAnswer, PaperDetail, SearchResult
from ..tools.papers import fetch_papers
from ..tools.search import search_papers
from .deps import AgentDeps


@dataclass(frozen=True)
class AgentRun:
    answer: AgentAnswer
    deps: AgentDeps


INSTRUCTIONS = """你是本地论文库研究助手。只能依据工具返回的论文证据回答。

规则：
1. 问题涉及方法或论文时，先调用 search_papers；查询可用英文关键词。
2. 需要引用任何论文前，必须调用 fetch_papers 获取摘要。
3. 比较两个对象时分别搜索两次，不要只搜一个合并字符串。
4. 只能引用 fetched 中真实存在的 paper_id，不得编造 ID、标题或结论。
5. 每条 claim 必须带 evidence_ids；没有足够证据时返回空 claims，并填写 no_answer_reason。
6. 不回答本地论文库无法支持的外部知识问题。
7. 最多 4 次 search、3 次 fetch；证据足够后立即停止调用工具。
8. 用户使用中文时，最终 claim 使用中文；论文标题保持原文。
"""


def _api_key(cfg: dict) -> str:
    llm = cfg.get("llm") or {}
    key = str(llm.get("api_key") or "").strip()
    env_name = str(llm.get("api_key_env") or "").strip()
    if not key and env_name:
        key = str(os.environ.get(env_name, "")).strip()
    if not key:
        raise RuntimeError("未配置 API key，无法启动算法 v2 agent")
    return key


def _normalize_base_url(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    return value[:-len(suffix)].rstrip("/") if value.endswith(suffix) else value


def _build_agent(cfg: dict) -> Agent[AgentDeps, AgentAnswer]:
    llm = cfg.get("llm") or {}
    base_url = _normalize_base_url(
        str(llm.get("base_url") or "https://api.deepseek.com/v1")
    )
    model_name = str(llm.get("model_pro") or llm.get("model_flash") or "deepseek-chat")
    provider = OpenAIProvider(base_url=base_url, api_key=_api_key(cfg))
    model = OpenAIChatModel(model_name, provider=provider)
    agent = Agent(
        model,
        deps_type=AgentDeps,
        output_type=AgentAnswer,
        instructions=INSTRUCTIONS,
        retries=2,
    )

    @agent.tool
    def search_tool(
        ctx: RunContext[AgentDeps],
        query: str,
        keywords: list[str] | None = None,
        years: list[int] | None = None,
        venues: list[str] | None = None,
        limit: int = 12,
    ) -> SearchResult:
        """Search local papers. Use concise English queries and optional synonyms."""
        if ctx.deps.search_calls >= 4:
            raise ModelRetry("search budget exhausted; answer with current evidence")
        ctx.deps.search_calls += 1
        result = search_papers(
            ctx.deps.store,
            query,
            keywords=keywords,
            years=years,
            venues=venues,
            limit=limit,
        )
        ctx.deps.seen_ids.update(hit.paper_id for hit in result.results)
        ctx.deps.events.append({
            "tool": "search_papers",
            "query": query,
            "returned": len(result.results),
            "ids": [hit.paper_id for hit in result.results],
        })
        return result

    @agent.tool
    def fetch_tool(
        ctx: RunContext[AgentDeps],
        paper_ids: list[str],
    ) -> list[PaperDetail]:
        """Fetch title and abstract for paper IDs returned by search."""
        if ctx.deps.fetch_calls >= 3:
            raise ModelRetry("fetch budget exhausted; answer with fetched evidence")
        ctx.deps.fetch_calls += 1
        unknown = [pid for pid in paper_ids if pid not in ctx.deps.seen_ids]
        if unknown:
            raise ModelRetry(f"paper IDs were not returned by search: {unknown}")
        result = fetch_papers(ctx.deps.store, paper_ids)
        ctx.deps.fetched.update(paper.paper_id for paper in result)
        ctx.deps.events.append({
            "tool": "fetch_papers",
            "requested": len(paper_ids),
            "returned": len(result),
            "ids": [paper.paper_id for paper in result],
        })
        return result

    @agent.output_validator
    def validate_grounding(
        ctx: RunContext[AgentDeps],
        output: AgentAnswer,
    ) -> AgentAnswer:
        output.claims = [
            claim for claim in output.claims
            if any(str(pid).strip() for pid in claim.evidence_ids)
        ]
        if not output.claims:
            output.no_answer_reason = (
                output.no_answer_reason.strip()
                or "没有产生带可验证证据 ID 的 claim"
            )
            return output
        used = {
            pid
            for claim in output.claims
            for pid in claim.evidence_ids
            if str(pid).strip()
        }
        if not used:
            raise ModelRetry("every claim must include at least one evidence ID")
        missing_fetch = sorted(used - ctx.deps.fetched)
        if missing_fetch:
            raise ModelRetry(
                "fetch these evidence IDs before answering: "
                + ", ".join(missing_fetch)
            )
        return output

    return agent


def run_research_agent(
    question: str,
    *,
    cfg: dict,
    history: list[dict] | None = None,
    timeout: float = 180.0,
) -> AgentRun:
    deps = AgentDeps(cfg=cfg)
    if not deps.store.available:
        raise RuntimeError(
            "LanceDB index missing; run scripts/build_lance_index.py"
        )
    prompt = question
    if history:
        recent = "\n".join(
            f"{item.get('role')}: {item.get('text')}"
            for item in history[-6:]
            if item.get("text")
        )
        if recent:
            prompt = f"最近对话：\n{recent}\n\n当前问题：{question}"
    result = _build_agent(cfg).run_sync(
        prompt,
        deps=deps,
        timeout=timeout,
        usage_limits=UsageLimits(
            request_limit=8,
            tool_calls_limit=8,
        ),
    )
    return AgentRun(answer=result.output, deps=deps)
