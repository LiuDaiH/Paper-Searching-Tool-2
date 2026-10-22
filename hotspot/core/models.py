"""Pydantic models for the v2 agent/tool boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    years: list[int] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    paper_id: str
    title: str
    year: int | None = None
    venue: str = ""
    score: float = 0.0
    snippet: str = ""


class SearchResult(BaseModel):
    query: str
    results: list[SearchHit] = Field(default_factory=list)
    n_candidates: int = 0
    truncated: bool = False


class PaperDetail(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    subfields: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    claims: list[AnswerClaim] = Field(default_factory=list)
    no_answer_reason: str = ""
    limitations: list[str] = Field(default_factory=list)
