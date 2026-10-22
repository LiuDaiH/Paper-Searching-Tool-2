"""Internal-only contracts used while maturing tools.

These dataclasses deliberately stay outside ``hotspot.api.contract``. A tool is
not considered public merely because it can produce a normalized observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^\d+\.\d+$")


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class ToolObservation:
    """Normalized tool result; policy should not depend on implementation scores."""

    status: Literal["ok", "empty", "error"]
    ids: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    flags: dict[str, bool | str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    elapsed_s: float | None = None
    tool_version: str = "0.0"

    def __post_init__(self) -> None:
        if self.status == "error" and self.error is None:
            raise ValueError("error observation requires ToolError")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error observation cannot carry ToolError")
        if self.ranks and len(self.ranks) != len(self.ids):
            raise ValueError("ranks and ids must have equal length")
        if self.scores and len(self.scores) != len(self.ids):
            raise ValueError("scores and ids must have equal length")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ids": list(self.ids),
            "ranks": list(self.ranks),
            "scores": list(self.scores),
            "counts": dict(self.counts),
            "flags": dict(self.flags),
            "data": dict(self.data),
            "error": self.error.to_dict() if self.error else None,
            "elapsed_s": self.elapsed_s,
            "tool_version": self.tool_version,
        }

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        tool_version: str = "0.0",
    ) -> "ToolObservation":
        return cls(
            status="error",
            error=ToolError(code=code, message=message, retryable=retryable),
            tool_version=tool_version,
        )


@dataclass(frozen=True)
class ToolSpec:
    """A proposed internal contract, not a public registration."""

    name: str
    version: str
    level: Literal["primitive", "macro"]
    args_schema: dict[str, Any]
    returns_schema: dict[str, Any]
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    cost_model: str = ""
    deterministic: bool = True
    fallback: str = ""
    owner: str = ""

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"invalid tool name: {self.name}")
        if not _VERSION.fullmatch(self.version):
            raise ValueError(f"invalid tool version: {self.version}")
        if self.level not in {"primitive", "macro"}:
            raise ValueError(f"invalid tool level: {self.level}")
        if not self.args_schema or not self.returns_schema:
            raise ValueError("tool schemas cannot be empty")
        if not self.cost_model:
            raise ValueError("cost_model is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "version": self.version,
            "level": self.level,
            "args_schema": dict(self.args_schema),
            "returns_schema": dict(self.returns_schema),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "cost_model": self.cost_model,
            "deterministic": self.deterministic,
            "fallback": self.fallback,
            "owner": self.owner,
        }
