"""Internal tool contracts and maturity gates.

Nothing in this package is part of the public ``hotspot.api`` contract.
"""

from .contract import ToolError, ToolObservation, ToolSpec

__all__ = ["ToolError", "ToolObservation", "ToolSpec"]
