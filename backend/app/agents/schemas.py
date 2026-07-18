"""Structured-output schemas for agent turns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalystTurn(BaseModel):
    """One analyst decision: either run code against the dataset, or finish."""

    action: Literal["run_code", "finish"]
    code: str = Field(
        default="",
        description="Python to execute in the sandbox (required when action='run_code').",
    )
    findings: str = Field(
        default="",
        description="Findings summary with concrete numbers (required when action='finish').",
    )
