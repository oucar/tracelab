"""Structured-output schemas for agent turns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.runtime.reconcile import CriticFinding
from app.runtime.state import Claim, PlanStep

__all__ = ["AnalystTurn", "PlannerTurn", "CriticTurn", "CriticFinding", "JudgeTurn"]


class PlannerTurn(BaseModel):
    """The planner's structured plan: 1..N independent analysis steps."""

    steps: list[PlanStep] = Field(
        description=(
            "Independent analysis steps; they run in parallel and must not depend on each other."
        )
    )
    rationale: str = Field(default="", description="One short paragraph: why this decomposition.")


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
    claims: list[Claim] = Field(
        default_factory=list,
        description="Atomic checkable claims backing the findings (required when action='finish').",
    )


class CriticTurn(BaseModel):
    """One critic decision: either run verification code, or finish with findings."""

    action: Literal["run_code", "finish"]
    code: str = Field(
        default="",
        description="Python to execute in the sandbox (required when action='run_code').",
    )
    findings: list[CriticFinding] = Field(
        default_factory=list,
        description="One finding per claim (required when action='finish').",
    )


class JudgeTurn(BaseModel):
    """Tier-2 rubric scores, 1 (poor) to 5 (excellent)."""

    clarity: int = Field(ge=1, le=5)
    uncertainty_honesty: int = Field(ge=1, le=5)
    chart_appropriateness: int = Field(ge=1, le=5)
    methodological_soundness: int = Field(ge=1, le=5)
    rationale: str = ""
