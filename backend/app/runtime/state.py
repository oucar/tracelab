"""Typed graph state. LangGraph nodes read and return partial updates of this."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field


class SandboxResult(BaseModel):
    code: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisStep(BaseModel):
    iteration: int
    result: SandboxResult


class RunState(BaseModel):
    """State flowing through the graph. M1: single analyst → composer.

    M2 adds: plan, parallel analyst results (with an `operator.add` reducer),
    critic verdicts, retry counters.
    """

    run_id: str
    question: str
    dataset_path: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)

    # Analyst work log — appended across iterations.
    steps: Annotated[list[AnalysisStep], operator.add] = Field(default_factory=list)
    analyst_findings: str = ""
    analyst_failed: bool = False
    failure_reason: str = ""

    final_answer: str = ""
