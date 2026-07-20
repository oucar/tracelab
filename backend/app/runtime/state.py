"""Typed graph state. LangGraph nodes read and return partial updates of this."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.runtime.chartspec import ChartSpec


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


class Methodology(BaseModel):
    """How a statistical claim was derived — rendered as a methodology chip."""

    method: str  # e.g. "Welch t-test", "Mann-Whitney U", "Pearson correlation"
    n: int
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str = ""  # "Cohen's d", "r", "rank-biserial"
    assumptions: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """One atomic, checkable statement an analyst makes."""

    id: str = ""  # assigned by the analyst node: "<step_id>-<n>"
    step_id: int = 0
    text: str
    kind: Literal["numeric", "categorical", "statistical"]
    value: float | str | None = None  # numeric value or category label
    direction: Literal["higher", "lower", "none"] | None = None  # statistical only
    significant: bool | None = None  # statistical only, at alpha
    methodology: Methodology | None = None


class PlanStep(BaseModel):
    id: int = 0
    description: str
    method: Literal[
        "descriptive",
        "mean_comparison",
        "correlation",
        "regression",
        "clustering",
        "timeseries_backtest",
        "anomaly_detection",
    ] = "descriptive"


class Plan(BaseModel):
    steps: list[PlanStep]
    rationale: str = ""


class Verdict(BaseModel):
    """The critic's judgment of one claim after deterministic reconciliation."""

    claim_id: str
    status: Literal["verified", "discrepancy", "unverifiable"]
    critic_value: float | str | None = None
    reason: str = ""
    methodology_ok: bool | None = None  # statistical claims only


class AnalystResult(BaseModel):
    step_id: int
    findings: str = ""
    claims: list[Claim] = Field(default_factory=list)
    chart_specs: list[ChartSpec] = Field(default_factory=list)
    chart_rejections: list[str] = Field(default_factory=list)
    iterations: list[AnalysisStep] = Field(default_factory=list)
    failed: bool = False
    failure_reason: str = ""


class VerifiedClaim(BaseModel):
    claim: Claim
    status: Literal["verified", "unverified"]
    detail: str = ""  # discrepancy/unchecked explanation


class FinalAnswer(BaseModel):
    narrative: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    failed: bool = False


class AnalystTask(BaseModel):
    """Send-payload for one analyst branch (also the retry payload)."""

    run_id: str
    question: str
    dataset_path: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    root_span_id: str = ""  # Send payloads don't see RunState; the tree root rides along
    step: PlanStep
    critic_feedback: str = ""


class RunState(BaseModel):
    """State flowing through the M2 graph: planner → analysts (Send) → critic → composer."""

    run_id: str
    question: str
    dataset_path: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    root_span_id: str = ""  # span_id of the run_started event; parents the whole tree

    route: str = ""  # set by router_node: "simple" | "multi_step" | "statistical"
    route_reason: str = ""

    plan: Plan | None = None
    planner_failed: bool = False
    planner_failure_reason: str = ""

    # Parallel analyst branches append here; last result per step_id wins.
    analyst_results: Annotated[list[AnalystResult], operator.add] = Field(default_factory=list)

    verdicts: list[Verdict] = Field(default_factory=list)
    retry_count: int = 0
    retry_steps: list[int] = Field(default_factory=list)  # set by critic when retrying

    final_answer: str = ""
    final: FinalAnswer | None = None
