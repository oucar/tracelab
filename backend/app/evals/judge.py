"""Tier-2 LLM judge: rubric-scores a FinalAnswer. Calibrated against human labels."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agents.llm import LLMUsage, _structured_fn
from app.agents.schemas import JudgeTurn
from app.config import settings
from app.runtime.state import FinalAnswer

JudgeFn = Callable[[list], tuple[JudgeTurn, LLMUsage]]
DIMENSIONS = ("clarity", "uncertainty_honesty", "chart_appropriateness",
              "methodological_soundness")
_PROMPT = (Path(__file__).resolve().parents[1] / "agents" / "prompts" / "judge.md")


def real_judge(model: str | None = None) -> JudgeFn:
    """The tier-2 judge, pinned to `judge_model` unless a caller overrides it.

    Passing no model at all would resolve through `Settings.model_for("judge")`,
    which collapses to `analyst_model` under `cheap_mode` — so a default-config
    judged run was gpt-4o-mini scoring gpt-4o-mini's own answers. Pinning here
    keeps the judge independent of the config under test, which is the entire
    reason the judge is a separately configured model.
    """
    return _structured_fn("judge", JudgeTurn, model or settings().judge_model)


def _answer_digest(final: FinalAnswer) -> str:
    claims = [
        {"text": v.claim.text, "kind": v.claim.kind, "value": v.claim.value,
         "status": v.status,
         "methodology": v.claim.methodology.model_dump() if v.claim.methodology else None}
        for v in final.claims
    ]
    return json.dumps({"narrative": final.narrative, "claims": claims,
                       "charts": len(final.charts), "failed": final.failed},
                      indent=2, default=str)


def judge_answer(question: str, final: FinalAnswer,
                 judge: JudgeFn) -> tuple[JudgeTurn, LLMUsage]:
    prompt = _PROMPT.read_text().format(question=question, answer=_answer_digest(final))
    return judge([HumanMessage(content=prompt)])
