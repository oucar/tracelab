"""Generate starter questions for a freshly uploaded dataset (a cheap mini call)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage

from app.agents.llm import LLMUsage, _structured_fn
from app.agents.schemas import SuggestionsTurn

SuggestFn = Callable[[list], tuple[SuggestionsTurn, LLMUsage]]
_PROMPT = Path(__file__).parent / "prompts" / "suggest.md"


def real_suggester() -> SuggestFn:
    # Reuse the planner's (mini) model — suggestions never need a strong model.
    return _structured_fn("planner", SuggestionsTurn)


def suggest_questions(profile: dict, suggester: SuggestFn | None = None) -> list[str]:
    suggester = suggester or real_suggester()
    prompt = _PROMPT.read_text().format(profile=json.dumps(profile, indent=2, default=str))
    turn, _usage = suggester([HumanMessage(content=prompt)])
    return [q.strip() for q in turn.questions if q.strip()][:4]
