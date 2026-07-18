"""LLM access, isolated so tests can inject stubs.

The graph never imports an OpenAI client directly; it receives two callables
via `GraphDeps`. Tests provide plain Python functions — the whole graph runs
without an API key. `GraphDeps.default()` wires the real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import BaseMessage

from app.agents.schemas import AnalystTurn
from app.config import settings


def _real_analyst_turn() -> Callable[[list[BaseMessage]], AnalystTurn]:
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(
        model=cfg.model_for("analyst"), temperature=0, api_key=cfg.openai_api_key
    ).with_structured_output(AnalystTurn)

    def turn(messages: list[BaseMessage]) -> AnalystTurn:
        return model.invoke(messages)

    return turn


def _real_compose() -> Callable[[list[BaseMessage]], str]:
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(model=cfg.model_for("composer"), temperature=0, api_key=cfg.openai_api_key)

    def compose(messages: list[BaseMessage]) -> str:
        return model.invoke(messages).content

    return compose


@dataclass
class GraphDeps:
    """Injected model callables. Tests pass stubs; production uses `default()`."""

    analyst_turn: Callable[[list[BaseMessage]], AnalystTurn]
    compose: Callable[[list[BaseMessage]], str]

    @classmethod
    def default(cls) -> "GraphDeps":
        return cls(analyst_turn=_real_analyst_turn(), compose=_real_compose())
