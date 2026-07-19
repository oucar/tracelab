"""LLM access, isolated so tests can inject stubs.

The graph never imports an OpenAI client directly; it receives callables via
`GraphDeps`, each returning `(output, LLMUsage)`. Tests provide plain Python
functions — the whole graph runs without an API key. `GraphDeps.default()`
wires the real models with structured output and one repair retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from app.agents.schemas import AnalystTurn, CriticTurn, PlannerTurn
from app.config import settings
from app.runtime.state import SandboxResult


@dataclass(frozen=True)
class LLMUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""  # attached by _usage_of; "" for stubs → cost 0


class MalformedOutputError(Exception):
    def __init__(self, role: str, detail: str) -> None:
        self.role = role
        super().__init__(
            f"{role} produced malformed structured output after repair retry: {detail}"
        )


def _usage_of(raw: object) -> LLMUsage:
    meta = getattr(raw, "usage_metadata", None) or {}
    resp = getattr(raw, "response_metadata", None) or {}
    return LLMUsage(
        tokens_in=meta.get("input_tokens", 0),
        tokens_out=meta.get("output_tokens", 0),
        model=resp.get("model_name", ""),
    )


def invoke_structured(
    structured_invoke: Callable[[list[BaseMessage]], dict],
    messages: list[BaseMessage],
    role: str,
) -> tuple[BaseModel, LLMUsage]:
    """Invoke a `with_structured_output(include_raw=True)` runnable with ONE repair retry."""
    out = structured_invoke(messages)
    usage = _usage_of(out["raw"])
    if out["parsed"] is not None:
        return out["parsed"], usage

    repair = list(messages) + [
        HumanMessage(
            content=(
                f"Your previous response failed validation: {out['parsing_error']}. "
                "Respond again, matching the required schema exactly."
            )
        )
    ]
    out = structured_invoke(repair)
    second = _usage_of(out["raw"])
    usage = LLMUsage(
        usage.tokens_in + second.tokens_in,
        usage.tokens_out + second.tokens_out,
        model=usage.model or second.model,
    )
    if out["parsed"] is None:
        raise MalformedOutputError(role, str(out["parsing_error"]))
    return out["parsed"], usage


def _structured_fn(role: str, schema: type[BaseModel]):
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(model=cfg.model_for(role), temperature=0, api_key=cfg.openai_api_key)
    runnable = model.with_structured_output(schema, include_raw=True)

    def call(messages: list[BaseMessage]):
        return invoke_structured(runnable.invoke, messages, role)

    return call


def _real_compose() -> "ComposeFn":
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(model=cfg.model_for("composer"), temperature=0, api_key=cfg.openai_api_key)

    def compose(messages: list[BaseMessage]) -> tuple[str, LLMUsage]:
        response = model.invoke(messages)
        return response.content, _usage_of(response)

    return compose


PlannerFn = Callable[[list[BaseMessage]], tuple[PlannerTurn, LLMUsage]]
AnalystFn = Callable[[list[BaseMessage]], tuple[AnalystTurn, LLMUsage]]
CriticFn = Callable[[list[BaseMessage]], tuple[CriticTurn, LLMUsage]]
ComposeFn = Callable[[list[BaseMessage]], tuple[str, LLMUsage]]
SandboxFn = Callable[[str, str], SandboxResult]


@dataclass
class GraphDeps:
    """Injected model callables. Tests pass stubs; production uses `default()`."""

    planner: PlannerFn
    analyst_turn: AnalystFn
    critic_turn: CriticFn
    compose: ComposeFn
    run_code: SandboxFn | None = None

    def __post_init__(self) -> None:
        if self.run_code is None:
            from app.sandbox.executor import run_code

            self.run_code = run_code

    @classmethod
    def default(cls) -> "GraphDeps":
        return cls(
            planner=_structured_fn("planner", PlannerTurn),
            analyst_turn=_structured_fn("analyst", AnalystTurn),
            critic_turn=_structured_fn("critic", CriticTurn),
            compose=_real_compose(),
        )
