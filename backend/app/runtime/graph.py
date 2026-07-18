"""The M1 graph: analyst → composer.

Deliberately minimal — the point of M1 is that the vertical slice (upload →
graph → sandbox → streamed answer) works end to end with events flowing.
M2 replaces the entry edge with a planner, adds parallel analysts via Send,
and inserts the critic's verification gate as conditional edges.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.llm import GraphDeps
from app.config import settings
from app.runtime.events import AgentEvent, EventType, bus
from app.runtime.state import AnalysisStep, RunState
from app.sandbox.executor import run_code

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents" / "prompts"


def _prompt(name: str, **kwargs: object) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return text.format(**kwargs) if kwargs else text


def _emit(state: RunState, agent: str, type_: EventType, payload: dict, started: float) -> None:
    bus.emit(
        AgentEvent(
            run_id=state.run_id,
            agent=agent,
            type=type_,
            payload=payload,
            started_at=started,
            duration_ms=int((time.time() - started) * 1000),
        )
    )


def analyst_node(state: RunState, deps: GraphDeps) -> dict:
    cfg = settings()
    system = SystemMessage(
        content=_prompt(
            "analyst",
            max_iterations=cfg.max_analyst_iterations,
            profile=state.dataset_profile,
        )
    )
    messages: list = [system, HumanMessage(content=state.question)]
    steps: list[AnalysisStep] = []

    for iteration in range(1, cfg.max_analyst_iterations + 1):
        t0 = time.time()
        turn = deps.analyst_turn(messages)
        _emit(
            state,
            "analyst",
            EventType.LLM_CALL,
            {"iteration": iteration, "action": turn.action},
            t0,
        )

        if turn.action == "finish":
            return {"steps": steps, "analyst_findings": turn.findings}

        t0 = time.time()
        result = run_code(turn.code, state.dataset_path)
        steps.append(AnalysisStep(iteration=iteration, result=result))
        _emit(
            state,
            "analyst",
            EventType.TOOL_CALL,
            {
                "iteration": iteration,
                "code": turn.code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
            t0,
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Execution result (exit={result.exit_code}, timed_out={result.timed_out})\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            )
        )

    return {
        "steps": steps,
        "analyst_failed": True,
        "failure_reason": f"analyst exhausted {cfg.max_analyst_iterations} iterations without findings",
    }


def composer_node(state: RunState, deps: GraphDeps) -> dict:
    t0 = time.time()
    if state.analyst_failed:
        context = f"The analysis FAILED: {state.failure_reason}. Compose an honest failure answer."
    else:
        context = f"Analyst findings:\n{state.analyst_findings}"
    answer = deps.compose(
        [
            SystemMessage(content=_prompt("composer")),
            HumanMessage(content=f"Question: {state.question}\n\n{context}"),
        ]
    )
    _emit(state, "composer", EventType.LLM_CALL, {"answer": answer}, t0)
    return {"final_answer": answer}


def build_graph(deps: GraphDeps):
    g = StateGraph(RunState)
    g.add_node("analyst", lambda s: analyst_node(s, deps))
    g.add_node("composer", lambda s: composer_node(s, deps))
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "composer")
    g.add_edge("composer", END)
    return g.compile()


def execute_run(state: RunState, deps: GraphDeps | None = None) -> RunState:
    """Run the graph for one question, emitting lifecycle events."""
    deps = deps or GraphDeps.default()
    t0 = time.time()
    bus.emit(
        AgentEvent(
            run_id=state.run_id,
            agent="system",
            type=EventType.RUN_STARTED,
            payload={"question": state.question},
        )
    )
    try:
        final = build_graph(deps).invoke(state)
        final_state = RunState.model_validate(final)
        _emit(final_state, "system", EventType.RUN_FINISHED, {"answer": final_state.final_answer}, t0)
        return final_state
    except Exception as exc:  # surface, don't swallow — the UI shows honest errors
        bus.emit(
            AgentEvent(
                run_id=state.run_id,
                agent="system",
                type=EventType.ERROR,
                payload={"error": str(exc)},
            )
        )
        raise
