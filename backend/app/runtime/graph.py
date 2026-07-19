"""The M2 graph: planner → parallel analysts (Send) → critic gate → composer.

Orchestration logic lives in the conditional edges:
  - after planner: fan out one analyst branch per independent plan step (Send API),
    or straight to composer on planner failure (honest failure path);
  - after critic: all verified → composer; discrepancy → ONE bounded retry of the
    disputed steps with the critic's findings injected; else composer ships the
    answer with explicit unverified flags.

The critic never sees analyst code — only the question, the dataset profile, and
the claims. Independent derivation is the point (see docs/architecture.md).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.llm import GraphDeps, LLMUsage, MalformedOutputError
from app.config import settings
from app.runtime.budget import AgentBudget, BudgetExceeded
from app.runtime.chartspec import ChartSpec, extract_chart_specs
from app.runtime.events import AgentEvent, EventType, bus
from app.runtime.reconcile import reconcile_claims
from app.runtime.state import (
    AnalysisStep,
    AnalystResult,
    AnalystTask,
    Claim,
    FinalAnswer,
    Plan,
    RunState,
    VerifiedClaim,
)
from app.sandbox.executor import run_code

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents" / "prompts"


def _prompt(name: str, **kwargs: object) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return text.format(**kwargs) if kwargs else text


def _emit(
    run_id: str,
    agent: str,
    type_: EventType,
    payload: dict,
    started: float,
    usage: LLMUsage | None = None,
) -> None:
    bus.emit(
        AgentEvent(
            run_id=run_id,
            agent=agent,
            type=type_,
            payload=payload,
            tokens_in=usage.tokens_in if usage else 0,
            tokens_out=usage.tokens_out if usage else 0,
            started_at=started,
            duration_ms=int((time.time() - started) * 1000),
        )
    )


def _latest_results(results: list[AnalystResult]) -> dict[int, AnalystResult]:
    """Last result per step wins — a retried step's result supersedes the original."""
    latest: dict[int, AnalystResult] = {}
    for r in results:
        latest[r.step_id] = r
    return dict(sorted(latest.items()))


# ── planner ──────────────────────────────────────────────────────────────────


def planner_node(state: RunState, deps: GraphDeps) -> dict:
    cfg = settings()
    budget = AgentBudget.for_role("planner")
    t0 = time.time()
    try:
        turn, usage = deps.planner(
            [
                SystemMessage(
                    content=_prompt(
                        "planner", max_steps=cfg.max_plan_steps, profile=state.dataset_profile
                    )
                ),
                HumanMessage(content=state.question),
            ]
        )
        budget.spend_llm(usage.tokens_in, usage.tokens_out)
    except (BudgetExceeded, MalformedOutputError) as exc:
        _emit(state.run_id, "planner", EventType.ERROR, {"error": str(exc)}, t0)
        return {"planner_failed": True, "planner_failure_reason": str(exc)}

    steps = turn.steps[: cfg.max_plan_steps]
    for i, step in enumerate(steps, start=1):
        step.id = i
    _emit(
        state.run_id,
        "planner",
        EventType.LLM_CALL,
        {"plan": [s.model_dump() for s in steps], "rationale": turn.rationale},
        t0,
        usage,
    )
    if not steps:
        return {"planner_failed": True, "planner_failure_reason": "planner produced an empty plan"}
    _emit(
        state.run_id,
        "planner",
        EventType.HANDOFF,
        {"to": "analyst", "steps": [s.id for s in steps]},
        time.time(),
    )
    return {"plan": Plan(steps=steps, rationale=turn.rationale)}


def fan_out(state: RunState):
    if state.planner_failed or state.plan is None or not state.plan.steps:
        return "composer"
    return [
        Send(
            "analyst",
            AnalystTask(
                run_id=state.run_id,
                question=state.question,
                dataset_path=state.dataset_path,
                dataset_profile=state.dataset_profile,
                step=step,
            ).model_dump(),
        )
        for step in state.plan.steps
    ]


# ── analyst (Send target; also the retry target) ─────────────────────────────


def _assign_claim_ids(step_id: int, claims: list[Claim]) -> list[Claim]:
    for i, claim in enumerate(claims, start=1):
        claim.id = f"{step_id}-{i}"
        claim.step_id = step_id
    return claims


def analyst_node(task: AnalystTask | dict, deps: GraphDeps) -> dict:
    task = AnalystTask.model_validate(task)  # Send delivers a plain dict payload
    cfg = settings()
    budget = AgentBudget.for_role("analyst")
    step = task.step
    columns = [c.get("name", "") for c in task.dataset_profile.get("columns", [])]

    messages: list = [
        SystemMessage(
            content=_prompt(
                "analyst",
                max_iterations=cfg.max_analyst_iterations,
                profile=task.dataset_profile,
                step_description=step.description,
                method=step.method,
                alpha=cfg.alpha,
            )
        ),
        HumanMessage(
            content=f"Overall question: {task.question}\n[step {step.id}] {step.description}"
        ),
    ]
    if task.critic_feedback:
        messages.append(
            HumanMessage(
                content=(
                    "A previous attempt at this step was disputed by the critic:\n"
                    f"{task.critic_feedback}\n"
                    "Re-derive the result carefully and resolve the discrepancy."
                )
            )
        )

    iterations: list[AnalysisStep] = []
    chart_specs: list[ChartSpec] = []
    chart_rejections: list[str] = []

    def failure(reason: str) -> dict:
        return {
            "analyst_results": [
                AnalystResult(
                    step_id=step.id, iterations=iterations, failed=True, failure_reason=reason
                )
            ]
        }

    for iteration in range(1, cfg.max_analyst_iterations + 1):
        t0 = time.time()
        try:
            turn, usage = deps.analyst_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            _emit(
                task.run_id,
                "analyst",
                EventType.ERROR,
                {"step_id": step.id, "error": str(exc)},
                t0,
            )
            return failure(str(exc))
        _emit(
            task.run_id,
            "analyst",
            EventType.LLM_CALL,
            {"step_id": step.id, "iteration": iteration, "action": turn.action},
            t0,
            usage,
        )

        if turn.action == "finish":
            return {
                "analyst_results": [
                    AnalystResult(
                        step_id=step.id,
                        findings=turn.findings,
                        claims=_assign_claim_ids(step.id, turn.claims),
                        chart_specs=chart_specs,
                        chart_rejections=chart_rejections,
                        iterations=iterations,
                    )
                ]
            }

        try:
            budget.spend_tool()
        except BudgetExceeded as exc:
            return failure(str(exc))
        t0 = time.time()
        result = run_code(turn.code, task.dataset_path)
        iterations.append(AnalysisStep(iteration=iteration, result=result))
        specs, rejections = extract_chart_specs(result.artifacts, columns)
        chart_specs.extend(specs)
        chart_rejections.extend(rejections)
        _emit(
            task.run_id,
            "analyst",
            EventType.TOOL_CALL,
            {
                "step_id": step.id,
                "iteration": iteration,
                "code": turn.code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "charts_accepted": len(specs),
                "charts_rejected": rejections,
            },
            t0,
        )
        feedback = (
            f"Execution result (exit={result.exit_code}, timed_out={result.timed_out})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        if result.exit_code == 0 and not result.stdout.strip():
            feedback += (
                "\nWARNING: your script printed NOTHING to stdout. Only printed output "
                "comes back to you — wrap every result in print(...)."
            )
        if rejections:
            feedback += "\nRejected charts:\n" + "\n".join(rejections)
        messages.append(HumanMessage(content=feedback))

    return failure(f"analyst exhausted {cfg.max_analyst_iterations} iterations without findings")


# ── critic ───────────────────────────────────────────────────────────────────


def critic_node(state: RunState, deps: GraphDeps) -> dict:
    cfg = settings()
    latest = _latest_results(state.analyst_results)
    claims = [c for r in latest.values() if not r.failed for c in r.claims]
    if not claims:
        return {"verdicts": [], "retry_steps": []}

    budget = AgentBudget.for_role("critic")
    claims_json = json.dumps([c.model_dump() for c in claims], indent=2, default=str)
    messages: list = [
        SystemMessage(
            content=_prompt(
                "critic", max_iterations=cfg.max_critic_iterations, profile=state.dataset_profile
            )
        ),
        HumanMessage(content=f"Question: {state.question}\n\nClaims to verify:\n{claims_json}"),
    ]

    findings = []
    for iteration in range(1, cfg.max_critic_iterations + 2):
        t0 = time.time()
        try:
            turn, usage = deps.critic_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            _emit(state.run_id, "critic", EventType.ERROR, {"error": str(exc)}, t0)
            break  # claims fall through as unverifiable
        _emit(
            state.run_id,
            "critic",
            EventType.LLM_CALL,
            {"iteration": iteration, "action": turn.action},
            t0,
            usage,
        )
        if turn.action == "finish":
            findings = turn.findings
            break
        try:
            budget.spend_tool()
        except BudgetExceeded:
            break
        t0 = time.time()
        result = run_code(turn.code, state.dataset_path)
        _emit(
            state.run_id,
            "critic",
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
        critic_feedback = (
            f"Execution result (exit={result.exit_code}, timed_out={result.timed_out})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        if result.exit_code == 0 and not result.stdout.strip():
            critic_feedback += (
                "\nWARNING: your script printed NOTHING to stdout. Only printed output "
                "comes back to you — wrap every result in print(...)."
            )
        messages.append(HumanMessage(content=critic_feedback))

    verdicts = reconcile_claims(claims, findings, cfg.numeric_rel_tolerance)
    for verdict in verdicts:
        _emit(state.run_id, "critic", EventType.VERDICT, verdict.model_dump(), time.time())

    claim_step = {c.id: c.step_id for c in claims}
    disputed = sorted({claim_step[v.claim_id] for v in verdicts if v.status == "discrepancy"})
    if disputed and state.retry_count < cfg.max_retries:
        _emit(
            state.run_id,
            "critic",
            EventType.HANDOFF,
            {"to": "analyst", "retry_steps": disputed},
            time.time(),
        )
        return {
            "verdicts": verdicts,
            "retry_count": state.retry_count + 1,
            "retry_steps": disputed,
        }
    return {"verdicts": verdicts, "retry_steps": []}


def route_after_critic(state: RunState):
    if not state.retry_steps or state.plan is None:
        return "composer"
    step_map = {s.id: s for s in state.plan.steps}
    claim_step = {c.id: c.step_id for r in state.analyst_results for c in r.claims}
    feedback: dict[int, list[str]] = {sid: [] for sid in state.retry_steps}
    for v in state.verdicts:
        sid = claim_step.get(v.claim_id)
        if sid in feedback and v.status == "discrepancy":
            feedback[sid].append(f"- {v.claim_id}: {v.reason}")
    return [
        Send(
            "analyst",
            AnalystTask(
                run_id=state.run_id,
                question=state.question,
                dataset_path=state.dataset_path,
                dataset_profile=state.dataset_profile,
                step=step_map[sid],
                critic_feedback="\n".join(feedback[sid]) or "the critic disputed this step",
            ).model_dump(),
        )
        for sid in state.retry_steps
        if sid in step_map
    ]


# ── composer ─────────────────────────────────────────────────────────────────


def composer_node(state: RunState, deps: GraphDeps) -> dict:
    t0 = time.time()
    latest = _latest_results(state.analyst_results)
    verdict_map = {v.claim_id: v for v in state.verdicts}

    verified_claims: list[VerifiedClaim] = []
    for result in latest.values():
        for claim in result.claims:
            verdict = verdict_map.get(claim.id)
            if verdict is not None and verdict.status == "verified":
                verified_claims.append(VerifiedClaim(claim=claim, status="verified"))
            else:
                detail = verdict.reason if verdict else "not checked by the critic"
                verified_claims.append(
                    VerifiedClaim(claim=claim, status="unverified", detail=detail)
                )

    charts = [spec for r in latest.values() for spec in r.chart_specs]
    all_failed = bool(latest) and all(r.failed for r in latest.values())
    failed = state.planner_failed or not latest or all_failed

    if state.planner_failed:
        context = (
            f"Planning FAILED: {state.planner_failure_reason}. Compose an honest failure answer."
        )
    else:
        parts: list[str] = []
        for result in latest.values():
            if result.failed:
                parts.append(f"Step {result.step_id} FAILED: {result.failure_reason}")
            else:
                parts.append(f"Step {result.step_id} findings:\n{result.findings}")
        parts.append("Claims with verification status:")
        for vc in verified_claims:
            line = f"- [{vc.status}] {vc.claim.text}"
            if vc.claim.value is not None:
                line += f" = {vc.claim.value}"
            if vc.detail:
                line += f" ({vc.detail})"
            parts.append(line)
        context = "\n\n".join(parts)

    answer, usage = deps.compose(
        [
            SystemMessage(content=_prompt("composer")),
            HumanMessage(content=f"Question: {state.question}\n\n{context}"),
        ]
    )
    _emit(state.run_id, "composer", EventType.LLM_CALL, {"answer": answer}, t0, usage)
    final = FinalAnswer(narrative=answer, claims=verified_claims, charts=charts, failed=failed)
    return {"final_answer": answer, "final": final}


# ── graph assembly ───────────────────────────────────────────────────────────


def build_graph(deps: GraphDeps, checkpointer=None):
    g = StateGraph(RunState)
    g.add_node("planner", lambda s: planner_node(s, deps))
    g.add_node("analyst", lambda t: analyst_node(t, deps), input_schema=AnalystTask)
    g.add_node("critic", lambda s: critic_node(s, deps))
    g.add_node("composer", lambda s: composer_node(s, deps))
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["analyst", "composer"])
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["analyst", "composer"])
    g.add_edge("composer", END)
    return g.compile(checkpointer=checkpointer)


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
        # Local imports keep unit tests (which call build_graph directly) off the
        # checkpoint DB; SqliteSaver checkpoints every superstep under thread_id=run_id.
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(settings().checkpoints_db_path), check_same_thread=False)
        try:
            graph = build_graph(deps, checkpointer=SqliteSaver(conn))
            final = graph.invoke(state, {"configurable": {"thread_id": state.run_id}})
        finally:
            conn.close()
        final_state = RunState.model_validate(final)
        _emit(
            final_state.run_id,
            "system",
            EventType.RUN_FINISHED,
            {
                "answer": final_state.final_answer,
                "final": final_state.final.model_dump() if final_state.final else None,
            },
            t0,
        )
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
