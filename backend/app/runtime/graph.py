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
from app.tracing import pricing
from app.runtime.reconcile import reconcile_claims
from app.runtime.state import (
    AnalysisStep,
    AnalystResult,
    AnalystTask,
    Claim,
    FinalAnswer,
    Plan,
    PlanStep,
    RunState,
    VerifiedClaim,
)

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
    parent: str | None = None,
) -> str:
    event = AgentEvent(
        run_id=run_id,
        parent_span_id=parent,
        agent=agent,
        type=type_,
        payload=payload,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
        cost_usd=pricing.cost_usd(usage.model, usage.tokens_in, usage.tokens_out)
        if usage
        else 0.0,
        started_at=started,
        duration_ms=int((time.time() - started) * 1000),
    )
    bus.emit(event)
    return event.span_id


def _latest_results(results: list[AnalystResult]) -> dict[int, AnalystResult]:
    """Last result per step wins — a retried step's result supersedes the original."""
    latest: dict[int, AnalystResult] = {}
    for r in results:
        latest[r.step_id] = r
    return dict(sorted(latest.items()))


# ── router ───────────────────────────────────────────────────────────────────


def router_node(state: RunState, deps: GraphDeps) -> dict:
    t0 = time.time()
    if deps.router is None:
        route, reason, usage = "multi_step", "no router configured", None
    else:
        budget = AgentBudget.for_role("router")
        try:
            turn, usage = deps.router(
                [
                    HumanMessage(
                        content=_prompt(
                            "router", profile=state.dataset_profile, question=state.question
                        )
                    )
                ]
            )
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            _emit(
                state.run_id,
                "router",
                EventType.ERROR,
                {"error": str(exc)},
                t0,
                parent=state.root_span_id,
            )
            route, reason, usage = "multi_step", f"router failed, defaulting: {exc}", None
        else:
            route, reason = turn.route, turn.reason

    _emit(
        state.run_id,
        "router",
        EventType.HANDOFF,
        {"route": route, "reason": reason, "to": "analyst" if route == "simple" else "planner"},
        t0,
        usage,
        parent=state.root_span_id,
    )
    out: dict = {"route": route, "route_reason": reason}
    if route == "simple":
        step = PlanStep(id=1, description=state.question, method="descriptive")
        out["plan"] = Plan(steps=[step], rationale=f"router: simple — {reason}")
    return out


def route_from_router(state: RunState):
    if state.route == "simple" and state.plan and state.plan.steps:
        step = state.plan.steps[0]
        return [
            Send(
                "analyst",
                AnalystTask(
                    run_id=state.run_id,
                    question=state.question,
                    dataset_path=state.dataset_path,
                    dataset_profile=state.dataset_profile,
                    root_span_id=state.root_span_id,
                    step=step,
                ).model_dump(),
            )
        ]
    return "planner"


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
        _emit(
            state.run_id,
            "planner",
            EventType.ERROR,
            {"error": str(exc)},
            t0,
            parent=state.root_span_id,
        )
        return {"planner_failed": True, "planner_failure_reason": str(exc)}

    steps = turn.steps[: cfg.max_plan_steps]
    for i, step in enumerate(steps, start=1):
        step.id = i
    span = _emit(  # the plan llm_call — the planner's node root
        state.run_id,
        "planner",
        EventType.LLM_CALL,
        {"plan": [s.model_dump() for s in steps], "rationale": turn.rationale},
        t0,
        usage,
        parent=state.root_span_id,
    )
    if not steps:
        return {"planner_failed": True, "planner_failure_reason": "planner produced an empty plan"}
    _emit(
        state.run_id,
        "planner",
        EventType.HANDOFF,
        {"to": "analyst", "steps": [s.id for s in steps]},
        time.time(),
        parent=span,
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
                root_span_id=state.root_span_id,
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
    branch_root: str | None = None
    cfg = settings()
    budget = AgentBudget.for_role("analyst")
    step = task.step
    columns = [c.get("name", "") for c in task.dataset_profile.get("columns", [])]
    wants_chart = any(k in task.question.lower() for k in ("chart", "graph", "plot", "visuali"))

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
        # A step can fail to FINISH cleanly yet still have produced a valid,
        # accepted chart — never discard it. The composer surfaces it honestly.
        return {
            "analyst_results": [
                AnalystResult(
                    step_id=step.id,
                    iterations=iterations,
                    chart_specs=chart_specs,
                    chart_rejections=chart_rejections,
                    failed=True,
                    failure_reason=reason,
                )
            ]
        }

    for iteration in range(1, cfg.max_analyst_iterations + 1):
        t0 = time.time()
        try:
            turn, usage = deps.analyst_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            sid = _emit(
                task.run_id,
                "analyst",
                EventType.ERROR,
                {"step_id": step.id, "error": str(exc)},
                t0,
                parent=branch_root or task.root_span_id,
            )
            branch_root = branch_root or sid
            return failure(str(exc))
        sid = _emit(
            task.run_id,
            "analyst",
            EventType.LLM_CALL,
            {"step_id": step.id, "iteration": iteration, "action": turn.action},
            t0,
            usage,
            parent=branch_root or task.root_span_id,
        )
        branch_root = branch_root or sid

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
        result = deps.run_code(turn.code, task.dataset_path)
        iterations.append(AnalysisStep(iteration=iteration, result=result))
        specs, rejections = extract_chart_specs(result.artifacts, columns)
        chart_specs.extend(specs)
        chart_rejections.extend(rejections)
        sid = _emit(
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
            parent=branch_root or task.root_span_id,
        )
        branch_root = branch_root or sid
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
        if result.exit_code == 0 and result.stdout.strip() and not rejections:
            # Deterministic pressure so gpt-4o-mini doesn't loop re-running near-identical
            # code and never emit `finish`. If the user asked for a chart and none exists
            # yet, steer to produce it in ONE more script; otherwise, finish now.
            if wants_chart and not chart_specs:
                feedback += (
                    "\n\nYour script succeeded, but the user asked for a CHART and none "
                    "has been written yet. In ONE more script, recompute the values and "
                    "write the chart JSON to ./artifacts/chart_<name>.json (a JSON spec, "
                    "NOT matplotlib), then finish."
                )
            else:
                charted = " Your chart was accepted." if specs else ""
                feedback += (
                    f"\n\nYour script succeeded and produced output.{charted} Your NEXT "
                    "action MUST be `finish` with findings and claims. Do NOT run more "
                    "code unless the output shows an error or is missing something specific."
                )
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
    branch_root: str | None = None
    for iteration in range(1, cfg.max_critic_iterations + 2):
        t0 = time.time()
        try:
            turn, usage = deps.critic_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            sid = _emit(
                state.run_id,
                "critic",
                EventType.ERROR,
                {"error": str(exc)},
                t0,
                parent=branch_root or state.root_span_id,
            )
            branch_root = branch_root or sid
            break  # claims fall through as unverifiable
        sid = _emit(
            state.run_id,
            "critic",
            EventType.LLM_CALL,
            {"iteration": iteration, "action": turn.action},
            t0,
            usage,
            parent=branch_root or state.root_span_id,
        )
        branch_root = branch_root or sid
        if turn.action == "finish":
            findings = turn.findings
            break
        try:
            budget.spend_tool()
        except BudgetExceeded:
            break
        t0 = time.time()
        result = deps.run_code(turn.code, state.dataset_path)
        sid = _emit(
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
            parent=branch_root or state.root_span_id,
        )
        branch_root = branch_root or sid
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
        _emit(
            state.run_id,
            "critic",
            EventType.VERDICT,
            verdict.model_dump(),
            time.time(),
            parent=branch_root or state.root_span_id,
        )

    claim_step = {c.id: c.step_id for c in claims}
    disputed = sorted({claim_step[v.claim_id] for v in verdicts if v.status == "discrepancy"})
    if disputed and state.retry_count < cfg.max_retries:
        _emit(
            state.run_id,
            "critic",
            EventType.HANDOFF,
            {"to": "analyst", "retry_steps": disputed},
            time.time(),
            parent=branch_root or state.root_span_id,
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
                root_span_id=state.root_span_id,
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

    # Folding is the simple-route fast path specifically (Task 13), not a generic
    # "any single verified step" shortcut — a planner-derived single-step plan on
    # the multi_step/statistical routes still goes through the composer LLM.
    single = state.route == "simple" and len(latest) == 1 and not state.planner_failed
    only = next(iter(latest.values())) if single else None
    all_verified = (
        single
        and only is not None
        and not only.failed
        and bool(state.verdicts)
        and all(v.status == "verified" for v in state.verdicts)
    )
    if all_verified:
        _emit(
            state.run_id,
            "composer",
            EventType.HANDOFF,
            {"folded": True, "reason": "single verified finding"},
            t0,
            parent=state.root_span_id,
        )
        final = FinalAnswer(
            narrative=only.findings, claims=verified_claims, charts=charts, failed=False
        )
        return {"final_answer": only.findings, "final": final}

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
        if charts:
            chart_lines = [
                f"- {c.kind} chart: {c.title}" if c.title else f"- {c.kind} chart" for c in charts
            ]
            parts.append(
                "Charts already produced and rendered in the UI beside your answer "
                "(reference them; never claim no visualization exists):\n"
                + "\n".join(chart_lines)
            )
        context = "\n\n".join(parts)

    answer, usage = deps.compose(
        [
            SystemMessage(content=_prompt("composer")),
            HumanMessage(content=f"Question: {state.question}\n\n{context}"),
        ]
    )
    _emit(
        state.run_id,
        "composer",
        EventType.LLM_CALL,
        {"answer": answer},
        t0,
        usage,
        parent=state.root_span_id,
    )
    final = FinalAnswer(narrative=answer, claims=verified_claims, charts=charts, failed=failed)
    return {"final_answer": answer, "final": final}


# ── graph assembly ───────────────────────────────────────────────────────────


def build_graph(deps: GraphDeps, checkpointer=None):
    g = StateGraph(RunState)
    g.add_node("router", lambda s: router_node(s, deps))
    g.add_node("planner", lambda s: planner_node(s, deps))
    g.add_node("analyst", lambda t: analyst_node(t, deps), input_schema=AnalystTask)
    g.add_node("critic", lambda s: critic_node(s, deps))
    g.add_node("composer", lambda s: composer_node(s, deps))
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_from_router, ["planner", "analyst"])
    g.add_conditional_edges("planner", fan_out, ["analyst", "composer"])
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["analyst", "composer"])
    g.add_edge("composer", END)
    return g.compile(checkpointer=checkpointer)


def execute_run(state: RunState, deps: GraphDeps | None = None) -> RunState:
    """Run the graph for one question, emitting lifecycle events."""
    deps = deps or GraphDeps.default()
    t0 = time.time()
    root = AgentEvent(
        run_id=state.run_id,
        agent="system",
        type=EventType.RUN_STARTED,
        payload={"question": state.question},
    )
    bus.emit(root)
    state.root_span_id = root.span_id
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
            parent=final_state.root_span_id,
        )
        return final_state
    except Exception as exc:  # surface, don't swallow — the UI shows honest errors
        bus.emit(
            AgentEvent(
                run_id=state.run_id,
                agent="system",
                type=EventType.ERROR,
                payload={"error": str(exc)},
                parent_span_id=root.span_id,
            )
        )
        raise
