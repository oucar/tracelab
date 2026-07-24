"""M2 graph routing with stubbed LLMs — the whole graph runs without an API key."""

import json
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn, RouterTurn
from app.runtime.graph import execute_run
from app.runtime.events import bus
from app.runtime.state import Claim, PlanStep, RunState

U = LLMUsage(tokens_in=10, tokens_out=5)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip,day\n10,2,Sat\n20,5,Sun\n30,9,Mon\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="What is the total fare and which day is busiest?",
        dataset_path=str(dataset),
        dataset_profile={
            "rows": 3,
            "columns": [{"name": "fare"}, {"name": "tip"}, {"name": "day"}],
        },
    )


def one_step_planner(messages):
    return (
        PlannerTurn(steps=[PlanStep(description="Compute the total fare", method="descriptive")]),
        U,
    )


def two_step_planner(messages):
    return (
        PlannerTurn(
            steps=[
                PlanStep(description="Compute the total fare", method="descriptive"),
                PlanStep(description="Find the busiest day", method="descriptive"),
            ]
        ),
        U,
    )


def count_claim(value: float) -> Claim:
    return Claim(text="row count", kind="numeric", value=value)


def finishing_analyst(findings_by_step: dict[int, tuple[str, list[Claim]]]):
    """Analyst stub that finishes immediately with per-step findings and claims."""

    def turn(messages):
        text = "\n".join(m.content for m in messages)
        for step_id, (findings, claims) in findings_by_step.items():
            if f"[step {step_id}]" in text:
                return AnalystTurn(action="finish", findings=findings, claims=claims), U
        raise AssertionError(f"no step marker in prompt: {text[-200:]}")

    return turn


def verifying_critic(messages):
    """Critic that confirms every claim it is shown."""
    text = next(m.content for m in messages if "Claims to verify:" in m.content)
    claims = json.loads(text.split("Claims to verify:\n", 1)[1])
    return (
        CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
        ),
        U,
    )


def test_planner_fans_out_and_verified_claims_compose(dataset):
    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst(
            {
                1: ("Total fare is 60.", [Claim(text="total fare", kind="numeric", value=60.0)]),
                2: ("Busiest day is Sat.", [Claim(text="busiest day", kind="categorical", value="Sat")]),
            }
        ),
        critic_turn=verifying_critic,
        compose=lambda m: ("Total fare is 60; Saturday is busiest.", U),
    )
    final = execute_run(make_state(dataset, "run-fanout"), deps)

    assert len(final.analyst_results) == 2
    assert {r.step_id for r in final.analyst_results} == {1, 2}
    assert all(v.status == "verified" for v in final.verdicts) and len(final.verdicts) == 2
    assert final.retry_count == 0
    assert final.final is not None
    assert all(vc.status == "verified" for vc in final.final.claims)


def test_discrepancy_triggers_one_bounded_retry_with_feedback(dataset):
    analyst_calls = []

    def analyst(messages):
        analyst_calls.append("\n".join(m.content for m in messages))
        return (
            AnalystTurn(
                action="finish",
                findings="Total fare is 55.",
                claims=[Claim(text="total fare", kind="numeric", value=55.0)],
            ),
            U,
        )

    def critic(messages):
        return (
            CriticTurn(action="finish", findings=[CriticFinding(claim_id="1-1", value=60.0)]),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=critic,
        compose=lambda m: ("Composed with caveats.", U),
    )
    final = execute_run(make_state(dataset, "run-retry"), deps)

    assert final.retry_count == 1
    assert len(analyst_calls) == 2  # original + exactly one retry
    assert "critic" in analyst_calls[1].lower()  # feedback injected
    assert final.verdicts[0].status == "discrepancy"  # still wrong after retry
    assert final.final.claims[0].status == "unverified"  # shipped with explicit flag
    assert "60" in final.final.claims[0].detail


def test_planner_failure_routes_to_honest_composer(dataset):
    def bad_planner(messages):
        return PlannerTurn(steps=[]), U

    deps = GraphDeps(
        planner=bad_planner,
        analyst_turn=lambda m: (_ for _ in ()).throw(AssertionError("analyst must not run")),
        critic_turn=lambda m: (_ for _ in ()).throw(AssertionError("critic must not run")),
        compose=lambda m: ("I could not analyze this question.", U),
    )
    final = execute_run(make_state(dataset, "run-noplan"), deps)
    assert final.planner_failed
    assert final.final.failed
    assert "could not" in final.final_answer.lower()


def test_analyst_budget_exhaustion_is_a_typed_failure(dataset):
    big = LLMUsage(tokens_in=30_000, tokens_out=0)  # over max_tokens_per_agent in one call

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=lambda m: (AnalystTurn(action="run_code", code="print(1)"), big),
        critic_turn=lambda m: (CriticTurn(action="finish", findings=[]), U),
        compose=lambda m: ("The analysis could not be completed: budget exhausted.", U),
    )
    final = execute_run(make_state(dataset, "run-budget"), deps)
    assert final.analyst_results[0].failed
    assert "budget" in final.analyst_results[0].failure_reason
    assert final.final.failed


def test_analyst_code_loop_still_works_end_to_end(dataset):
    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                AnalystTurn(
                    action="run_code",
                    code=(
                        "import pandas as pd\n"
                        "print('total', pd.read_csv('data.csv')['fare'].sum())"
                    ),
                ),
                U,
            )
        assert "total 60" in messages[-1].content
        return (
            AnalystTurn(
                action="finish",
                findings="Total fare is 60.",
                claims=[Claim(text="total fare", kind="numeric", value=60.0)],
            ),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("The total fare is 60.", U),
    )
    final = execute_run(make_state(dataset, "run-loop"), deps)
    assert final.final_answer == "The total fare is 60."
    assert final.analyst_results[0].iterations[0].result.exit_code == 0
    assert final.verdicts[0].status == "verified"


def test_events_flow_for_all_agents(dataset):
    from app.runtime.events import bus

    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst(
            {
                1: ("A", [Claim(text="total fare", kind="numeric", value=60.0)]),
                2: ("B", [Claim(text="busiest day", kind="categorical", value="Sat")]),
            }
        ),
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-events"), deps)
    events = bus.history("run-events")
    agents = {e.agent for e in events}
    types = [e.type.value for e in events]
    assert {"system", "planner", "analyst", "critic", "composer"} <= agents
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert "verdict" in types and "handoff" in types
    assert any(e.tokens_in > 0 for e in events)  # usage flows onto events


def test_span_tree_is_rooted_and_connected(dataset):
    from app.runtime.events import EventType, bus

    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst(
            {
                1: ("A", [Claim(text="total fare", kind="numeric", value=60.0)]),
                2: ("B", [Claim(text="busiest day", kind="categorical", value="Sat")]),
            }
        ),
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-tree"), deps)
    events = bus.history("run-tree")

    ids = {e.span_id for e in events}
    roots = [e for e in events if e.parent_span_id is None]
    assert len(roots) == 1 and roots[0].type == EventType.RUN_STARTED
    # every non-root event points at a span that exists in the same run
    assert all(e.parent_span_id in ids for e in events if e.parent_span_id is not None)
    # planner's llm_call is a direct child of the run root
    planner_call = next(e for e in events if e.agent == "planner" and e.type == EventType.LLM_CALL)
    assert planner_call.parent_span_id == roots[0].span_id


def test_analyst_iterations_nest_under_branch_root(dataset):
    from app.runtime.events import bus

    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('x')"), U
        return (
            AnalystTurn(
                action="finish",
                findings="F",
                claims=[Claim(text="t", kind="numeric", value=1.0)],
            ),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="one step")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-nest"), deps)
    events = bus.history("run-nest")

    analyst_events = [e for e in events if e.agent == "analyst"]
    branch_root = analyst_events[0]
    root = next(e for e in events if e.parent_span_id is None)
    assert branch_root.parent_span_id == root.span_id
    assert all(e.parent_span_id == branch_root.span_id for e in analyst_events[1:])


def test_sandbox_is_injectable_through_deps(dataset):
    from app.runtime.state import SandboxResult

    executed: list[str] = []

    def fake_run_code(code: str, dataset_path: str) -> SandboxResult:
        executed.append(code)
        return SandboxResult(code=code, stdout="CANNED OUTPUT", exit_code=0)

    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('never runs')"), U
        assert "CANNED OUTPUT" in messages[-1].content  # the fake result came back
        return (
            AnalystTurn(
                action="finish", findings="F",
                claims=[Claim(text="t", kind="numeric", value=1.0)],
            ),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="s")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
        run_code=fake_run_code,
    )
    final = execute_run(make_state(dataset, "run-fakebox"), deps)
    assert executed == ["print('never runs')"]
    assert final.analyst_results[0].iterations[0].result.stdout == "CANNED OUTPUT"


def test_plan_step_accepts_m3_methods():
    for method in ("regression", "clustering", "timeseries_backtest", "anomaly_detection"):
        assert PlanStep(description="x", method=method).method == method


# ── Task 13: router node, simple-route Send, conditional composer ────────────


def _router(route: str):
    def fn(_msgs):
        return RouterTurn(route=route, reason=f"stubbed {route}"), U

    return fn


def test_simple_route_skips_planner(dataset):
    planner_calls = []

    def spying_planner(msgs):
        planner_calls.append(msgs)
        return one_step_planner(msgs)

    deps = GraphDeps(
        planner=spying_planner,
        analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
        critic_turn=verifying_critic,
        compose=lambda m: ("unused", U),
        router=_router("simple"),
    )
    out = execute_run(make_state(dataset, "run-simple"), deps)
    assert not planner_calls  # planner LLM never called
    events = bus.history("run-simple")
    assert not [e for e in events if e.agent == "planner"]
    router_events = [e for e in events if e.agent == "router"]
    assert router_events and router_events[0].payload["route"] == "simple"
    assert out.final is not None and not out.final.failed


def test_simple_route_folds_composer(dataset):
    compose_calls = []

    def spying_compose(msgs):
        compose_calls.append(msgs)
        return ("unused", U)

    deps = GraphDeps(
        planner=one_step_planner,
        analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
        critic_turn=verifying_critic,
        compose=spying_compose,
        router=_router("simple"),
    )
    out = execute_run(make_state(dataset, "run-folded"), deps)
    assert not compose_calls  # composer LLM skipped
    folded = [
        e
        for e in bus.history("run-folded")
        if e.agent == "composer" and e.payload.get("folded")
    ]
    assert folded
    assert out.final.narrative  # narrative taken from the analyst's findings


def test_statistical_route_uses_planner(dataset):
    deps = GraphDeps(
        planner=one_step_planner,
        analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
        critic_turn=verifying_critic,
        compose=lambda m: ("composed", U),
        router=_router("statistical"),
    )
    execute_run(make_state(dataset, "run-stat"), deps)
    events = bus.history("run-stat")
    assert [e for e in events if e.agent == "planner"]
    assert [e for e in events if e.agent == "router"][0].payload["route"] == "statistical"


def test_no_router_behaves_like_multi_step(dataset):
    deps = GraphDeps(
        planner=one_step_planner,
        analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
        critic_turn=verifying_critic,
        compose=lambda m: ("composed", U),
    )
    out = execute_run(make_state(dataset, "run-norouter"), deps)
    assert out.final is not None
    route_events = [e for e in bus.history("run-norouter") if e.agent == "router"]
    assert route_events and route_events[0].payload["route"] == "multi_step"


def test_multi_finding_run_still_uses_composer(dataset):
    compose_calls = []

    def spying_compose(msgs):
        compose_calls.append(msgs)
        return ("composed", U)

    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst(
            {1: ("A.", [count_claim(3)]), 2: ("B.", [count_claim(3)])}
        ),
        critic_turn=verifying_critic,
        compose=spying_compose,
        router=_router("multi_step"),
    )
    execute_run(make_state(dataset, "run-multi"), deps)
    assert compose_calls


def test_composer_is_told_about_produced_charts(dataset):
    """Regression: the composer denied a chart that existed because its context
    never mentioned the produced charts. It must now be told about them."""
    from app.runtime.chartspec import ChartSpec
    from app.runtime.graph import composer_node
    from app.runtime.state import AnalystResult, RunState, Verdict

    spec = ChartSpec(
        kind="bar",
        title="Top 10 companies by frequency",
        x="company",
        y=["count"],
        data=[{"company": "Acme", "count": 2}],
        source_columns=["company"],
    )
    claim = Claim(id="c1", step_id=1, text="Acme appears 2 times", kind="numeric", value=2)
    result = AnalystResult(
        step_id=1, findings="Acme appears 2 times.", claims=[claim], chart_specs=[spec]
    )
    state = RunState(
        run_id="run-compose",
        question="can you generate a cool graph?",
        dataset_path=str(dataset),
        route="multi_step",  # not "simple" -> real composer LLM path, no fold
        analyst_results=[result],
        verdicts=[Verdict(claim_id="c1", status="verified")],
    )

    seen: dict = {}

    def spy_compose(messages):
        seen["prompt"] = "\n".join(str(m.content) for m in messages)
        return ("Acme appears twice — see the bar chart.", U)

    deps = GraphDeps(
        planner=lambda m: None,
        analyst_turn=lambda m: None,
        critic_turn=lambda m: None,
        compose=spy_compose,
    )
    out = composer_node(state, deps)

    assert "chart" in seen["prompt"].lower()
    assert "Top 10 companies by frequency" in seen["prompt"]
    assert len(out["final"].charts) == 1


def test_charts_survive_a_step_that_never_finishes(dataset):
    """Regression: an analyst that loops without ever emitting `finish` still keeps
    the chart it produced — the failure path must not discard chart_specs."""
    from app.runtime.graph import analyst_node
    from app.runtime.state import AnalystTask, PlanStep, SandboxResult

    chart_artifact = {
        "name": "chart_countries",
        "kind": "json",
        "data": {
            "kind": "bar",
            "title": "Customers by country",
            "x": "country",
            "y": ["count"],
            "data": [{"country": "Liechtenstein", "count": 12}],
            "source_columns": ["country"],
        },
    }

    def never_finishes(_messages):
        return AnalystTurn(action="run_code", code="print('again')", findings="", claims=[]), U

    def fake_run_code(code, _path):
        return SandboxResult(code=code, stdout="ok", artifacts=[chart_artifact])

    deps = GraphDeps(
        planner=lambda m: None,
        analyst_turn=never_finishes,
        critic_turn=lambda m: None,
        compose=lambda m: ("", U),
        run_code=fake_run_code,
    )
    task = AnalystTask(
        run_id="run-noloop",
        question="which countries have the most customers?",
        dataset_path=str(dataset),
        dataset_profile={"columns": [{"name": "country"}]},
        root_span_id="root",
        step=PlanStep(id=1, description="count customers by country", method="descriptive"),
    )

    out = analyst_node(task, deps)
    result = out["analyst_results"][0]

    assert result.failed is True  # it never finished
    assert result.chart_specs, "a produced chart must survive a failed step"
    assert result.chart_specs[0].title == "Customers by country"
