"""Recording captures the nondeterministic boundary: LLM turns + sandbox runs."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn, RouterTurn
from app.runtime.events import bus
from app.runtime.graph import execute_run
from app.runtime.recording import Recorder, recording_deps
from app.runtime.state import Claim, PlanStep, RunState, SandboxResult
from app.tracing.store import Store

U = LLMUsage(tokens_in=10, tokens_out=5, model="gpt-4o-mini")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip\n10,2\n20,5\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="total fare?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 2, "columns": [{"name": "fare"}, {"name": "tip"}]},
    )


def coding_then_finishing_analyst():
    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('total 30')"), U
        return (
            AnalystTurn(
                action="finish", findings="Total fare is 30.",
                claims=[Claim(text="total fare", kind="numeric", value=30.0)],
            ),
            U,
        )

    return analyst


def verifying_critic(messages):
    text = next(m.content for m in messages if "Claims to verify:" in m.content)
    claims = json.loads(text.split("Claims to verify:\n", 1)[1])
    return (
        CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
        ),
        U,
    )


def stub_deps() -> GraphDeps:
    return GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=coding_then_finishing_analyst(),
        critic_turn=verifying_critic,
        compose=lambda m: ("Total fare is 30.", U),
        run_code=lambda code, path: SandboxResult(code=code, stdout="total 30\n", exit_code=0),
    )


def test_recording_captures_every_llm_and_sandbox_call(dataset, tmp_path):
    store = Store(tmp_path / "rec.db")
    execute_run(make_state(dataset, "run-rec1"), recording_deps(stub_deps(), Recorder(store, "run-rec1")))

    rows = store.recordings_for_run("run-rec1")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("sandbox") == 1
    # planner + 2 analyst turns + critic + composer = 5 llm recordings
    assert kinds.count("llm") == 5
    sandbox = next(r for r in rows if r["kind"] == "sandbox")
    assert sandbox["response"]["stdout"] == "total 30\n"
    llm = next(r for r in rows if r["kind"] == "llm")
    assert "usage" in llm["response"] and llm["response"]["usage"]["model"] == "gpt-4o-mini"


def test_request_keys_are_deterministic_across_runs(dataset, tmp_path):
    store = Store(tmp_path / "rec2.db")
    for run_id in ("run-a", "run-b"):
        execute_run(
            make_state(dataset, run_id), recording_deps(stub_deps(), Recorder(store, run_id))
        )
    keys_a = sorted((r["kind"], r["key"], r["seq"]) for r in store.recordings_for_run("run-a"))
    keys_b = sorted((r["kind"], r["key"], r["seq"]) for r in store.recordings_for_run("run-b"))
    assert keys_a == keys_b  # same stub conversation → identical keys, both runs replayable


def test_identical_requests_get_increasing_seq(tmp_path):
    store = Store(tmp_path / "rec3.db")
    rec = Recorder(store, "run-seq")
    rec.record("llm", "same-key", {"n": 1})
    rec.record("llm", "same-key", {"n": 2})
    rows = store.recordings_for_run("run-seq")
    assert [(r["seq"], r["response"]["n"]) for r in rows] == [(0, 1), (1, 2)]


# ── router joins the recorded boundary (regression: M4's router was dead on the
# live path because recording_deps/replay_deps never wired it through) ────────


def stub_router(route: str):
    def fn(messages):
        return RouterTurn(route=route, reason="stub"), U

    return fn


def test_recording_deps_runs_router_live(dataset, tmp_path):
    """Wrapping deps with a router in recording_deps must actually invoke it —
    not silently fall back to the router=None / multi_step behavior."""
    store = Store(tmp_path / "rec-router-live.db")
    deps = replace(stub_deps(), router=stub_router("simple"))
    execute_run(
        make_state(dataset, "run-router-live"),
        recording_deps(deps, Recorder(store, "run-router-live")),
    )
    events = bus.history("run-router-live")
    router_events = [e for e in events if e.agent == "router"]
    assert router_events, "router node did not run"
    assert router_events[0].payload["route"] == "simple"


def test_recording_deps_records_the_router_call(dataset, tmp_path):
    """The router turn is captured alongside the other structured LLM calls."""
    store = Store(tmp_path / "rec-router-count.db")
    # "statistical" routes through the planner exactly like the router=None
    # fallback, so this run's shape matches
    # test_recording_captures_every_llm_and_sandbox_call's baseline (5 llm +
    # 1 sandbox) plus exactly one extra recording for the router call itself.
    deps = replace(stub_deps(), router=stub_router("statistical"))
    execute_run(
        make_state(dataset, "run-router-count"),
        recording_deps(deps, Recorder(store, "run-router-count")),
    )
    rows = store.recordings_for_run("run-router-count")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("sandbox") == 1
    assert kinds.count("llm") == 6  # router + planner + 2 analyst turns + critic + composer
