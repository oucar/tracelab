"""Deterministic replay: a recorded run re-executes offline to the same answer."""

import json
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.events import bus
from app.runtime.graph import execute_run
from app.runtime.recording import Recorder, ReplayMiss, recording_deps, replay_deps
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


def stub_deps() -> GraphDeps:
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

    def critic(messages):
        text = next(m.content for m in messages if "Claims to verify:" in m.content)
        claims = json.loads(text.split("Claims to verify:\n", 1)[1])
        return (
            CriticTurn(
                action="finish",
                findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
            ),
            U,
        )

    return GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=critic,
        compose=lambda m: ("Total fare is 30.", U),
        run_code=lambda code, path: SandboxResult(code=code, stdout="total 30\n", exit_code=0),
    )


def test_replay_reproduces_the_run_offline(dataset, tmp_path):
    store = Store(tmp_path / "replay.db")
    original = execute_run(
        make_state(dataset, "run-orig"), recording_deps(stub_deps(), Recorder(store, "run-orig"))
    )

    # Replay uses ONLY the recordings — no stubs, no sandbox, no network.
    replayed = execute_run(
        make_state(dataset, "run-replay"), replay_deps(store.recordings_for_run("run-orig"))
    )

    assert replayed.final_answer == original.final_answer
    assert replayed.final == original.final
    assert [v.model_dump() for v in replayed.verdicts] == [
        v.model_dump() for v in original.verdicts
    ]
    # replayed events keep token counts but cost nothing
    events = bus.history("run-replay")
    assert any(e.tokens_in > 0 for e in events)
    assert all(e.cost_usd == 0 for e in events)


def test_replay_missing_recording_fails_loudly(dataset):
    deps = replay_deps([])  # nothing recorded
    with pytest.raises(Exception) as exc:
        execute_run(make_state(dataset, "run-miss"), deps)
    assert "ReplayMiss" in exc.value.__class__.__name__ or isinstance(exc.value, ReplayMiss)
