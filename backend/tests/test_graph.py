"""Graph routing with stubbed LLMs — the whole graph runs without an API key."""

from pathlib import Path

import pytest

from app.agents.llm import GraphDeps
from app.agents.schemas import AnalystTurn
from app.runtime.graph import execute_run
from app.runtime.state import RunState


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip\n10,2\n20,5\n30,9\n")
    return csv


def make_state(dataset: Path, run_id: str = "test-run") -> RunState:
    return RunState(
        run_id=run_id,
        question="What is the total fare?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 3},
    )


def test_analyst_runs_code_then_finishes_then_composer_answers(dataset):
    calls = {"n": 0}

    def analyst_turn(messages) -> AnalystTurn:
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(
                action="run_code",
                code="import pandas as pd\nprint('total', pd.read_csv('data.csv')['fare'].sum())",
            )
        # Second turn sees the execution result in the transcript and finishes.
        assert "total 60" in messages[-1].content
        return AnalystTurn(action="finish", findings="Total fare is 60.")

    deps = GraphDeps(analyst_turn=analyst_turn, compose=lambda m: "The total fare is 60.")
    final = execute_run(make_state(dataset, "run-ok"), deps)

    assert final.final_answer == "The total fare is 60."
    assert final.analyst_findings == "Total fare is 60."
    assert len(final.steps) == 1 and final.steps[0].result.exit_code == 0


def test_exhausted_iterations_routes_to_honest_failure(dataset):
    deps = GraphDeps(
        analyst_turn=lambda m: AnalystTurn(action="run_code", code="print('still looking')"),
        compose=lambda m: "I could not determine the answer.",
    )
    final = execute_run(make_state(dataset, "run-fail"), deps)

    assert final.analyst_failed
    assert "exhausted" in final.failure_reason
    assert "could not" in final.final_answer.lower()


def test_events_are_emitted_for_the_run(dataset):
    from app.runtime.events import bus

    deps = GraphDeps(
        analyst_turn=lambda m: AnalystTurn(action="finish", findings="n/a"),
        compose=lambda m: "answer",
    )
    execute_run(make_state(dataset, "run-events"), deps)

    types = [e.type.value for e in bus.history("run-events")]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert "llm_call" in types
