"""Mid-run budget enforcement, end to end through the graph.

`test_run_budget.py` covers the budget object. This covers the thing that was
actually broken: that the graph consults it at all, and that running out of
money produces an honest answer instead of a stack trace.
"""

import json
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.config import settings
from app.runtime.graph import execute_run
from app.runtime.state import Claim, PlanStep, RunState

FREE = LLMUsage(tokens_in=10, tokens_out=5)  # model "" — a stub, genuinely free
#: gpt-4o input is $2.50/1M, so this single call costs $0.0025.
PRICEY = LLMUsage(tokens_in=1_000, tokens_out=0, model="gpt-4o")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare\n10\n20\n30\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="What is the total fare?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 3, "columns": [{"name": "fare"}]},
    )


@pytest.fixture
def tiny_cap(monkeypatch):
    """A cap smaller than one gpt-4o call, so the second call is the one refused."""
    monkeypatch.setenv("MAX_COST_PER_RUN_USD", "0.001")
    settings.cache_clear()
    yield
    settings.cache_clear()


def verifying_critic(messages):
    """Confirms every claim it is shown, for free."""
    text = next(m.content for m in messages if "Claims to verify:" in m.content)
    claims = json.loads(text.split("Claims to verify:\n", 1)[1])
    return (
        CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
        ),
        FREE,
    )


def _deps(planner_usage, compose_calls):
    return GraphDeps(
        planner=lambda m: (
            PlannerTurn(steps=[PlanStep(description="Total the fare", method="descriptive")]),
            planner_usage,
        ),
        analyst_turn=lambda m: (
            AnalystTurn(
                action="finish",
                findings="Total fare is 60.",
                claims=[Claim(text="total fare", kind="numeric", value=60.0)],
            ),
            FREE,
        ),
        critic_turn=verifying_critic,
        compose=lambda m: (compose_calls.append(m), ("composed", FREE))[1],
    )


def test_run_stops_once_the_dollar_cap_is_crossed(dataset, tiny_cap):
    compose_calls: list = []
    final = execute_run(make_state(dataset, "run-budget-stop"), _deps(PRICEY, compose_calls))

    # The planner's own call is allowed through — the cap is checked before a
    # call, so a run overshoots by at most the one call already in flight.
    assert final.plan is not None
    # Everything downstream is refused.
    assert final.analyst_results and final.analyst_results[0].failed
    assert "budget exhausted" in final.analyst_results[0].failure_reason
    assert final.final is not None and final.final.failed
    # The composer reports the stop itself, without spending another call on it.
    assert compose_calls == []
    assert "run_cost" in final.final_answer


def test_a_free_run_is_never_stopped(dataset, tiny_cap):
    """Stub usage costs nothing, so even a $0.001 cap must not interfere."""
    compose_calls: list = []
    final = execute_run(make_state(dataset, "run-budget-free"), _deps(FREE, compose_calls))

    assert final.analyst_results and not final.analyst_results[0].failed
    assert len(compose_calls) == 1
    assert final.final is not None and not final.final.failed


def test_daily_headroom_is_honoured_independently_of_the_run_cap(dataset):
    """Admission said yes with $0.001 left; the graph must still stop the run."""
    compose_calls: list = []
    final = execute_run(
        make_state(dataset, "run-budget-daily"),
        _deps(PRICEY, compose_calls),
        daily_headroom_usd=0.001,
    )
    assert final.final is not None and final.final.failed
    assert "daily_cost" in final.final_answer


def test_budget_registry_is_cleaned_up_after_a_run(dataset, tiny_cap):
    from app.runtime.budget import get_run_budget

    execute_run(make_state(dataset, "run-budget-cleanup"), _deps(FREE, []))
    assert get_run_budget("run-budget-cleanup") is None
