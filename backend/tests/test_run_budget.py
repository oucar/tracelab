"""The per-run dollar budget is a real cap, not an admission check.

The gap these cover: `daily_budget_usd` used to be tested only at
`POST /api/runs`, which admits a run and then never looks again. A run admitted
with $0.01 of headroom could fan out four analysts and spend freely, because
per-agent budgets are denominated in tokens and say nothing about dollars.
"""

import math

import pytest

from app.config import settings
from app.runtime.budget import (
    AgentBudget,
    BudgetExceeded,
    RunBudget,
    close_run_budget,
    get_run_budget,
    open_run_budget,
)


@pytest.fixture
def _clean_registry():
    yield
    close_run_budget("r1")


def test_run_cost_cap_trips():
    b = RunBudget(run_id="r1", max_run_usd=0.10)
    b.charge("gpt-4o", 0.04)
    b.check("analyst")  # still under
    b.charge("gpt-4o", 0.07)
    with pytest.raises(BudgetExceeded) as exc:
        b.check("analyst")
    assert exc.value.kind == "run_cost"
    assert exc.value.role == "analyst"


def test_daily_headroom_cap_trips_before_run_cap():
    """A run admitted with almost no headroom left must stop on the daily layer."""
    b = RunBudget(run_id="r1", max_run_usd=10.0, daily_headroom_usd=0.02)
    b.charge("gpt-4o", 0.03)
    with pytest.raises(BudgetExceeded) as exc:
        b.check("critic")
    assert exc.value.kind == "daily_cost"


def test_no_headroom_argument_means_only_the_run_cap_applies():
    b = RunBudget(run_id="r1", max_run_usd=1.0)
    assert b.daily_headroom_usd == math.inf
    b.charge("gpt-4o", 0.99)
    b.check("planner")  # no raise


def test_free_models_never_trip_the_cap():
    """Stubs and replays cost nothing by construction, so they must not count."""
    b = RunBudget(run_id="r1", max_run_usd=0.01)
    for _ in range(50):
        b.charge("replay", 0.0)
        b.charge("", 0.0)
    b.check("analyst")
    assert b.unpriced_calls == 0


def test_unpriced_model_trips_the_cap_instead_of_silently_passing():
    """The dangerous $0: a real model with no price entry.

    Reporting $0.00 for it would make every dollar budget unenforceable for
    that model, so the run stops and says why rather than running uncapped.
    """
    b = RunBudget(run_id="r1", max_run_usd=1.0)
    b.charge("some-unlisted-model-v9", 0.0)
    assert b.spent_usd == 0.0
    with pytest.raises(BudgetExceeded) as exc:
        b.check("analyst")
    assert exc.value.kind == "run_cost"
    assert "price table" in exc.value.detail


def test_agent_budget_delegates_to_the_registered_run_budget(_clean_registry):
    open_run_budget("r1", daily_headroom_usd=None)
    budget = AgentBudget.for_role("analyst", "r1")
    budget.check_run_cost()  # nothing spent yet
    get_run_budget("r1").charge("gpt-4o", settings().max_cost_per_run_usd + 1)
    with pytest.raises(BudgetExceeded):
        budget.check_run_cost()


def test_agent_budget_without_a_run_id_is_unaffected():
    """Existing per-agent-only construction stays valid and never touches the registry."""
    budget = AgentBudget.for_role("analyst")
    assert budget.run_id is None
    budget.check_run_cost()  # no raise, no registry lookup


def test_open_and_close_do_not_leak(_clean_registry):
    open_run_budget("r1")
    assert get_run_budget("r1") is not None
    close_run_budget("r1")
    assert get_run_budget("r1") is None
