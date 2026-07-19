"""Per-agent budgets are hard stops, not suggestions."""

import pytest

from app.runtime.budget import AgentBudget, BudgetExceeded


def test_llm_call_budget_exhausts():
    b = AgentBudget(role="planner", max_llm_calls=2, max_tool_calls=0, max_tokens=10_000)
    b.spend_llm(100, 50)
    b.spend_llm(100, 50)
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_llm(100, 50)
    assert exc.value.role == "planner" and exc.value.kind == "llm_calls"


def test_token_budget_exhausts():
    b = AgentBudget(role="analyst", max_llm_calls=100, max_tool_calls=10, max_tokens=500)
    b.spend_llm(300, 100)
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_llm(300, 100)
    assert exc.value.kind == "tokens"


def test_tool_call_budget_exhausts():
    b = AgentBudget(role="critic", max_llm_calls=10, max_tool_calls=1, max_tokens=10_000)
    b.spend_tool()
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_tool()
    assert exc.value.kind == "tool_calls"


def test_for_role_reads_settings():
    b = AgentBudget.for_role("analyst")
    assert b.max_tool_calls == 3  # cfg.max_analyst_iterations
    assert b.max_llm_calls == 4  # iterations + 1 finishing turn
    assert b.max_tokens == 24_000
    assert AgentBudget.for_role("composer").max_tool_calls == 0
