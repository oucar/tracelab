"""Price table: deliberately hardcoded, prefix-matched, unknown models cost 0."""

import pytest

from app.agents.llm import LLMUsage, _usage_of
from app.tracing import pricing


def test_known_model_cost():
    assert pricing.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert pricing.cost_usd("gpt-4o", 1_000, 2_000) == pytest.approx(0.0025 + 0.02)


def test_versioned_model_name_prefix_matches():
    assert pricing.cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    # "gpt-4o-mini-..." must match gpt-4o-mini, never the shorter gpt-4o prefix
    assert pricing.cost_usd("gpt-4o-2024-08-06", 0, 1_000_000) == pytest.approx(10.0)


def test_unknown_model_costs_zero():
    assert pricing.cost_usd("", 5000, 5000) == 0.0
    assert pricing.cost_usd("replay", 5000, 5000) == 0.0


def test_usage_of_extracts_model_name():
    class Raw:
        usage_metadata = {"input_tokens": 7, "output_tokens": 3}
        response_metadata = {"model_name": "gpt-4o-mini-2024-07-18"}

    usage = _usage_of(Raw())
    assert usage == LLMUsage(tokens_in=7, tokens_out=3, model="gpt-4o-mini-2024-07-18")
