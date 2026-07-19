"""Malformed structured output gets exactly one repair retry (plan §3.1)."""

import pytest

from app.agents.llm import LLMUsage, MalformedOutputError, invoke_structured
from app.agents.schemas import AnalystTurn


def raw(tokens=(10, 5)):
    class Raw:
        usage_metadata = {"input_tokens": tokens[0], "output_tokens": tokens[1]}

    return Raw()


def test_valid_output_passes_through_with_usage():
    out = {
        "raw": raw(),
        "parsed": AnalystTurn(action="finish", findings="ok"),
        "parsing_error": None,
    }
    turn, usage = invoke_structured(lambda messages: out, [], "analyst")
    assert turn.findings == "ok"
    assert usage == LLMUsage(tokens_in=10, tokens_out=5)


def test_malformed_output_is_repaired_once():
    outputs = [
        {"raw": raw(), "parsed": None, "parsing_error": ValueError("bad json")},
        {
            "raw": raw(),
            "parsed": AnalystTurn(action="finish", findings="fixed"),
            "parsing_error": None,
        },
    ]
    calls: list[list] = []

    def invoke(messages):
        calls.append(list(messages))
        return outputs[len(calls) - 1]

    turn, usage = invoke_structured(invoke, [], "analyst")
    assert turn.findings == "fixed"
    assert len(calls) == 2
    assert "failed validation" in calls[1][-1].content  # repair instruction appended
    assert usage == LLMUsage(tokens_in=20, tokens_out=10)  # both calls billed


def test_second_malformed_output_raises():
    bad = {"raw": raw(), "parsed": None, "parsing_error": ValueError("nope")}
    with pytest.raises(MalformedOutputError):
        invoke_structured(lambda messages: bad, [], "analyst")
