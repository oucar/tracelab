import pytest
from pydantic import ValidationError

from app.agents.llm import LLMUsage
from app.agents.schemas import JudgeTurn
from app.evals.judge import DIMENSIONS, judge_answer
from app.runtime.state import FinalAnswer


def test_judge_turn_bounds():
    with pytest.raises(ValidationError):
        JudgeTurn(clarity=6, uncertainty_honesty=1, chart_appropriateness=1,
                  methodological_soundness=1, rationale="x")
    turn = JudgeTurn(clarity=4, uncertainty_honesty=5, chart_appropriateness=3,
                     methodological_soundness=2, rationale="ok")
    assert [getattr(turn, d) for d in DIMENSIONS] == [4, 5, 3, 2]


def test_judge_answer_builds_prompt_and_returns_turn():
    seen: dict = {}

    def stub_judge(messages):
        seen["prompt"] = messages[-1].content
        return (JudgeTurn(clarity=5, uncertainty_honesty=4, chart_appropriateness=3,
                          methodological_soundness=5, rationale="solid"),
                LLMUsage(tokens_in=100, tokens_out=20, model="gpt-4o"))

    final = FinalAnswer(narrative="Weekend fares are higher (p=0.001).",
                        claims=[], charts=[], failed=False)
    turn, usage = judge_answer("Is fare higher on weekends?", final, stub_judge)
    assert turn.clarity == 5
    assert usage.tokens_in == 100
    assert "Is fare higher on weekends?" in seen["prompt"]
    assert "Weekend fares are higher" in seen["prompt"]
