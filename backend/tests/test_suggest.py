from app.agents.llm import LLMUsage
from app.agents.schemas import SuggestionsTurn
from app.agents.suggest import suggest_questions


def test_suggest_questions_builds_prompt_and_returns_questions():
    seen: dict = {}

    def stub(messages):
        seen["prompt"] = str(messages[-1].content)
        return (
            SuggestionsTurn(
                questions=[
                    "What is the average fare?",
                    "Is the average fare higher on weekends?",
                    "  ",  # blanks are dropped
                ]
            ),
            LLMUsage(tokens_in=50, tokens_out=10),
        )

    profile = {"rows": 900, "columns": [{"name": "fare", "dtype": "float64"}]}
    out = suggest_questions(profile, suggester=stub)

    assert out == ["What is the average fare?", "Is the average fare higher on weekends?"]
    assert "fare" in seen["prompt"]  # the profile made it into the prompt


def test_suggest_questions_caps_at_four():
    def stub(_messages):
        return (
            SuggestionsTurn(questions=["a", "b", "c", "d"]),
            LLMUsage(),
        )

    out = suggest_questions({"rows": 1, "columns": []}, suggester=stub)
    assert len(out) == 4
