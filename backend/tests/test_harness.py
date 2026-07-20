"""M4 harness: sweeps the golden set through the graph, scores both tiers, persists rows."""
import json

import pandas as pd

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticTurn, JudgeTurn, PlannerTurn
from app.evals.golden import GoldenDataset, GoldenExpected, GoldenQuestion
from app.evals.harness import run_eval
from app.runtime.reconcile import CriticFinding
from app.runtime.state import Claim, PlanStep
from app.tracing.store import Store

U = LLMUsage(tokens_in=10, tokens_out=5)


def _claim(value):
    return Claim(id="c1", step_id=1, text=f"answer is {value}", kind="numeric", value=value)


def _deps(answer_value):
    def planner(_msgs):
        return PlannerTurn(
            steps=[PlanStep(id=1, description="count rows", method="descriptive")],
            rationale="one step",
        ), U

    def analyst(_msgs):
        return AnalystTurn(
            action="finish", findings=f"value {answer_value}", claims=[_claim(answer_value)]
        ), U

    def critic(_msgs):
        return CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id="c1", value=answer_value, methodology_ok=True)],
        ), U

    return GraphDeps(
        planner=planner,
        analyst_turn=analyst,
        critic_turn=critic,
        compose=lambda _m: (f"The answer is {answer_value}.", U),
        run_code=lambda code, ws: None,
    )


def _golden(tmp_path, expected_value):
    csv = tmp_path / "tiny.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv, index=False)
    return [
        GoldenDataset(
            name="tiny",
            csv=csv.name,
            questions=[
                GoldenQuestion(
                    id="tiny-001",
                    question="How many rows?",
                    expected=GoldenExpected(
                        kind="numeric", value=expected_value, tolerance=0.0
                    ),
                ),
            ],
        )
    ]


def test_harness_scores_pass_and_fail(tmp_path):
    st = Store(tmp_path / "t.db")
    eval_id = run_eval(
        st, _golden(tmp_path, 3), lambda: _deps(3), label="stub", repo_root=tmp_path,
        enforce_budget=False,
    )
    run = st.list_eval_runs()[0]
    assert run["id"] == eval_id
    assert run["questions_total"] == 1
    assert run["tier1_scorable"] == 1 and run["tier1_passed"] == 1
    assert run["judge_avg"] is None

    eval_id2 = run_eval(
        st, _golden(tmp_path, 3), lambda: _deps(99), label="wrong", repo_root=tmp_path,
        enforce_budget=False,
    )
    wrong = [r for r in st.list_eval_runs() if r["id"] == eval_id2][0]
    assert wrong["tier1_passed"] == 0


def test_harness_records_judge_scores(tmp_path):
    st = Store(tmp_path / "t.db")

    def judge(_msgs):
        return JudgeTurn(
            clarity=4, uncertainty_honesty=4, chart_appropriateness=3,
            methodological_soundness=5, rationale="fine",
        ), U

    eval_id = run_eval(
        st, _golden(tmp_path, 3), lambda: _deps(3), judge=judge, repo_root=tmp_path,
        enforce_budget=False,
    )
    run = [r for r in st.list_eval_runs() if r["id"] == eval_id][0]
    assert run["judge_avg"] == 4.0
    row = st.eval_results(eval_id)[0]
    assert json.loads(row["judge"])["clarity"] == 4


def test_harness_survives_a_crashing_question(tmp_path):
    st = Store(tmp_path / "t.db")

    def exploding_deps():
        d = _deps(3)

        def bad_planner(_msgs):
            raise RuntimeError("boom")

        return GraphDeps(
            planner=bad_planner, analyst_turn=d.analyst_turn, critic_turn=d.critic_turn,
            compose=d.compose, run_code=d.run_code,
        )

    eval_id = run_eval(
        st, _golden(tmp_path, 3), exploding_deps, repo_root=tmp_path, enforce_budget=False,
    )
    row = st.eval_results(eval_id)[0]
    assert row["tier1_passed"] == 0
    assert "boom" in row["tier1_detail"] or "failed" in row["tier1_detail"].lower()
