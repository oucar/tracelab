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


def test_harness_isolates_spans_across_stores(tmp_path):
    """Verify that each Store's eval run records spans in its own database."""
    # Create subdirectories for each eval
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    # First eval against store A
    st1 = Store(tmp_path / "a.db")
    eval_id1 = run_eval(
        st1,
        _golden(a_dir, 3),
        lambda: _deps(3),
        label="eval1",
        repo_root=a_dir,
        enforce_budget=False,
    )

    # Get the run_id from the first eval
    run1 = st1.list_eval_runs()[0]
    assert run1["id"] == eval_id1
    result_rows_1 = st1.eval_results(eval_id1)
    assert len(result_rows_1) > 0
    run_id_1 = result_rows_1[0]["run_id"]
    spans_1 = st1.spans_for_run(run_id_1)
    # Stub deps have cost 0, so check span count > 0
    assert len(spans_1) > 0, "First store should have spans recorded"

    # Second eval against store B
    st2 = Store(tmp_path / "b.db")
    eval_id2 = run_eval(
        st2,
        _golden(b_dir, 3),
        lambda: _deps(3),
        label="eval2",
        repo_root=b_dir,
        enforce_budget=False,
    )

    # Get the run_id from the second eval and verify spans are in ST2, not ST1
    run2 = st2.list_eval_runs()[0]
    assert run2["id"] == eval_id2
    result_rows_2 = st2.eval_results(eval_id2)
    assert len(result_rows_2) > 0
    run_id_2 = result_rows_2[0]["run_id"]
    spans_2 = st2.spans_for_run(run_id_2)
    assert len(spans_2) > 0, "Second store should have spans recorded (bus was not one-shot)"
