"""Checkpointing: every superstep persists; a run's state is recoverable by thread_id."""

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.graph import build_graph
from app.runtime.state import Claim, PlanStep, RunState

U = LLMUsage(tokens_in=1, tokens_out=1)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare\n10\n20\n30\n")
    return csv


def test_run_state_is_checkpointed_per_superstep(dataset, tmp_path):
    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=lambda m: (
            AnalystTurn(
                action="finish",
                findings="Total is 60.",
                claims=[Claim(text="total", kind="numeric", value=60.0)],
            ),
            U,
        ),
        critic_turn=lambda m: (
            CriticTurn(action="finish", findings=[CriticFinding(claim_id="1-1", value=60.0)]),
            U,
        ),
        compose=lambda m: ("Total is 60.", U),
    )
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite3")) as saver:
        graph = build_graph(deps, checkpointer=saver)
        config = {"configurable": {"thread_id": "run-ckpt"}}
        graph.invoke(
            RunState(
                run_id="run-ckpt",
                question="total?",
                dataset_path=str(dataset),
                dataset_profile={"columns": [{"name": "fare"}]},
            ),
            config,
        )
        checkpoints = list(saver.list(config))
        assert len(checkpoints) >= 4  # input + planner + analyst + critic + composer supersteps
        latest = graph.get_state(config)
        assert latest.values["final_answer"] == "Total is 60."
        assert latest.next == ()  # run completed; nothing pending
