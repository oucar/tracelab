"""Run creation, SSE event stream, and run/span retrieval."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agents.llm import GraphDeps
from app.config import settings
from app.deps import store
from app.runtime.events import EventType, bus
from app.runtime.graph import execute_run
from app.runtime.recording import Recorder, recording_deps, replay_deps
from app.runtime.state import RunState
from app.tracing.store import utc_midnight

router = APIRouter(prefix="/api/runs", tags=["runs"])


class AskRequest(BaseModel):
    dataset_id: str
    question: str


def _execute(run_id: str, dataset: dict, question: str, deps: GraphDeps | None = None) -> None:
    """Runs in a worker thread; recording on for real runs, off for replays."""
    if deps is None:
        deps = recording_deps(GraphDeps.default(), Recorder(store(), run_id))
    state = RunState(
        run_id=run_id,
        question=question,
        dataset_path=dataset["path"],
        dataset_profile=dataset["profile"],
    )
    try:
        final = execute_run(state, deps)
        result = final.final.model_dump_json() if final.final else ""
        store().finish_run(run_id, final.final_answer, "finished", result)
    except Exception as exc:
        store().finish_run(run_id, f"error: {exc}", "error")


@router.post("")
async def create_run(req: AskRequest) -> dict:
    dataset = store().get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    cfg = settings()
    spent = store().cost_since(utc_midnight())
    if spent >= cfg.daily_budget_usd:
        raise HTTPException(
            429,
            f"daily budget of ${cfg.daily_budget_usd:.2f} exhausted (${spent:.2f} spent today)",
        )
    run_id = store().create_run(req.dataset_id, req.question.strip())
    asyncio.get_running_loop().run_in_executor(
        None, _execute, run_id, dataset, req.question.strip()
    )
    return {"run_id": run_id}


@router.post("/{run_id}/replay")
async def replay_run(run_id: str) -> dict:
    """Re-execute a past run offline from its recordings. Free, keyless, deterministic."""
    source = store().get_run(run_id)
    if source is None:
        raise HTTPException(404, "run not found")
    recordings = store().recordings_for_run(run_id)
    if not recordings:
        raise HTTPException(400, "run has no recordings to replay")
    dataset = store().get_dataset(source["dataset_id"])
    if dataset is None:
        raise HTTPException(404, "dataset no longer exists")
    new_id = store().create_run(source["dataset_id"], source["question"], replay_of=run_id)
    asyncio.get_running_loop().run_in_executor(
        None, _execute, new_id, dataset, source["question"], replay_deps(recordings)
    )
    return {"run_id": new_id}


@router.get("/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    if store().get_run(run_id) is None:
        raise HTTPException(404, "run not found")

    async def generator():
        async for event in bus.subscribe(run_id):
            yield {"event": event.type.value, "data": event.model_dump_json()}
            if event.type in (EventType.RUN_FINISHED, EventType.ERROR):
                return

    return EventSourceResponse(generator())


@router.get("")
def list_runs() -> list[dict]:
    return store().list_runs()


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    run = store().get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    run["result"] = json.loads(run["result"]) if run.get("result") else None
    return {**run, "spans": store().spans_for_run(run_id)}
