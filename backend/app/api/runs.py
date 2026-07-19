"""Run creation, SSE event stream, and run/span retrieval."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.deps import store
from app.runtime.events import EventType, bus
from app.runtime.graph import execute_run
from app.runtime.state import RunState

router = APIRouter(prefix="/api/runs", tags=["runs"])


class AskRequest(BaseModel):
    dataset_id: str
    question: str


def _execute(run_id: str, dataset: dict, question: str) -> None:
    """Runs in a worker thread; persists every event, then finalizes the run row."""
    state = RunState(
        run_id=run_id,
        question=question,
        dataset_path=dataset["path"],
        dataset_profile=dataset["profile"],
    )
    try:
        final = execute_run(state)
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
    run_id = store().create_run(req.dataset_id, req.question.strip())
    asyncio.get_running_loop().run_in_executor(
        None, _execute, run_id, dataset, req.question.strip()
    )
    return {"run_id": run_id}


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
