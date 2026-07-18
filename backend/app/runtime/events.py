"""AgentEvent — the load-bearing type of the codebase.

The SSE stream, the trace store, the replay engine, and the live graph UI all
consume this one shape. If you change it, you change everything downstream;
that is intentional and documented in docs/architecture.md.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"  # sandbox executions surface as tool calls
    HANDOFF = "handoff"
    VERDICT = "verdict"
    ANSWER_CHUNK = "answer_chunk"
    RUN_FINISHED = "run_finished"
    ERROR = "error"


class AgentEvent(BaseModel):
    run_id: str
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_span_id: str | None = None
    agent: str  # "planner" | "analyst" | "critic" | "composer" | "system"
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: float = Field(default_factory=time.time)
    duration_ms: int = 0


class EventBus:
    """Per-run fan-out of AgentEvents to any number of SSE subscribers.

    Also keeps the full event history so a subscriber that connects late
    (e.g. the browser opening the run view mid-run) receives every event.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[AgentEvent | None]]] = {}
        self._history: dict[str, list[AgentEvent]] = {}
        self._finished: set[str] = set()

    def emit(self, event: AgentEvent) -> None:
        self._history.setdefault(event.run_id, []).append(event)
        if event.type in (EventType.RUN_FINISHED, EventType.ERROR):
            self._finished.add(event.run_id)
        for q in self._queues.get(event.run_id, []):
            q.put_nowait(event)
        if event.run_id in self._finished:
            for q in self._queues.get(event.run_id, []):
                q.put_nowait(None)  # sentinel: stream over

    async def subscribe(self, run_id: str):
        """Async iterator over a run's events: full history first, then live."""
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._queues.setdefault(run_id, []).append(q)
        try:
            for event in list(self._history.get(run_id, [])):
                yield event
            if run_id in self._finished:
                return
            while True:
                event = await q.get()
                if event is None:
                    return
                yield event
        finally:
            self._queues.get(run_id, []).remove(q)

    def history(self, run_id: str) -> list[AgentEvent]:
        return list(self._history.get(run_id, []))


bus = EventBus()
