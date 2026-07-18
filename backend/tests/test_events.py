"""EventBus: history replay for late subscribers, live fan-out, stream termination."""

import asyncio

from app.runtime.events import AgentEvent, EventBus, EventType


def make_event(run_id: str, type_: EventType = EventType.LLM_CALL) -> AgentEvent:
    return AgentEvent(run_id=run_id, agent="analyst", type=type_)


async def test_late_subscriber_gets_history_then_completes():
    bus = EventBus()
    bus.emit(make_event("r1"))
    bus.emit(make_event("r1", EventType.RUN_FINISHED))

    received = [e async for e in bus.subscribe("r1")]
    assert [e.type for e in received] == [EventType.LLM_CALL, EventType.RUN_FINISHED]


async def test_live_subscriber_receives_events_as_emitted():
    bus = EventBus()
    received: list[AgentEvent] = []

    async def consume():
        async for event in bus.subscribe("r2"):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    bus.emit(make_event("r2"))
    bus.emit(make_event("r2", EventType.RUN_FINISHED))
    await asyncio.wait_for(task, timeout=1)
    assert len(received) == 2


async def test_runs_are_isolated():
    bus = EventBus()
    bus.emit(make_event("a"))
    bus.emit(make_event("a", EventType.RUN_FINISHED))
    bus.emit(make_event("b", EventType.RUN_FINISHED))
    a_events = [e async for e in bus.subscribe("a")]
    assert all(e.run_id == "a" for e in a_events)
