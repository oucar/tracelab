# M3 — Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every run fully inspectable and reproducible: a priced cost meter on every LLM call, a real span tree persisted live, a run view with a live agent graph (reactflow) + span inspector, a runs dashboard with cost/latency/quality columns, deterministic offline replay from recorded calls, a daily budget cap, and the remaining stats toolkit (regression, clustering+PCA, time-series backtest, anomaly detection).

**Architecture:** Cost is computed at the single `_emit` chokepoint from a hardcoded price table (`tracing/pricing.py`) using the model name carried on `LLMUsage`. Spans become a tree by threading `parent_span_id` through the existing `_emit` helper (run root → node root → iteration events) and persist live via an `EventBus` sink instead of the end-of-run flush. Replay records at the `GraphDeps` boundary — the same seam tests already use for stubs — keyed by a content hash of each request, so parallel analyst branches replay correctly regardless of completion order; the sandbox joins that seam via a new injectable `GraphDeps.run_code`. The frontend gains react-router with three pages (Workbench, RunsDashboard, RunView); the agent graph is derived purely from the `AgentEvent` list, so it works identically for live SSE runs and past runs loaded from the spans table.

**Tech Stack:** Existing backend (FastAPI, LangGraph 1.2.9, Pydantic v2, sqlite3) + statsmodels, scikit-learn (sandbox stats). Frontend: existing React 18 + MUI 6 + @mui/x-data-grid (community) + new `react-router-dom` v7 and `@xyflow/react` v12.

## Global Constraints

- Python 3.11+ (venv is 3.14), ruff line length 100.
- Tests must pass with **no API key** — all LLMs stubbed via `GraphDeps`; the whole M3 test suite (including replay) runs offline.
- Backend tests: `cd backend && .venv/bin/pytest`. Lint: `cd backend && .venv/bin/ruff check .`. Frontend: `cd frontend && npm run typecheck`.
- `AgentEvent` stays the single event shape; no new event types. The span tree is expressed through the existing `parent_span_id` field.
- Deterministic logic (pricing, reconciliation, replay lookup, budget cap) is plain testable Python — never delegated to the LLM.
- MUI X community tier only (no Pro components).
- Sandbox rlimits stay best-effort on macOS (Darwin rejects RLIMIT_AS) — don't touch executor limits.
- Checkpoint DB is isolated per test by `tests/conftest.py`; new tests that call `execute_run` inherit that automatically.
- Commit after every task. **Never push.**
- Working directory for all commands below: `/Users/onurucar/Developer/tracelab`.

**Files overview** (Create/Modify):

| File | Task |
|---|---|
| C `backend/app/tracing/pricing.py`, C `backend/tests/test_pricing.py`, M `backend/app/agents/llm.py`, M `backend/app/runtime/graph.py` | 1 |
| M `backend/app/runtime/events.py`, M `backend/app/runtime/state.py`, M `backend/app/runtime/graph.py`, M `backend/app/api/runs.py`, M `backend/app/main.py`, M `backend/tests/test_events.py`, M `backend/tests/test_graph.py` | 2 |
| M `backend/app/config.py`, M `backend/app/tracing/store.py`, C `backend/app/api/config.py`, M `backend/app/api/runs.py`, M `backend/app/main.py`, C `backend/tests/test_api_budget.py`, M `backend/tests/test_store.py` | 3 |
| M `backend/app/agents/llm.py`, M `backend/app/runtime/graph.py`, M `backend/tests/test_graph.py` | 4 |
| C `backend/app/runtime/recording.py`, M `backend/app/tracing/store.py`, M `backend/app/api/runs.py`, C `backend/tests/test_recording.py` | 5 |
| M `backend/app/runtime/recording.py`, M `backend/app/tracing/store.py`, M `backend/app/api/runs.py`, C `backend/tests/test_replay.py`, M `backend/tests/test_store.py` | 6 |
| M `backend/app/runtime/state.py`, M `backend/pyproject.toml`, M `backend/app/agents/prompts/planner.md`, M `.../analyst.md`, M `.../critic.md`, M `backend/tests/test_sandbox.py`, M `backend/tests/test_graph.py` | 7 |
| M `backend/app/tracing/store.py`, M `backend/app/api/runs.py`, M `backend/tests/test_store.py` | 8 |
| M `frontend/package.json`, M `frontend/src/lib/types.ts`, M `frontend/src/lib/api.ts`, M `frontend/src/App.tsx`, C `frontend/src/pages/RunsDashboard.tsx` | 9 |
| C `frontend/src/lib/graphModel.ts`, C `frontend/src/components/AgentGraph.tsx`, C `frontend/src/components/SpanInspector.tsx`, C `frontend/src/components/CostMeter.tsx`, C `frontend/src/hooks/useRunEvents.ts`, C `frontend/src/pages/RunView.tsx`, M `frontend/src/App.tsx` | 10 |
| M `frontend/src/pages/Workbench.tsx`, M `MILESTONES.md` | 11 |

---

### Task 1: Price table + cost on every LLM event

`cost_usd` exists on `AgentEvent` and the spans table but is always 0. Add a hardcoded price table, carry the model name on `LLMUsage`, and compute cost at the `_emit` chokepoint.

**Files:**
- Create: `backend/app/tracing/pricing.py`
- Modify: `backend/app/agents/llm.py` (LLMUsage.model, `_usage_of`)
- Modify: `backend/app/runtime/graph.py` (`_emit` computes cost)
- Test: `backend/tests/test_pricing.py`

**Interfaces:**
- Produces: `pricing.cost_usd(model: str, tokens_in: int, tokens_out: int) -> float` (prefix-matches versioned model names like `gpt-4o-mini-2024-07-18`; unknown models return 0.0). `LLMUsage` gains `model: str = ""` as the LAST field (existing positional constructions like `LLMUsage(10, 5)` keep working). Later tasks rely on: replay sets `model="replay"` (not in the price table) so replayed events cost $0 while keeping token counts.

- [ ] **Step 1: Write the failing test** (`backend/tests/test_pricing.py`)

```python
"""Price table: deliberately hardcoded, prefix-matched, unknown models cost 0."""

import pytest

from app.agents.llm import LLMUsage, _usage_of
from app.tracing import pricing


def test_known_model_cost():
    assert pricing.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert pricing.cost_usd("gpt-4o", 1_000, 2_000) == pytest.approx(0.0025 + 0.02)


def test_versioned_model_name_prefix_matches():
    assert pricing.cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    # "gpt-4o-mini-..." must match gpt-4o-mini, never the shorter gpt-4o prefix
    assert pricing.cost_usd("gpt-4o-2024-08-06", 0, 1_000_000) == pytest.approx(10.0)


def test_unknown_model_costs_zero():
    assert pricing.cost_usd("", 5000, 5000) == 0.0
    assert pricing.cost_usd("replay", 5000, 5000) == 0.0


def test_usage_of_extracts_model_name():
    class Raw:
        usage_metadata = {"input_tokens": 7, "output_tokens": 3}
        response_metadata = {"model_name": "gpt-4o-mini-2024-07-18"}

    usage = _usage_of(Raw())
    assert usage == LLMUsage(tokens_in=7, tokens_out=3, model="gpt-4o-mini-2024-07-18")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_pricing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.tracing.pricing'`

- [ ] **Step 3: Implement** — create `backend/app/tracing/pricing.py`:

```python
"""Model price table and cost math.

Prices are USD per 1M tokens, hardcoded deliberately: an observability tool
that silently fetches prices is harder to trust than one you can read.
Unknown models (including stubs and "replay") cost 0 — better an honest $0
than a fabricated number. Versioned API names ("gpt-4o-mini-2024-07-18")
prefix-match their base entry; the longest prefix wins.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # model: (usd_per_1m_input_tokens, usd_per_1m_output_tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICES.get(model)
    if price is None and model:
        for name in sorted(PRICES, key=len, reverse=True):
            if model.startswith(name):
                price = PRICES[name]
                break
    if price is None:
        return 0.0
    price_in, price_out = price
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
```

In `backend/app/agents/llm.py`, extend `LLMUsage` and `_usage_of`:

```python
@dataclass(frozen=True)
class LLMUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""  # attached by _usage_of; "" for stubs → cost 0
```

```python
def _usage_of(raw: object) -> LLMUsage:
    meta = getattr(raw, "usage_metadata", None) or {}
    resp = getattr(raw, "response_metadata", None) or {}
    return LLMUsage(
        tokens_in=meta.get("input_tokens", 0),
        tokens_out=meta.get("output_tokens", 0),
        model=resp.get("model_name", ""),
    )
```

In `invoke_structured`, the combined-usage line must preserve the model:

```python
    usage = LLMUsage(
        usage.tokens_in + second.tokens_in,
        usage.tokens_out + second.tokens_out,
        model=usage.model or second.model,
    )
```

In `backend/app/runtime/graph.py`, import the module (`from app.tracing import pricing`) and change `_emit`'s `AgentEvent` construction to stamp cost:

```python
            cost_usd=pricing.cost_usd(usage.model, usage.tokens_in, usage.tokens_out)
            if usage
            else 0.0,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_pricing.py -v && .venv/bin/pytest -q`
Expected: test_pricing PASSes and the full suite stays green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/tracing/pricing.py backend/tests/test_pricing.py backend/app/agents/llm.py backend/app/runtime/graph.py
git commit -m "feat(m3): price table + per-call cost stamped on every LLM event"
```

---

### Task 2: Span tree + live persistence

Spans are currently flat (`parent_span_id` always NULL) and persisted only when the run ends. Thread parents through `_emit` (run root → node/branch root → iteration events) and persist every event the moment it is emitted via an `EventBus` sink.

**Files:**
- Modify: `backend/app/runtime/events.py` (EventBus sinks)
- Modify: `backend/app/runtime/state.py` (`RunState.root_span_id`, `AnalystTask.root_span_id`)
- Modify: `backend/app/runtime/graph.py` (parent threading)
- Modify: `backend/app/api/runs.py` (remove end-of-run flush)
- Modify: `backend/app/main.py` (register store sink)
- Test: `backend/tests/test_events.py`, `backend/tests/test_graph.py`

**Interfaces:**
- Consumes: `Store.add_span` is idempotent (`INSERT OR REPLACE`), so a sink may double-write safely.
- Produces: `EventBus.add_sink(sink: Callable[[AgentEvent], None])` (registering the same callable twice is a no-op). `_emit(...)` gains keyword `parent: str | None = None` and now **returns the emitted span_id** (str). `RunState.root_span_id: str` and `AnalystTask.root_span_id: str` (both default `""`). Tree shape later tasks/UI rely on: exactly one span per run has `parent_span_id=None` (the `run_started` span); every node's first event is a child of the root; an analyst/critic branch's subsequent events are children of that branch's first event.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_events.py`:

```python
def test_sink_receives_every_event_at_emit_time():
    from app.runtime.events import AgentEvent, EventBus, EventType

    bus = EventBus()
    seen: list[str] = []
    sink = lambda e: seen.append(e.type.value)  # noqa: E731
    bus.add_sink(sink)
    bus.add_sink(sink)  # double registration is a no-op

    bus.emit(AgentEvent(run_id="r-sink", agent="system", type=EventType.RUN_STARTED))
    bus.emit(AgentEvent(run_id="r-sink", agent="system", type=EventType.RUN_FINISHED))
    assert seen == ["run_started", "run_finished"]
```

Append to `backend/tests/test_graph.py` (reuses the module's existing `dataset` fixture and stub helpers):

```python
def test_span_tree_is_rooted_and_connected(dataset):
    from app.runtime.events import EventType, bus

    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst(
            {
                1: ("A", [Claim(text="total fare", kind="numeric", value=60.0)]),
                2: ("B", [Claim(text="busiest day", kind="categorical", value="Sat")]),
            }
        ),
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-tree"), deps)
    events = bus.history("run-tree")

    ids = {e.span_id for e in events}
    roots = [e for e in events if e.parent_span_id is None]
    assert len(roots) == 1 and roots[0].type == EventType.RUN_STARTED
    # every non-root event points at a span that exists in the same run
    assert all(e.parent_span_id in ids for e in events if e.parent_span_id is not None)
    # planner's llm_call is a direct child of the run root
    planner_call = next(e for e in events if e.agent == "planner" and e.type == EventType.LLM_CALL)
    assert planner_call.parent_span_id == roots[0].span_id


def test_analyst_iterations_nest_under_branch_root(dataset):
    from app.runtime.events import EventType, bus

    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('x')"), U
        return (
            AnalystTurn(
                action="finish",
                findings="F",
                claims=[Claim(text="t", kind="numeric", value=1.0)],
            ),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="one step")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-nest"), deps)
    events = bus.history("run-nest")

    analyst_events = [e for e in events if e.agent == "analyst"]
    branch_root = analyst_events[0]
    root = next(e for e in events if e.parent_span_id is None)
    assert branch_root.parent_span_id == root.span_id
    assert all(e.parent_span_id == branch_root.span_id for e in analyst_events[1:])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_events.py tests/test_graph.py -v`
Expected: the three new tests FAIL (`add_sink` missing; `parent_span_id` is None everywhere).

- [ ] **Step 3: Implement**

`backend/app/runtime/events.py` — add sinks to `EventBus`:

```python
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[AgentEvent | None]]] = {}
        self._history: dict[str, list[AgentEvent]] = {}
        self._finished: set[str] = set()
        self._sinks: list = []

    def add_sink(self, sink) -> None:
        """Register a synchronous callback invoked for every emitted event.

        Sinks persist events as they happen (live span persistence); they must
        not raise — a broken sink would take the run down with it, honestly.
        """
        if sink not in self._sinks:
            self._sinks.append(sink)
```

and at the top of `emit`, right after the history append:

```python
        for sink in self._sinks:
            sink(event)
```

`backend/app/runtime/state.py` — add to `RunState` (after `dataset_profile`):

```python
    root_span_id: str = ""  # span_id of the run_started event; parents the whole tree
```

and to `AnalystTask` (after `dataset_profile`):

```python
    root_span_id: str = ""  # Send payloads don't see RunState; the tree root rides along
```

`backend/app/runtime/graph.py`:

1. `_emit` gains a `parent` keyword and returns the span id:

```python
def _emit(
    run_id: str,
    agent: str,
    type_: EventType,
    payload: dict,
    started: float,
    usage: LLMUsage | None = None,
    parent: str | None = None,
) -> str:
    event = AgentEvent(
        run_id=run_id,
        parent_span_id=parent,
        agent=agent,
        type=type_,
        payload=payload,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
        cost_usd=pricing.cost_usd(usage.model, usage.tokens_in, usage.tokens_out)
        if usage
        else 0.0,
        started_at=started,
        duration_ms=int((time.time() - started) * 1000),
    )
    bus.emit(event)
    return event.span_id
```

2. `execute_run`: build the root event explicitly, remember its span id, thread it into state, and parent the closing events:

```python
    root = AgentEvent(
        run_id=state.run_id,
        agent="system",
        type=EventType.RUN_STARTED,
        payload={"question": state.question},
    )
    bus.emit(root)
    state.root_span_id = root.span_id
```

and the `RUN_FINISHED` `_emit(...)` call gains `parent=final_state.root_span_id`; the `except` branch's ERROR event gains `parent_span_id=root.span_id` in its `AgentEvent(...)` construction.

3. `planner_node`: capture the llm_call span and parent the rest on it:

```python
    # error path (BudgetExceeded/MalformedOutputError):
        _emit(..., parent=state.root_span_id)
    # success path:
    span = _emit(  # the plan llm_call — the planner's node root
        state.run_id, "planner", EventType.LLM_CALL, {...}, t0, usage,
        parent=state.root_span_id,
    )
    ...
    _emit(state.run_id, "planner", EventType.HANDOFF, {...}, time.time(), parent=span)
```

(The empty-plan early return sits between the two emits — move the `span = _emit(...)` assignment so both the handoff and nothing else need it; the empty-plan branch emits nothing new.)

4. `fan_out` and `route_after_critic`: add `root_span_id=state.root_span_id` to every `AnalystTask(...)` construction.

5. `analyst_node`: thread a branch root through the loop. After `task = AnalystTask.model_validate(task)` add `branch_root: str | None = None`, then change every `_emit(...)` in the function to capture/parent:

```python
        sid = _emit(..., parent=branch_root or task.root_span_id)
        branch_root = branch_root or sid
```

(Apply the same two-line pattern at the error `_emit`, the llm_call `_emit`, and the tool_call `_emit`.)

6. `critic_node`: same pattern with `state.root_span_id` — `branch_root: str | None = None` before the loop; error/llm_call/tool_call emits use `parent=branch_root or state.root_span_id` and set `branch_root` after; the verdict and handoff emits use `parent=branch_root or state.root_span_id` (do not reassign there).

7. `composer_node`: its llm_call `_emit` gains `parent=state.root_span_id`.

`backend/app/api/runs.py` — delete the end-of-run flush in `_execute` (the sink persists live):

```python
    finally:
        for event in bus.history(run_id):
            store().add_span(event)
```

becomes: remove the whole `finally` block (keep the `try/except` with `finish_run`). Remove the now-unused `bus` import if nothing else uses it (the SSE endpoint still uses `bus` — keep the import).

`backend/app/main.py` — register the persistence sink (after the routers):

```python
from app.deps import store
from app.runtime.events import bus

# Live span persistence: every AgentEvent lands in SQLite the moment it is
# emitted, so a crash mid-run still leaves an inspectable partial trace.
bus.add_sink(lambda event: store().add_span(event))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green, including the two new tree tests and the sink test.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime/events.py backend/app/runtime/state.py backend/app/runtime/graph.py backend/app/api/runs.py backend/app/main.py backend/tests/test_events.py backend/tests/test_graph.py
git commit -m "feat(m3): span tree (parent threading) + live persistence via event sink"
```

---

### Task 3: Daily budget cap + /api/config

Reject new runs once today's spend (UTC) crosses `daily_budget_usd`, and expose config (CHEAP_MODE, models, budget, spend) to the UI.

**Files:**
- Modify: `backend/app/config.py` (`daily_budget_usd`)
- Modify: `backend/app/tracing/store.py` (`cost_since`, `utc_midnight`)
- Create: `backend/app/api/config.py`
- Modify: `backend/app/api/runs.py` (budget guard in `create_run`)
- Modify: `backend/app/main.py` (include router)
- Test: `backend/tests/test_api_budget.py`, `backend/tests/test_store.py`

**Interfaces:**
- Produces: `Store.cost_since(since_ts: float) -> float`; module function `app.tracing.store.utc_midnight() -> float`; `GET /api/config` → `{"cheap_mode": bool, "daily_budget_usd": float, "spent_today": float, "models": {"planner": str, "analyst": str, "critic": str, "composer": str}}`; `POST /api/runs` → HTTP 429 when spent ≥ cap. Replay (Task 6) is exempt — it costs nothing.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_store.py`:

```python
def test_cost_since_sums_span_costs(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    run_id = store.create_run(ds, "q")
    for cost, ts in [(0.5, 100.0), (0.25, 200.0), (1.0, 50.0)]:
        store.add_span(
            AgentEvent(
                run_id=run_id, agent="analyst", type=EventType.LLM_CALL,
                cost_usd=cost, started_at=ts,
            )
        )
    assert store.cost_since(60.0) == 0.75  # the ts=50 span is before the window
    assert store.cost_since(0.0) == 1.75
```

Create `backend/tests/test_api_budget.py`:

```python
"""Daily budget cap: POST /api/runs is refused once today's spend crosses the cap."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.deps import store as store_dep
from app.runtime.events import AgentEvent, EventType


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.sqlite3"))
    settings.cache_clear()
    store_dep.cache_clear()
    from app.main import app

    yield TestClient(app)
    settings.cache_clear()
    store_dep.cache_clear()


def test_run_refused_when_daily_budget_exhausted(client):
    dataset_id = store_dep().add_dataset("d.csv", "/tmp/d.csv", {"rows": 1})
    run_id = store_dep().create_run(dataset_id, "warmup")
    store_dep().add_span(
        AgentEvent(
            run_id=run_id, agent="analyst", type=EventType.LLM_CALL,
            cost_usd=settings().daily_budget_usd + 1.0,
        )
    )
    res = client.post("/api/runs", json={"dataset_id": dataset_id, "question": "again?"})
    assert res.status_code == 429
    assert "budget" in res.json()["detail"].lower()


def test_config_endpoint_reports_spend_and_cheap_mode(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["cheap_mode"] is True
    assert body["daily_budget_usd"] == settings().daily_budget_usd
    assert set(body["models"]) == {"planner", "analyst", "critic", "composer"}
    assert body["spent_today"] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_api_budget.py tests/test_store.py -v`
Expected: FAIL (`cost_since` missing; /api/config 404; POST returns 200-ish path, not 429).

- [ ] **Step 3: Implement**

`backend/app/config.py` — add under the M2 block:

```python
    # M3 — observability.
    daily_budget_usd: float = 2.0  # hard cap on real-run spend per UTC day
```

`backend/app/tracing/store.py` — add near the top (after imports):

```python
from datetime import datetime, timezone


def utc_midnight() -> float:
    """Start of the current UTC day as a unix timestamp — the budget window."""
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
```

and a method in `Store` (spans section):

```python
    def cost_since(self, since_ts: float) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM spans WHERE started_at >= ?",
                (since_ts,),
            ).fetchone()
        return float(row["c"])
```

Create `backend/app/api/config.py`:

```python
"""Read-only config surface for the UI: models, CHEAP_MODE, budget state."""

from fastapi import APIRouter

from app.config import settings
from app.deps import store
from app.tracing.store import utc_midnight

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
def get_config() -> dict:
    cfg = settings()
    return {
        "cheap_mode": cfg.cheap_mode,
        "daily_budget_usd": cfg.daily_budget_usd,
        "spent_today": store().cost_since(utc_midnight()),
        "models": {role: cfg.model_for(role) for role in ("planner", "analyst", "critic", "composer")},
    }
```

`backend/app/api/runs.py` — in `create_run`, after the dataset/question validation and before `store().create_run(...)`:

```python
    cfg = settings()
    spent = store().cost_since(utc_midnight())
    if spent >= cfg.daily_budget_usd:
        raise HTTPException(
            429,
            f"daily budget of ${cfg.daily_budget_usd:.2f} exhausted (${spent:.2f} spent today)",
        )
```

with imports `from app.config import settings` and `from app.tracing.store import utc_midnight`.

`backend/app/main.py` — `from app.api import config, datasets, runs` and `app.include_router(config.router)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py backend/app/tracing/store.py backend/app/api/config.py backend/app/api/runs.py backend/app/main.py backend/tests/test_api_budget.py backend/tests/test_store.py
git commit -m "feat(m3): daily budget cap (429 past the cap) + /api/config for the UI"
```

---

### Task 4: Sandbox becomes injectable via GraphDeps

Replay must intercept sandbox executions the same way it intercepts LLM calls. Move the `run_code` call behind `GraphDeps` — the seam tests already use.

**Files:**
- Modify: `backend/app/agents/llm.py` (`GraphDeps.run_code`)
- Modify: `backend/app/runtime/graph.py` (two call sites)
- Test: `backend/tests/test_graph.py`

**Interfaces:**
- Produces: `GraphDeps.run_code: Callable[[str, str], SandboxResult]` — signature `(code, dataset_path) -> SandboxResult`, defaulting to the real `app.sandbox.executor.run_code` via `__post_init__` (existing 4-arg `GraphDeps(...)` constructions keep working). `SandboxFn` type alias exported from `app.agents.llm`.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_graph.py`)

```python
def test_sandbox_is_injectable_through_deps(dataset):
    from app.runtime.state import SandboxResult

    executed: list[str] = []

    def fake_run_code(code: str, dataset_path: str) -> SandboxResult:
        executed.append(code)
        return SandboxResult(code=code, stdout="CANNED OUTPUT", exit_code=0)

    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('never runs')"), U
        assert "CANNED OUTPUT" in messages[-1].content  # the fake result came back
        return (
            AnalystTurn(
                action="finish", findings="F",
                claims=[Claim(text="t", kind="numeric", value=1.0)],
            ),
            U,
        )

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="s")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
        run_code=fake_run_code,
    )
    final = execute_run(make_state(dataset, "run-fakebox"), deps)
    assert executed == ["print('never runs')"]
    assert final.analyst_results[0].iterations[0].result.stdout == "CANNED OUTPUT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_graph.py::test_sandbox_is_injectable_through_deps -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'run_code'`

- [ ] **Step 3: Implement**

`backend/app/agents/llm.py`:

```python
from app.runtime.state import SandboxResult

SandboxFn = Callable[[str, str], SandboxResult]  # (code, dataset_path) -> result
```

(place the alias next to the other `*Fn` aliases) and extend `GraphDeps`:

```python
@dataclass
class GraphDeps:
    """Injected model callables. Tests pass stubs; production uses `default()`."""

    planner: PlannerFn
    analyst_turn: AnalystFn
    critic_turn: CriticFn
    compose: ComposeFn
    run_code: SandboxFn | None = None  # None → the real sandbox (set in __post_init__)

    def __post_init__(self) -> None:
        if self.run_code is None:
            from app.sandbox.executor import run_code

            self.run_code = run_code
```

`backend/app/runtime/graph.py`: delete `from app.sandbox.executor import run_code`; change the analyst call site `result = run_code(turn.code, task.dataset_path)` to `result = deps.run_code(turn.code, task.dataset_path)` and the critic call site `result = run_code(turn.code, state.dataset_path)` to `result = deps.run_code(turn.code, state.dataset_path)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green (existing graph tests still execute the real sandbox by default).

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/llm.py backend/app/runtime/graph.py backend/tests/test_graph.py
git commit -m "refactor(m3): inject sandbox through GraphDeps — the replay seam"
```

---

### Task 5: Recording layer

Record every LLM call and sandbox execution of a real run, keyed by a content hash of the request, so parallel branches replay deterministically regardless of completion order.

**Files:**
- Create: `backend/app/runtime/recording.py`
- Modify: `backend/app/tracing/store.py` (recordings table + methods)
- Modify: `backend/app/api/runs.py` (wrap production deps)
- Test: `backend/tests/test_recording.py`

**Interfaces:**
- Consumes: `GraphDeps` (incl. `run_code` from Task 4), `Store`.
- Produces:
  - `recording.request_key(role: str, messages: list[BaseMessage]) -> str` (sha256 hex) and `recording.sandbox_key(code: str) -> str`.
  - `Recorder(store: Store, run_id: str)` with `.record(kind: str, key: str, response: dict)` — thread-safe per-key sequence numbers.
  - `recording_deps(inner: GraphDeps, recorder: Recorder) -> GraphDeps` — pass-through wrapper that records after each call. LLM responses are stored as `{"turn": <model_dump>, "usage": {"tokens_in", "tokens_out", "model"}}` (composer: `{"text": str, "usage": {...}}`); sandbox responses as `SandboxResult.model_dump()`.
  - `Store.add_recording(run_id, key, seq, kind, response: dict)` and `Store.recordings_for_run(run_id) -> list[dict]` (each row: `run_id, key, seq, kind, response` with response JSON-decoded; ordered by key, seq).

- [ ] **Step 1: Write the failing test** (`backend/tests/test_recording.py`)

```python
"""Recording captures the nondeterministic boundary: LLM turns + sandbox runs."""

import json
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.graph import execute_run
from app.runtime.recording import Recorder, recording_deps, request_key
from app.runtime.state import Claim, PlanStep, RunState, SandboxResult
from app.tracing.store import Store

U = LLMUsage(tokens_in=10, tokens_out=5, model="gpt-4o-mini")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip\n10,2\n20,5\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="total fare?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 2, "columns": [{"name": "fare"}, {"name": "tip"}]},
    )


def coding_then_finishing_analyst():
    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('total 30')"), U
        return (
            AnalystTurn(
                action="finish", findings="Total fare is 30.",
                claims=[Claim(text="total fare", kind="numeric", value=30.0)],
            ),
            U,
        )

    return analyst


def verifying_critic(messages):
    text = next(m.content for m in messages if "Claims to verify:" in m.content)
    claims = json.loads(text.split("Claims to verify:\n", 1)[1])
    return (
        CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
        ),
        U,
    )


def stub_deps() -> GraphDeps:
    return GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=coding_then_finishing_analyst(),
        critic_turn=verifying_critic,
        compose=lambda m: ("Total fare is 30.", U),
        run_code=lambda code, path: SandboxResult(code=code, stdout="total 30\n", exit_code=0),
    )


def test_recording_captures_every_llm_and_sandbox_call(dataset, tmp_path):
    store = Store(tmp_path / "rec.db")
    execute_run(make_state(dataset, "run-rec1"), recording_deps(stub_deps(), Recorder(store, "run-rec1")))

    rows = store.recordings_for_run("run-rec1")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("sandbox") == 1
    # planner + 2 analyst turns + critic + composer = 5 llm recordings
    assert kinds.count("llm") == 5
    sandbox = next(r for r in rows if r["kind"] == "sandbox")
    assert sandbox["response"]["stdout"] == "total 30\n"
    llm = next(r for r in rows if r["kind"] == "llm")
    assert "usage" in llm["response"] and llm["response"]["usage"]["model"] == "gpt-4o-mini"


def test_request_keys_are_deterministic_across_runs(dataset, tmp_path):
    store = Store(tmp_path / "rec2.db")
    for run_id in ("run-a", "run-b"):
        execute_run(
            make_state(dataset, run_id), recording_deps(stub_deps(), Recorder(store, run_id))
        )
    keys_a = sorted((r["kind"], r["key"], r["seq"]) for r in store.recordings_for_run("run-a"))
    keys_b = sorted((r["kind"], r["key"], r["seq"]) for r in store.recordings_for_run("run-b"))
    assert keys_a == keys_b  # same stub conversation → identical keys, both runs replayable


def test_identical_requests_get_increasing_seq(tmp_path):
    store = Store(tmp_path / "rec3.db")
    rec = Recorder(store, "run-seq")
    rec.record("llm", "same-key", {"n": 1})
    rec.record("llm", "same-key", {"n": 2})
    rows = store.recordings_for_run("run-seq")
    assert [(r["seq"], r["response"]["n"]) for r in rows] == [(0, 1), (1, 2)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_recording.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.runtime.recording'`

- [ ] **Step 3: Implement**

`backend/app/tracing/store.py` — append to `_SCHEMA` (before the index line):

```sql
CREATE TABLE IF NOT EXISTS recordings (
    run_id TEXT NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,          -- sha256 of the request content (role + messages / code)
    seq INTEGER NOT NULL,       -- per-key ordinal: identical requests replay in order
    kind TEXT NOT NULL,         -- 'llm' | 'sandbox'
    response TEXT NOT NULL,     -- JSON: recorded output + usage
    PRIMARY KEY (run_id, key, seq)
);
```

(`CREATE TABLE IF NOT EXISTS` in `_SCHEMA` doubles as the migration for existing DBs — same pattern as the other tables.) Add methods to `Store`:

```python
    # ── recordings (deterministic replay) ─────────────────────────────────
    def add_recording(self, run_id: str, key: str, seq: int, kind: str, response: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO recordings VALUES (?, ?, ?, ?, ?)",
                (run_id, key, seq, kind, json.dumps(response)),
            )

    def recordings_for_run(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM recordings WHERE run_id = ? ORDER BY key, seq", (run_id,)
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["response"] = json.loads(d["response"])
            out.append(d)
        return out
```

Create `backend/app/runtime/recording.py`:

```python
"""Record/replay of the nondeterministic boundary: LLM calls and the sandbox.

Recording wraps GraphDeps — the same seam tests use for stubs. Every call is
keyed by a sha256 of its request content, so parallel analyst branches replay
correctly no matter which order they complete in. Identical requests within a
run replay in first-recorded order via a per-key sequence number. Replay never
touches the network or the sandbox: a recorded run replays on a plane.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict

from langchain_core.messages import BaseMessage

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticTurn, PlannerTurn
from app.runtime.state import SandboxResult
from app.tracing.store import Store


def request_key(role: str, messages: list[BaseMessage]) -> str:
    content = json.dumps([[m.__class__.__name__, str(m.content)] for m in messages])
    return hashlib.sha256(f"{role}\n{content}".encode()).hexdigest()


def sandbox_key(code: str) -> str:
    return hashlib.sha256(f"sandbox\n{code}".encode()).hexdigest()


def _usage_dict(usage: LLMUsage) -> dict:
    return {"tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out, "model": usage.model}


class Recorder:
    """Appends recordings with thread-safe per-key sequence numbers."""

    def __init__(self, store: Store, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self._seq: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, kind: str, key: str, response: dict) -> None:
        with self._lock:
            seq = self._seq[key]
            self._seq[key] += 1
        self.store.add_recording(self.run_id, key, seq, kind, response)


def recording_deps(inner: GraphDeps, recorder: Recorder) -> GraphDeps:
    """Wrap deps so every call passes through unchanged but is recorded."""

    def structured(role: str, fn):
        def call(messages):
            turn, usage = fn(messages)
            recorder.record(
                "llm",
                request_key(role, messages),
                {"turn": turn.model_dump(), "usage": _usage_dict(usage)},
            )
            return turn, usage

        return call

    def compose(messages):
        text, usage = inner.compose(messages)
        recorder.record(
            "llm", request_key("composer", messages), {"text": text, "usage": _usage_dict(usage)}
        )
        return text, usage

    def run_code(code: str, dataset_path: str) -> SandboxResult:
        result = inner.run_code(code, dataset_path)
        recorder.record("sandbox", sandbox_key(code), result.model_dump())
        return result

    return GraphDeps(
        planner=structured("planner", inner.planner),
        analyst_turn=structured("analyst", inner.analyst_turn),
        critic_turn=structured("critic", inner.critic_turn),
        compose=compose,
        run_code=run_code,
    )
```

`backend/app/api/runs.py` — real runs always record. Change `_execute` to accept optional deps and build the recording wrapper by default:

```python
from app.agents.llm import GraphDeps
from app.runtime.recording import Recorder, recording_deps


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime/recording.py backend/app/tracing/store.py backend/app/api/runs.py backend/tests/test_recording.py
git commit -m "feat(m3): record LLM + sandbox calls per run, keyed by request hash"
```

---

### Task 6: Deterministic replay

Rebuild `GraphDeps` from a run's recordings and re-execute the graph offline: same conversation, same sandbox outputs, $0 cost, no API key, no sandbox processes.

**Files:**
- Modify: `backend/app/runtime/recording.py` (`ReplayMiss`, `replay_deps`)
- Modify: `backend/app/tracing/store.py` (`runs.replay_of` column + migration; `create_run(replay_of=)`)
- Modify: `backend/app/api/runs.py` (`POST /api/runs/{run_id}/replay`)
- Test: `backend/tests/test_replay.py`, `backend/tests/test_store.py`

**Interfaces:**
- Consumes: `recordings_for_run` rows, `request_key`/`sandbox_key`, `GraphDeps.run_code` seam.
- Produces: `recording.replay_deps(recordings: list[dict]) -> GraphDeps` (raises `ReplayMiss` on an unrecorded request — a replay that diverges fails loudly, never silently re-calls the network). Replayed usage carries recorded token counts with `model="replay"` → cost 0 by the price table. `Store.create_run(dataset_id, question, replay_of="")`; run dicts now include `replay_of`. `POST /api/runs/{run_id}/replay` → `{"run_id": <new>}` (404 unknown run, 400 no recordings).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_store.py`:

```python
def test_replay_of_column_roundtrip_and_migration(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    original = store.create_run(ds, "q")
    replay = store.create_run(ds, "q", replay_of=original)
    assert store.get_run(original)["replay_of"] == ""
    assert store.get_run(replay)["replay_of"] == original

    # a pre-M3 database gains the column on open
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', answer TEXT DEFAULT '',
            result TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL
        );
        INSERT INTO runs (id, dataset_id, question, created_at) VALUES ('r1', 'd1', 'q', 0);
        """
    )
    conn.commit()
    conn.close()
    assert Store(db).get_run("r1")["replay_of"] == ""
```

Create `backend/tests/test_replay.py`:

```python
"""Deterministic replay: a recorded run re-executes offline to the same answer."""

import json
from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.events import bus
from app.runtime.graph import execute_run
from app.runtime.recording import Recorder, ReplayMiss, recording_deps, replay_deps
from app.runtime.state import Claim, PlanStep, RunState, SandboxResult
from app.tracing.store import Store

U = LLMUsage(tokens_in=10, tokens_out=5, model="gpt-4o-mini")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip\n10,2\n20,5\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="total fare?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 2, "columns": [{"name": "fare"}, {"name": "tip"}]},
    )


def stub_deps() -> GraphDeps:
    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(action="run_code", code="print('total 30')"), U
        return (
            AnalystTurn(
                action="finish", findings="Total fare is 30.",
                claims=[Claim(text="total fare", kind="numeric", value=30.0)],
            ),
            U,
        )

    def critic(messages):
        text = next(m.content for m in messages if "Claims to verify:" in m.content)
        claims = json.loads(text.split("Claims to verify:\n", 1)[1])
        return (
            CriticTurn(
                action="finish",
                findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
            ),
            U,
        )

    return GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=critic,
        compose=lambda m: ("Total fare is 30.", U),
        run_code=lambda code, path: SandboxResult(code=code, stdout="total 30\n", exit_code=0),
    )


def test_replay_reproduces_the_run_offline(dataset, tmp_path):
    store = Store(tmp_path / "replay.db")
    original = execute_run(
        make_state(dataset, "run-orig"), recording_deps(stub_deps(), Recorder(store, "run-orig"))
    )

    # Replay uses ONLY the recordings — no stubs, no sandbox, no network.
    replayed = execute_run(
        make_state(dataset, "run-replay"), replay_deps(store.recordings_for_run("run-orig"))
    )

    assert replayed.final_answer == original.final_answer
    assert replayed.final == original.final
    assert [v.model_dump() for v in replayed.verdicts] == [
        v.model_dump() for v in original.verdicts
    ]
    # replayed events keep token counts but cost nothing
    events = bus.history("run-replay")
    assert any(e.tokens_in > 0 for e in events)
    assert all(e.cost_usd == 0 for e in events)


def test_replay_missing_recording_fails_loudly(dataset):
    deps = replay_deps([])  # nothing recorded
    with pytest.raises(Exception) as exc:
        execute_run(make_state(dataset, "run-miss"), deps)
    assert "ReplayMiss" in exc.value.__class__.__name__ or isinstance(exc.value, ReplayMiss)
```

Note: the planner's `ReplayMiss` is caught by no node (planner_node catches only `BudgetExceeded`/`MalformedOutputError`), so it propagates out of `execute_run` — that is the loud failure we assert.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_replay.py tests/test_store.py -v`
Expected: FAIL (`ReplayMiss`/`replay_deps` not importable; `create_run` rejects `replay_of` kwarg).

- [ ] **Step 3: Implement**

`backend/app/tracing/store.py`:

1. `_SCHEMA` runs table gains `replay_of TEXT NOT NULL DEFAULT ''` (after `result`).
2. `_migrate` gains:

```python
        if "replay_of" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN replay_of TEXT NOT NULL DEFAULT ''")
```

3. `create_run` becomes:

```python
    def create_run(self, dataset_id: str, question: str, replay_of: str = "") -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, dataset_id, question, replay_of, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, dataset_id, question, replay_of, time.time()),
            )
        return run_id
```

`backend/app/runtime/recording.py` — append:

```python
class ReplayMiss(Exception):
    """A replayed run issued a request that was never recorded."""

    def __init__(self, kind: str, key: str) -> None:
        super().__init__(
            f"replay miss: no recorded {kind} response for request {key[:12]}… — "
            "the replayed graph diverged from the original run"
        )


def replay_deps(recordings: list[dict]) -> GraphDeps:
    """GraphDeps that answer every request from recordings. Fully offline."""

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(recordings, key=lambda r: r["seq"]):
        buckets[row["key"]].append(row["response"])
    lock = threading.Lock()

    def pop(kind: str, key: str) -> dict:
        with lock:
            bucket = buckets.get(key)
            if not bucket:
                raise ReplayMiss(kind, key)
            return bucket.pop(0)

    def replay_usage(rec: dict) -> LLMUsage:
        usage = rec.get("usage", {})
        # model="replay" is deliberately unpriced: replays keep tokens, cost $0
        return LLMUsage(usage.get("tokens_in", 0), usage.get("tokens_out", 0), model="replay")

    def structured(role: str, schema):
        def call(messages):
            rec = pop("llm", request_key(role, messages))
            return schema.model_validate(rec["turn"]), replay_usage(rec)

        return call

    def compose(messages):
        rec = pop("llm", request_key("composer", messages))
        return rec["text"], replay_usage(rec)

    def run_code(code: str, dataset_path: str) -> SandboxResult:
        return SandboxResult.model_validate(pop("sandbox", sandbox_key(code)))

    return GraphDeps(
        planner=structured("planner", PlannerTurn),
        analyst_turn=structured("analyst", AnalystTurn),
        critic_turn=structured("critic", CriticTurn),
        compose=compose,
        run_code=run_code,
    )
```

`backend/app/api/runs.py` — add the endpoint (imports: `from app.runtime.recording import Recorder, recording_deps, replay_deps`):

```python
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
```

(No budget guard here: replays cost nothing. `_execute` with explicit deps skips recording — replays are not re-recorded.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green — the replay test proves record → offline re-execution → identical `FinalAnswer`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime/recording.py backend/app/tracing/store.py backend/app/api/runs.py backend/tests/test_replay.py backend/tests/test_store.py
git commit -m "feat(m3): deterministic offline replay from recorded calls + replay endpoint"
```

---

### Task 7: Remaining stats toolkit

Four new planner methods — `regression`, `clustering`, `timeseries_backtest`, `anomaly_detection` — with analyst playbooks, critic review guidance, and statsmodels/scikit-learn in the sandbox.

**Files:**
- Modify: `backend/app/runtime/state.py` (`PlanStep.method` literals)
- Modify: `backend/pyproject.toml` (statsmodels, scikit-learn)
- Modify: `backend/app/agents/prompts/planner.md`, `analyst.md`, `critic.md`
- Test: `backend/tests/test_sandbox.py`, `backend/tests/test_graph.py`

**Interfaces:**
- Produces: `PlanStep.method: Literal["descriptive", "mean_comparison", "correlation", "regression", "clustering", "timeseries_backtest", "anomaly_detection"]`. Sandbox imports `statsmodels` and `sklearn`. (The frontend `MethodologyChip` renders `Methodology` generically — no frontend change needed.)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_graph.py`:

```python
def test_plan_step_accepts_m3_methods():
    for method in ("regression", "clustering", "timeseries_backtest", "anomaly_detection"):
        assert PlanStep(description="x", method=method).method == method
```

Append to `backend/tests/test_sandbox.py` (the module already imports `run_code` and has a `dataset` fixture — reuse both):

```python
def test_stats_libraries_are_importable_in_sandbox(dataset):
    result = run_code(
        "import statsmodels.api, sklearn.cluster, sklearn.decomposition\nprint('stats ok')",
        str(dataset),
    )
    assert result.exit_code == 0 and "stats ok" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_graph.py::test_plan_step_accepts_m3_methods tests/test_sandbox.py -v`
Expected: PlanStep test FAILs with a Pydantic validation error; sandbox test FAILs with `ModuleNotFoundError` in stdout/stderr (exit_code != 0).

- [ ] **Step 3: Implement**

`backend/app/runtime/state.py`:

```python
class PlanStep(BaseModel):
    id: int = 0
    description: str
    method: Literal[
        "descriptive",
        "mean_comparison",
        "correlation",
        "regression",
        "clustering",
        "timeseries_backtest",
        "anomaly_detection",
    ] = "descriptive"
```

`backend/pyproject.toml` dependencies — add:

```toml
    "statsmodels>=0.14",
    "scikit-learn>=1.5",
```

Install: `backend/.venv/bin/pip install -e "backend[dev]"` (run from repo root).

`backend/app/agents/prompts/planner.md` — extend the method list (after the `correlation` bullet):

```markdown
- `regression` — how one numeric outcome depends on one or more numeric predictors
  ("what drives price?"). Triggers OLS with diagnostics.
- `clustering` — discovering natural groups of rows ("segment the customers").
  Triggers standardized k-means with a PCA projection chart.
- `timeseries_backtest` — forecasting a value ordered by a date/time column
  ("how will sales develop?"). Triggers a time-ordered train/holdout backtest
  against a naive baseline.
- `anomaly_detection` — finding unusual rows or outliers ("any suspicious
  transactions?"). Triggers robust outlier scoring.
```

`backend/app/agents/prompts/analyst.md` — change the libraries rule to:

```markdown
- Libraries available: pandas, numpy, scipy, statsmodels, sklearn.
```

add a reproducibility rule to the Rules list:

```markdown
- Set `random_state=0` on every stochastic method (KMeans, IsolationForest,
  sampling). The critic must be able to reproduce your numbers exactly.
```

and append these playbooks after the `correlation` playbook:

```markdown
- `regression`: OLS via statsmodels (`sm.add_constant`, `sm.OLS(...).fit()`). Report n,
  each coefficient with its p-value, R² and adjusted R². Diagnostics are mandatory,
  not optional: residual normality (shapiro on ≤5000 residuals, else skewness),
  heteroscedasticity (`statsmodels.stats.diagnostic.het_breuschpagan`), and
  multicollinearity (VIF; flag predictors with VIF > 10) — list what passed and
  failed in the methodology assumptions. The key statistical claim is the main
  predictor's direction and significance at alpha = {alpha}; effect size is
  adjusted R² (name it "adjusted R²").
- `clustering`: standardize numeric columns (`sklearn.preprocessing.StandardScaler`).
  Choose k in 2..6 by silhouette score (`sklearn.metrics.silhouette_score`), then
  `KMeans(n_clusters=k, n_init=10, random_state=0)`. Report chosen k, silhouette,
  cluster sizes, and per-cluster means of the most distinguishing columns (numeric
  claims). Chart: PCA 2-component scatter (`sklearn.decomposition.PCA`) of ≤500
  sampled rows with a "cluster" field as a categorical series.
- `timeseries_backtest`: sort by the time column; hold out the final ~20% of rows.
  Baseline = naive last-value (or seasonal-naive when an obvious period exists).
  Model = rolling mean or `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
  Report MAE and MAPE for model AND baseline on the holdout (numeric claims). A
  model that cannot beat the naive baseline must be reported as exactly that —
  "does not beat naive" is a valid, honest finding.
- `anomaly_detection`: robust z-score (median/MAD) or IQR fences per numeric column;
  for multivariate anomalies use `sklearn.ensemble.IsolationForest(random_state=0)`.
  Report the anomaly count, the share of rows (numeric claims), and the top 5 most
  anomalous rows with the columns that make them anomalous in the findings text.
```

`backend/app/agents/prompts/critic.md` — change the libraries rule to `pandas, numpy, scipy, statsmodels, sklearn` and add one bullet to the Rules list:

```markdown
- Methodology review covers the full toolkit: regression (were diagnostics actually
  run? enough rows per predictor?), clustering (is k justified by silhouette, or
  arbitrary?), backtests (was the split time-ordered? was a naive baseline compared?),
  and anomaly detection (is the threshold defensible?). Stochastic methods without
  `random_state=0` are unreproducible — flag them.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all green (the sandbox import test may take a few seconds — cold statsmodels/sklearn imports).

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime/state.py backend/pyproject.toml backend/app/agents/prompts/planner.md backend/app/agents/prompts/analyst.md backend/app/agents/prompts/critic.md backend/tests/test_sandbox.py backend/tests/test_graph.py
git commit -m "feat(m3): stats toolkit — regression, clustering+PCA, backtest, anomaly detection"
```

---

### Task 8: Runs list with cost/latency/quality rollups

The dashboard needs one query that returns every run with its cost, token, latency, and claim-quality aggregates.

**Files:**
- Modify: `backend/app/tracing/store.py` (`list_runs_with_stats`)
- Modify: `backend/app/api/runs.py` (`GET /api/runs` uses it)
- Test: `backend/tests/test_store.py`

**Interfaces:**
- Produces: `Store.list_runs_with_stats() -> list[dict]`, newest first, each dict: `id, dataset_id, question, status, replay_of, created_at, cost_usd, tokens_in, tokens_out, duration_ms, claims_total, claims_verified` (no `answer`/`result` payloads — dashboard rows stay light). `duration_ms` = wall clock from first span start to last span end. `GET /api/runs` returns exactly these rows.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_store.py`)

```python
def test_list_runs_with_stats_rolls_up_cost_latency_quality(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    run_id = store.create_run(ds, "q")
    store.add_span(
        AgentEvent(
            run_id=run_id, agent="planner", type=EventType.LLM_CALL,
            tokens_in=100, tokens_out=50, cost_usd=0.001, started_at=10.0, duration_ms=500,
        )
    )
    store.add_span(
        AgentEvent(
            run_id=run_id, agent="composer", type=EventType.LLM_CALL,
            tokens_in=200, tokens_out=100, cost_usd=0.002, started_at=12.0, duration_ms=1000,
        )
    )
    result = {
        "narrative": "n",
        "failed": False,
        "charts": [],
        "claims": [
            {"claim": {"text": "a", "kind": "numeric"}, "status": "verified", "detail": ""},
            {"claim": {"text": "b", "kind": "numeric"}, "status": "unverified", "detail": "d"},
        ],
    }
    import json as _json

    store.finish_run(run_id, "answer", "finished", _json.dumps(result))

    empty_run = store.create_run(ds, "no spans yet")

    rows = store.list_runs_with_stats()
    assert [r["id"] for r in rows] == [empty_run, run_id]  # newest first
    row = next(r for r in rows if r["id"] == run_id)
    assert row["cost_usd"] == pytest.approx(0.003)
    assert row["tokens_in"] == 300 and row["tokens_out"] == 150
    # first span starts at 10.0, last ends at 12.0 + 1.0s → 3000ms wall clock
    assert row["duration_ms"] == 3000
    assert row["claims_total"] == 2 and row["claims_verified"] == 1
    assert "result" not in row and "answer" not in row

    empty = next(r for r in rows if r["id"] == empty_run)
    assert empty["cost_usd"] == 0 and empty["duration_ms"] == 0 and empty["claims_total"] == 0
```

Add `import pytest` to the top of `test_store.py` if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'list_runs_with_stats'`

- [ ] **Step 3: Implement** — add to `Store` (runs section):

```python
    def list_runs_with_stats(self) -> list[dict]:
        """Dashboard rows: run columns + cost/token/latency rollups + claim quality."""
        sql = """
        SELECT r.id, r.dataset_id, r.question, r.status, r.replay_of, r.created_at, r.result,
               COALESCE(s.cost_usd, 0) AS cost_usd,
               COALESCE(s.tokens_in, 0) AS tokens_in,
               COALESCE(s.tokens_out, 0) AS tokens_out,
               COALESCE(s.duration_ms, 0) AS duration_ms
        FROM runs r
        LEFT JOIN (
            SELECT run_id,
                   SUM(cost_usd) AS cost_usd,
                   SUM(tokens_in) AS tokens_in,
                   SUM(tokens_out) AS tokens_out,
                   CAST((MAX(started_at + duration_ms / 1000.0) - MIN(started_at)) * 1000
                        AS INTEGER) AS duration_ms
            FROM spans GROUP BY run_id
        ) s ON s.run_id = r.id
        ORDER BY r.created_at DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            claims = []
            result = d.pop("result")
            if result:
                claims = json.loads(result).get("claims", [])
            d["claims_total"] = len(claims)
            d["claims_verified"] = sum(1 for c in claims if c.get("status") == "verified")
            out.append(d)
        return out
```

`backend/app/api/runs.py` — change `list_runs`:

```python
@router.get("")
def list_runs() -> list[dict]:
    return store().list_runs_with_stats()
```

(The plain `Store.list_runs` stays — other code/tests use it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest -q && .venv/bin/ruff check .`
Expected: all green, lint clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/tracing/store.py backend/app/api/runs.py backend/tests/test_store.py
git commit -m "feat(m3): runs list with cost/latency/quality rollups for the dashboard"
```

---

### Task 9: Frontend shell — router, nav, config chip, runs dashboard

Turn the single-page app into three routes and build the dashboard. RunView itself lands in Task 10; until then `/runs/:runId` renders nothing (same-milestone gap, closed next task).

**Files:**
- Modify: `frontend/package.json` (react-router-dom, @xyflow/react)
- Modify: `frontend/src/lib/types.ts` (RunSummary, RunDetail, AppConfig)
- Modify: `frontend/src/lib/api.ts` (listRuns, getRun, replayRun, getConfig)
- Modify: `frontend/src/App.tsx` (router + nav layout)
- Create: `frontend/src/pages/RunsDashboard.tsx`

**Interfaces:**
- Produces: routes `/` (Workbench), `/runs` (RunsDashboard); Task 10 adds `/runs/:runId`. API client: `listRuns(): Promise<RunSummary[]>`, `getRun(id: string): Promise<RunDetail>`, `replayRun(id: string): Promise<{ run_id: string }>`, `getConfig(): Promise<AppConfig>`. Types below are what Tasks 10–11 import.

- [ ] **Step 1: Install the new dependencies**

```bash
cd frontend && npm install react-router-dom@^7 @xyflow/react@^12
```

- [ ] **Step 2: Extend types and API client**

Append to `frontend/src/lib/types.ts`:

```ts
export type RunStatusValue = "running" | "finished" | "error";

/** One row of GET /api/runs — dashboard aggregates, no heavy payloads. */
export interface RunSummary {
  id: string;
  dataset_id: string;
  question: string;
  status: RunStatusValue;
  replay_of: string;
  created_at: number;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  duration_ms: number;
  claims_total: number;
  claims_verified: number;
}

/** GET /api/runs/:id — the full run including its span tree. */
export interface RunDetail {
  id: string;
  dataset_id: string;
  question: string;
  status: RunStatusValue;
  answer: string;
  result: FinalAnswer | null;
  replay_of: string;
  created_at: number;
  spans: AgentEvent[];
}

export interface AppConfig {
  cheap_mode: boolean;
  daily_budget_usd: number;
  spent_today: number;
  models: Record<string, string>;
}
```

Append to `frontend/src/lib/api.ts`:

```ts
import type { AppConfig, RunDetail, RunSummary } from "./types";

export async function listRuns(): Promise<RunSummary[]> {
  return json(await fetch("/api/runs"));
}

export async function getRun(runId: string): Promise<RunDetail> {
  return json(await fetch(`/api/runs/${runId}`));
}

export async function replayRun(runId: string): Promise<{ run_id: string }> {
  return json(await fetch(`/api/runs/${runId}/replay`, { method: "POST" }));
}

export async function getConfig(): Promise<AppConfig> {
  return json(await fetch("/api/config"));
}
```

(Merge the type import with the existing `import type { Dataset }` line.)

- [ ] **Step 3: Router + nav shell** — replace `frontend/src/App.tsx`:

```tsx
import { AppBar, Box, Button, Chip, CssBaseline, Stack, ThemeProvider, Toolbar, Typography, createTheme } from "@mui/material";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { BrowserRouter, Link as RouterLink, Route, Routes } from "react-router-dom";
import { getConfig } from "./lib/api";
import { RunsDashboard } from "./pages/RunsDashboard";
import { Workbench } from "./pages/Workbench";

const theme = createTheme({
  palette: { mode: "dark", primary: { main: "#7aa2f7" }, background: { default: "#0f1115" } },
  typography: { fontFamily: "Inter, system-ui, sans-serif" },
  shape: { borderRadius: 10 },
});

const queryClient = new QueryClient();

function ConfigChips() {
  const { data } = useQuery({ queryKey: ["config"], queryFn: getConfig, refetchInterval: 60_000 });
  if (!data) return null;
  return (
    <Stack direction="row" spacing={1}>
      {data.cheap_mode && (
        <Chip size="small" color="warning" variant="outlined" label="CHEAP_MODE" />
      )}
      <Chip
        size="small"
        variant="outlined"
        label={`$${data.spent_today.toFixed(2)} / $${data.daily_budget_usd.toFixed(2)} today`}
      />
    </Stack>
  );
}

function Nav() {
  return (
    <AppBar position="sticky" color="transparent" elevation={0} sx={{ borderBottom: 1, borderColor: "divider", backdropFilter: "blur(8px)" }}>
      <Toolbar variant="dense" sx={{ gap: 2 }}>
        <Typography
          variant="h6"
          fontWeight={700}
          component={RouterLink}
          to="/"
          sx={{ color: "inherit", textDecoration: "none" }}
        >
          tracelab
        </Typography>
        <Button size="small" color="inherit" component={RouterLink} to="/runs">
          Runs
        </Button>
        <Box sx={{ flexGrow: 1 }} />
        <ConfigChips />
      </Toolbar>
    </AppBar>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Nav />
          <Routes>
            <Route path="/" element={<Workbench />} />
            <Route path="/runs" element={<RunsDashboard />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 4: Runs dashboard** — create `frontend/src/pages/RunsDashboard.tsx`:

```tsx
import ReplayIcon from "@mui/icons-material/Replay";
import { Alert, Chip, Container, Stack, Typography } from "@mui/material";
import { DataGrid, GridActionsCellItem, type GridColDef } from "@mui/x-data-grid";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { listRuns, replayRun } from "../lib/api";
import type { RunSummary } from "../lib/types";

const statusColor = { running: "info", finished: "success", error: "error" } as const;

export function RunsDashboard() {
  const navigate = useNavigate();
  const runs = useQuery({ queryKey: ["runs"], queryFn: listRuns, refetchInterval: 5_000 });
  const replay = useMutation({
    mutationFn: replayRun,
    onSuccess: ({ run_id }) => navigate(`/runs/${run_id}`),
  });

  const columns: GridColDef<RunSummary>[] = [
    {
      field: "created_at",
      headerName: "When",
      width: 170,
      valueFormatter: (value: number) => new Date(value * 1000).toLocaleString(),
    },
    { field: "question", headerName: "Question", flex: 1, minWidth: 240 },
    {
      field: "status",
      headerName: "Status",
      width: 110,
      renderCell: ({ row }) => (
        <Chip size="small" variant="outlined" color={statusColor[row.status]} label={row.status} />
      ),
    },
    {
      field: "duration_ms",
      headerName: "Latency",
      width: 90,
      valueFormatter: (value: number) => (value ? `${(value / 1000).toFixed(1)}s` : "—"),
    },
    {
      field: "tokens",
      headerName: "Tokens",
      width: 90,
      valueGetter: (_value, row) => row.tokens_in + row.tokens_out,
    },
    {
      field: "cost_usd",
      headerName: "Cost",
      width: 100,
      valueFormatter: (value: number) => (value ? `$${value.toFixed(4)}` : "$0"),
    },
    {
      field: "claims_verified",
      headerName: "Verified",
      width: 90,
      valueGetter: (_value, row) =>
        row.claims_total ? `${row.claims_verified}/${row.claims_total}` : "—",
    },
    {
      field: "replay_of",
      headerName: "",
      width: 90,
      renderCell: ({ row }) =>
        row.replay_of ? <Chip size="small" variant="outlined" label="replay" /> : null,
    },
    {
      field: "actions",
      type: "actions",
      width: 60,
      getActions: ({ row }) => [
        <GridActionsCellItem
          key="replay"
          icon={<ReplayIcon fontSize="small" />}
          label="Replay offline"
          disabled={row.status === "running" || replay.isPending}
          onClick={() => replay.mutate(row.id)}
        />,
      ],
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Typography variant="h5" fontWeight={700}>
          Runs
        </Typography>
        {replay.isError && <Alert severity="error">{String(replay.error)}</Alert>}
        <DataGrid
          rows={runs.data ?? []}
          columns={columns}
          loading={runs.isLoading}
          density="compact"
          disableRowSelectionOnClick
          onRowClick={({ row }) => navigate(`/runs/${row.id}`)}
          initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
          pageSizeOptions={[25, 50]}
          sx={{ "& .MuiDataGrid-row": { cursor: "pointer" } }}
          autoHeight
        />
      </Stack>
    </Container>
  );
}
```

- [ ] **Step 5: Typecheck**

Run: `cd frontend && npm run typecheck`
Expected: clean exit.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/App.tsx frontend/src/pages/RunsDashboard.tsx
git commit -m "feat(m3): router + nav shell, config chips, runs dashboard (DataGrid)"
```

---

### Task 10: Run view — live agent graph + span inspector + cost meter

One page that works identically for live runs (SSE) and past runs (spans from SQLite): a reactflow graph of the agents, click a node to inspect its spans, cost meter on top.

**Files:**
- Create: `frontend/src/lib/graphModel.ts` (pure events → nodes/edges)
- Create: `frontend/src/components/AgentGraph.tsx`
- Create: `frontend/src/components/SpanInspector.tsx`
- Create: `frontend/src/components/CostMeter.tsx`
- Create: `frontend/src/hooks/useRunEvents.ts`
- Create: `frontend/src/pages/RunView.tsx`
- Modify: `frontend/src/App.tsx` (route)

**Interfaces:**
- Consumes: `RunDetail`/`getRun`/`replayRun` (Task 9), `AgentEvent` payload conventions: planner llm_call carries `payload.plan` (array of `{id, description, method}`), analyst events carry `payload.step_id`, analyst finish is `payload.action === "finish"`, critic handoff carries `payload.retry_steps`.
- Produces: `buildAgentGraph(events: AgentEvent[]): { nodes: AgentNodeModel[]; edges: AgentEdgeModel[] }`; `<CostMeter events={AgentEvent[]} />` (reused by Workbench in Task 11); route `/runs/:runId`.

- [ ] **Step 1: Graph model** — create `frontend/src/lib/graphModel.ts`:

```ts
import type { AgentEvent } from "./types";

export type NodeStatus = "pending" | "active" | "done" | "failed";

export interface AgentNodeModel {
  id: string; // "planner" | "analyst-<step>" | "critic" | "composer"
  agent: string;
  stepId: number | null;
  label: string;
  sublabel: string;
  status: NodeStatus;
  tokens: number;
  costUsd: number;
}

export interface AgentEdgeModel {
  id: string;
  source: string;
  target: string;
  retry: boolean;
}

interface PlanStepPayload {
  id: number;
  description: string;
  method: string;
}

/** Derive the agent graph purely from the event list — works for live SSE and stored spans. */
export function buildAgentGraph(events: AgentEvent[]): {
  nodes: AgentNodeModel[];
  edges: AgentEdgeModel[];
} {
  const planEvent = events.find(
    (e) => e.agent === "planner" && e.type === "llm_call" && Array.isArray(e.payload.plan),
  );
  const steps = (planEvent?.payload.plan ?? []) as PlanStepPayload[];
  const finished = events.some((e) => e.type === "run_finished");

  const rollup = (pred: (e: AgentEvent) => boolean) => {
    const sel = events.filter(pred);
    return {
      any: sel.length > 0,
      error: sel.some((e) => e.type === "error"),
      tokens: sel.reduce((a, e) => a + e.tokens_in + e.tokens_out, 0),
      costUsd: sel.reduce((a, e) => a + e.cost_usd, 0),
    };
  };
  const status = (r: { any: boolean; error: boolean }, done: boolean): NodeStatus =>
    r.error ? "failed" : done ? "done" : r.any ? "active" : "pending";

  const nodes: AgentNodeModel[] = [];

  const planner = rollup((e) => e.agent === "planner");
  nodes.push({
    id: "planner",
    agent: "planner",
    stepId: null,
    label: "planner",
    sublabel: steps.length ? `${steps.length}-step plan` : "",
    status: status(planner, planner.any),
    tokens: planner.tokens,
    costUsd: planner.costUsd,
  });

  for (const step of steps) {
    const r = rollup((e) => e.agent === "analyst" && e.payload.step_id === step.id);
    const done =
      finished ||
      events.some(
        (e) => e.agent === "analyst" && e.payload.step_id === step.id && e.payload.action === "finish",
      );
    nodes.push({
      id: `analyst-${step.id}`,
      agent: "analyst",
      stepId: step.id,
      label: `analyst ${step.id}`,
      sublabel: `${step.method} · ${step.description}`,
      status: status(r, done),
      tokens: r.tokens,
      costUsd: r.costUsd,
    });
  }

  const critic = rollup((e) => e.agent === "critic");
  const verdicts = events.filter((e) => e.type === "verdict").length;
  nodes.push({
    id: "critic",
    agent: "critic",
    stepId: null,
    label: "critic",
    sublabel: verdicts ? `${verdicts} verdicts` : "",
    status: status(critic, verdicts > 0 || finished),
    tokens: critic.tokens,
    costUsd: critic.costUsd,
  });

  const composer = rollup((e) => e.agent === "composer");
  nodes.push({
    id: "composer",
    agent: "composer",
    stepId: null,
    label: "composer",
    sublabel: "",
    status: status(composer, finished),
    tokens: composer.tokens,
    costUsd: composer.costUsd,
  });

  const edges: AgentEdgeModel[] = [];
  for (const step of steps) {
    edges.push({ id: `p-a${step.id}`, source: "planner", target: `analyst-${step.id}`, retry: false });
    edges.push({ id: `a${step.id}-c`, source: `analyst-${step.id}`, target: "critic", retry: false });
  }
  if (steps.length === 0) {
    // honest-failure path: planner straight to composer
    edges.push({ id: "p-comp", source: "planner", target: "composer", retry: false });
  }
  edges.push({ id: "c-comp", source: "critic", target: "composer", retry: false });
  for (const e of events) {
    if (e.agent === "critic" && e.type === "handoff" && Array.isArray(e.payload.retry_steps)) {
      for (const sid of e.payload.retry_steps as number[]) {
        edges.push({ id: `retry-${sid}`, source: "critic", target: `analyst-${sid}`, retry: true });
      }
    }
  }
  return { nodes, edges };
}
```

- [ ] **Step 2: Agent graph component** — create `frontend/src/components/AgentGraph.tsx`:

```tsx
import { Box, Paper, Typography } from "@mui/material";
import {
  Background,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import type { AgentEdgeModel, AgentNodeModel } from "../lib/graphModel";

const STATUS_COLOR: Record<AgentNodeModel["status"], string> = {
  pending: "#3b3f51",
  active: "#7aa2f7",
  done: "#9ece6a",
  failed: "#f7768e",
};

type AgentFlowNode = Node<{ model: AgentNodeModel }, "agent">;

function AgentNode({ data }: NodeProps<AgentFlowNode>) {
  const m = data.model;
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1,
        width: 200,
        borderColor: STATUS_COLOR[m.status],
        borderWidth: 2,
        boxShadow: m.status === "active" ? `0 0 12px ${STATUS_COLOR.active}66` : "none",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <Typography variant="subtitle2" fontWeight={700}>
        {m.label}
      </Typography>
      {m.sublabel && (
        <Typography variant="caption" color="text.secondary" noWrap display="block">
          {m.sublabel}
        </Typography>
      )}
      {(m.tokens > 0 || m.costUsd > 0) && (
        <Typography variant="caption" color="text.secondary">
          {m.tokens.toLocaleString()} tok · ${m.costUsd.toFixed(4)}
        </Typography>
      )}
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </Paper>
  );
}

const nodeTypes = { agent: AgentNode };
const COLUMN_X: Record<string, number> = { planner: 0, analyst: 280, critic: 560, composer: 840 };

export function AgentGraph({
  nodes,
  edges,
  onSelect,
}: {
  nodes: AgentNodeModel[];
  edges: AgentEdgeModel[];
  onSelect: (node: AgentNodeModel) => void;
}) {
  const analystCount = nodes.filter((n) => n.agent === "analyst").length;
  const midY = (Math.max(analystCount, 1) - 1) * 55;

  const flowNodes: AgentFlowNode[] = useMemo(() => {
    let analystIndex = 0;
    return nodes.map((m) => {
      const y = m.agent === "analyst" ? analystIndex++ * 110 : midY;
      return {
        id: m.id,
        type: "agent" as const,
        position: { x: COLUMN_X[m.agent] ?? 0, y },
        data: { model: m },
      };
    });
  }, [nodes, midY]);

  const flowEdges: Edge[] = useMemo(
    () =>
      edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        animated: e.retry,
        style: e.retry
          ? { stroke: STATUS_COLOR.failed, strokeDasharray: "6 4" }
          : { stroke: "#3b3f51" },
      })),
    [edges],
  );

  return (
    <Box sx={{ height: 340, border: 1, borderColor: "divider", borderRadius: 2 }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodeClick={(_evt, node) => onSelect(node.data.model)}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Background gap={24} color="#1c2030" />
      </ReactFlow>
    </Box>
  );
}
```

- [ ] **Step 3: Span inspector + cost meter** — create `frontend/src/components/SpanInspector.tsx`:

```tsx
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import type { AgentNodeModel } from "../lib/graphModel";
import type { AgentEvent } from "../lib/types";

const PRE_KEYS = ["code", "stdout", "stderr", "answer"] as const;

function PayloadView({ payload }: { payload: Record<string, unknown> }) {
  const rest = Object.fromEntries(
    Object.entries(payload).filter(([k, v]) => !PRE_KEYS.includes(k as never) && v !== ""),
  );
  return (
    <Stack spacing={1}>
      {PRE_KEYS.map(
        (key) =>
          typeof payload[key] === "string" &&
          payload[key] !== "" && (
            <Box key={key}>
              <Typography variant="caption" color="text.secondary">
                {key}
              </Typography>
              <Box
                component="pre"
                sx={{
                  m: 0, p: 1, bgcolor: "background.default", borderRadius: 1,
                  fontSize: 12, overflow: "auto", maxHeight: 240,
                }}
              >
                {payload[key] as string}
              </Box>
            </Box>
          ),
      )}
      {Object.keys(rest).length > 0 && (
        <Box
          component="pre"
          sx={{
            m: 0, p: 1, bgcolor: "background.default", borderRadius: 1,
            fontSize: 12, overflow: "auto", maxHeight: 240,
          }}
        >
          {JSON.stringify(rest, null, 2)}
        </Box>
      )}
    </Stack>
  );
}

export function SpanInspector({
  events,
  selection,
}: {
  events: AgentEvent[];
  selection: AgentNodeModel | null;
}) {
  if (!selection) {
    return (
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography color="text.secondary">
          Click a node in the agent graph to inspect its spans.
        </Typography>
      </Paper>
    );
  }
  const spans = events.filter((e) =>
    selection.agent === "analyst"
      ? e.agent === "analyst" && e.payload.step_id === selection.stepId
      : e.agent === selection.agent,
  );
  return (
    <Paper variant="outlined" sx={{ p: 1 }}>
      <Typography variant="subtitle2" sx={{ px: 1, py: 0.5 }}>
        {selection.label} — {spans.length} span{spans.length === 1 ? "" : "s"}
      </Typography>
      {spans.map((e) => (
        <Accordion key={e.span_id} disableGutters variant="outlined" sx={{ bgcolor: "transparent" }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
              <Chip size="small" variant="outlined" label={e.type} />
              <Typography variant="caption" color="text.secondary" noWrap>
                {e.duration_ms}ms
                {e.tokens_in + e.tokens_out > 0 && ` · ${e.tokens_in + e.tokens_out} tok`}
                {e.cost_usd > 0 && ` · $${e.cost_usd.toFixed(4)}`}
              </Typography>
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <PayloadView payload={e.payload} />
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );
}
```

Create `frontend/src/components/CostMeter.tsx`:

```tsx
import { Chip, Paper, Stack, Typography } from "@mui/material";
import type { AgentEvent } from "../lib/types";

/** Per-run cost rollup with per-agent breakdown, derived live from events. */
export function CostMeter({ events }: { events: AgentEvent[] }) {
  const total = events.reduce((a, e) => a + e.cost_usd, 0);
  const tokens = events.reduce((a, e) => a + e.tokens_in + e.tokens_out, 0);
  const byAgent = new Map<string, number>();
  for (const e of events) {
    if (e.cost_usd > 0) byAgent.set(e.agent, (byAgent.get(e.agent) ?? 0) + e.cost_usd);
  }
  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="subtitle2">
          ${total.toFixed(4)} · {tokens.toLocaleString()} tokens
        </Typography>
        {[...byAgent.entries()].map(([agent, cost]) => (
          <Chip key={agent} size="small" variant="outlined" label={`${agent} $${cost.toFixed(4)}`} />
        ))}
        {total === 0 && events.length > 0 && (
          <Typography variant="caption" color="text.secondary">
            free run (stub or replay)
          </Typography>
        )}
      </Stack>
    </Paper>
  );
}
```

- [ ] **Step 4: Events hook + page** — create `frontend/src/hooks/useRunEvents.ts`:

```ts
import { useCallback, useEffect, useState } from "react";
import { getRun } from "../lib/api";
import type { AgentEvent, RunDetail } from "../lib/types";

const EVENT_TYPES = [
  "run_started",
  "llm_call",
  "tool_call",
  "handoff",
  "verdict",
  "answer_chunk",
  "run_finished",
  "error",
];

/**
 * One event source for the run view: persisted spans first, then — while the
 * run is live — the SSE stream, deduped by span_id (the bus replays history).
 */
export function useRunEvents(runId: string | undefined) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);

  const refresh = useCallback(async () => {
    if (!runId) return;
    const detail = await getRun(runId);
    setRun(detail);
    setEvents(detail.spans);
  }, [runId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!runId || run?.status !== "running") return;
    const source = new EventSource(`/api/runs/${runId}/events`);
    const onMessage = (e: MessageEvent) => {
      const event = JSON.parse(e.data) as AgentEvent;
      setEvents((prev) =>
        prev.some((p) => p.span_id === event.span_id)
          ? prev.map((p) => (p.span_id === event.span_id ? event : p))
          : [...prev, event],
      );
      if (event.type === "run_finished" || event.type === "error") {
        source.close();
        void refresh(); // pick up final status/result from the store
      }
    };
    EVENT_TYPES.forEach((t) => source.addEventListener(t, onMessage));
    source.onerror = () => source.close();
    return () => source.close();
  }, [runId, run?.status, refresh]);

  return { run, events };
}
```

Create `frontend/src/pages/RunView.tsx`:

```tsx
import ReplayIcon from "@mui/icons-material/Replay";
import { Alert, Box, Button, Chip, Container, Stack, Typography } from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useNavigate, useParams, Link as RouterLink } from "react-router-dom";
import { AgentGraph } from "../components/AgentGraph";
import { AnswerPanel } from "../components/AnswerPanel";
import { CostMeter } from "../components/CostMeter";
import { SpanInspector } from "../components/SpanInspector";
import { useRunEvents } from "../hooks/useRunEvents";
import { replayRun } from "../lib/api";
import { buildAgentGraph, type AgentNodeModel } from "../lib/graphModel";

const statusColor = { running: "info", finished: "success", error: "error" } as const;

export function RunView() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { run, events } = useRunEvents(runId);
  const [selection, setSelection] = useState<AgentNodeModel | null>(null);
  const { nodes, edges } = useMemo(() => buildAgentGraph(events), [events]);
  const replay = useMutation({
    mutationFn: replayRun,
    onSuccess: ({ run_id }) => navigate(`/runs/${run_id}`),
  });

  if (!run) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography color="text.secondary">loading run…</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
          <Typography variant="h6" fontWeight={700} sx={{ flexGrow: 1, minWidth: 200 }}>
            {run.question}
          </Typography>
          <Chip size="small" variant="outlined" color={statusColor[run.status]} label={run.status} />
          {run.replay_of && (
            <Chip
              size="small"
              variant="outlined"
              label={`replay of ${run.replay_of}`}
              component={RouterLink}
              to={`/runs/${run.replay_of}`}
              clickable
            />
          )}
          <Button
            size="small"
            startIcon={<ReplayIcon />}
            disabled={run.status === "running" || replay.isPending}
            onClick={() => runId && replay.mutate(runId)}
          >
            Replay offline
          </Button>
        </Stack>
        {replay.isError && <Alert severity="error">{String(replay.error)}</Alert>}

        <CostMeter events={events} />
        <AgentGraph nodes={nodes} edges={edges} onSelect={setSelection} />

        <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="flex-start">
          <Box sx={{ flex: 1, minWidth: 0, width: "100%" }}>
            <SpanInspector events={events} selection={selection} />
          </Box>
          <Box sx={{ flex: 1, minWidth: 0, width: "100%" }}>
            {run.result && <AnswerPanel final={run.result} />}
          </Box>
        </Stack>
      </Stack>
    </Container>
  );
}
```

In `frontend/src/App.tsx`, add the route and import:

```tsx
import { RunView } from "./pages/RunView";
```

```tsx
            <Route path="/runs/:runId" element={<RunView />} />
```

- [ ] **Step 5: Typecheck + visual check**

Run: `cd frontend && npm run typecheck`
Expected: clean. (`AnswerPanel`'s prop is `final: FinalAnswer` — matches the usage above.)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/graphModel.ts frontend/src/components/AgentGraph.tsx frontend/src/components/SpanInspector.tsx frontend/src/components/CostMeter.tsx frontend/src/hooks/useRunEvents.ts frontend/src/pages/RunView.tsx frontend/src/App.tsx
git commit -m "feat(m3): run view — live agent graph, span inspector, cost meter"
```

---

### Task 11: Workbench wiring + milestone checkoff

Surface the live cost meter on the Workbench, link every started run to its run view, and check off M3.

**Files:**
- Modify: `frontend/src/pages/Workbench.tsx`
- Modify: `MILESTONES.md`

**Interfaces:**
- Consumes: `CostMeter` (Task 10), route `/runs/:runId` (Task 10).

- [ ] **Step 1: Wire the Workbench**

In `frontend/src/pages/Workbench.tsx`:

```tsx
import { Link as RouterLink } from "react-router-dom";
import { CostMeter } from "../components/CostMeter";
```

Inside the `status !== "idle"` block, change the header row and add the meter:

```tsx
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Run
              </Typography>
              {runId && (
                <Button size="small" component={RouterLink} to={`/runs/${runId}`}>
                  Open run view →
                </Button>
              )}
            </Stack>
            <CostMeter events={events} />
```

(replacing the bare `<Typography variant="h6">Run</Typography>`).

- [ ] **Step 2: Typecheck + full backend suite one last time**

Run: `cd frontend && npm run typecheck && cd ../backend && .venv/bin/pytest -q && .venv/bin/ruff check .`
Expected: everything green.

- [ ] **Step 3: Smoke-check the app manually**

Run: `make dev` — upload a sample CSV, ask a question (CHEAP_MODE keeps it at ~cents), watch the graph animate in the run view, click nodes, open `/runs`, hit the replay action on the finished run and confirm the replayed run reaches the same answer at $0 cost. Kill the server after.

- [ ] **Step 4: Check off M3 in MILESTONES.md**

Mark all seven `## M3 — Observability` items `[x]`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Workbench.tsx MILESTONES.md
git commit -m "feat(m3): workbench cost meter + run view link; check off M3"
```
