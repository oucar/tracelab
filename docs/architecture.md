# Architecture

This doc goes deeper than the README: the full system diagram, why
`AgentEvent` is the load-bearing type of the codebase, the two independent
layers that make runs replayable, and the design decisions behind the
router, the sandbox, and the budget model. Code references point at the
actual files, not paraphrases — if this doc and the code disagree, the code
is right and this doc is stale.

## 1. System overview

```
┌───────────────────────────────── frontend (React + Vite + TS) ─────────────────────────────────┐
│                                                                                                  │
│   Workbench ──ask──► useRunStream(runId) [Zustand]  ◄──SSE── AgentGraph / SpanInspector          │
│   RunView   ──click node──► SpanInspector (prompt, response, code, stdout/stderr, cost)          │
│   RunsDashboard ──TanStack Query──► RunsGrid (DataGrid), CostMeter                                │
│   EvalsScreen ──TanStack Query──► TradeoffChart, CalibrationGrid, score-over-time LineChart       │
└───────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                             │ REST + SSE (GET /runs/{id}/events)
┌────────────────────────────────────────────┴───────────────────────────────────────────────────┐
│                               backend (FastAPI, Python 3.12)                                     │
│                                                                                                    │
│  api/        upload.py · runs.py (create, list, replay) · ask.py (SSE) · evals.py                │
│  runtime/    graph.py (nodes + edges) · state.py (RunState) · events.py (AgentEvent, EventBus)    │
│              reconcile.py (tolerance policy) · budget.py (per-agent hard caps)                     │
│              recording.py (record/replay of the nondeterministic boundary) · chartspec.py         │
│  agents/     llm.py (GraphDeps, model wiring) · schemas.py (structured-output types) · prompts/   │
│  sandbox/    executor.py (subprocess isolation + rlimits)                                          │
│  tracing/    store.py (SQLite persistence) · pricing.py (cost table)                               │
│  evals/      golden.py + derivations.py (self-verifying golden set) · harness.py (run_eval)       │
│              scoring.py (tier 1) · judge.py (tier 2) · calibration.py · study.py · report.py       │
└────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                             │
                              SQLite (backend/data/tracelab.sqlite3 — runs, spans,
                                       datasets, recordings, eval_runs, eval_results)
                              SQLite (backend/data/checkpoints.sqlite3 — LangGraph SqliteSaver)
                              ./data/samples (bundled CSVs), ./data/uploads (user CSVs)
                              OpenAI API (the only external dependency)
```

Two separate SQLite files matter: `tracelab.sqlite3` is the application's own
store (spans, eval results, recordings — `app/tracing/store.py`), written
directly by app code. `checkpoints.sqlite3` is LangGraph's own
`SqliteSaver` database, written by the framework itself and never touched
directly. They serve different jobs (§5) and conflating them would be a
design mistake, not a simplification.

## 2. The graph, node by node

`backend/app/runtime/graph.py` builds one `StateGraph(RunState)`:

```python
g.add_node("router", ...)
g.add_node("planner", ...)
g.add_node("analyst", ..., input_schema=AnalystTask)   # a Send target, not a normal node
g.add_node("critic", ...)
g.add_node("composer", ...)
g.add_edge(START, "router")
g.add_conditional_edges("router", route_from_router, ["planner", "analyst"])
g.add_conditional_edges("planner", fan_out, ["analyst", "composer"])
g.add_edge("analyst", "critic")
g.add_conditional_edges("critic", route_after_critic, ["analyst", "composer"])
g.add_edge("composer", END)
```

Note `analyst` is declared with `input_schema=AnalystTask` rather than
`RunState` — it's a **`Send` target**, so it never sees the whole run state,
only the one `AnalystTask` payload (question, dataset path/profile, its
single `PlanStep`, and optional `critic_feedback` on a retry). This is
deliberate isolation: an analyst branch cannot read another branch's partial
results, which is what makes true parallelism safe.

### 2.1 The router (M4)

`router_node` makes exactly one structured-output call to a pinned mini
model (`router_model`, always `gpt-4o-mini` regardless of `cheap_mode` —
see `app/config.py:model_for`) classifying the question `simple |
multi_step | statistical`. On `simple`, it synthesizes a single-step `Plan`
itself (`PlanStep(id=1, description=question, method="descriptive")`) so the
rest of the graph doesn't need a special case — `route_from_router` sends
straight to one `analyst` branch with that step, skipping the planner call
entirely. On anything else, control passes to `planner`.

Why this exists: a fixed planner → fan-out → critic → composer pipeline is
overkill for "what's the average fare?" — two ceremonial LLM calls, extra
latency and cost, zero value. The critic still always runs; verification is
never optional, only the planning and composition ceremony scales down.

If `deps.router` is `None` (a stub graph in a test, or `GraphDeps`
constructed without a router), the router degrades to `route="multi_step"`
— the pre-M4 behavior — rather than crashing. A failed router call (budget
exceeded, malformed output) degrades the same way, logged as an `error`
event, not a hard stop.

### 2.2 The planner

Structured output (`PlannerTurn`: `steps: list[PlanStep]`, `rationale:
str`), capped at `max_plan_steps` (4). Plan steps get sequential `id`s
assigned after generation so downstream code (claim IDs, retry targeting)
can address them. An empty or malformed plan sets `planner_failed=True`
rather than raising — the composer is required to surface this honestly
("Planning FAILED: ...") instead of hallucinating an answer.

### 2.3 The analyst (Send target + retry target)

Runs a bounded tool loop (`max_analyst_iterations`, default 3): the model
either emits Python to run or `action: "finish"` with a `findings` narrative
plus structured `Claim`s. Each `Claim` gets an id of `"{step_id}-{n}"`. Chart
artifacts written by the sandboxed script are validated against the
dataset's real columns (`chartspec.py:extract_chart_specs`) — a chart
referencing a column that doesn't exist is rejected the same way a bad
number would be, not silently rendered.

On a critic-triggered retry, the *same* node runs again with
`critic_feedback` appended to its message history — there's no separate
"retry node." `_latest_results` in `graph.py` picks the newest
`AnalystResult` per `step_id` when downstream nodes read `analyst_results`, so
a superseded first attempt doesn't leak into the critic's or composer's
view.

### 2.4 The critic (the reconciliation gate)

Builds its context from `state.question` + `state.dataset_profile` + the
claims JSON — **never** the analyst's code (§4). Runs its own bounded tool
loop against the sandbox to re-derive values, then hands its raw
`CriticFinding`s to `reconcile_claims()` (`reconcile.py`), a plain
deterministic function — the LLM proposes a value, code decides whether it
matches. `route_after_critic` computes per-step retry feedback from the
verdicts and sends `Send("analyst", ...)` only for the disputed steps,
carrying the critic's specific reasoning as `critic_feedback` — not the
whole verdict list, just what's relevant to that step.

### 2.5 The composer

Two paths, chosen by `state.route == "simple" and len(latest) == 1 and not
state.planner_failed` plus every verdict being `verified`: the **folded**
path (no extra LLM call — the single finding *is* the answer) and the
**full synthesis** path (one call, given every step's findings/failure plus
each claim's verification status, asked to write an honest narrative,
including saying "could not compute X because..." when steps failed).
Folding is scoped tightly to the simple route on purpose — a planner-derived
single-step plan on the `multi_step`/`statistical` routes still goes through
the full composer call, because the planner chose to decompose for a
reason even if it landed on one step.

## 3. `RunState` and the reducer that makes fan-out safe

```python
class RunState(BaseModel):
    ...
    analyst_results: Annotated[list[AnalystResult], operator.add] = Field(default_factory=list)
    ...
```

`operator.add` as the channel reducer means N parallel `Send("analyst",
...)` branches, each returning `{"analyst_results": [one_result]}`, get
**concatenated** by LangGraph after the superstep, not overwritten. No
manual join, no lock — the type annotation *is* the concurrency contract.
This is the one line in `state.py` that makes the whole parallel-fan-out
design work; miss it and parallel analysts would clobber each other.

## 4. `AgentEvent` — the load-bearing type

Everything downstream of a running graph — the SSE stream to the browser,
the SQLite span store, the replay engine, the live agent-graph UI, and the
cost meter — consumes exactly one shape (`backend/app/runtime/events.py`):

```python
class EventType(str, Enum):
    RUN_STARTED = "run_started"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"     # sandbox executions surface as tool calls
    HANDOFF = "handoff"
    VERDICT = "verdict"
    ANSWER_CHUNK = "answer_chunk"
    RUN_FINISHED = "run_finished"
    ERROR = "error"

class AgentEvent(BaseModel):
    run_id: str
    span_id: str
    parent_span_id: str | None
    agent: str
    type: EventType
    payload: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
    started_at: float
    duration_ms: int
```

Design decisions worth naming:

- **One flat shape for every kind of thing that happens**, distinguished by
  `type` + `payload`, rather than a type hierarchy per event kind. That
  keeps the SSE wire format, the SQLite schema, and the frontend's event
  reducer all trivial — one `switch` on `type`, one table, one Zustand fold.
  The cost is that `payload` is an untyped `dict` on the wire; the frontend
  narrows it per `type` (`useRunEvents.ts`) rather than the backend
  enforcing a payload schema per event type. That's a real tradeoff, made
  deliberately for a small system — a larger one would want a discriminated
  union of payload schemas.
- **Nodes emit directly**, via an inline `_emit(...)` helper in `graph.py`,
  rather than deriving events from LangGraph's `astream_events` stream. The
  build plan originally proposed mapping `astream_events`; the actual
  implementation calls `bus.emit()` inline in every node instead. This gives
  exact control over span parenting (each analyst branch roots its own
  events under its first span, `handoff` events carry the *reason* for a
  routing decision, not just the destination) at the cost of one more line
  of bookkeeping per LLM/tool call. Worth it: parenting is what makes the
  run view render one legible subtree per parallel branch instead of an
  interleaved flat log.
- **`EventBus` keeps full history per run** (`self._history`), so a browser
  tab that opens the run view mid-run still replays everything from the
  start before going live — `subscribe()` yields the backlog, then streams.
  A synchronous `sink` list (`add_sink`) lets the trace store persist spans
  as they happen without the emitter caring who's listening; sinks are
  documented as must-not-raise, because a broken sink shouldn't take a run
  down.
- **Cost is computed at emit time**, not after the fact: `_emit()` calls
  `pricing.cost_usd(usage.model, tokens_in, tokens_out)` inline using a
  hardcoded price table (`tracing/pricing.py` — deliberately not a live
  pricing API fetch: "an observability tool that silently fetches prices is
  harder to trust than one you can read"). Unknown models, including
  `"replay"` (see §5), price at $0 rather than guessing.

## 5. Replay — two independent layers

The build plan describes replay as two layers; the actual implementation
has both, and they solve different problems.

**Layer 1 — LangGraph's checkpointer.** `SqliteSaver` (wired in
`execute_run`, keyed by `thread_id=state.run_id`) persists the full channel
state after every superstep. This is *state-level* time travel: LangGraph
itself can rewind a thread to any checkpoint and resume or fork from there.
It's what a crashed run resumes from, and it's entirely the framework's
machinery — tracelab doesn't implement any of it, just wires the
checkpointer in.

**Layer 2 — recorded-call replay** (`backend/app/runtime/recording.py`,
built on top of the same `GraphDeps` seam tests use for stubs). Every LLM
call and sandbox execution the graph makes gets recorded, keyed by a
`sha256` of its request content:

```python
def request_key(role: str, messages: list[BaseMessage]) -> str:
    content = json.dumps([[m.__class__.__name__, str(m.content)] for m in messages])
    return hashlib.sha256(f"{role}\n{content}".encode()).hexdigest()

def sandbox_key(code: str) -> str:
    return hashlib.sha256(f"sandbox\n{code}".encode()).hexdigest()
```

`recording_deps(inner, recorder)` wraps a real `GraphDeps` so every call
passes through unchanged but is also written to the `recordings` table
(`kind`, `key`, a per-key sequence number, and the raw response). `POST
/runs/{id}/replay` (`app/api/runs.py`) then builds `replay_deps(recordings)`
— a `GraphDeps` that answers every request by popping the next recorded
response for that key, in first-recorded order (a per-key sequence number
disambiguates identical requests, so two parallel analyst branches issuing
the same prompt replay in the order they were originally recorded, not
whichever order they happen to re-run in). Replayed usage is tagged
`model="replay"`, which prices at $0 in the cost table — a replay costs
nothing and touches neither the network nor the sandbox; it runs on a
plane, as the module docstring puts it.

A replayed call with no matching recording raises `ReplayMiss` rather than
silently falling through to a live call — a replay that quietly diverges
from the original run would be worse than one that fails loudly.

**A known gap, stated honestly rather than glossed over:** `recording_deps`
and `replay_deps` wrap `planner`, `analyst_turn`, `critic_turn`, and
`compose`, but not `router`. A replayed `GraphDeps` therefore has
`router=None`, which makes `router_node` fall back to `route="multi_step"`
regardless of what the original run actually routed to. Replay of a
`simple`-routed run today re-derives the same *analyst* recording (the
`Send` target is identical either way) but takes the `multi_step` graph
shape to get there. Worth fixing before replay is used to reproduce a
routing bug specifically; not yet done.

Together the two layers give you: (a) a debugger for a nondeterministic
system — checkpoint-level for "where did this run get to," recording-level
for "what exactly did the model say"; (b) free UI development against real
traces, no API key required; (c) eval reruns that only re-bill for the
judge tier, since the graph execution itself replays from recordings when
you re-score an old run rather than re-running the agents.

**LangSmith interop.** A `LANGCHAIN_TRACING_V2` env toggle is the one place
tracelab defers to ecosystem tooling instead of the custom store — flipped
on, the same graph also exports to LangSmith. The custom dashboards (run
view, cost meter, evals) remain the actual product either way; the toggle
exists to show the ecosystem tool is understood, not to replace what's
built.

## 6. Sandbox: isolation layers

`backend/app/sandbox/executor.py`. Every execution:

1. **Fresh subprocess**, `start_new_session=True` so it owns its own process
   group — a timeout kills the whole group (`os.killpg(..., SIGKILL)`), not
   just the parent, which matters the moment the script itself spawns a
   child (e.g. a library shelling out).
2. **`resource.setrlimit`** for CPU seconds, address space (memory), file
   size, and process count — applied by a small `_PRELUDE` script prepended
   to every submission, each limit wrapped in a best-effort `try/except`
   because platform support is inconsistent (macOS rejects `RLIMIT_AS`
   outright; some sandboxed environments cap `RLIMIT_NPROC` further than
   requested — the code clamps to whatever the OS's hard limit actually
   allows rather than crashing on an unsupported limit).
3. **A throwaway workdir** containing only a fresh copy of the dataset as
   `data.csv`; the subprocess's `HOME` is redirected there too.
4. **No network materials in the environment** — no `HTTP(S)_PROXY`, no API
   keys, `NO_PROXY=*` set explicitly. This blocks proxied traffic and keeps
   secrets out of a compromised script's environment, but it is **not** a
   kernel-level network namespace; a script that opens a raw socket to a
   reachable host is not stopped by this layer alone. That's the honest
   boundary — see the README's Tradeoffs section for the production path
   (gVisor/Firecracker).
5. **Output capture and truncation** (20k chars each of stdout/stderr) and
   artifact collection from a designated `./artifacts` directory — only
   `.json` files are read back (chart specs, structured findings); anything
   else in that directory is ignored, not executed or served.

## 7. Budgets

`backend/app/runtime/budget.py`. Every node instantiates its own
`AgentBudget.for_role(role)` — parallel analyst branches each get an
independent counter, "per agent" meaning per *instance*, not per role
globally. Caps are call-count and token-count based (`max_llm_calls`,
`max_tool_calls`, `max_tokens`), not dollar-based directly — dollars are a
derived rollup from the cost meter. `BudgetExceeded` is caught at each call
site and converted into a typed failure path (an `error` event, an
`AnalystResult(failed=True, ...)` or `planner_failed=True`) that the composer
is required to surface honestly, rather than an unhandled exception taking
the whole run down. Separately, `run_eval`'s daily budget check
(`cost_since(utc_midnight()) >= daily_budget_usd`) is a *sweep-level* guard
against runaway spend across an entire eval run, independent of per-agent
budgets.

## 8. Frontend state: two different problems, two different tools

- **Zustand** owns the live run store: `useRunStream(runId)` folds incoming
  SSE `AgentEvent`s into it, and the agent graph, event log, and cost ticker
  all derive from that one store. This is inherently a streaming,
  append-only, single-source-of-truth problem.
- **TanStack Query** owns everything that's a normal HTTP fetch with
  caching semantics: the runs list, eval history, dataset profiles. Using a
  live-stream store for that would mean hand-rolling cache invalidation;
  using a fetch-cache library for the live event stream would mean fighting
  its request/response model. The split is the point, not an accident of
  having two libraries installed.
