# tracelab — Build Plan

**An agentic data analyst you can watch think.**

Upload a CSV, ask questions in plain English. A team of agents plans the analysis, writes and executes Python in a sandbox, and a critic independently verifies every number before you see it. Every run is traced, replayable, costed, and scored by a calibrated LLM judge, with regression tracking across prompt and model versions.

The pitch in one line: most agent demos show you the answer; tracelab shows you the reasoning, the verification, the cost, and the receipts.

---

## 1. Why this project (the portfolio thesis)

Three claims a hiring manager should walk away with:

1. **Deep fluency with the industry-standard stack.** The orchestration is built on LangGraph the way companies actually use it: a typed `StateGraph`, conditional edges as verification gates, parallel fan-out via the `Send` API, and a SQLite checkpointer for durability and time-travel. The README explains not just the graph but what LangGraph is doing under the hood, same read-it-top-to-bottom philosophy as MOOD's `util/ai.ts`.
2. **Treats LLM output as untrusted.** Sandboxed code execution, a critic agent that re-derives results independently, and reconciliation gates before anything renders. This is the "production judgment" signal.
3. **Measures everything.** Per-agent token cost and latency, deterministic replay, a golden-dataset eval harness, an LLM-as-judge calibrated against human labels, and quality vs cost vs latency curves across model configs.

Anti-goals (write these in the README too, restraint is a senior signal):

- Not a general AutoGPT. Agents do one job: analyze tabular data.
- Not a BI tool. No dashboards-as-a-product, no SQL builder. The artifact is the analysis and its trace.
- No vector DB, no RAG. That story lives in MOOD. tracelab is the orchestration + evals story.
- Not a framework tour. LangGraph for orchestration, and that's the only agent framework; the evals, tracing, and replay layers are deliberately custom.

---

## 2. Product surface (what a visitor sees)

Four screens:

1. **Workbench** — drop a CSV (or pick a bundled sample dataset), see an auto-generated profile (columns, types, null counts, preview) plus 3 suggested questions, at least one statistical ("is the weekend fare difference significant?"). Ask anything. Results stream in: narrative answer, tables, charts. Each numeric claim carries a badge: `verified` (critic reconciled it) or `unverified` (with the discrepancy shown). Statistical claims additionally carry a methodology chip (test used, p-value, effect size).
2. **Run view** — the live agent graph. Nodes light up as planner, analysts, and critic execute. Click any node: exact prompt, response, code written, sandbox stdout/stderr, tokens, cost, latency. This screen is the portfolio money shot.
3. **Runs dashboard** — every past run with cost, latency, quality score. Filter by model config. The quality vs cost vs latency scatter/frontier chart lives here.
4. **Evals** — golden dataset results per commit/config, judge calibration stats (agreement with human labels), and the regression timeline: score over time with annotations for prompt/model changes.

Demo flow (this is the storyboard for the README GIF and the 60-second video, since there's no public deploy): pick the sample dataset → click a suggested question → watch the graph execute live → click the critic node and see it re-derive the number → flash the methodology chip on a statistical claim → open Runs and show the cost of what just happened. Record it once, well, at M5; that recording is what 95% of viewers will ever see of the app.

---

## 3. Architecture

```
┌─────────────────────────── frontend (React + Vite + TS) ───────────────────────────┐
│  Workbench        Run view (agent graph)        Runs dashboard        Evals        │
│        └──────────────── SSE stream of AgentEvents ────────────────┘              │
└────────────────────────────────────────┬───────────────────────────────────────────┘
                                         │ REST + SSE
┌────────────────────────────────────────┴───────────────────────────────────────────┐
│                              backend (FastAPI, Python 3.12)                        │
│                                                                                    │
│   api/            routes: upload, runs, ask (SSE), replay, evals                   │
│   runtime/        the hand-rolled engine: loop, tools, handoffs, events            │
│   agents/         planner, analyst, critic, judge (prompts + configs)              │
│   sandbox/        subprocess executor: no network, CPU/mem/time limits             │
│   tracing/        spans, cost meter, SQLite persistence, replay                    │
│   evals/          golden datasets, harness, judge calibration, regressions         │
└────────────────────────────────────────┬───────────────────────────────────────────┘
                                         │
                              SQLite (runs, spans, evals)
                              ./data/    (uploaded + sample CSVs)
                              OpenAI API (only external dependency)
```

Deployment: none, and that's deliberate. tracelab is local-first: `docker compose up` (or `make dev`) with your own `OPENAI_API_KEY`. A public demo of an app that executes LLM-written code and bills per question is a cost and abuse liability with near-zero hiring upside; recruiters evaluate the README, the GIF, and the demo video, not a URL. The README says exactly this in one sentence, which turns "no deploy" into a judgment call instead of a gap.

### 3.1 The graph (the crown jewel)

Built on **LangGraph** (`langgraph` + `langchain-openai`), in `runtime/graph.py` + `runtime/state.py`. Core concepts:

- **Typed state** — a Pydantic `RunState`: question, dataset profile, plan, per-step analyst results, critic verdicts, retry counters, budgets. Reducers (e.g. `operator.add` on analyst results) let parallel branches merge safely.
- **Nodes** — `planner`, `analyst`, `critic`, `composer`, each a plain function over `RunState`. Analysts internally run a small tool loop (write python → sandbox exec → interpret, ≤3 iterations) using native tool calling with structured output.
- **Edges** — the orchestration logic lives in conditional edges: planner fans out one `analyst` branch per independent plan step via the **`Send` API**; a conditional edge after `critic` routes to `composer` (all verified), one bounded retry (discrepancy, with the critic's findings injected), or honest-failure composition.
- **Checkpointing** — `SqliteSaver` checkpoints state at every superstep. This buys durability (a crashed run resumes from its last checkpoint), time-travel debugging, and the backbone of replay. Interviewers who use LangGraph will recognize this immediately.
- **`AgentEvent`** — the single event type everything hangs off: `run_id, span_id, parent_span_id, agent, type (llm_call | tool_call | handoff | verdict | error), payload, tokens_in/out, cost_usd, started_at, duration_ms`. Produced by mapping LangGraph's `astream_events` stream (plus sandbox callbacks) into this one shape. The SSE stream, the trace store, the replay engine, and the live graph UI all consume it. Design this first; it is the load-bearing decision of the codebase.

**Adaptive routing (lands in M4, after the full pipeline exists in M2).** A fixed planner → fan-out → critic → composer pipeline is overkill for "what's the average fare?" — two ceremonial LLM calls, extra latency and cost, zero value. So the graph scales with the question instead of running at fixed size:

- A cheap **complexity router** at graph entry (mini model, one call, structured output: `simple | multi_step | statistical`) decides the shape. Simple questions skip the planner entirely and go straight to one analyst; the planner and `Send` fan-out only run for multi-step or statistical questions. The critic always runs — verification is never optional.
- The **composer is conditional** too: a run with a single verified finding folds composition into the final step; the full composer call only happens when there are multiple findings to synthesize (or a failure to explain honestly).
- Router decisions are emitted as `handoff` events with the chosen route and reason, so the run view literally shows different graph shapes for different questions — a strong demo moment, and the README line writes itself: "the orchestration scales with the question; most questions don't need orchestration."

Why LangGraph (put this reasoning in the README): it is the de facto industry standard for agent orchestration, and using it well is a hiring signal in itself. The senior differentiation moves up the stack: verification gates, budgets, evals, and observability are all custom. The README's runtime section walks through the compiled graph and explains what LangGraph does under the hood (Pregel-style supersteps, channels, checkpoint semantics), so the framework reads as a deliberate choice, not a crutch.

Orchestration shape for a question:

```
question ──> PLANNER  (decomposes into 1..N analysis steps, structured plan)
                │
                ▼  (parallel where steps are independent)
             ANALYST(s)  (each: write python → sandbox exec → interpret → repeat ≤3)
                │
                ▼
             CRITIC  (re-derives key numbers with independently written code,
                      reconciles within tolerance, verdict per claim)
                │
        ┌───────┴────────┐
   all verified      discrepancy
        │                 │
        ▼                 ▼
     COMPOSER        one bounded retry with critic's findings injected,
     (final answer)  else ship with explicit "unverified" flags
```

The critic never sees the analyst's code, only the question and the dataset. Independent derivation is the point; that design decision goes in the README.

**Failure handling (build in from day one):** every agent has a hard budget; sandbox timeouts kill the process group; malformed structured output gets one repair retry; an analyst that fails 3 times returns a typed failure the composer must surface honestly ("could not compute X because...") rather than hallucinating. Honest failure is a feature to demo, not a bug to hide.

### 3.2 Sandbox

`sandbox/executor.py`: each execution is a fresh subprocess with

- no network (block via `unshare`/no-net where available; also strip proxies and document the layers)
- resource limits via `resource.setrlimit`: CPU seconds, memory, file size, process count
- a temp working dir containing only the dataset copy; nothing else readable/writable
- stdout/stderr captured and truncated; artifacts (chart PNGs, result JSON, ChartSpecs) collected from a designated output dir
- allowlisted imports documented (pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn); enforcement is best-effort at this isolation level

README gets an honest tradeoffs section: subprocess isolation is right-sized for a personal demo; production would use gVisor/Firecracker/containers-per-execution. Naming the next step instead of overclaiming is exactly the senior signal we want.

### 3.2b The statistics & ML toolkit

The sandbox ships scipy, statsmodels, and scikit-learn, and the planner knows a menu of methods it can route to. This is what elevates tracelab from "GPT wrapper over pandas" to "data analyst":

- **Descriptive + distributions** — summaries, histograms, box plots, outlier flags (IQR).
- **Correlation & association** — correlation matrix with significance, categorical association (chi-square, Cramér's V).
- **Hypothesis testing** — t-test / Mann-Whitney / ANOVA / chi-square chosen by the planner based on data types; always reported with effect size, not just p-value.
- **Regression** — OLS with diagnostics (residual plots, R², condition checks), logistic for binary targets.
- **Clustering & structure** — k-means with silhouette score, PCA projection to 2D rendered as a scatter ChartSpec.
- **Time series** — trend/seasonality decomposition and a simple backtested forecast (Holt-Winters), reported with MAPE from a holdout, never an unvalidated forecast.
- **Anomaly detection** — IQR fences or isolation forest, surfaced as flagged rows in a DataGrid.

The senior signal is the rigor gate, not the method list. The critic's job extends beyond numeric reconciliation to **methodology review**: was the test appropriate for the data types and distribution, was sample size sane, are assumptions violated (normality, independence), is the claim's strength proportionate to the effect size? A statistically shaky answer gets flagged exactly like a wrong number. Every statistical claim renders with a methodology chip in the UI: test, n, p-value, effect size, assumptions checked.

Guardrail: this is analysis, not AutoML. No model training pipelines, no hyperparameter search, no "predict my column" as a product. Methods exist to answer questions about the data with appropriate rigor.

### 3.3 Tracing and replay

- Every `AgentEvent` persists to SQLite as a span; a run is a tree of spans.
- **Cost meter:** central price table per model; every LLM call records tokens and computed USD (via LangChain callbacks). Per-agent and per-run rollups.
- **Replay:** two layers. LangGraph's checkpointer already gives state-level time travel (rewind to any superstep, fork from there). On top, every LLM call and sandbox execution is recorded keyed by `(span_path, input_hash)`, so replay mode re-executes the graph resolving calls from the recording: deterministic, free, instant. This gives you (a) a debugger for nondeterministic systems, (b) free UI development against real traces, (c) eval reruns that only re-bill for the judge.
- **LangSmith interop (optional):** a `LANGCHAIN_TRACING_V2` toggle so runs also export to LangSmith. The custom dashboards remain the product; the toggle shows you know the ecosystem tooling too.

### 3.4 Evals (the differentiator, so be precise)

**Golden datasets.** 3 bundled CSVs (pick real public data: e.g. a taxi-trip sample, a retail sales set, a weather set). Per dataset, 10 to 15 questions with human-verified answers in `evals/golden/*.yaml`:

```yaml
- id: taxi-007
  question: "Which pickup hour has the highest average fare?"
  expected:
    kind: numeric            # numeric | categorical | narrative
    value: 5
    tolerance: 0             # exact for categorical/int, pct band for floats
  tags: [aggregation, groupby]
- id: taxi-012
  question: "Is the average fare significantly higher on weekends?"
  expected:
    kind: statistical
    direction: higher          # expected direction of effect
    significant: true          # at alpha = 0.05, human-verified
    method_family: mean-comparison   # t-test or Mann-Whitney both acceptable
  tags: [stats, hypothesis-testing]
```

Golden sets deliberately include statistical questions (tagged `stats`) so the eval measures methodology, not just arithmetic: scoring checks the direction, significance conclusion, and whether an acceptable method family was used.

**Two-tier scoring.** Tier 1 is programmatic: for numeric/categorical questions, extract the claimed value from the structured answer and compare directly. Cheap, exact, covers most of the set. Tier 2 is the LLM judge: for narrative questions and for grading answer *quality* on a rubric, 1 to 5, across four dimensions: clarity, honesty about uncertainty, chart appropriateness, and **methodological soundness** (right test, assumptions acknowledged, effect size reported).

**Judge calibration (say the real thing, do the real thing).** You hand-label ~40 answers against the rubric once. The harness reports judge vs human agreement: exact + within-1 agreement and Cohen's kappa, plus per-rubric-dimension breakdown. If agreement is weak, iterate the judge prompt and re-report. The README shows the calibration table. This single table beats 90% of "LLM-as-judge" portfolio claims, which never check the judge.

**Regression tracking.** `python -m evals run --config configs/gpt4o-mini.yaml` writes a scored result set to SQLite tagged with git SHA + config hash. The Evals screen plots score over time; a CI job runs the programmatic tier on PRs and fails if score drops more than a threshold. (Judge tier stays manual/nightly to control spend.)

**The tradeoff study.** Run the full golden set across 3 or 4 configs (e.g. mini-everywhere / strong-planner+mini-analysts / strong-everywhere) and publish the quality vs cost vs latency chart in the README with a paragraph of conclusions. This is the artifact people will screenshot.

### 3.5 Frontend notes

- Vite + React + TS + **Material UI + MUI X** (no Next.js: there's no SEO story and the API is Python). You already know MUI from two projects, and MUI X is a real hiring-relevant library. **MUI X Charts** renders every visualization in the app: agent-generated charts, the tradeoff frontier, latency/cost timelines, regression charts. **MUI X DataGrid** handles the runs table, eval results, and span listings.
- **Agent charts are typed specs, not code.** Analysts emit a constrained `ChartSpec` (Pydantic on the backend, mirrored Zod schema on the front): `kind (line | bar | scatter | pie | histogram | box) + x/y/series + labels`. The spec is validated against the dataset's actual columns before render (a chart referencing a nonexistent column is rejected the same way a bad number is), then rendered with MUI X Charts. Charts become just another piece of verified structured output, which is the whole thesis. Matplotlib PNGs from the sandbox remain a fallback artifact for exotic plots (e.g. QQ plots), shown alongside native charts.
- Note: a few MUI X chart types (e.g. heatmaps) are in the paid Pro tier. The judge calibration matrix is a small custom grid component instead; everything else fits the community tier.
- State management: **Zustand** for the live run store (the `useRunStream(runId)` hook folds SSE `AgentEvent`s into it; the graph, log panel, and cost ticker all derive from that one store) and **TanStack Query** for plain server state (runs list, eval history, dataset profiles). Live-stream state and fetch-cache state are different problems; using the right tool for each is the point.
- Agent graph: nodes + edges from the span tree. Don't hand-roll layout; use reactflow. Animate state transitions (pending → running → done/failed).

---

## 4. Repo structure

```
tracelab/
├── README.md                  # the product page; see skeleton below
├── docs/
│   ├── architecture.md        # diagrams + the design decisions and tradeoffs
│   └── evals.md               # methodology, calibration results, tradeoff study
├── backend/
│   ├── pyproject.toml         # uv; ruff + pytest configured
│   ├── app/
│   │   ├── main.py
│   │   ├── api/               # upload.py, runs.py, ask.py (SSE), evals.py
│   │   ├── runtime/           # graph.py, state.py, events.py, tools.py
│   │   ├── agents/            # planner.py, analyst.py, critic.py, composer.py, judge.py
│   │   │   └── prompts/       # versioned prompt files, not inline strings
│   │   ├── sandbox/           # executor.py, limits.py
│   │   ├── tracing/           # spans.py, costs.py, store.py, replay.py
│   │   └── evals/             # harness.py, scoring.py, calibration.py, golden/
│   └── tests/                 # graph routing (conditional edges), state reducers,
│                              # sandbox limits, scoring, cost math — the deterministic core
├── frontend/
│   ├── package.json
│   └── src/
│       ├── pages/             # Workbench, RunView, RunsDashboard, Evals
│       ├── components/        # AgentGraph, SpanInspector, ClaimBadge, MethodologyChip,
│       │                      # ChartSpecRenderer (ChartSpec → MUI X Charts),
│       │                      # TradeoffChart, CalibrationGrid, RunsGrid (DataGrid)...
│       ├── hooks/             # useRunStream.ts
│       └── lib/               # api client, event types, chartSpec.ts (Zod mirror)
├── data/samples/              # 3 bundled datasets + attribution
└── .github/workflows/         # lint+test, eval-regression gate
```

---

## 5. Milestones

Ordered so there is a demoable app at the end of every milestone. Estimates assume nights-and-weekends pace.

**M1 — Vertical slice (1 to 2 weekends).**
Upload CSV → profile → a minimal two-node LangGraph graph (single analyst → composer) → sandbox exec → streamed answer in a minimal UI. `AgentEvent` schema and the `astream_events` → SSE pipeline done properly here, even though the UI is ugly. Exit: a stranger can upload a CSV and get a real answer.

**M2 — Multi-agent + critic (1 to 2 weekends).**
Full graph: planner with structured plans, parallel analysts via `Send`, independent critic with reconciliation gates as conditional edges, composer, verified/unverified badges, bounded retry, budgets and honest failure paths, checkpointing on. First two statistical methods wired (mean comparison + correlation) with methodology chips; the rest of the stats toolkit lands incrementally during M3/M4. Exit: a question that requires 2+ analysis steps runs end to end, and a deliberately tricky question shows the critic catching something (a wrong number or an inappropriate test).

**M3 — Observability (1 weekend).**
Span persistence, cost meter, run view with live agent graph + span inspector, runs dashboard, deterministic replay. Exit: click any node in a past run and see everything; replay a run offline.

**M4 — Evals + adaptive routing (2 to 3 weekends, don't rush it).**
Golden datasets and questions, programmatic scoring, LLM judge + rubric, the 40-label calibration set and agreement stats, regression storage + Evals screen, CI gate on the programmatic tier. Plus adaptive routing from section 3.1: the complexity router at graph entry (planner skipped for simple questions), the conditional composer, and router decisions visible in the trace. Landing it alongside evals is deliberate — the golden set proves routing changes don't regress quality, and the cost meter quantifies what the skipped calls save. Exit: the calibration table and the first regression chart exist with real numbers; a simple and a multi-step question visibly produce different graph shapes in the run view, at unchanged eval scores.

**M5 — The study + polish (1 weekend).**
Multi-config tradeoff study, quality vs cost vs latency chart, README with architecture diagram and honest tradeoffs, the README GIF and 60-second demo video (recorded from the storyboard in section 2), blog post on blog.ucaronur.com ("I calibrated my LLM judge" or "Watching agents think: tracing a hand-rolled runtime" are both strong angles). Then add it to the portfolio grid.

Scope guardrails: local-only (no deploy, no auth, no multi-tenancy), CSV only (no Excel/DB connectors), one provider (OpenAI; model-agnostic design proven by config, not by adapters).

---

## 6. README skeleton (write it as you build, not after)

```
# tracelab
An agentic data analyst you can watch think.
[demo GIF: the agent graph executing]
60-sec video · blog post · why there's no live demo (one honest sentence)

## Why
3 short paragraphs: untrusted-LLM thesis, verification-first design, measure everything.

## How it works
Architecture diagram. One paragraph per agent. The critic-independence design decision.

## The orchestration
The compiled LangGraph graph (rendered diagram), typed state, conditional-edge gates,
Send fan-out, checkpointing. A "what LangGraph does under the hood" subsection. Event model.

## Verification
How reconciliation works, tolerance policy, what happens on discrepancy (with screenshot).
The methodology gate: how statistical claims are reviewed (test choice, assumptions,
effect sizes) and the methodology chip UI.

## Evals
Methodology, golden sets, two-tier scoring.
THE JUDGE CALIBRATION TABLE (agreement %, kappa, per-dimension).
THE QUALITY vs COST vs LATENCY CHART + 3 conclusions drawn from it.
Regression chart from CI.

## Tradeoffs & limits
Sandbox isolation level and the production path. In-process orchestration vs durable
queues and when that flips. What I would build next.

## Run locally
Standard quickstart.
```

---

## 7. Risks and pre-decisions

- **Your own API spend during development:** a per-run and daily budget cap enforced in the cost meter (it already meters everything, so the cap is nearly free), plus a `CHEAP_MODE=1` env flag that forces mini models everywhere for day-to-day dev. The eval judge tier runs only on demand. Replay makes UI work free.
- **Sandbox honesty:** even local-only, keep the rlimits and no-net design; it's the portfolio point, and it protects your machine from a runaway `while True`.
- **Critic false alarms:** floats need tolerance policy (relative epsilon per claim kind) or the critic becomes noise. Design the tolerance rules in M2, test them.
- **Judge spend:** judge tier only runs on demand/nightly, and replay means re-scoring doesn't re-run agents.
- **Scope creep:** the moment you're tempted to add SQL connectors or auth, stop and polish the evals screen instead. The evals are the portfolio; connectors are commodity.
```
