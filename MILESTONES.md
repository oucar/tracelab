# tracelab — Milestones (local working checklist, gitignored)

Full plan: `tracelab-plan.md`. Each milestone ends demoable. Check things off as they land.

## M1 — Vertical slice
- [x] Repo scaffold: backend (FastAPI + LangGraph), frontend (Vite + React + MUI), Makefile, README
- [x] `.gitignore`, `.env.example`
- [x] Dataset upload endpoint + profiling (columns, dtypes, nulls, preview)
- [x] Sandbox executor: subprocess, rlimits (CPU/mem/fsize/nproc), timeout, temp workdir, artifact collection
- [x] `AgentEvent` schema + per-run event bus + SSE endpoint
- [x] Minimal graph: analyst (write python → sandbox → interpret, ≤3 iters) → composer
- [x] SQLite trace store (runs + spans)
- [x] Workbench page: upload, profile, ask, live event log, streamed answer
- [x] Unit tests: sandbox limits, event bus, store, graph routing with stubbed LLM
- [x] First real end-to-end run with an OpenAI key (local)

## M2 — Multi-agent + critic
- [x] Planner node with structured plans
- [x] Parallel analysts via `Send` fan-out
- [x] Critic: independent re-derivation, reconciliation tolerances, verdict per claim
- [x] Conditional edges: verified → composer; discrepancy → one bounded retry; honest failure path
- [x] Verified/unverified claim badges in UI
- [x] Budgets (tokens, tool calls) enforced per agent
- [x] Checkpointing on (SqliteSaver)
- [x] First stats methods: mean comparison + correlation, with methodology chips
- [x] ChartSpec (Pydantic + Zod mirror) → MUI X Charts renderer, column validation

## M3 — Observability
- [x] Span persistence complete (tree per run)
- [x] Cost meter: price table, per-call cost, per-agent + per-run rollups
- [x] Run view: live agent graph (reactflow), span inspector
- [x] Runs dashboard (DataGrid) with cost/latency/quality columns
- [x] Deterministic replay from recorded calls
- [x] Daily budget cap + CHEAP_MODE flag
- [x] Remaining stats toolkit: regression w/ diagnostics, clustering + PCA, time series backtest, anomaly detection

## M4 — Evals + adaptive routing
- [ ] 3 golden datasets + 10–15 questions each (incl. stats-tagged)
- [ ] Programmatic scoring tier (numeric/categorical/statistical)
- [ ] LLM judge + 4-dimension rubric (clarity, uncertainty honesty, chart appropriateness, methodological soundness)
- [ ] ~40 hand-labeled answers; calibration report (agreement %, within-1, Cohen's kappa, per-dimension)
- [ ] Regression tracking in SQLite, tagged with git SHA + config hash
- [ ] Evals screen: score-over-time, calibration grid
- [ ] CI: lint + tests + programmatic eval gate on PRs
- [ ] Adaptive routing: complexity router at graph entry — simple questions skip the planner and go straight to a single analyst; plan → fan-out only for multi-step/statistical questions
- [ ] Conditional composer: single verified finding folds composition into the final step; full composer only for multi-finding runs
- [ ] Router decisions visible in the trace (`handoff` event with route + reason) → different graph shapes per question in the run view
- [ ] Tests: router routes trivial/multi-step/statistical questions correctly with stubbed LLMs; eval scores unchanged after routing lands

## M5 — The study + polish
- [ ] Tradeoff study across 3–4 model configs
- [ ] Quality vs cost vs latency chart + written conclusions
- [ ] README finalized (architecture diagram, tradeoffs, calibration table, study chart)
- [ ] Demo GIF + 60-second video from the section-2 storyboard
- [ ] Blog post on blog.ucaronur.com
- [ ] Add tracelab to portfolio grid
