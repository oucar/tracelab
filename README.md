# tracelab

**An agentic data analyst you can watch think.**

Upload a CSV and ask questions in plain English. A team of agents, orchestrated with LangGraph, plans the analysis, writes Python, and executes it in a sandbox; a critic independently re-derives every number before you see it. Every run is traced, replayable, and costed, and the whole system is scored by a calibrated LLM judge with regression tracking.

> **Status: M1 (vertical slice) — in progress.** Upload → analyst agent → sandboxed execution → streamed answer works end to end. Planner, critic, observability dashboards, and the eval harness land in M2–M4.

*(demo GIF placeholder — recorded at M5)*

## Why

Most agent demos show you the answer. tracelab is built on the premise that LLM output is untrusted by default: code runs in a sandbox with resource limits, numeric claims must be independently re-derived by a critic agent before they render, and answer quality is measured by an eval harness rather than asserted.

There is intentionally no hosted demo: an app that executes LLM-written code and bills per question is a cost and abuse liability as a public toy. It runs locally in one command with your own API key.

## How it works

```
question ──> analyst (writes python → sandbox → interprets, ≤3 iterations)
                │                        [M2: planner fan-out, parallel analysts,
                ▼                         critic verification gate]
             composer ──> streamed answer over SSE
```

- **Orchestration:** LangGraph `StateGraph` over a typed Pydantic `RunState`. Model calls are injected as dependencies, so the entire graph runs in tests with stubbed LLMs and no API key.
- **Sandbox:** each execution is a fresh subprocess with CPU/memory/file-size rlimits, a timeout with process-group kill, a temp workdir containing only the dataset copy, and no API keys or proxies in its environment.
- **Events:** one type, `AgentEvent`, feeds the SSE stream, the SQLite trace store, and the UI. Late subscribers replay history, then go live.
- **Frontend:** React + Vite + TypeScript, Material UI + MUI X (Charts, DataGrid), Zustand for the live run store, TanStack Query for server state.

## Run locally

Requirements: Python 3.11+, Node 20+.

```bash
make install                      # backend venv + frontend npm install
cp backend/.env.example backend/.env   # add your OPENAI_API_KEY
make dev                          # backend :8000 + frontend :5173
```

Then open http://localhost:5173, upload a CSV, and ask a question.

```bash
make test    # backend unit tests (no API key needed)
make lint    # ruff + tsc
```

## Roadmap

- **M2** — planner, parallel analysts (`Send` fan-out), critic verification gate as conditional edges, checkpointing, first statistical methods with methodology chips, typed ChartSpec → MUI X Charts.
- **M3** — full tracing UI (live agent graph, span inspector), cost meter, deterministic replay.
- **M4** — golden-dataset evals, LLM-as-judge calibrated against human labels, regression tracking with a CI gate.
- **M5** — quality vs cost vs latency study across model configs, demo video.

## Tradeoffs & limits

Subprocess isolation is right-sized for a local personal tool; a multi-tenant product would use gVisor/Firecracker-class isolation per execution. Orchestration is in-process; at scale you would move analyst fan-out onto a durable queue. These are deliberate scope decisions for a portfolio project, not oversights.
