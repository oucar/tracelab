"""tracelab backend — FastAPI app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import config, datasets, evals, runs
from app.deps import store
from app.runtime.events import bus

app = FastAPI(title="tracelab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # vite dev server; local-only app
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(evals.router)

# Live span persistence: every AgentEvent lands in SQLite the moment it is
# emitted, so a crash mid-run still leaves an inspectable partial trace.
bus.add_sink(lambda event: store().add_span(event))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
