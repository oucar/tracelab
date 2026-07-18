"""tracelab backend — FastAPI app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, runs

app = FastAPI(title="tracelab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # vite dev server; local-only app
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(runs.router)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
