"""SQLite persistence for runs and spans (AgentEvents).

Plain sqlite3 + tiny helpers, no ORM: the schema is two tables and the point
is that you can read this file in one sitting.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.runtime.events import AgentEvent


def utc_midnight() -> float:
    """Start of the current UTC day as a unix timestamp — the budget window."""
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',   -- running | finished | error
    answer TEXT DEFAULT '',
    result TEXT NOT NULL DEFAULT '',          -- JSON FinalAnswer (claims, charts, verdicts)
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    parent_span_id TEXT,
    agent TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    started_at REAL NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    profile TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, started_at);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._conn() as conn:
            self._migrate(conn)
            conn.executescript(_SCHEMA)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive migrations for databases created before a column existed."""
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "runs" not in tables:
            return  # fresh database; _SCHEMA creates everything current
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "result" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN result TEXT NOT NULL DEFAULT ''")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── datasets ──────────────────────────────────────────────────────────
    def add_dataset(self, name: str, path: str, profile: dict) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?)",
                (dataset_id, name, path, json.dumps(profile), time.time()),
            )
        return dataset_id

    def get_dataset(self, dataset_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["profile"] = json.loads(d["profile"])
        return d

    def list_datasets(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["profile"] = json.loads(d["profile"])
            out.append(d)
        return out

    # ── runs ──────────────────────────────────────────────────────────────
    def create_run(self, dataset_id: str, question: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, dataset_id, question, created_at) VALUES (?, ?, ?, ?)",
                (run_id, dataset_id, question, time.time()),
            )
        return run_id

    def finish_run(
        self, run_id: str, answer: str, status: str = "finished", result: str = ""
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, answer = ?, result = ? WHERE id = ?",
                (status, answer, result, run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ── spans ─────────────────────────────────────────────────────────────
    def add_span(self, event: AgentEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.span_id,
                    event.run_id,
                    event.parent_span_id,
                    event.agent,
                    event.type.value,
                    json.dumps(event.payload),
                    event.tokens_in,
                    event.tokens_out,
                    event.cost_usd,
                    event.started_at,
                    event.duration_ms,
                ),
            )

    def spans_for_run(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at", (run_id,)
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def cost_since(self, since_ts: float) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM spans WHERE started_at >= ?",
                (since_ts,),
            ).fetchone()
        return float(row["c"])
