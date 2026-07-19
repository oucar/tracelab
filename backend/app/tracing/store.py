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
    replay_of TEXT NOT NULL DEFAULT '',       -- run_id this run replays, '' if not a replay
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
CREATE TABLE IF NOT EXISTS recordings (
    run_id TEXT NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,          -- sha256 of the request content (role + messages / code)
    seq INTEGER NOT NULL,       -- per-key ordinal: identical requests replay in order
    kind TEXT NOT NULL,         -- 'llm' | 'sandbox'
    response TEXT NOT NULL,     -- JSON: recorded output + usage
    PRIMARY KEY (run_id, key, seq)
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, started_at);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    git_sha TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    questions_total INTEGER NOT NULL DEFAULT 0,
    tier1_scorable INTEGER NOT NULL DEFAULT 0,
    tier1_passed INTEGER NOT NULL DEFAULT 0,
    judge_avg REAL,
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS eval_results (
    eval_run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    dataset TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    tier1_scorable INTEGER NOT NULL,
    tier1_passed INTEGER NOT NULL,
    tier1_detail TEXT NOT NULL DEFAULT '',
    judge TEXT,
    judge_rationale TEXT NOT NULL DEFAULT '',
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (eval_run_id, question_id)
);
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
        if "replay_of" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN replay_of TEXT NOT NULL DEFAULT ''")

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
    def create_run(self, dataset_id: str, question: str, replay_of: str = "") -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, dataset_id, question, replay_of, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, dataset_id, question, replay_of, time.time()),
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

    # ── eval runs and results ─────────────────────────────────────────────
    def add_eval_run(
        self,
        *,
        id: str,
        created_at: float,
        label: str,
        git_sha: str,
        config_hash: str,
        config_json: str,
        questions_total: int,
        tier1_scorable: int,
        tier1_passed: int,
        judge_avg: float | None,
        cost_usd: float,
        duration_ms: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    id,
                    created_at,
                    label,
                    git_sha,
                    config_hash,
                    config_json,
                    questions_total,
                    tier1_scorable,
                    tier1_passed,
                    judge_avg,
                    cost_usd,
                    duration_ms,
                ),
            )

    def add_eval_result(
        self,
        *,
        eval_run_id: str,
        question_id: str,
        run_id: str,
        dataset: str,
        tags_json: str,
        tier1_scorable: bool,
        tier1_passed: bool,
        tier1_detail: str,
        judge_json: str | None,
        judge_rationale: str,
        cost_usd: float,
        duration_ms: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eval_run_id,
                    question_id,
                    run_id,
                    dataset,
                    tags_json,
                    int(tier1_scorable),
                    int(tier1_passed),
                    tier1_detail,
                    judge_json,
                    judge_rationale,
                    cost_usd,
                    duration_ms,
                ),
            )

    def list_eval_runs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def eval_results(self, eval_run_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE eval_run_id = ? ORDER BY question_id",
                (eval_run_id,),
            ).fetchall()
        return [dict(r) for r in rows]
